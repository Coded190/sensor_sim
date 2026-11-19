"""Procedural tabletop assets and one room chair.

Meshes are Z-up meters with the resting contact at local z=0. OBJ bytes come
from ``mesh_io.write_obj``. Mitsuba 3.9.1 computes smooth vertex normals when
the OBJ has no ``vn`` lines, so this module does not add a second normal
writer. Packages are published only through ``create_package`` and
``publish_candidate``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import (
    create_package,
    publish_candidate,
    read_index,
    set_candidate_status,
)
from sensor_sim.scene_gen.asset_validate import orient_outward, polygon_normal, validate_mesh
from sensor_sim.scene_gen.indoor_transform import identity_matrix, serialize_matrix
from sensor_sim.scene_gen.mesh_io import write_obj
from sensor_sim.scene_gen.structural import CLEARANCE_TRIANGLE_LIMIT, box_mesh, json_vector, support_quad

logger = logging.getLogger("sensor_sim.scene_gen.tabletop_assets")

GENERATOR = "tabletop"
GENERATOR_VERSION = "1.0.0"
VERSION = "1.0.0"
RESERVED_TOKENS = frozenset({"book", "cup", "mug", "table", "shelf", "crate", "container", "sphere"})
_TOKEN = re.compile(r"[a-z0-9_]+")

# RGB reflectances are the diffuse color. pplastic stores the same numbers as
# diffuse_reflectance. Conductors do not use a reflectance triple.
PALETTE = {
    "prop_bowl": [0.72, 0.70, 0.64],
    "prop_plate": [0.82, 0.82, 0.78],
    "prop_bottle": [0.20, 0.45, 0.62],
    "prop_marker": [0.12, 0.18, 0.55],
    "prop_phone": [0.15, 0.15, 0.16],
    "prop_tape_roll": [0.78, 0.74, 0.52],
    "prop_apple": [0.62, 0.14, 0.12],
    "prop_small_box": [0.62, 0.48, 0.32],
    "prop_chair": [0.40, 0.26, 0.16],
}


def material_for(asset_id: str) -> dict:
    """Return the package BSDF dict for one asset id."""
    if asset_id in {"prop_bowl", "prop_plate", "prop_apple", "prop_small_box", "prop_chair"}:
        return {
            "id": "surface",
            "type": "diffuse",
            "reflectance": list(PALETTE[asset_id if asset_id != "prop_bowl_glossy" else "prop_bowl"]),
            "research_role": "depolarizing_reference",
            "polarization_note": "mitsuba_diffuse_depolarizes",
        }
    if asset_id == "prop_bowl_glossy":
        return _pplastic(PALETTE["prop_bowl"], role="approximate_polarized_plastic")
    if asset_id in {"prop_bottle", "prop_marker", "prop_phone", "prop_tape_roll"}:
        return _pplastic(PALETTE[asset_id], role="approximate_polarized_plastic")
    if asset_id == "prop_can":
        return {
            "id": "surface",
            "type": "conductor",
            "material": "Al",
            "research_role": "smooth_conductor",
            "polarization_note": "mitsuba_conductor_from_tabulated_metal",
        }
    if asset_id == "prop_spoon":
        return {
            "id": "surface",
            "type": "conductor",
            "material": "Cu",
            "research_role": "smooth_conductor",
            "polarization_note": "mitsuba_conductor_from_tabulated_metal",
        }
    if asset_id == "prop_can_rough":
        return {
            "id": "surface",
            "type": "roughconductor",
            "material": "Al",
            "distribution": "beckmann",
            "alpha": 0.1,
            "alpha_meaning": "beckmann_rms_slope",
            "research_role": "rough_conductor",
            "polarization_note": "mitsuba_roughconductor_beckmann",
        }
    raise KeyError(asset_id)


def _pplastic(rgb: list[float], role: str) -> dict:
    return {
        "id": "surface",
        "type": "pplastic",
        "int_ior": 1.5,
        "ext_ior": 1.0,
        "diffuse_reflectance": list(rgb),
        "distribution": "beckmann",
        "alpha": 0.1,
        "alpha_meaning": "beckmann_rms_slope",
        "research_role": role,
        "polarization_note": "mitsuba_pplastic_is_an_approximate_polarized_plastic",
    }


def build_asset(asset_id: str) -> dict:
    """Build one mesh. Variants reuse the base geometry."""
    source = "prop_bowl" if asset_id == "prop_bowl_glossy" else "prop_can" if asset_id == "prop_can_rough" else asset_id
    builders = {
        "prop_bowl": _bowl,
        "prop_plate": _plate,
        "prop_bottle": _bottle,
        "prop_can": _can,
        "prop_spoon": _spoon,
        "prop_marker": _marker,
        "prop_phone": _phone,
        "prop_tape_roll": _tape,
        "prop_apple": _apple,
        "prop_small_box": _box,
        "prop_chair": _chair,
    }
    if source not in builders:
        raise KeyError(asset_id)
    built = builders[source]()
    vertices = np.asarray(built["vertices"], dtype=np.float64)
    if built["profile"] == "multi_component":
        polygons = _orient_components(vertices, built["polygons"], built["components"])
    else:
        polygons = orient_outward(vertices, _consistent_faces(built["polygons"]))
    report = validate_mesh(
        vertices,
        polygons,
        built["profile"],
        components=built.get("components"),
        thickness_axis=built.get("thickness_axis"),
    )
    if not report["passed"]:
        raise RuntimeError(f"{asset_id} failed mesh validation issues={report['issues']}")
    if int(report["triangle_count"]) > CLEARANCE_TRIANGLE_LIMIT:
        raise RuntimeError(f"{asset_id} triangle_count={report['triangle_count']} exceeds {CLEARANCE_TRIANGLE_LIMIT}")
    _require_contact(asset_id, vertices)
    logger.info(
        "tabletop_assets event=mesh_built asset_id=%s profile=%s vertices=%d polygons=%d triangles=%d bounds_min=%s bounds_max=%s",
        asset_id,
        built["profile"],
        len(vertices),
        report["polygon_count"],
        report["triangle_count"],
        [round(value, 4) for value in report["bounds_min"]],
        [round(value, 4) for value in report["bounds_max"]],
    )
    return {
        "vertices": vertices,
        "polygons": polygons,
        "profile": built["profile"],
        "components": built.get("components") or [],
        "support": built["support"],
        "category": built["category"],
        "aliases": built["aliases"],
        "report": report,
        "geometry_source": source,
    }


def package_asset(catalog: Path, asset_id: str, *, geometry_from: str | None = None) -> dict:
    """Create, publish, and mark one candidate technically_validated.

    ``geometry_from`` copies that package's OBJ bytes exactly. The variant
    package hash still differs because the manifest material differs.
    """
    catalog = Path(catalog)
    built = build_asset(asset_id)
    if geometry_from is None:
        render = write_obj(built["vertices"], built["polygons"], name=asset_id)
        collision = write_obj(built["vertices"], built["polygons"], name=f"{asset_id}_collision")
    else:
        source = catalog / geometry_from / VERSION / "geometry"
        render = (source / "render.obj").read_bytes()
        collision = (source / "collision.obj").read_bytes()
    geometry_sha = hashlib.sha256(render).hexdigest()
    _reject_reserved_text(asset_id, built)
    manifest = _manifest(asset_id, built, geometry_sha)
    staging = Path(tempfile.mkdtemp(prefix=".tabletop-", dir=catalog if catalog.exists() else None))
    destination = staging / "package"
    try:
        sealed = create_package(
            destination,
            manifest,
            {"geometry/render.obj": render, "geometry/collision.obj": collision},
            {"geometry/render.obj": "render", "geometry/collision.obj": "collision"},
        )
        publish_candidate(catalog, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    set_candidate_status(catalog, asset_id, VERSION, "technically_validated")
    logger.info(
        "tabletop_assets event=published asset_id=%s package_sha256=%s geometry_sha256=%s status=technically_validated",
        asset_id,
        sealed["package_sha256"],
        geometry_sha,
    )
    return sealed


def package_catalog(catalog: Path) -> list[dict]:
    """Publish every new tabletop asset. Existing index statuses are preserved."""
    catalog = Path(catalog)
    before = _status_map(catalog)
    published = []
    shared = {"prop_bowl_glossy": "prop_bowl", "prop_can_rough": "prop_can"}
    for asset_id in ASSET_IDS:
        published.append(package_asset(catalog, asset_id, geometry_from=shared.get(asset_id)))
    after = _status_map(catalog)
    for key, status in before.items():
        if after.get(key) != status:
            raise RuntimeError(f"existing catalog status changed asset={key} before={status} after={after.get(key)}")
    glossy = _geometry_hash(catalog, "prop_bowl_glossy")
    matte = _geometry_hash(catalog, "prop_bowl")
    rough = _geometry_hash(catalog, "prop_can_rough")
    polished = _geometry_hash(catalog, "prop_can")
    if glossy != matte or rough != polished:
        raise RuntimeError("material variants do not share geometry bytes")
    logger.info("tabletop_assets event=catalog_complete count=%d", len(published))
    return published


ASSET_IDS = (
    "prop_bowl",
    "prop_bowl_glossy",
    "prop_plate",
    "prop_bottle",
    "prop_can",
    "prop_can_rough",
    "prop_spoon",
    "prop_marker",
    "prop_phone",
    "prop_tape_roll",
    "prop_apple",
    "prop_small_box",
    "prop_chair",
)
TABLETOP_IDS = tuple(asset_id for asset_id in ASSET_IDS if asset_id != "prop_chair")
FURNITURE_IDS = ("prop_chair",)


def _manifest(asset_id: str, built: dict, geometry_sha: str) -> dict:
    report = built["report"]
    components = []
    for component in built["components"]:
        entry = {"id": component["id"], "profile": component["profile"]}
        if "thickness_axis" in component:
            entry["thickness_axis"] = json_vector(component["thickness_axis"])
        components.append(entry)
    if not components:
        components = [{"id": built["category"], "profile": built["profile"]}]
    return {
        "asset_id": asset_id,
        "version": VERSION,
        "units": "m",
        "frame": {
            "up_axis": "Z",
            "handedness": "right",
            "transform_convention": "row_major_column_vector",
            "matrix": serialize_matrix(identity_matrix()),
        },
        "provenance": {
            "source": "sensor_sim",
            "generator": GENERATOR,
            "generator_version": GENERATOR_VERSION,
            "seed": 17101,
            "parameters": {"geometry_source": built["geometry_source"]},
            "license": "project-owned procedural geometry",
            "license_status": "permitted",
        },
        "geometry": {
            "profile": built["profile"],
            "units": "m",
            "category": built["category"],
            "aliases": list(built["aliases"]),
            "polygon_count": report["polygon_count"],
            "triangle_count": report["triangle_count"],
            "bounds_min": json_vector(report["bounds_min"]),
            "bounds_max": json_vector(report["bounds_max"]),
            "clearance_triangle_limit": CLEARANCE_TRIANGLE_LIMIT,
            "clearance_from_render_mesh": True,
            "geometry_sha256": geometry_sha,
            "components": components,
        },
        "collision": [
            {
                "id": "proxy",
                "path": "geometry/collision.obj",
                "render_path": "geometry/render.obj",
                "triangle_count": report["triangle_count"],
                "qualified": True,
            }
        ],
        "support_surfaces": built["support"],
        "materials": [material_for(asset_id)],
    }


def _status_map(catalog: Path) -> dict[tuple[str, str], str]:
    index_path = Path(catalog) / "index.json"
    if not index_path.exists():
        return {}
    index = read_index(index_path)
    found = {}
    for asset_id, versions in index.get("assets", {}).items():
        if not isinstance(versions, dict):
            continue
        for version, entry in versions.items():
            if isinstance(entry, dict) and "status" in entry:
                found[(str(asset_id), str(version))] = str(entry["status"])
    return found


def _geometry_hash(catalog: Path, asset_id: str) -> str:
    path = Path(catalog) / asset_id / VERSION / "geometry" / "render.obj"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_reserved_text(asset_id: str, built: dict) -> None:
    parts = [asset_id.replace("_", " "), built["profile"], built["category"], *built["aliases"]]
    parts.extend(component["id"].replace("_", " ") for component in built["components"])
    parts.extend(surface.get("id", "").replace("_", " ") for surface in built["support"])
    tokens = _TOKEN.findall(" ".join(parts).lower())
    found = sorted(set(tokens) & RESERVED_TOKENS)
    if found:
        raise RuntimeError(f"{asset_id} retrieval text contains reserved tokens {found}")


def _require_contact(asset_id: str, vertices: np.ndarray) -> None:
    minimum = float(vertices[:, 2].min())
    if abs(minimum) > 1e-6:
        raise RuntimeError(f"{asset_id} contact z={minimum} is not 0")


def _consistent_faces(polygons: list[list[int]]) -> list[list[int]]:
    """Give neighboring faces opposite directions along every shared edge."""
    oriented = [list(polygon) for polygon in polygons]
    neighbors: list[list[int]] = [[] for _ in polygons]
    owner: dict[tuple[int, int], int] = {}
    for face_index, polygon in enumerate(polygons):
        count = len(polygon)
        for index in range(count):
            first, second = int(polygon[index]), int(polygon[(index + 1) % count])
            key = (first, second) if first < second else (second, first)
            if key in owner:
                other = owner[key]
                neighbors[face_index].append(other)
                neighbors[other].append(face_index)
            else:
                owner[key] = face_index
    seen: set[int] = set()
    stack = [0]
    seen.add(0)
    while stack:
        face_index = stack.pop()
        directed = set()
        polygon = oriented[face_index]
        for index in range(len(polygon)):
            directed.add((int(polygon[index]), int(polygon[(index + 1) % len(polygon)])))
        for other in neighbors[face_index]:
            if other in seen:
                continue
            current = oriented[other]
            same = False
            for index in range(len(current)):
                edge = (int(current[index]), int(current[(index + 1) % len(current)]))
                if edge in directed:
                    same = True
                    break
            if same:
                oriented[other] = list(reversed(current))
            seen.add(other)
            stack.append(other)
    return oriented


def _orient_components(vertices, polygons, components):
    oriented = [list(polygon) for polygon in polygons]
    for component in components:
        indexes = component["polygon_indexes"]
        subset = [polygons[index] for index in indexes]
        fixed = orient_outward(vertices, subset)
        for index, polygon in zip(indexes, fixed):
            oriented[index] = polygon
    return oriented


def _lathe(profile: list[tuple[float, float]], segments: int) -> tuple[np.ndarray, list[list[int]]]:
    rings: list[tuple[int, int]] = []
    vertices: list[list[float]] = []
    for radius, height in profile:
        radius = float(radius)
        height = float(height)
        if radius <= 1e-9:
            rings.append((len(vertices), 1))
            vertices.append([0.0, 0.0, height])
            continue
        start = len(vertices)
        angles = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False)
        for angle in angles:
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), height])
        rings.append((start, int(segments)))
    polygons: list[list[int]] = []
    for (start_a, count_a), (start_b, count_b) in zip(rings, rings[1:]):
        if count_a == 1 and count_b == segments:
            center = start_a
            for index in range(segments):
                nxt = (index + 1) % segments
                polygons.append([center, start_b + index, start_b + nxt])
        elif count_b == 1 and count_a == segments:
            center = start_b
            for index in range(segments):
                nxt = (index + 1) % segments
                polygons.append([center, start_a + nxt, start_a + index])
        elif count_a == segments and count_b == segments:
            for index in range(segments):
                nxt = (index + 1) % segments
                polygons.append([start_a + index, start_a + nxt, start_b + nxt, start_b + index])
        else:
            raise ValueError("lathe rings must be a point or the same segment count")
    return np.asarray(vertices, dtype=np.float64), polygons


def _support_disk(radius: float, z_value: float = 0.0) -> list[dict]:
    extent = float(radius) * 0.7071
    surface = support_quad((-extent, -extent), (extent, extent), z_value, (0.0, 0.0, 1.0))
    surface["id"] = "base"
    return [surface]


def _support_rect(x0: float, y0: float, x1: float, y1: float, z_value: float = 0.0) -> list[dict]:
    surface = support_quad((x0, y0), (x1, y1), z_value, (0.0, 0.0, 1.0))
    surface["id"] = "base"
    return [surface]


def _closed(vertices, polygons, category: str, aliases: list[str], support: list[dict]) -> dict:
    return {
        "vertices": vertices,
        "polygons": polygons,
        "profile": "closed_solid",
        "components": [],
        "support": support,
        "category": category,
        "aliases": aliases,
    }


def _bowl() -> dict:
    profile = [
        (0.0, 0.0),
        (0.040, 0.0),
        (0.052, 0.008),
        (0.055, 0.024),
        (0.055, 0.045),
        (0.051, 0.045),
        (0.048, 0.028),
        (0.040, 0.010),
        (0.0, 0.004),
    ]
    vertices, polygons = _lathe(profile, 32)
    return _closed(vertices, polygons, "bowl", ["bowl", "ceramic", "concave"], _support_disk(0.040))


def _plate() -> dict:
    profile = [
        (0.0, 0.0),
        (0.048, 0.0),
        (0.060, 0.004),
        (0.070, 0.008),
        (0.070, 0.012),
        (0.063, 0.012),
        (0.056, 0.006),
        (0.036, 0.003),
        (0.0, 0.003),
    ]
    vertices, polygons = _lathe(profile, 36)
    return _closed(vertices, polygons, "plate", ["plate", "ceramic", "rim"], _support_disk(0.048))


def _bottle() -> dict:
    profile = [
        (0.0, 0.0),
        (0.022, 0.0),
        (0.028, 0.014),
        (0.028, 0.086),
        (0.022, 0.108),
        (0.013, 0.124),
        (0.012, 0.146),
        (0.016, 0.150),
        (0.016, 0.160),
        (0.0, 0.160),
    ]
    vertices, polygons = _lathe(profile, 28)
    return _closed(vertices, polygons, "bottle", ["bottle", "plastic"], _support_disk(0.022))


def _can() -> dict:
    profile = [
        (0.0, 0.0),
        (0.026, 0.0),
        (0.034, 0.003),
        (0.030, 0.008),
        (0.030, 0.106),
        (0.034, 0.111),
        (0.026, 0.115),
        (0.0, 0.115),
    ]
    vertices, polygons = _lathe(profile, 32)
    return _closed(vertices, polygons, "can", ["can", "metal", "aluminum"], _support_disk(0.026))


def _tape() -> dict:
    major, minor = 0.026, 0.012
    major_count, minor_count = 28, 14
    vertices = []
    for major_index in range(major_count):
        phi = 2.0 * np.pi * major_index / major_count
        for minor_index in range(minor_count):
            theta = 2.0 * np.pi * minor_index / minor_count
            radius = major + minor * np.cos(theta)
            vertices.append([radius * np.cos(phi), radius * np.sin(phi), minor * (1.0 + np.sin(theta))])
    array = np.asarray(vertices, dtype=np.float64)
    array[:, 2] -= float(array[:, 2].min())
    polygons = []
    for major_index in range(major_count):
        major_next = (major_index + 1) % major_count
        for minor_index in range(minor_count):
            minor_next = (minor_index + 1) % minor_count
            a = major_index * minor_count + minor_index
            b = major_next * minor_count + minor_index
            c = major_next * minor_count + minor_next
            d = major_index * minor_count + minor_next
            polygons.append([a, d, c, b])
    return _closed(array, polygons, "tape", ["tape", "roll", "plastic"], _support_disk(major + minor))


def _marker() -> dict:
    profile = [
        (0.0, 0.0),
        (0.008, 0.0),
        (0.008, 0.098),
        (0.0062, 0.104),
        (0.008, 0.110),
        (0.008, 0.130),
        (0.0, 0.130),
    ]
    vertices, polygons = _lathe(profile, 20)
    turned = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=1)
    turned[:, 2] -= float(turned[:, 2].min())
    return _closed(turned, polygons, "marker", ["marker", "plastic"], _support_rect(0.0, -0.008, 0.130, 0.008))


def _box() -> dict:
    vertices, polygons = box_mesh([0.0, 0.0, 0.0], [0.070, 0.045, 0.040])
    return _closed(vertices, polygons, "box", ["box", "cardboard"], _support_rect(0.0, 0.0, 0.070, 0.045))


def _apple() -> dict:
    rings, segments = 18, 28
    contact_theta = 2.45
    vertices: list[list[float]] = []

    def radius(theta: float, phi: float) -> float:
        lobe = 0.012 * np.sin(theta) ** 2 * np.cos(3.0 * phi)
        shoulder = 0.006 * np.cos(2.0 * theta)
        dent = -0.008 * np.exp(-((theta - 0.38) / 0.20) ** 2)
        return max(0.020, 0.040 + lobe + shoulder + dent)

    top = radius(0.0, 0.0)
    vertices.append([0.0, 0.0, top])
    for ring in range(1, rings):
        theta = contact_theta * ring / rings
        for segment in range(segments):
            phi = 2.0 * np.pi * segment / segments
            radial = radius(theta, phi)
            vertices.append([radial * np.sin(theta) * np.cos(phi), radial * np.sin(theta) * np.sin(phi), radial * np.cos(theta)])
    contact_z = float(np.cos(contact_theta) * radius(contact_theta, 0.0))
    vertices.append([0.0, 0.0, contact_z])
    vertices_array = np.asarray(vertices, dtype=np.float64)
    polygons: list[list[int]] = []
    for segment in range(segments):
        nxt = (segment + 1) % segments
        polygons.append([0, 1 + segment, 1 + nxt])
    for ring in range(rings - 2):
        row = 1 + ring * segments
        nxt_row = row + segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            polygons.append([row + segment, nxt_row + segment, nxt_row + nxt, row + nxt])
    last = 1 + (rings - 2) * segments
    south = len(vertices_array) - 1
    for segment in range(segments):
        nxt = (segment + 1) % segments
        polygons.append([south, last + nxt, last + segment])
    vertices_array[:, 2] -= float(vertices_array[:, 2].min())
    return _closed(vertices_array, polygons, "apple", ["apple", "fruit"], _support_disk(0.012))


def _phone() -> dict:
    width, depth, height = 0.145, 0.070, 0.009
    vertices, polygons = _rounded_phone(width, depth, height, 0.008, 0.008, 0.0016, 5)
    return _closed(vertices, polygons, "phone", ["phone", "handset"], _support_rect(0.0, 0.0, width, depth))


def _rounded_phone(width: float, depth: float, height: float, radius: float, inset: float, pocket: float, steps: int):
    outer = _rounded_loop(0.0, 0.0, width, depth, radius, steps)
    inner_radius = max(radius * 0.45, 0.003)
    inner = _rounded_loop(inset, inset, width - inset, depth - inset, inner_radius, steps)
    if len(outer) != len(inner):
        raise RuntimeError(f"phone loops differ outer={len(outer)} inner={len(inner)}")
    count = len(outer)
    vertices = []
    vertices.extend([[point[0], point[1], 0.0] for point in outer])
    vertices.extend([[point[0], point[1], height] for point in outer])
    vertices.extend([[point[0], point[1], height] for point in inner])
    vertices.extend([[point[0], point[1], height - pocket] for point in inner])
    polygons: list[list[int]] = [list(reversed(range(count)))]
    for index in range(count):
        nxt = (index + 1) % count
        polygons.append([index, nxt, count + nxt, count + index])
        polygons.append([count + index, count + nxt, 2 * count + nxt, 2 * count + index])
        polygons.append([2 * count + index, 2 * count + nxt, 3 * count + nxt, 3 * count + index])
    polygons.append(list(range(3 * count, 4 * count)))
    return np.asarray(vertices, dtype=np.float64), polygons


def _rounded_loop(x0: float, y0: float, x1: float, y1: float, radius: float, steps: int) -> list[tuple[float, float]]:
    corners = [
        (x1 - radius, y0 + radius, -0.5 * np.pi, 0.0),
        (x1 - radius, y1 - radius, 0.0, 0.5 * np.pi),
        (x0 + radius, y1 - radius, 0.5 * np.pi, np.pi),
        (x0 + radius, y0 + radius, np.pi, 1.5 * np.pi),
    ]
    points = []
    for center_x, center_y, start, stop in corners:
        for step in range(steps):
            angle = start + (stop - start) * step / steps
            points.append((center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)))
    return points


def _spoon() -> dict:
    length = 0.140
    nx, ny = 34, 18
    xs = np.linspace(0.0, length, nx)
    ys = np.linspace(-0.017, 0.017, ny)
    head_cx, head_rx, head_ry = 0.022, 0.020, 0.016

    def inside(x: float, y: float) -> bool:
        head = ((x - head_cx) / head_rx) ** 2 + (y / head_ry) ** 2 <= 1.0
        handle = head_cx <= x <= length + 1e-9 and abs(y) <= 0.008
        return head or handle

    def top(x: float, y: float) -> float:
        nx_ = (x - head_cx) / head_rx
        ny_ = y / head_ry
        radial = nx_ * nx_ + ny_ * ny_
        if x < 0.050 and radial <= 1.0:
            bowl = 0.009 - 0.006 * (1.0 - min(radial, 1.0))
            blend = float(np.clip((x - 0.030) / 0.018, 0.0, 1.0))
            return (1.0 - blend) * bowl + blend * 0.004
        return 0.004

    mask = np.array([[inside(float(x), float(y)) for x in xs] for y in ys])
    cells = [(i, j) for j in range(ny - 1) for i in range(nx - 1) if mask[j, i] and mask[j, i + 1] and mask[j + 1, i] and mask[j + 1, i + 1]]
    cell_set = set(cells)
    index: dict[tuple[int, int, int], int] = {}
    vertices: list[list[float]] = []

    def vid(i: int, j: int, layer: int) -> int:
        key = (i, j, layer)
        if key not in index:
            index[key] = len(vertices)
            z_value = 0.0 if layer == 0 else top(float(xs[i]), float(ys[j]))
            vertices.append([float(xs[i]), float(ys[j]), z_value])
        return index[key]

    polygons: list[list[int]] = []
    for i, j in cells:
        b00, b10, b11, b01 = vid(i, j, 0), vid(i + 1, j, 0), vid(i + 1, j + 1, 0), vid(i, j + 1, 0)
        t00, t10, t11, t01 = vid(i, j, 1), vid(i + 1, j, 1), vid(i + 1, j + 1, 1), vid(i, j + 1, 1)
        polygons.append([b00, b01, b11, b10])
        polygons.append([t00, t10, t11, t01])
        if (i - 1, j) not in cell_set:
            polygons.append([b00, t00, t01, b01])
        if (i + 1, j) not in cell_set:
            polygons.append([b10, b11, t11, t10])
        if (i, j - 1) not in cell_set:
            polygons.append([b00, b10, t10, t00])
        if (i, j + 1) not in cell_set:
            polygons.append([b01, t01, t11, b11])
    array = np.asarray(vertices, dtype=np.float64)
    polygons = _outward_faces(array, polygons)
    return _closed(array, polygons, "spoon", ["spoon", "utensil", "metal"], _support_rect(0.0, -0.016, length, 0.016))


def _outward_faces(vertices: np.ndarray, polygons: list[list[int]]) -> list[list[int]]:
    center = vertices.mean(axis=0)
    oriented = []
    for polygon in polygons:
        normal = polygon_normal(vertices, polygon)
        point = vertices[int(polygon[0])]
        if float(np.dot(normal, point - center)) < 0.0:
            oriented.append(list(reversed(polygon)))
        else:
            oriented.append(list(polygon))
    return oriented


def _chair() -> dict:
    pieces = [
        ("seat", [0.04, 0.04, 0.42], [0.46, 0.42, 0.46], [0.0, 0.0, 1.0]),
        ("backrest", [0.04, 0.38, 0.46], [0.46, 0.44, 0.88], [0.0, 1.0, 0.0]),
        ("leg_0", [0.06, 0.06, 0.0], [0.10, 0.10, 0.42], [1.0, 0.0, 0.0]),
        ("leg_1", [0.36, 0.06, 0.0], [0.40, 0.10, 0.42], [1.0, 0.0, 0.0]),
        ("leg_2", [0.06, 0.30, 0.0], [0.10, 0.34, 0.42], [1.0, 0.0, 0.0]),
        ("leg_3", [0.36, 0.30, 0.0], [0.40, 0.34, 0.42], [1.0, 0.0, 0.0]),
    ]
    vertices = []
    polygons: list[list[int]] = []
    components = []
    for name, minimum, maximum, axis in pieces:
        local_vertices, local_polygons = box_mesh(minimum, maximum)
        offset = sum(len(block) for block in vertices) if vertices else 0
        start = len(polygons)
        polygons.extend([[index + offset for index in polygon] for polygon in local_polygons])
        vertices.append(local_vertices)
        components.append(
            {
                "id": name,
                "profile": "structural_solid",
                "polygon_indexes": list(range(start, start + len(local_polygons))),
                "thickness_axis": axis,
            }
        )
    stacked = np.vstack(vertices)
    return {
        "vertices": stacked,
        "polygons": polygons,
        "profile": "multi_component",
        "components": components,
        "support": _support_rect(0.06, 0.06, 0.40, 0.34),
        "category": "chair",
        "aliases": ["chair", "seat", "furniture"],
    }

"""Rectangular rooms with real openings, isolated packages, and room replay.

The interior finished floor corner is the origin. ``build_room`` remains the
reference for piece sizes. Its exterior-corner origin is only a comparison shift.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import canonical_json_bytes, create_package, verify_package
from sensor_sim.scene_gen.asset_validate import triangulate, validate_mesh
from sensor_sim.scene_gen.indoor_contact import measure_support, unintended_intersection, world_vertices
from sensor_sim.scene_gen.indoor_mitsuba import _bitmap_names, render_stokes, require_variant, write_rgb_png
from sensor_sim.scene_gen.indoor_transform import (
    apply_points,
    compose,
    deserialize_matrix,
    identity_matrix,
    rotation_matrix,
    serialize_matrix,
    translation_matrix,
)
from sensor_sim.scene_gen.mesh_io import read_obj, write_obj
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES
from sensor_sim.scene_gen.narrow import mesh_clearance
from sensor_sim.scene_gen.room_schema import (
    REFERENCE_SHIFT_NOTE,
    RoomSpecError,
    diversity_specifications,
    reference_specification,
    validate_specification,
)
from sensor_sim.scene_gen.scene_quality import close_table_camera
from sensor_sim.scene_gen.structural import box_mesh, build_room, support_quad
from sensor_sim.scene_gen.tabletop_layout import load_mesh
from sensor_sim.scene_gen.tabletop_render import (
    LIGHT_RADIANCE,
    SCENE_AC_BSDF,
    _ceiling_light,
    _project_point,
    _scene,
    instance_visibility,
    mitsuba_bsdf,
)

logger = logging.getLogger("sensor_sim.scene_gen.room_generator")

ROOT = Path("dataset/qualification/room_diversity")
CATALOG = Path("assets/candidates")
SOURCE_BUNDLE = Path("dataset/qualification/scene_quality_v2/quality_controlled/ed940a5f855d1588")
SOURCE_SEED = 17201
SOURCE_RUN = "ed940a5f855d1588"
VERSION = "1.0.0"
BUNDLE_SCHEMA = "room_bundle_v1"
RENDER_SEED = 17311
SCENE_SPP = 16
FURNITURE_CLEARANCE_M = 0.08
WALL_GAP_M = 0.05
WINDOW_KEEPOUT_M = 0.20
DOOR_CORRIDOR_M = 0.90
DOOR_CORRIDOR_MIN_WIDTH_M = 0.75
SEEDS = {
    ("room_compact", "central"): 17301,
    ("room_compact", "perimeter"): 17302,
    ("room_standard", "central"): 17303,
    ("room_standard", "perimeter"): 17304,
    ("room_spacious", "central"): 17305,
    ("room_spacious", "perimeter"): 17306,
}
DAYLIGHT_RADIANCE = 8.0
CEILING_SIZE_M = (0.80, 0.50)
CEILING_DROP_M = 0.02


def assemble(spec: dict, *, frames: bool = True, leaf: bool = True) -> dict:
    """Return named boxes for one rectangular room. Openings are missing boxes, not painted holes."""
    spec = validate_specification(spec)
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    height = float(spec["interior_height_m"])
    wall = float(spec["wall_thickness_m"])
    floor_t = float(spec["floor_thickness_m"])
    ceiling_t = float(spec["ceiling_thickness_m"])
    pieces = [
        _piece("floor", [-wall, -wall, -floor_t], [width + wall, depth + wall, 0.0], [0.0, 0.0, 1.0]),
        _piece("ceiling", [-wall, -wall, height], [width + wall, depth + wall, height + ceiling_t], [0.0, 0.0, -1.0]),
    ]
    by_wall: dict[str, list[dict]] = {name: [] for name in ("south", "north", "east", "west")}
    for opening in spec["openings"]:
        by_wall[opening["wall"]].append(opening)
    for name in ("south", "north", "east", "west"):
        pieces.extend(_wall_pieces(spec, name, by_wall[name], frames=frames, leaf=leaf))
    _require_no_volume_overlap(pieces)
    groups = _groups(pieces)
    return {"spec": spec, "pieces": pieces, "groups": groups}


def compare_reference() -> dict:
    """Match ``build_room`` piece AABBs after the documented origin shift."""
    spec = reference_specification()
    shift = np.array([spec["wall_thickness_m"], spec["wall_thickness_m"], spec["floor_thickness_m"]], dtype=np.float64)
    built = assemble(spec, frames=False, leaf=False)
    reference = build_room()
    found = {_bounds(piece): piece["id"] for piece in built["pieces"]}
    mismatches = []
    compared = []
    for piece in reference["pieces"]:
        minimum = np.asarray(piece["minimum"], dtype=np.float64)
        maximum = np.asarray(piece["maximum"], dtype=np.float64)
        key = (tuple(np.round(minimum - shift, 6)), tuple(np.round(maximum - shift, 6)))
        # Compare by shifting the new piece onto the reference origin.
        candidate = None
        for new in built["pieces"]:
            moved_min = np.asarray(new["minimum"], dtype=np.float64) + shift
            moved_max = np.asarray(new["maximum"], dtype=np.float64) + shift
            if np.allclose(moved_min, minimum, atol=1e-6) and np.allclose(moved_max, maximum, atol=1e-6):
                candidate = new["id"]
                break
        compared.append({"reference_id": piece["id"], "assembler_id": candidate})
        if candidate is None:
            mismatches.append(piece["id"])
    if len(built["pieces"]) != len(reference["pieces"]):
        mismatches.append(f"count:{len(built['pieces'])}!={len(reference['pieces'])}")
    logger.info("room_generator event=parity mismatches=%d pieces=%d", len(mismatches), len(compared))
    return {"matches": mismatches == [], "mismatches": mismatches, "pairs": compared, "shift_m": shift.tolist(), "note": REFERENCE_SHIFT_NOTE, "unused": found}


def choose_compact_size(catalog: Path = CATALOG) -> dict:
    """Use 3.20 x 3.00 x 2.40 m when both arrangements fit. Otherwise freeze 3.40 x 3.20 x 2.40 m."""
    catalog = Path(catalog)
    tried = []
    for size in ((3.20, 3.00, 2.40), (3.40, 3.20, 2.40)):
        spec = diversity_specifications(size)["room_compact"]
        reports = {name: place_arrangement(spec, name, catalog) for name in ("central", "perimeter")}
        tried.append({"size_m": list(size), "central": reports["central"]["valid"], "perimeter": reports["perimeter"]["valid"]})
        if all(item["valid"] for item in reports.values()):
            logger.info("room_generator event=compact_size width=%.2f depth=%.2f height=%.2f", *size)
            return {"size_m": list(size), "tried": tried, "spec": spec}
    raise RoomSpecError(f"reason=compact_does_not_fit tried={tried}")


def place_arrangement(spec: dict, arrangement: str, catalog: Path = CATALOG) -> dict:
    """Place table, shelf, and chair from the room bounds. Do not nudge a failed pose."""
    spec = validate_specification(spec)
    catalog = Path(catalog)
    table = load_mesh(catalog, "table_rect", VERSION)
    shelf = load_mesh(catalog, "shelf_two", VERSION)
    chair = load_mesh(catalog, "prop_chair", VERSION)
    _require_chair_backrest(chair["vertices"])
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    table_size = _size(table["vertices"])
    if arrangement == "central":
        table_center = np.array([width * 0.5, depth * 0.5], dtype=np.float64)
        shelf_wall = "east"
    elif arrangement == "perimeter":
        table_center = np.array([width * 0.55, depth - WALL_GAP_M - 0.5 * table_size[1]], dtype=np.float64)
        shelf_wall = "west"
    else:
        raise RoomSpecError(f"reason=arrangement value={arrangement}")
    shelf_span = float(_oriented_extents(shelf["vertices"], {"east": 0.5 * np.pi, "west": -0.5 * np.pi}[shelf_wall])[1])
    shelf_along = _shelf_along(spec, shelf_wall, shelf_span)
    table_matrix = _yaw_about_footprint(table["vertices"], 0.0, table_center, 0.0)
    table_world = world_vertices(table["vertices"], table_matrix)
    table_center = table_world[:, :2].mean(axis=0)
    shelf_matrix = _against_wall(shelf["vertices"], shelf_wall, spec, shelf_along)
    shelf_world = world_vertices(shelf["vertices"], shelf_matrix)
    chair_center = _chair_center(table_world, chair["vertices"], arrangement)
    desired = table_center - chair_center
    yaw = _seat_yaw(desired)
    chair_matrix = _yaw_about_footprint(chair["vertices"], yaw, chair_center, 0.0)
    chair_world = world_vertices(chair["vertices"], chair_matrix)
    bodies = {
        "table_1": {"vertices": table_world, "faces": table["faces"], "matrix": table_matrix, "asset_id": "table_rect"},
        "shelf_1": {"vertices": shelf_world, "faces": shelf["faces"], "matrix": shelf_matrix, "asset_id": "shelf_two"},
        "chair_1": {"vertices": chair_world, "faces": chair["faces"], "matrix": chair_matrix, "asset_id": "prop_chair"},
    }
    floor = _floor_support(spec)
    checks = []
    for instance_id, body in bodies.items():
        contact = measure_support(body["vertices"], body["faces"], floor)
        checks.append({"name": f"support:{instance_id}", "passed": bool(contact["passed"]), "gap_m": contact["gap_m"], "reason": contact["reason"]})
    pairs = (("table_1", "chair_1"), ("table_1", "shelf_1"), ("chair_1", "shelf_1"))
    for left_id, right_id in pairs:
        clearance = float(mesh_clearance(bodies[left_id], bodies[right_id])["clearance_m"])
        checks.append({"name": f"clearance:{left_id}:{right_id}", "passed": clearance >= FURNITURE_CLEARANCE_M, "clearance_m": clearance, "limit_m": FURNITURE_CLEARANCE_M})
    shell = _shell_meshes(spec)
    for instance_id, body in bodies.items():
        for wall_id, wall in shell.items():
            if wall_id == "floor":
                continue
            found = unintended_intersection(body, wall)
            checks.append({"name": f"intersection:{instance_id}:{wall_id}", "passed": not found["unintended"], "clearance_m": found["clearance_m"]})
    corridor = _door_corridor(spec)
    for instance_id, body in bodies.items():
        hits = _footprint_hits_rectangle(body["vertices"], body["faces"], corridor)
        checks.append({"name": f"door_corridor:{instance_id}", "passed": not hits, "corridor_m": corridor})
    for opening in spec["openings"]:
        if opening["kind"] != "window":
            continue
        keepout = _keepout_mesh(opening, spec)
        for instance_id, body in bodies.items():
            found = unintended_intersection(body, keepout)
            checks.append({"name": f"window_keepout:{instance_id}:{opening['id']}", "passed": not found["unintended"], "clearance_m": found["clearance_m"]})
    seating = _world_vector(chair_matrix, np.array([0.0, -1.0, 0.0]))
    toward = table_world[:, :3].mean(axis=0) - chair_world[:, :3].mean(axis=0)
    toward = toward / np.linalg.norm(toward)
    facing = float(np.dot(seating, toward))
    checks.append({"name": "chair_faces_table", "passed": facing > 0.9, "dot": facing, "seating_direction": seating.tolist()})
    inside = _inside_interior(bodies, spec)
    checks.append({"name": "inside_interior", "passed": inside})
    failed = [item for item in checks if not item["passed"]]
    return {
        "valid": not failed,
        "arrangement": arrangement,
        "checks": checks,
        "failed": failed,
        "poses": {key: serialize_matrix(value["matrix"]) for key, value in bodies.items()},
        "matrices": {key: value["matrix"] for key, value in bodies.items()},
        "bodies": bodies,
        "seating_direction": seating.tolist(),
        "door_corridor_m": corridor,
    }


def package_profile(spec: dict, catalog: Path) -> dict:
    spec = validate_specification(spec)
    catalog = Path(catalog)
    built = assemble(spec, frames=True, leaf=True)
    written = []
    for group_name, pieces in built["groups"].items():
        asset_id = f"{spec['room_id']}_{group_name}"
        destination = catalog / asset_id / VERSION
        manifest = _package_group(destination, asset_id, pieces, spec, group_name)
        verified = verify_package(destination)
        geometry_hash = hashlib.sha256((destination / "geometry" / "render.obj").read_bytes()).hexdigest()
        evidence = catalog / "evidence" / f"{asset_id}.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        report = _validation_report(pieces)
        evidence.write_text(json.dumps({"passed": report["passed"], "issues": report["issues"], "triangle_count": report["triangle_count"]}, indent=2) + "\n", encoding="utf-8")
        if not report["passed"]:
            raise RoomSpecError(f"reason=geometry asset={asset_id} issues={report['issues']}")
        written.append(
            {
                "profile_id": spec["room_id"],
                "asset_id": asset_id,
                "version": VERSION,
                "path": f"{asset_id}/{VERSION}",
                "package_sha256": verified["package_sha256"],
                "geometry_sha256": geometry_hash,
                "components": [piece["id"] for piece in pieces],
                "geometry_validation": "passed",
                "renderer_validation": "pending",
                "evidence": str(evidence),
            }
        )
        logger.info("room_generator event=package asset_id=%s package_sha256=%s", asset_id, verified["package_sha256"])
    return {"profile_id": spec["room_id"], "packages": written, "spec": spec}


def table_local_poses(source: Path = SOURCE_BUNDLE) -> dict:
    """Object matrices in the source table frame. The source run is fixed."""
    source = Path(source)
    if source.name != SOURCE_RUN:
        raise RoomSpecError(f"reason=source_run path={source}")
    bundle = json.loads((source / "bundle.json").read_text(encoding="utf-8"))
    if int(bundle["task"]["seed"]) != SOURCE_SEED or bundle["task"]["task_id"] != "quality_controlled":
        raise RoomSpecError("reason=source_bundle")
    poses = {item["instance_id"]: deserialize_matrix(item["matrix"]) for item in bundle["resolved"]["poses"]}
    table = poses["table_1"]
    inverse = np.linalg.inv(table)
    local = {}
    for instance_id, matrix in poses.items():
        if not instance_id.startswith("prop_"):
            continue
        local[instance_id] = inverse @ matrix
    return {
        "local": local,
        "bundle_sha256": hashlib.sha256((source / "bundle.json").read_bytes()).hexdigest(),
        "relations": [item for item in bundle["resolved"]["relations"] if item["instances"][0].startswith("prop_")],
        "assets": [item for item in bundle["resolved"]["assets"] if item["instance_id"].startswith("prop_")],
    }


def scene_document(bundle: dict) -> dict:
    return {
        "schema_version": 1,
        "task_id": bundle["task_id"],
        "seed": bundle["seed"],
        "qualification": True,
        "visual_review": "pending",
        "admitted": False,
        "valid": True,
        "instances": bundle["instances"],
        "relations": bundle["relations"],
    }


def replay_room(bundle_dir: Path, catalog: Path) -> dict:
    """Rebuild scene.json from saved records. This does not mesh, retrieve, or place."""
    bundle_dir = Path(bundle_dir)
    catalog = Path(catalog)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (bundle_dir / "bundle.json", bundle_dir / "room_spec.json", bundle_dir / "scene.json")}
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    spec_bytes = (bundle_dir / "room_spec.json").read_bytes()
    if hashlib.sha256(spec_bytes).hexdigest() != bundle["room_spec_sha256"]:
        raise RoomSpecError("reason=spec_hash")
    validate_specification(json.loads(spec_bytes.decode("utf-8")))
    _validate_room_bundle(bundle, catalog)
    document = scene_document(bundle)
    rebuilt = canonical_json_bytes(document)
    if rebuilt != (bundle_dir / "scene.json").read_bytes():
        raise RoomSpecError("reason=replay_mismatch")
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    if after != before:
        raise RoomSpecError("reason=replay_modified_source")
    logger.info("room_generator event=replay task=%s result=ok", bundle["task_id"])
    return {"valid": True, "task_id": bundle["task_id"]}


def build_scenes(specs: dict, room_catalog: Path, asset_catalog: Path = CATALOG, output: Path = ROOT) -> list[dict]:
    output = Path(output)
    source = table_local_poses(SOURCE_BUNDLE)
    written = []
    for spec in specs.values():
        for arrangement in ("central", "perimeter"):
            placed = place_arrangement(spec, arrangement, asset_catalog)
            destination = output / spec["room_id"]
            if not placed["valid"]:
                failed = destination / f"failed_{arrangement}"
                failed.mkdir(parents=True, exist_ok=True)
                (failed / "failure.json").write_text(json.dumps({"failed": placed["failed"], "checks": placed["checks"]}, indent=2) + "\n", encoding="utf-8")
                raise RoomSpecError(f"reason=arrangement_failed profile={spec['room_id']} arrangement={arrangement}")
            bundle = _bundle(spec, arrangement, placed, source, room_catalog, asset_catalog)
            run = hashlib.sha256(canonical_json_bytes({"profile": spec["room_id"], "arrangement": arrangement, "seed": bundle["seed"]})).hexdigest()[:16]
            path = destination / run
            path.mkdir(parents=True, exist_ok=False)
            (path / "room_spec.json").write_bytes(canonical_json_bytes(spec))
            (path / "bundle.json").write_bytes(canonical_json_bytes(bundle))
            (path / "scene.json").write_bytes(canonical_json_bytes(scene_document(bundle)))
            replay_room(path, room_catalog)
            written.append({"path": str(path), "task_id": bundle["task_id"], "seed": bundle["seed"]})
            logger.info("room_generator event=scene task=%s path=%s", bundle["task_id"], path)
    return written


def render_scene(bundle_dir: Path, room_catalog: Path, asset_catalog: Path = CATALOG, *, daylight: bool = False) -> dict:
    bundle_dir = Path(bundle_dir)
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    instances = _render_instances(bundle, room_catalog, asset_catalog)
    output = bundle_dir / ("render_daylight_v1" if daylight else "render_ceiling_v1")
    if (output / "validation.json").exists():
        raise RoomSpecError(f"reason=output_exists path={output}")
    output.mkdir(parents=True)
    views = [("room", bundle["cameras"]["overview"], (512, 384))]
    if not daylight:
        views.append(("tabletop", bundle["cameras"]["tabletop"], (1600, 900)))
    else:
        for index, camera in enumerate(bundle["cameras"].get("window_views") or []):
            views.append((f"window_{index}", camera, (512, 384)))
    images = {}
    support = _table_support_world(asset_catalog, bundle)
    for name, camera, resolution in views:
        scene = _scene(instances, camera, resolution, _room_light(bundle, daylight))
        if daylight:
            for index, emitter in enumerate(_daylight_emitters(bundle)):
                scene[f"daylight_{index}"] = emitter["mitsuba"]
        exr = output / f"{name}.exr"
        png = output / f"{name}.png"
        stokes = render_stokes(scene, exr, SCENE_SPP, RENDER_SEED)
        write_rgb_png(exr, png)
        polygon = support if name == "tabletop" else None
        report = instance_visibility(instances, camera, resolution, support_polygon=polygon, return_image=name == "tabletop")
        image = report.pop("id_image", None)
        report.pop("roi_mask", None)
        if image is not None:
            np.save(output / f"{name}_instance_ids.npy", image.astype(np.int32))
            if list(image.shape) != stokes["shape"][:2]:
                raise RoomSpecError(f"reason=mask_shape image={image.shape} exr={stokes['shape']}")
        images[name] = {"png": str(png), "exr": str(exr), "shape": stokes["shape"], "channel_names": stokes["channel_names"], "dtype": stokes["dtype"], "finite": stokes["finite"]}
    validation = {"task_id": bundle["task_id"], "lighting_id": "ceiling_plus_daylight_v1" if daylight else "ceiling_v1", "images": images, "render_seed": RENDER_SEED, "spp": SCENE_SPP}
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    logger.info("room_generator event=rendered task=%s lighting=%s", bundle["task_id"], validation["lighting_id"])
    return validation


def write_gallery(review_path: Path, root: Path = ROOT) -> None:
    review_path = Path(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Room diversity</title>",
        "<style>body{font-family:sans-serif;margin:2rem;background:#111;color:#eee} a{color:#9cf} img{max-width:480px;background:#222} figure{display:inline-block;margin:1rem;vertical-align:top} .notice{border:1px solid #c90;padding:1rem}</style>",
        "</head><body><h1>Room diversity</h1>",
        "<p class=\"notice\"><strong>These rooms are qualification candidates. Human visual review, admission, and registration are pending.</strong></p>",
    ]
    bundles = []
    for path in sorted(Path(root).glob("*/*/bundle.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        bundle["_dir"] = path.parent
        bundles.append(bundle)
    body.append("<h2>A. Room profiles</h2>")
    seen = set()
    for bundle in bundles:
        if bundle["profile_id"] in seen:
            continue
        seen.add(bundle["profile_id"])
        spec = json.loads((bundle["_dir"] / "room_spec.json").read_text(encoding="utf-8"))
        exterior = spec["exterior_bounds_m"]
        body.append("<p>" + html.escape(
            f"{spec['room_id']} interior {spec['interior_width_m']} x {spec['interior_depth_m']} x {spec['interior_height_m']} m "
            f"wall {spec['wall_thickness_m']} floor {spec['floor_thickness_m']} ceiling {spec['ceiling_thickness_m']} "
            f"exterior min {exterior['minimum_m']} max {exterior['maximum_m']}"
        ) + "</p>")
        for opening in spec["openings"]:
            body.append("<p>" + html.escape(
                f"{opening['id']} wall {opening['wall']} offset {opening['offset_m']} size {opening['width_m']} x {opening['height_m']} sill {opening['sill_m']}"
            ) + "</p>")
    body.append("<h2>B. Furniture arrangements</h2>")
    for bundle in bundles:
        clearance = bundle["collision"]["furniture_clearance_m"]
        body.append("<p>" + html.escape(
            f"{bundle['task_id']} seed {bundle['seed']} corridor {bundle['collision']['door_corridor_m']} clearance {clearance}"
        ) + "</p>")
    body.append("<h2>C. Visual renders</h2>")
    for bundle in bundles:
        ceiling = bundle["_dir"] / "render_ceiling_v1" / "validation.json"
        if ceiling.is_file():
            _gallery_validation(body, review_path.parent, json.loads(ceiling.read_text(encoding="utf-8")))
        daylight = bundle["_dir"] / "render_daylight_v1" / "validation.json"
        if daylight.is_file() and bundle["arrangement"] == "central":
            body.append("<h3>" + html.escape(bundle["profile_id"] + " daylight") + "</h3>")
            _gallery_validation(body, review_path.parent, json.loads(daylight.read_text(encoding="utf-8")))
    body.append("<h2>D. Architecture diagnostics</h2>")
    catalog = json.loads((Path(root) / "catalog" / "room_catalog.json").read_text(encoding="utf-8"))
    for package in catalog["packages"]:
        body.append("<p>" + html.escape(
            f"{package['asset_id']} {package['package_sha256']} geometry {package['geometry_sha256']} {package['geometry_validation']} {package['renderer_validation']}"
        ) + "</p>")
    body.append("</body></html>\n")
    review_path.write_text("\n".join(body), encoding="utf-8")


def _piece(name, minimum, maximum, axis) -> dict:
    vertices, polygons = box_mesh(minimum, maximum)
    return {
        "id": name,
        "vertices": vertices,
        "polygons": polygons,
        "thickness_axis": [float(value) for value in axis],
        "minimum": [float(value) for value in minimum],
        "maximum": [float(value) for value in maximum],
        "wall": name.split("_", 1)[0] if name not in {"floor", "ceiling"} else name,
    }


def _wall_pieces(spec, wall: str, openings: list[dict], *, frames: bool, leaf: bool) -> list[dict]:
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    height = float(spec["interior_height_m"])
    thickness = float(spec["wall_thickness_m"])
    if wall == "south":
        bounds = (-thickness, 0.0, -thickness, width + thickness)
        tangent = "x"
    elif wall == "north":
        bounds = (depth, depth + thickness, -thickness, width + thickness)
        tangent = "x"
    elif wall == "west":
        bounds = (-thickness, 0.0, 0.0, depth)
        tangent = "y"
    else:
        bounds = (width, width + thickness, 0.0, depth)
        tangent = "y"
    fixed0, fixed1, along0, along1 = bounds
    if not openings:
        return [_wall_box(wall, wall, fixed0, fixed1, along0, along1, 0.0, height, tangent)]
    pieces = []
    for opening in openings:
        offset = float(opening["offset_m"])
        hole_w = float(opening["width_m"])
        hole_h = float(opening["height_m"])
        sill = float(opening["sill_m"])
        start = along0 if tangent == "x" else 0.0
        # Interior tangent origin is 0. Corner returns exist only on north and south.
        interior_start = 0.0
        hole0 = interior_start + offset
        hole1 = hole0 + hole_w
        pieces.append(_wall_box(f"{wall}_left", wall, fixed0, fixed1, along0, hole0, 0.0, height, tangent))
        pieces.append(_wall_box(f"{wall}_right", wall, fixed0, fixed1, hole1, along1, 0.0, height, tangent))
        if opening["kind"] == "window":
            pieces.append(_wall_box(f"{wall}_sill", wall, fixed0, fixed1, hole0, hole1, 0.0, sill, tangent))
            pieces.append(_wall_box(f"{wall}_header", wall, fixed0, fixed1, hole0, hole1, sill + hole_h, height, tangent))
        else:
            pieces.append(_wall_box(f"{wall}_header", wall, fixed0, fixed1, hole0, hole1, hole_h, height, tangent))
        if frames:
            pieces.extend(_frame_pieces(spec, opening))
        if leaf and opening["kind"] == "door":
            door_leaf = _leaf_piece(spec, opening)
            if door_leaf is not None:
                pieces.append(door_leaf)
    _ = start
    return pieces


def _wall_box(name, wall, fixed0, fixed1, along0, along1, z0, z1, tangent) -> dict:
    if along1 <= along0 or z1 <= z0:
        raise RoomSpecError(f"reason=empty_segment id={name}")
    if tangent == "x":
        minimum = [along0, fixed0, z0]
        maximum = [along1, fixed1, z1]
        axis = [0.0, 1.0, 0.0]
    else:
        minimum = [fixed0, along0, z0]
        maximum = [fixed1, along1, z1]
        axis = [1.0, 0.0, 0.0]
    return _piece(name, minimum, maximum, axis)


def _frame_pieces(spec, opening: dict) -> list[dict]:
    wide = float(spec["frame_m"]["width_m"])
    depth = float(spec["frame_m"]["depth_m"])
    offset = float(opening["offset_m"])
    hole_w = float(opening["width_m"])
    hole_h = float(opening["height_m"])
    sill = float(opening["sill_m"])
    wall = opening["wall"]
    z0, z1 = sill, sill + hole_h
    boxes = []
    if wall in {"south", "north"}:
        y0, y1 = (0.0, depth) if wall == "south" else (float(spec["interior_depth_m"]) - depth, float(spec["interior_depth_m"]))
        boxes.append((f"{wall}_frame_left", [offset - wide, y0, z0], [offset, y1, z1], [0.0, 1.0, 0.0]))
        boxes.append((f"{wall}_frame_right", [offset + hole_w, y0, z0], [offset + hole_w + wide, y1, z1], [0.0, 1.0, 0.0]))
        boxes.append((f"{wall}_frame_header", [offset - wide, y0, z1], [offset + hole_w + wide, y1, z1 + wide], [0.0, 1.0, 0.0]))
        if opening["kind"] == "window":
            boxes.append((f"{wall}_frame_sill", [offset - wide, y0, z0 - wide], [offset + hole_w + wide, y1, z0], [0.0, 1.0, 0.0]))
    else:
        x0, x1 = (0.0, depth) if wall == "west" else (float(spec["interior_width_m"]) - depth, float(spec["interior_width_m"]))
        boxes.append((f"{wall}_frame_left", [x0, offset - wide, z0], [x1, offset, z1], [1.0, 0.0, 0.0]))
        boxes.append((f"{wall}_frame_right", [x0, offset + hole_w, z0], [x1, offset + hole_w + wide, z1], [1.0, 0.0, 0.0]))
        boxes.append((f"{wall}_frame_header", [x0, offset - wide, z1], [x1, offset + hole_w + wide, z1 + wide], [1.0, 0.0, 0.0]))
        if opening["kind"] == "window":
            boxes.append((f"{wall}_frame_sill", [x0, offset - wide, z0 - wide], [x1, offset + hole_w + wide, z0], [1.0, 0.0, 0.0]))
    return [_piece(name, minimum, maximum, axis) for name, minimum, maximum, axis in boxes]


def _leaf_piece(spec, opening: dict) -> dict | None:
    wide = float(spec["frame_m"]["width_m"])
    depth = float(spec["frame_m"]["depth_m"])
    offset = float(opening["offset_m"])
    hole_w = float(opening["width_m"])
    height = float(opening["height_m"])
    wall = opening["wall"]
    tangent_length = float(spec["interior_width_m"] if wall in {"south", "north"} else spec["interior_depth_m"])
    start = offset + hole_w + wide
    end = start + hole_w
    if end > tangent_length - WALL_GAP_M:
        logger.info("room_generator event=door_leaf omitted=pier_too_short wall=%s", wall)
        return None
    thickness = 0.03
    if wall == "south":
        minimum, maximum, axis = [start, depth, 0.0], [end, depth + thickness, height], [0.0, 1.0, 0.0]
    elif wall == "north":
        minimum, maximum, axis = [start, float(spec["interior_depth_m"]) - depth - thickness, 0.0], [end, float(spec["interior_depth_m"]) - depth, height], [0.0, 1.0, 0.0]
    elif wall == "west":
        minimum, maximum, axis = [depth, start, 0.0], [depth + thickness, end, height], [1.0, 0.0, 0.0]
    else:
        minimum, maximum, axis = [float(spec["interior_width_m"]) - depth - thickness, start, 0.0], [float(spec["interior_width_m"]) - depth, end, height], [1.0, 0.0, 0.0]
    return _piece(f"{wall}_leaf", minimum, maximum, axis)


def _require_no_volume_overlap(pieces: list[dict]) -> None:
    for index, left in enumerate(pieces):
        for right in pieces[index + 1 :]:
            if _overlap_volume(left["minimum"], left["maximum"], right["minimum"], right["maximum"]) > 1e-12:
                raise RoomSpecError(f"reason=volume_overlap ids={left['id']},{right['id']}")


def _overlap_volume(a_min, a_max, b_min, b_max) -> float:
    lower = np.maximum(np.asarray(a_min, dtype=np.float64), np.asarray(b_min, dtype=np.float64))
    upper = np.minimum(np.asarray(a_max, dtype=np.float64), np.asarray(b_max, dtype=np.float64))
    span = np.maximum(upper - lower, 0.0)
    return float(np.prod(span))


def _groups(pieces: list[dict]) -> dict:
    groups = {"floor": [], "ceiling": []}
    for piece in pieces:
        if piece["id"] in {"floor", "ceiling"}:
            groups[piece["id"]].append(piece)
        else:
            groups.setdefault(f"wall_{piece['wall']}", []).append(piece)
    return groups


def _bounds(piece: dict) -> tuple:
    return (tuple(np.round(piece["minimum"], 6)), tuple(np.round(piece["maximum"], 6)))


def _size(vertices: np.ndarray) -> np.ndarray:
    return vertices.max(axis=0) - vertices.min(axis=0)


def _require_chair_backrest(vertices: np.ndarray) -> None:
    high = vertices[:, 2] > 0.75 * float(vertices[:, 2].max())
    if float(vertices[high, 1].mean()) <= float(vertices[:, 1].mean()):
        raise RoomSpecError("reason=chair_backrest")


def _yaw_about_footprint(vertices: np.ndarray, yaw: float, world_xy: np.ndarray, floor_z: float) -> np.ndarray:
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = translation_matrix([-0.5 * (low[0] + high[0]), -0.5 * (low[1] + high[1]), 0.0])
    spun = compose(rotation_matrix("z", float(yaw)), center)
    return compose(translation_matrix([float(world_xy[0]), float(world_xy[1]), floor_z]), spun)


def _oriented_extents(vertices: np.ndarray, yaw: float) -> np.ndarray:
    parked = _yaw_about_footprint(vertices, yaw, np.zeros(2), 0.0)
    world = world_vertices(vertices, parked)
    return world.max(axis=0) - world.min(axis=0)


def _shelf_along(spec: dict, wall: str, span: float) -> float:
    """Center the shelf on a tangent interval that misses openings and the door corridor."""
    length = float(spec["interior_width_m"] if wall in {"south", "north"} else spec["interior_depth_m"])
    blocked = []
    for opening in spec["openings"]:
        if opening["wall"] != wall:
            continue
        blocked.append((float(opening["offset_m"]) - 0.08, float(opening["offset_m"]) + float(opening["width_m"]) + 0.08))
    if wall == "west":
        blocked.append((0.0, DOOR_CORRIDOR_M + 0.08))
    free = []
    cursor = 0.0
    for low, high in sorted(blocked):
        if low > cursor:
            free.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < length:
        free.append((cursor, length))
    half = 0.5 * span
    for low, high in free:
        if high - low >= span + 2.0 * WALL_GAP_M:
            return low + WALL_GAP_M + half
    raise RoomSpecError(f"reason=shelf_interval wall={wall}")


def _against_wall(vertices: np.ndarray, wall: str, spec: dict, along: float) -> np.ndarray:
    yaw = {"east": 0.5 * np.pi, "west": -0.5 * np.pi, "north": 0.0, "south": np.pi}[wall]
    parked = _yaw_about_footprint(vertices, yaw, np.array([0.0, 0.0]), 0.0)
    world = world_vertices(vertices, parked)
    low, high = world.min(axis=0), world.max(axis=0)
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    shift = [0.0, 0.0, 0.0]
    if wall == "east":
        shift[0] = width - WALL_GAP_M - float(high[0])
        shift[1] = along - 0.5 * float(low[1] + high[1])
    elif wall == "west":
        shift[0] = WALL_GAP_M - float(low[0])
        shift[1] = along - 0.5 * float(low[1] + high[1])
    elif wall == "north":
        shift[1] = depth - WALL_GAP_M - float(high[1])
        shift[0] = along - 0.5 * float(low[0] + high[0])
    else:
        shift[1] = WALL_GAP_M - float(low[1])
        shift[0] = along - 0.5 * float(low[0] + high[0])
    return compose(translation_matrix(shift), parked)


def _chair_center(table_world: np.ndarray, chair_vertices: np.ndarray, arrangement: str) -> np.ndarray:
    low, high = table_world.min(axis=0), table_world.max(axis=0)
    chair_size = _size(chair_vertices)
    center_x = 0.5 * float(low[0] + high[0])
    if arrangement == "central":
        center_y = float(high[1]) + 0.10 + 0.5 * float(chair_size[1])
    else:
        center_y = float(low[1]) - 0.10 - 0.5 * float(chair_size[1])
    return np.array([center_x, center_y], dtype=np.float64)


def _seat_yaw(desired_xy: np.ndarray) -> float:
    direction = np.asarray(desired_xy, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    return float(np.atan2(direction[0], -direction[1]))


def _world_vector(matrix: np.ndarray, local: np.ndarray) -> np.ndarray:
    rotated = matrix[:3, :3] @ np.asarray(local, dtype=np.float64)
    return rotated / np.linalg.norm(rotated)


def _floor_support(spec: dict) -> np.ndarray:
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    return np.array([[0.0, 0.0, 0.0], [width, 0.0, 0.0], [width, depth, 0.0], [0.0, depth, 0.0]], dtype=np.float64)


def _shell_meshes(spec: dict) -> dict:
    built = assemble(spec, frames=True, leaf=True)
    meshes = {}
    for name, pieces in built["groups"].items():
        vertices, polygons, _components = _combine(pieces)
        meshes[name] = {"vertices": vertices, "faces": triangulate(polygons)}
    return meshes


def _door_corridor(spec: dict) -> list[float]:
    door = next(item for item in spec["openings"] if item["kind"] == "door")
    if float(door["width_m"]) < DOOR_CORRIDOR_MIN_WIDTH_M:
        raise RoomSpecError("reason=door_too_narrow")
    offset = float(door["offset_m"])
    return [offset, 0.0, offset + float(door["width_m"]), DOOR_CORRIDOR_M]


def _keepout_mesh(opening: dict, spec: dict) -> dict:
    minimum = np.array(opening["clear_aperture_m"]["minimum_m"], dtype=np.float64)
    maximum = np.array(opening["clear_aperture_m"]["maximum_m"], dtype=np.float64)
    wall = opening["wall"]
    if wall == "north":
        minimum[1] -= WINDOW_KEEPOUT_M
    elif wall == "south":
        maximum[1] += WINDOW_KEEPOUT_M
    elif wall == "east":
        minimum[0] -= WINDOW_KEEPOUT_M
    else:
        maximum[0] += WINDOW_KEEPOUT_M
    vertices, polygons = box_mesh(minimum, maximum)
    return {"vertices": vertices, "faces": triangulate(polygons)}


def _inside_interior(bodies: dict, spec: dict) -> bool:
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    height = float(spec["interior_height_m"])
    for body in bodies.values():
        low, high = body["vertices"].min(axis=0), body["vertices"].max(axis=0)
        if float(low[0]) < 0.02 or float(high[0]) > width - 0.02:
            return False
        if float(low[1]) < 0.02 or float(high[1]) > depth - 0.02:
            return False
        if float(low[2]) < -1e-6 or float(high[2]) > height - 0.02:
            return False
    return True


def _footprint_hits_rectangle(vertices: np.ndarray, faces: np.ndarray, rect: list[float]) -> bool:
    points = vertices[:, :2]
    for face in faces:
        triangle = points[face]
        if any(_point_in_rectangle(point, rect) for point in triangle):
            return True
        corners = np.array([[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]])
        if any(_point_in_triangle(corner, triangle) for corner in corners):
            return True
    return False


def _point_in_rectangle(point: np.ndarray, rect: list[float]) -> bool:
    return rect[0] <= float(point[0]) <= rect[2] and rect[1] <= float(point[1]) <= rect[3]


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    def sign(a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])

    area = sign(triangle[0], triangle[1], triangle[2])
    if abs(area) < 1e-12:
        return False
    d0 = sign(point, triangle[0], triangle[1])
    d1 = sign(point, triangle[1], triangle[2])
    d2 = sign(point, triangle[2], triangle[0])
    negative = d0 < -1e-9 or d1 < -1e-9 or d2 < -1e-9
    positive = d0 > 1e-9 or d1 > 1e-9 or d2 > 1e-9
    return not (negative and positive)


def _combine(pieces: list[dict]):
    vertices = []
    polygons: list[list[int]] = []
    components = []
    offset = 0
    for piece in pieces:
        local = [[int(index) + offset for index in polygon] for polygon in piece["polygons"]]
        start = len(polygons)
        polygons.extend(local)
        vertices.append(np.asarray(piece["vertices"], dtype=np.float64))
        components.append(
            {
                "id": piece["id"],
                "profile": "structural_solid",
                "polygon_indexes": list(range(start, start + len(local))),
                "thickness_axis": piece["thickness_axis"],
            }
        )
        offset += len(piece["vertices"])
    return np.vstack(vertices), polygons, components


def _validation_report(pieces: list[dict]) -> dict:
    vertices, polygons, components = _combine(pieces)
    if len(pieces) == 1:
        return validate_mesh(vertices, polygons, "structural_solid", thickness_axis=pieces[0]["thickness_axis"])
    return validate_mesh(vertices, polygons, "multi_component", components=components)


def _package_group(destination: Path, asset_id: str, pieces: list[dict], spec: dict, group_name: str) -> dict:
    vertices, polygons, components = _combine(pieces)
    profile = "structural_solid" if len(pieces) == 1 else "multi_component"
    report = _validation_report(pieces)
    if not report["passed"]:
        raise RoomSpecError(f"reason=geometry asset={asset_id} issues={report['issues']}")
    material_key = "floor" if group_name == "floor" else "ceiling" if group_name == "ceiling" else "wall"
    supports = []
    if group_name == "floor":
        surface = support_quad((0.0, 0.0), (float(spec["interior_width_m"]), float(spec["interior_depth_m"])), 0.0, (0.0, 0.0, 1.0))
        surface["id"] = "interior"
        supports.append(surface)
    manifest = {
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
            "generator": "room_generator",
            "generator_version": "1.0.0",
            "seed": int(spec["seed"]),
            "parameters": {"room_id": spec["room_id"], "group": group_name},
            "license": "project-owned procedural geometry",
            "license_status": "permitted",
        },
        "geometry": {
            "profile": profile,
            "units": "m",
            "polygon_count": report["polygon_count"],
            "triangle_count": report["triangle_count"],
            "bounds_min": [float(value) for value in report["bounds_min"]],
            "bounds_max": [float(value) for value in report["bounds_max"]],
            "components": [
                {"id": component["id"], "profile": "structural_solid", "thickness_axis": component["thickness_axis"]}
                for component in components
            ],
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
        "support_surfaces": supports,
        "materials": [{"id": "architectural", **spec["materials"][material_key]}],
    }
    payload = {
        "geometry/render.obj": write_obj(vertices, polygons, name=asset_id),
        "geometry/collision.obj": write_obj(vertices, polygons, name=f"{asset_id}_collision"),
    }
    return create_package(destination, manifest, payload, {"geometry/render.obj": "render", "geometry/collision.obj": "collision"})


def _bundle(spec, arrangement, placed, source, room_catalog: Path, asset_catalog: Path) -> dict:
    seed = SEEDS[(spec["room_id"], arrangement)]
    instances = []
    relations = []
    catalog_record = json.loads((room_catalog / "room_catalog.json").read_text(encoding="utf-8")) if False else None
    packages = []
    for group in ("floor", "ceiling", "wall_south", "wall_north", "wall_east", "wall_west"):
        asset_id = f"{spec['room_id']}_{group}"
        manifest = verify_package(room_catalog / asset_id / VERSION)
        geometry_hash = hashlib.sha256((room_catalog / asset_id / VERSION / "geometry" / "render.obj").read_bytes()).hexdigest()
        relative = f"{asset_id}/{VERSION}"
        packages.append({"asset_id": asset_id, "version": VERSION, "path": relative, "package_sha256": manifest["package_sha256"], "geometry_sha256": geometry_hash})
        instances.append(_instance(asset_id, asset_id, manifest["package_sha256"], identity_matrix()))
        if group != "floor":
            continue
    for instance_id, matrix in placed["poses"].items():
        asset_id = placed["bodies"][instance_id]["asset_id"]
        manifest = verify_package(asset_catalog / asset_id / VERSION)
        instances.append(_instance(instance_id, asset_id, manifest["package_sha256"], deserialize_matrix(matrix)))
        relations.append(
            {
                "type": "supported_on",
                "instances": [instance_id, f"{spec['room_id']}_floor"],
                "frame": "support_surface_local",
                "frame_instance": f"{spec['room_id']}_floor",
                "surface": "interior",
                "priority": "required",
                "constraints": {},
                "passed": True,
                "reason": "contact",
                "measured_m": 0.0,
            }
        )
    table_matrix = placed["matrices"]["table_1"]
    support = np.asarray(next(item["polygon"] for item in verify_package(asset_catalog / "table_rect" / VERSION)["support_surfaces"] if item["id"] == "top"), dtype=np.float64)
    for asset in source["assets"]:
        local = source["local"][asset["instance_id"]]
        world = table_matrix @ local
        instances.append(_instance(asset["instance_id"], asset["asset_id"], asset["package_sha256"], world))
        mesh = load_mesh(asset_catalog, asset["asset_id"], asset["version"])
        contact = measure_support(world_vertices(mesh["vertices"], world), mesh["faces"], apply_points(table_matrix, support))
        if not contact["passed"]:
            raise RoomSpecError(f"reason=tabletop_support instance={asset['instance_id']} detail={contact['reason']}")
        relations.append(
            {
                "type": "supported_on",
                "instances": [asset["instance_id"], "table_1"],
                "frame": "support_surface_local",
                "frame_instance": "table_1",
                "surface": "top",
                "priority": "required",
                "constraints": {},
                "passed": True,
                "reason": contact["reason"],
                "measured_m": contact["gap_m"],
                "gap_limit_m": 0.001,
                "penetration_limit_m": 0.0001,
            }
        )
    cameras = _cameras(spec, table_matrix, support)
    lighting = _lighting(spec)
    clearance = {item["name"]: item.get("clearance_m") for item in placed["checks"] if item["name"].startswith("clearance:")}
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "task_id": f"{spec['room_id']}_{arrangement}",
        "profile_id": spec["room_id"],
        "profile_version": spec["version"],
        "arrangement": arrangement,
        "seed": seed,
        "room_spec_sha256": hashlib.sha256(canonical_json_bytes(spec)).hexdigest(),
        "source_tabletop": {"task_id": "quality_controlled", "seed": SOURCE_SEED, "run_id": SOURCE_RUN, "bundle_sha256": source["bundle_sha256"]},
        "packages": packages,
        "openings": spec["openings"],
        "instances": instances,
        "relations": relations,
        "collision": {"checks": placed["checks"], "furniture_clearance_m": clearance, "door_corridor_m": placed["door_corridor_m"]},
        "cameras": cameras,
        "lighting": lighting,
        "renderer": {"variant": "cuda_ad_spectral_polarized", "spp": SCENE_SPP, "seed": RENDER_SEED},
        "validation": {"valid": True, "visual_verification": "pending", "admitted": False},
    }
    _ = catalog_record
    _validate_room_bundle(bundle, room_catalog)
    return bundle


def _instance(instance_id: str, asset_id: str, package_hash: str, matrix: np.ndarray) -> dict:
    return {
        "instance_id": instance_id,
        "asset_id": asset_id,
        "version": VERSION,
        "package_sha256": package_hash,
        "matrix": serialize_matrix(matrix),
    }


def _validate_room_bundle(bundle: dict, catalog: Path) -> None:
    if bundle.get("schema_version") != BUNDLE_SCHEMA:
        raise RoomSpecError("reason=bundle_schema")
    seen = set()
    for instance in bundle["instances"]:
        if instance["instance_id"] in seen:
            raise RoomSpecError(f"reason=duplicate_instance id={instance['instance_id']}")
        seen.add(instance["instance_id"])
        deserialize_matrix(instance["matrix"])
    for package in bundle["packages"]:
        path = _package_path(catalog, package["path"])
        manifest = verify_package(path)
        if manifest["package_sha256"] != package["package_sha256"]:
            raise RoomSpecError(f"reason=package_hash asset={package['asset_id']}")
        if not (path / "geometry" / "render.obj").is_file():
            raise RoomSpecError(f"reason=missing_mesh asset={package['asset_id']}")
    if not bundle["openings"]:
        raise RoomSpecError("reason=openings")
    if not bundle["relations"]:
        raise RoomSpecError("reason=relations")


def _package_path(catalog: Path, relative: str) -> Path:
    if ".." in Path(relative).parts or Path(relative).is_absolute():
        raise RoomSpecError(f"reason=path_escape path={relative}")
    path = (Path(catalog) / relative).resolve()
    root = Path(catalog).resolve()
    if root != path and root not in path.parents:
        raise RoomSpecError(f"reason=path_escape path={relative}")
    return path


def _cameras(spec: dict, table_matrix: np.ndarray, support_local: np.ndarray) -> dict:
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    height = float(spec["interior_height_m"])
    overview = {
        "name": "room_overview",
        "origin": [0.40, 0.40, min(1.50, height - 0.30)],
        "target": [width * 0.62, depth * 0.70, 1.05],
        "up": [0.0, 0.0, 1.0],
        "fov_deg": 58.0,
        "resolution": [512, 384],
    }
    support = apply_points(table_matrix, support_local)
    tabletop = close_table_camera(support, margin=0.08, aspect=1600 / 900)
    tabletop["resolution"] = [1600, 900]
    windows = [item for item in spec["openings"] if item["kind"] == "window"]
    visible = [item["id"] for item in windows if _projects(overview, _opening_center(item), (512, 384))]
    window_views = [_window_camera(spec, opening) for opening in windows]
    return {"overview": overview, "tabletop": tabletop, "window_views": window_views, "windows_in_overview": visible}


def _window_camera(spec: dict, opening: dict) -> dict:
    center = np.array(_opening_center(opening), dtype=np.float64)
    outward = {"south": np.array([0.0, -1.0, 0.0]), "north": np.array([0.0, 1.0, 0.0]), "west": np.array([-1.0, 0.0, 0.0]), "east": np.array([1.0, 0.0, 0.0])}[opening["wall"]]
    origin = center - outward * 1.8
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    height = float(spec["interior_height_m"])
    origin[0] = float(np.clip(origin[0], 0.35, width - 0.35))
    origin[1] = float(np.clip(origin[1], 0.35, depth - 0.35))
    origin[2] = float(np.clip(center[2], 1.2, height - 0.30))
    return {
        "name": f"window_{opening['id']}",
        "origin": origin.tolist(),
        "target": center.tolist(),
        "up": [0.0, 0.0, 1.0],
        "fov_deg": 58.0,
        "resolution": [512, 384],
        "shows": [opening["id"]],
    }


def _opening_center(opening: dict) -> list[float]:
    minimum = np.array(opening["clear_aperture_m"]["minimum_m"], dtype=np.float64)
    maximum = np.array(opening["clear_aperture_m"]["maximum_m"], dtype=np.float64)
    return ((minimum + maximum) * 0.5).tolist()


def _projects(camera: dict, point: list[float], resolution: tuple[int, int]) -> bool:
    pixel = _project_point(camera, resolution[0], resolution[1], np.asarray(point, dtype=np.float64))
    if pixel is None:
        return False
    return 16.0 <= pixel[0] < resolution[0] - 16.0 and 16.0 <= pixel[1] < resolution[1] - 16.0


def _lighting(spec: dict) -> dict:
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    height = float(spec["interior_height_m"])
    ceiling = {
        "id": "ceiling_v1",
        "center_m": [width * 0.5, depth * 0.5, height - CEILING_DROP_M],
        "size_m": list(CEILING_SIZE_M),
        "radiance": LIGHT_RADIANCE,
        "normal": [0.0, 0.0, -1.0],
        "polarized": False,
        "calibrated": False,
    }
    emitters = []
    for opening in spec["openings"]:
        if opening["kind"] != "window":
            continue
        emitters.append(_emitter_record(spec, opening))
    return {"ceiling": ceiling, "daylight": {"id": "ceiling_plus_daylight_v1", "emitters": emitters, "radiance": DAYLIGHT_RADIANCE, "polarized": False, "calibrated": False}}


def _emitter_record(spec: dict, opening: dict) -> dict:
    minimum = np.array(opening["clear_aperture_m"]["minimum_m"], dtype=np.float64)
    maximum = np.array(opening["clear_aperture_m"]["maximum_m"], dtype=np.float64)
    center = (minimum + maximum) * 0.5
    wall = opening["wall"]
    outward = {"south": np.array([0.0, -1.0, 0.0]), "north": np.array([0.0, 1.0, 0.0]), "west": np.array([-1.0, 0.0, 0.0]), "east": np.array([1.0, 0.0, 0.0])}[wall]
    margin = 0.05
    if wall in {"south", "north"}:
        size = [float(maximum[0] - minimum[0] - 2.0 * margin), float(maximum[2] - minimum[2] - 2.0 * margin)]
    else:
        size = [float(maximum[1] - minimum[1] - 2.0 * margin), float(maximum[2] - minimum[2] - 2.0 * margin)]
    if min(size) <= 0.05:
        raise RoomSpecError(f"reason=emitter_aperture id={opening['id']}")
    origin = center + outward * (float(spec["wall_thickness_m"]) + 0.08)
    return {
        "id": f"daylight_{opening['id']}",
        "opening_id": opening["id"],
        "center_m": origin.tolist(),
        "size_m": size,
        "outward_normal": outward.tolist(),
        "into_room_normal": (-outward).tolist(),
        "radiance": DAYLIGHT_RADIANCE,
    }


def _room_light(bundle: dict, daylight: bool) -> dict:
    import mitsuba as mi

    require_variant()
    ceiling = bundle["lighting"]["ceiling"]
    transform = (
        mi.ScalarTransform4f.translate(ceiling["center_m"])
        @ mi.ScalarTransform4f.rotate([1.0, 0.0, 0.0], 180.0)
        @ mi.ScalarTransform4f.scale([ceiling["size_m"][0] * 0.5, ceiling["size_m"][1] * 0.5, 1.0])
    )
    _ = daylight
    return {"type": "rectangle", "to_world": transform, "emitter": {"type": "area", "radiance": {"type": "rgb", "value": ceiling["radiance"]}}}


def _daylight_emitters(bundle: dict) -> list[dict]:
    import mitsuba as mi

    require_variant()
    emitters = []
    for record in bundle["lighting"]["daylight"]["emitters"]:
        into = np.array(record["into_room_normal"], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0])
        tangent = np.cross(up, into)
        tangent = tangent / np.linalg.norm(tangent)
        bitangent = np.cross(into, tangent)
        basis = np.eye(4)
        basis[:3, 0] = tangent
        basis[:3, 1] = bitangent
        basis[:3, 2] = into
        scale = np.diag([record["size_m"][0] * 0.5, record["size_m"][1] * 0.5, 1.0, 1.0])
        matrix = translation_matrix(record["center_m"]) @ basis @ scale
        emitters.append({"id": record["id"], "mitsuba": {"type": "rectangle", "to_world": mi.ScalarTransform4f(matrix), "emitter": {"type": "area", "radiance": {"type": "rgb", "value": record["radiance"]}}}})
    return emitters


def _render_instances(bundle: dict, room_catalog: Path, asset_catalog: Path) -> list[dict]:
    room_ids = {item["asset_id"] for item in bundle["packages"]}
    instances = []
    for instance in bundle["instances"]:
        if instance["asset_id"] in room_ids:
            root = _package_path(room_catalog, f"{instance['asset_id']}/{instance['version']}")
            material = verify_package(root)["materials"][0]
        else:
            root = asset_catalog / instance["asset_id"] / instance["version"]
            material = dict(SCENE_AC_BSDF.get(instance["asset_id"], verify_package(root)["materials"][0]))
        instances.append(
            {
                "id": instance["instance_id"],
                "asset_id": instance["asset_id"],
                "role": "contextual" if instance["asset_id"] in room_ids else "required",
                "mesh": root / "geometry" / "render.obj",
                "matrix": deserialize_matrix(instance["matrix"]),
                "bsdf": material,
            }
        )
    return instances


def _table_support_world(catalog: Path, bundle: dict) -> np.ndarray:
    table = next(item for item in bundle["instances"] if item["instance_id"] == "table_1")
    manifest = verify_package(Path(catalog) / "table_rect" / VERSION)
    local = np.asarray(next(item["polygon"] for item in manifest["support_surfaces"] if item["id"] == "top"), dtype=np.float64)
    return apply_points(deserialize_matrix(table["matrix"]), local)


def _gallery_validation(body: list[str], origin: Path, validation: dict) -> None:
    body.append("<h3>" + html.escape(f"{validation['task_id']} {validation['lighting_id']}") + "</h3>")
    for name, image in validation["images"].items():
        href = Path(os.path.relpath(image["png"], origin)).as_posix()
        exr = Path(os.path.relpath(image["exr"], origin)).as_posix()
        body.append(f"<figure><a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(name)}\"></a><figcaption>{html.escape(name)}</figcaption></figure>")
        body.append(f"<p><a href=\"{html.escape(exr)}\">Stokes EXR {html.escape(name)}</a></p>")
        if name == "tabletop":
            mask = Path(image["png"]).with_name("tabletop_instance_ids.npy")
            if mask.is_file():
                href = Path(os.path.relpath(mask, origin)).as_posix()
                body.append(f"<p><a href=\"{html.escape(href)}\">Instance id mask</a></p>")


def probe_openings(spec: dict) -> list[dict]:
    """Rays through each aperture miss the shell. Rays just outside it hit."""
    spec = validate_specification(spec)
    pieces = assemble(spec, frames=True, leaf=True)["pieces"]
    results = []
    for opening in spec["openings"]:
        center = np.array(_opening_center(opening), dtype=np.float64)
        outward = {"south": np.array([0.0, -1.0, 0.0]), "north": np.array([0.0, 1.0, 0.0]), "west": np.array([-1.0, 0.0, 0.0]), "east": np.array([1.0, 0.0, 0.0])}[opening["wall"]]
        direction = -outward
        minimum = np.array(opening["clear_aperture_m"]["minimum_m"], dtype=np.float64)
        maximum = np.array(opening["clear_aperture_m"]["maximum_m"], dtype=np.float64)
        delta = maximum - minimum
        axes = [axis for axis in range(3) if float(delta[axis]) > 1e-6]
        samples = [center]
        for scale in (0.25, 0.75):
            sample = minimum.copy()
            for axis in axes:
                sample[axis] = float(minimum[axis]) + scale * float(delta[axis])
            samples.append(sample)
        through = []
        for sample in samples:
            sample_origin = sample + outward * (float(spec["wall_thickness_m"]) + 0.2)
            through.append(not _ray_hits_pieces(sample_origin, direction, [piece for piece in pieces if piece["id"].startswith(opening["wall"])]))
        control_point = center.copy()
        if opening["wall"] in {"south", "north"}:
            control_point[0] = float(opening["offset_m"]) - 0.05
        else:
            control_point[1] = float(opening["offset_m"]) - 0.05
        control_origin = control_point + outward * (float(spec["wall_thickness_m"]) + 0.2)
        control_hit = _ray_hits_pieces(control_origin, direction, [piece for piece in pieces if piece["id"].startswith(opening["wall"])])
        results.append({"id": opening["id"], "wall": opening["wall"], "through_clear": all(through), "control_hit": control_hit})
        if not all(through) or not control_hit:
            raise RoomSpecError(f"reason=opening_ray id={opening['id']} through={through} control={control_hit}")
    logger.info("room_generator event=opening_rays profile=%s count=%d", spec["room_id"], len(results))
    return results


def _ray_hits_pieces(origin: np.ndarray, direction: np.ndarray, pieces: list[dict]) -> bool:
    direction = np.asarray(direction, dtype=np.float64)
    for piece in pieces:
        if _ray_hits_box(origin, direction, np.asarray(piece["minimum"], dtype=np.float64), np.asarray(piece["maximum"], dtype=np.float64)):
            return True
    return False


def _ray_hits_box(origin: np.ndarray, direction: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> bool:
    near = -np.inf
    far = np.inf
    for axis in range(3):
        if abs(float(direction[axis])) < 1e-12:
            if float(origin[axis]) < float(minimum[axis]) or float(origin[axis]) > float(maximum[axis]):
                return False
            continue
        t0 = (float(minimum[axis]) - float(origin[axis])) / float(direction[axis])
        t1 = (float(maximum[axis]) - float(origin[axis])) / float(direction[axis])
        near = max(near, min(t0, t1))
        far = min(far, max(t0, t1))
    return far >= max(near, 0.0)


def write_room_catalog(packages: list[dict], path: Path) -> None:
    document = {"schema_version": "room_catalog_v1", "packages": packages}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document))


def mark_renderer_validated(catalog_path: Path, profile_id: str) -> None:
    document = json.loads(catalog_path.read_text(encoding="utf-8"))
    for package in document["packages"]:
        if package["profile_id"] == profile_id:
            package["renderer_validation"] = "passed"
            package["status"] = "renderer_validated"
    catalog_path.write_bytes(canonical_json_bytes(document))


def check_emitters(spec: dict) -> dict:
    """Daylight planes sit outside the shell and still see through the aperture."""
    spec = validate_specification(spec)
    pieces = assemble(spec, frames=True, leaf=True)["pieces"]
    lighting = _lighting(spec)
    ceiling_z = float(lighting["ceiling"]["center_m"][2])
    if ceiling_z >= float(spec["interior_height_m"]) or ceiling_z < float(spec["interior_height_m"]) - 0.05:
        raise RoomSpecError(f"reason=ceiling_light z={ceiling_z}")
    for record in lighting["daylight"]["emitters"]:
        center = np.asarray(record["center_m"], dtype=np.float64)
        into = np.asarray(record["into_room_normal"], dtype=np.float64)
        opening = next(item for item in spec["openings"] if item["id"] == record["opening_id"])
        if _ray_hits_pieces(center, into, [piece for piece in pieces if piece["id"].startswith(opening["wall"])]):
            raise RoomSpecError(f"reason=emitter_blocked id={record['id']}")
        outward = -into
        exterior = np.asarray(spec["exterior_bounds_m"]["maximum_m"], dtype=np.float64)
        exterior_min = np.asarray(spec["exterior_bounds_m"]["minimum_m"], dtype=np.float64)
        if float(np.dot(center - exterior, np.maximum(outward, 0.0))) < 0.05 and float(np.dot(exterior_min - center, np.maximum(-outward, 0.0))) < 0.05:
            raise RoomSpecError(f"reason=emitter_inside id={record['id']}")
    return lighting

"""Project-owned indoor structure, furniture, and props.

Meshes are Z-up, right-handed, and in meters. An opening is a set of closed
wall, jamb, header, and sill boxes. It is not a missing wall. The same
parameters and seed rewrite the same OBJ bytes. Infinigen extraction does not
use this guarantee.

Thickness of a structural box is measured along the unit axis recorded with
that box. Floors and ceilings use +Z. North and south walls use +Y. East and
west walls use +X.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import create_package, publish_candidate, set_candidate_status
from sensor_sim.scene_gen.asset_validate import validate_mesh
from sensor_sim.scene_gen.indoor_transform import identity_matrix, serialize_matrix
from sensor_sim.scene_gen.mesh_io import write_obj

logger = logging.getLogger("sensor_sim.scene_gen.structural")

GENERATOR_VERSION = "1.0.0"
CLEARANCE_TRIANGLE_LIMIT = 2500


def json_number(value: float) -> float:
    return float(f"{float(value):.6f}")


def json_vector(values) -> list[float]:
    return [json_number(value) for value in values]


def default_room() -> dict:
    return {
        "interior_width_m": 4.0,
        "interior_depth_m": 3.2,
        "interior_height_m": 2.5,
        "wall_thickness_m": 0.12,
        "floor_thickness_m": 0.05,
        "ceiling_thickness_m": 0.05,
        "door": {"wall": "south", "offset_m": 0.3, "width_m": 0.9, "height_m": 2.1},
        "window": {"wall": "north", "offset_m": 0.5, "width_m": 1.2, "height_m": 1.0, "sill_m": 0.9},
    }


def box_mesh(minimum, maximum) -> tuple[np.ndarray, list[list[int]]]:
    """Closed box with outward right-handed quads."""
    x0, y0, z0 = (float(value) for value in minimum)
    x1, y1, z1 = (float(value) for value in maximum)
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    polygons = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ]
    return vertices, polygons


def cylinder_mesh(radius: float, height: float, segments: int = 24) -> tuple[np.ndarray, list[list[int]]]:
    angles = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False)
    bottom = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(segments)], axis=1)
    top = bottom.copy()
    top[:, 2] = height
    centers = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, height]], dtype=np.float64)
    vertices = np.vstack([bottom, top, centers])
    bottom_center = 2 * segments
    top_center = bottom_center + 1
    polygons: list[list[int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        polygons.append([index, nxt, segments + nxt, segments + index])
        polygons.append([bottom_center, nxt, index])
        polygons.append([top_center, segments + index, segments + nxt])
    return vertices, polygons


def cone_mesh(radius: float, height: float, segments: int = 24) -> tuple[np.ndarray, list[list[int]]]:
    angles = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(segments)], axis=1)
    vertices = np.vstack([ring, [[0.0, 0.0, height], [0.0, 0.0, 0.0]]])
    apex = segments
    center = segments + 1
    polygons = []
    for index in range(segments):
        nxt = (index + 1) % segments
        polygons.append([index, nxt, apex])
        polygons.append([center, nxt, index])
    return vertices, polygons


def sphere_mesh(radius: float, rings: int = 8, segments: int = 16) -> tuple[np.ndarray, list[list[int]]]:
    vertices = [[0.0, 0.0, radius]]
    for ring in range(1, rings):
        theta = np.pi * ring / rings
        z = radius * np.cos(theta)
        radial = radius * np.sin(theta)
        for segment in range(segments):
            phi = 2.0 * np.pi * segment / segments
            vertices.append([radial * np.cos(phi), radial * np.sin(phi), z])
    vertices.append([0.0, 0.0, -radius])
    vertices = np.asarray(vertices, dtype=np.float64)
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
    south = len(vertices) - 1
    last = 1 + (rings - 2) * segments
    for segment in range(segments):
        nxt = (segment + 1) % segments
        polygons.append([south, last + nxt, last + segment])
    return vertices, polygons


def support_quad(minimum_xy, maximum_xy, z_value: float, normal) -> dict:
    x0, y0 = minimum_xy
    x1, y1 = maximum_xy
    polygon = [[x0, y0, z_value], [x1, y0, z_value], [x1, y1, z_value], [x0, y1, z_value]]
    return {"normal": json_vector(normal), "origin": json_vector(polygon[0]), "polygon": [json_vector(point) for point in polygon]}


def build_room(parameters: dict | None = None) -> dict:
    """Return named structural meshes and the measured interior frame."""
    room = default_room()
    if parameters:
        room.update(parameters)
        for key in ("door", "window"):
            if key in parameters:
                merged = dict(default_room()[key])
                merged.update(parameters[key])
                room[key] = merged
    width = float(room["interior_width_m"])
    depth = float(room["interior_depth_m"])
    height = float(room["interior_height_m"])
    wall = float(room["wall_thickness_m"])
    floor_t = float(room["floor_thickness_m"])
    ceiling_t = float(room["ceiling_thickness_m"])
    if min(width, depth, height, wall, floor_t, ceiling_t) <= 0.0:
        raise ValueError("room dimensions must be positive")
    exterior_x = width + 2.0 * wall
    exterior_y = depth + 2.0 * wall
    floor_top = floor_t
    wall_top = floor_t + height
    floor = _piece("floor", [0.0, 0.0, 0.0], [exterior_x, exterior_y, floor_t], [0.0, 0.0, 1.0])
    ceiling = _piece("ceiling", [0.0, 0.0, wall_top], [exterior_x, exterior_y, wall_top + ceiling_t], [0.0, 0.0, 1.0])
    south = _opening_wall(
        "south",
        axis="x",
        fixed_min=0.0,
        fixed_max=wall,
        along_min=0.0,
        along_max=exterior_x,
        z0=floor_top,
        z1=wall_top,
        opening=room["door"],
        sill=False,
        interior_start=wall,
    )
    north = _opening_wall(
        "north",
        axis="x",
        fixed_min=wall + depth,
        fixed_max=wall + depth + wall,
        along_min=0.0,
        along_max=exterior_x,
        z0=floor_top,
        z1=wall_top,
        opening=room["window"],
        sill=True,
        interior_start=wall,
    )
    west = _piece("west", [0.0, wall, floor_top], [wall, wall + depth, wall_top], [1.0, 0.0, 0.0])
    east = _piece("east", [wall + width, wall, floor_top], [exterior_x, wall + depth, wall_top], [1.0, 0.0, 0.0])
    pieces = [floor, ceiling, west, east, *south, *north]
    _require_no_overlap(pieces)
    interior = {
        "origin_m": json_vector([wall, wall, floor_top]),
        "size_m": json_vector([width, depth, height]),
        "floor_top_z_m": json_number(floor_top),
        "ceiling_bottom_z_m": json_number(wall_top),
    }
    return {"parameters": room, "pieces": pieces, "interior": interior, "generator_version": GENERATOR_VERSION}


def build_table(seed: int = 17011) -> dict:
    rng = np.random.default_rng(int(seed))
    width = _jitter(rng, 1.2, 0.0)
    depth = _jitter(rng, 0.7, 0.0)
    height = _jitter(rng, 0.74, 0.0)
    top_t = 0.04
    leg = 0.06
    top = _piece("top", [0.0, 0.0, height - top_t], [width, depth, height], [0.0, 0.0, 1.0])
    legs = [
        _piece("leg_0", [0.05, 0.05, 0.0], [0.05 + leg, 0.05 + leg, height - top_t], [1.0, 0.0, 0.0]),
        _piece("leg_1", [width - 0.05 - leg, 0.05, 0.0], [width - 0.05, 0.05 + leg, height - top_t], [1.0, 0.0, 0.0]),
        _piece("leg_2", [0.05, depth - 0.05 - leg, 0.0], [0.05 + leg, depth - 0.05, height - top_t], [1.0, 0.0, 0.0]),
        _piece("leg_3", [width - 0.05 - leg, depth - 0.05 - leg, 0.0], [width - 0.05, depth - 0.05, height - top_t], [1.0, 0.0, 0.0]),
    ]
    _require_no_overlap([top, *legs])
    surface = support_quad((0.0, 0.0), (width, depth), height, (0.0, 0.0, 1.0))
    surface["id"] = "top"
    return {"pieces": [top, *legs], "support_surfaces": [surface], "seed": int(seed)}


def build_shelf(seed: int = 17011) -> dict:
    rng = np.random.default_rng(int(seed) + 17)
    width = _jitter(rng, 0.8, 0.0)
    depth = _jitter(rng, 0.35, 0.0)
    height = _jitter(rng, 1.2, 0.0)
    board = 0.02
    side_t = 0.02
    level_z = (0.02, height * 0.45, height - board)
    pieces = [
        _piece("side_0", [0.0, 0.0, 0.0], [side_t, depth, height], [1.0, 0.0, 0.0]),
        _piece("side_1", [width - side_t, 0.0, 0.0], [width, depth, height], [1.0, 0.0, 0.0]),
    ]
    surfaces = []
    for index, z_value in enumerate(level_z[:2]):
        board_piece = _piece(
            f"level_{index}",
            [side_t, 0.0, z_value],
            [width - side_t, depth, z_value + board],
            [0.0, 0.0, 1.0],
        )
        pieces.append(board_piece)
        surface = support_quad((side_t, 0.0), (width - side_t, depth), z_value + board, (0.0, 0.0, 1.0))
        surface["id"] = f"level_{index}"
        surfaces.append(surface)
    top = _piece("top", [side_t, 0.0, height - board], [width - side_t, depth, height], [0.0, 0.0, 1.0])
    pieces.append(top)
    _require_no_overlap(pieces)
    return {"pieces": pieces, "support_surfaces": surfaces, "seed": int(seed)}


def build_open_crate() -> dict:
    """Open-top container. The existing solid ``prop_crate`` has no interior."""
    outer_x, outer_y, outer_z = 0.40, 0.30, 0.20
    thickness = 0.015
    pieces = [
        _piece("bottom", [0.0, 0.0, 0.0], [outer_x, outer_y, thickness], [0.0, 0.0, 1.0]),
        _piece("wall_x0", [0.0, 0.0, thickness], [thickness, outer_y, outer_z], [1.0, 0.0, 0.0]),
        _piece("wall_x1", [outer_x - thickness, 0.0, thickness], [outer_x, outer_y, outer_z], [1.0, 0.0, 0.0]),
        _piece("wall_y0", [thickness, 0.0, thickness], [outer_x - thickness, thickness, outer_z], [0.0, 1.0, 0.0]),
        _piece("wall_y1", [thickness, outer_y - thickness, thickness], [outer_x - thickness, outer_y, outer_z], [0.0, 1.0, 0.0]),
    ]
    _require_no_overlap(pieces)
    interior_origin = [thickness, thickness, thickness]
    interior_size = [outer_x - 2.0 * thickness, outer_y - 2.0 * thickness, outer_z - thickness]
    surface = support_quad((thickness, thickness), (outer_x - thickness, outer_y - thickness), thickness, (0.0, 0.0, 1.0))
    surface["id"] = "interior_bottom"
    return {
        "pieces": pieces,
        "support_surfaces": [surface],
        "interior": {
            "id": "cavity",
            "frame": "container_interior_local",
            "origin_m": json_vector(interior_origin),
            "size_m": json_vector(interior_size),
            "bottom_z_m": json_number(thickness),
            "opening": "top",
            "margin_m": 0.005,
            "wall_component_ids": ["wall_x0", "wall_x1", "wall_y0", "wall_y1"],
        },
    }


def build_prop(kind: str, seed: int) -> dict:
    rng = np.random.default_rng(int(seed) + {"crate": 3, "cup": 5, "book": 7, "sphere": 11, "cone": 13}[kind])
    if kind == "crate":
        size = _jitter(rng, 0.28, 0.04)
        vertices, polygons = box_mesh((0.0, 0.0, 0.0), (size, size * 0.8, size * 0.6))
        axis = [0.0, 0.0, 1.0]
    elif kind == "cup":
        vertices, polygons = cylinder_mesh(_jitter(rng, 0.04, 0.005), _jitter(rng, 0.1, 0.01), 24)
        axis = [0.0, 0.0, 1.0]
    elif kind == "book":
        vertices, polygons = box_mesh((0.0, 0.0, 0.0), (_jitter(rng, 0.18, 0.02), _jitter(rng, 0.24, 0.02), 0.03))
        axis = [0.0, 0.0, 1.0]
    elif kind == "sphere":
        vertices, polygons = sphere_mesh(0.08, 8, 16)
        axis = [0.0, 0.0, 1.0]
    elif kind == "cone":
        vertices, polygons = cone_mesh(0.06, 0.12, 24)
        axis = [0.0, 0.0, 1.0]
    else:
        raise ValueError(kind)
    return {
        "pieces": [{"id": kind, "vertices": vertices, "polygons": polygons, "thickness_axis": axis, "minimum": vertices.min(0), "maximum": vertices.max(0)}],
        "support_surfaces": [],
        "seed": int(seed),
    }


def package_procedural_catalog(catalog: Path, seed: int = 17011, room: dict | None = None) -> list[dict]:
    """Write project-owned candidate packages and mark those that pass geometry checks."""
    catalog = Path(catalog)
    catalog.mkdir(parents=True, exist_ok=True)
    structure = build_room(room)
    south, north = _split_opening_pieces(structure)
    table = build_table(seed)
    shelf = build_shelf(seed)
    assets = [
        ("room_floor", [structure["pieces"][0]], _floor_supports(structure), "structural_solid"),
        ("room_ceiling", [structure["pieces"][1]], [], "structural_solid"),
        ("room_wall_west", [_named(structure, "west")], _bottom_support(_named(structure, "west")), "structural_solid"),
        ("room_wall_east", [_named(structure, "east")], _bottom_support(_named(structure, "east")), "structural_solid"),
        ("room_wall_south", south, _bottom_supports(south), "multi_component"),
        ("room_wall_north", north, _bottom_supports(north), "multi_component"),
        ("table_rect", table["pieces"], table["support_surfaces"], "multi_component"),
        ("shelf_two", shelf["pieces"], shelf["support_surfaces"], "multi_component"),
        ("prop_crate", build_prop("crate", seed)["pieces"], [], "structural_solid"),
        ("prop_cup", build_prop("cup", seed)["pieces"], [], "closed_solid"),
        ("prop_book", build_prop("book", seed)["pieces"], [], "structural_solid"),
        ("fixture_sphere", build_prop("sphere", seed)["pieces"], [], "closed_solid"),
        ("fixture_cone", build_prop("cone", seed)["pieces"], [], "closed_solid"),
    ]
    written = []
    for asset_id, pieces, supports, profile in assets:
        staging = Path(tempfile.mkdtemp(prefix=".stage-", dir=catalog))
        destination = staging / "package"
        try:
            manifest = _package_pieces(
                destination,
                asset_id,
                pieces,
                supports,
                profile,
                seed=seed,
                parameters=structure["parameters"] if asset_id.startswith("room_") else {"seed": seed},
            )
            publish_candidate(catalog, destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        report = _validate_pieces(pieces, profile)
        if report["passed"]:
            set_candidate_status(catalog, asset_id, "1.0.0", "technically_validated")
        written.append({"asset_id": asset_id, "package_sha256": manifest["package_sha256"], "geometry_passed": report["passed"], "issues": report["issues"]})
        logger.info(
            "structural event=package asset_id=%s package_sha256=%s geometry_passed=%s triangles=%s",
            asset_id,
            manifest["package_sha256"],
            report["passed"],
            report["triangle_count"],
        )
    return written


def package_open_crate(catalog: Path) -> dict:
    """Publish a new open-top crate. The solid ``prop_crate`` package is not rewritten."""
    catalog = Path(catalog)
    catalog.mkdir(parents=True, exist_ok=True)
    crate = build_open_crate()
    staging = Path(tempfile.mkdtemp(prefix=".stage-", dir=catalog))
    destination = staging / "package"
    try:
        manifest = _package_pieces(
            destination,
            "prop_crate_open",
            crate["pieces"],
            crate["support_surfaces"],
            "multi_component",
            seed=17011,
            parameters={"container": "open_top"},
            geometry_extra={"interior": crate["interior"], "category": "container", "aliases": ["crate", "container"]},
        )
        publish_candidate(catalog, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    set_candidate_status(catalog, "prop_crate_open", "1.0.0", "technically_validated")
    return manifest


def _package_pieces(destination, asset_id, pieces, supports, profile, *, seed, parameters, geometry_extra=None) -> dict:
    vertices, polygons, components = _combine(pieces)
    render = write_obj(vertices, polygons, name=asset_id)
    collision = write_obj(vertices, polygons, name=f"{asset_id}_collision")
    geometry_report = _validate_pieces(pieces, profile)
    if not geometry_report["passed"]:
        raise RuntimeError(f"{asset_id} failed geometry validation: {geometry_report['issues']}")
    manifest = {
        "asset_id": asset_id,
        "version": "1.0.0",
        "units": "m",
        "frame": {
            "up_axis": "Z",
            "handedness": "right",
            "transform_convention": "row_major_column_vector",
            "matrix": serialize_matrix(identity_matrix()),
        },
        "provenance": {
            "source": "sensor_sim",
            "generator": "structural",
            "generator_version": GENERATOR_VERSION,
            "seed": int(seed),
            "parameters": _json_ready(parameters),
            "license": "project-owned procedural geometry",
            "license_status": "permitted",
        },
        "geometry": {
            "profile": profile,
            "units": "m",
            "polygon_count": geometry_report["polygon_count"],
            "triangle_count": geometry_report["triangle_count"],
            "bounds_min": json_vector(geometry_report["bounds_min"]),
            "bounds_max": json_vector(geometry_report["bounds_max"]),
            "clearance_triangle_limit": CLEARANCE_TRIANGLE_LIMIT,
            "clearance_from_render_mesh": geometry_report["triangle_count"] <= CLEARANCE_TRIANGLE_LIMIT,
            "components": [
                {
                    "id": component["id"],
                    "profile": "structural_solid" if profile == "multi_component" else profile,
                    "thickness_axis": json_vector(component["thickness_axis"]),
                }
                for component in components
            ],
            **(geometry_extra or {}),
        },
        "collision": [
            {
                "id": "proxy",
                "path": "geometry/collision.obj",
                "render_path": "geometry/render.obj",
                "triangle_count": geometry_report["triangle_count"],
                "qualified": geometry_report["triangle_count"] <= CLEARANCE_TRIANGLE_LIMIT,
            }
        ],
        "support_surfaces": supports,
        "materials": [{"id": "default_diffuse", "type": "diffuse", "reflectance": [0.7, 0.7, 0.7]}],
    }
    return create_package(
        destination,
        manifest,
        {"geometry/render.obj": render, "geometry/collision.obj": collision},
        {"geometry/render.obj": "render", "geometry/collision.obj": "collision"},
    )


def _validate_pieces(pieces, profile):
    vertices, polygons, components = _combine(pieces)
    if profile == "multi_component":
        return validate_mesh(vertices, polygons, profile, components=components)
    return validate_mesh(vertices, polygons, profile, thickness_axis=pieces[0]["thickness_axis"])


def _combine(pieces):
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
    stacked = np.vstack(vertices) if vertices else np.zeros((0, 3))
    return stacked, polygons, components


def _piece(name, minimum, maximum, axis):
    vertices, polygons = box_mesh(minimum, maximum)
    return {
        "id": name,
        "vertices": vertices,
        "polygons": polygons,
        "thickness_axis": axis,
        "minimum": np.asarray(minimum, dtype=np.float64),
        "maximum": np.asarray(maximum, dtype=np.float64),
    }


def _opening_wall(name, *, axis, fixed_min, fixed_max, along_min, along_max, z0, z1, opening, sill, interior_start):
    offset = float(opening["offset_m"])
    hole_w = float(opening["width_m"])
    hole_h = float(opening["height_m"])
    start = interior_start + offset
    stop = start + hole_w
    if start <= along_min or stop >= along_max or hole_h <= 0.0 or hole_h >= (z1 - z0):
        raise ValueError(f"{name} opening does not fit the wall")
    pieces = [
        _span(f"{name}_left", axis, fixed_min, fixed_max, along_min, start, z0, z1),
        _span(f"{name}_right", axis, fixed_min, fixed_max, stop, along_max, z0, z1),
        _span(f"{name}_header", axis, fixed_min, fixed_max, start, stop, z0 + hole_h, z1),
    ]
    if sill:
        sill_height = float(opening["sill_m"])
        if sill_height <= 0.0 or z0 + sill_height >= z0 + hole_h:
            raise ValueError(f"{name} sill does not fit under the opening")
        pieces.append(_span(f"{name}_sill", axis, fixed_min, fixed_max, start, stop, z0, z0 + sill_height))
        # The header recorded above used the full hole height from the floor.
        # A window opening starts at the sill, so rebuild the header from the window head.
        pieces[2] = _span(f"{name}_header", axis, fixed_min, fixed_max, start, stop, z0 + sill_height + hole_h, z1)
        if z0 + sill_height + hole_h >= z1:
            raise ValueError(f"{name} window does not fit under the ceiling")
    return pieces


def _span(name, axis, fixed_min, fixed_max, along_min, along_max, z0, z1):
    if axis == "x":
        minimum = [along_min, fixed_min, z0]
        maximum = [along_max, fixed_max, z1]
        thickness_axis = [0.0, 1.0, 0.0]
    else:
        minimum = [fixed_min, along_min, z0]
        maximum = [fixed_max, along_max, z1]
        thickness_axis = [1.0, 0.0, 0.0]
    return _piece(name, minimum, maximum, thickness_axis)


def _require_no_overlap(pieces) -> None:
    for index, left in enumerate(pieces):
        for right in pieces[index + 1 :]:
            if _volume_overlap(left["minimum"], left["maximum"], right["minimum"], right["maximum"]) > 1e-12:
                raise ValueError(f"structural overlap between {left['id']} and {right['id']}")


def _volume_overlap(a_min, a_max, b_min, b_max) -> float:
    lower = np.maximum(np.asarray(a_min, dtype=np.float64), np.asarray(b_min, dtype=np.float64))
    upper = np.minimum(np.asarray(a_max, dtype=np.float64), np.asarray(b_max, dtype=np.float64))
    span = np.maximum(upper - lower, 0.0)
    return float(np.prod(span))


def _named(structure, name):
    for piece in structure["pieces"]:
        if piece["id"] == name:
            return piece
    raise KeyError(name)


def _split_opening_pieces(structure):
    south = [piece for piece in structure["pieces"] if piece["id"].startswith("south_")]
    north = [piece for piece in structure["pieces"] if piece["id"].startswith("north_")]
    return south, north


def _floor_supports(structure):
    interior = structure["interior"]
    origin = interior["origin_m"]
    size = interior["size_m"]
    z_value = interior["floor_top_z_m"]
    surface = support_quad((origin[0], origin[1]), (origin[0] + size[0], origin[1] + size[1]), z_value, (0.0, 0.0, 1.0))
    surface["id"] = "interior"
    return [surface]


def _bottom_support(piece):
    minimum, maximum = piece["minimum"], piece["maximum"]
    surface = support_quad((minimum[0], minimum[1]), (maximum[0], maximum[1]), float(minimum[2]), (0.0, 0.0, -1.0))
    surface["id"] = "bottom"
    return [surface]


def _bottom_supports(pieces):
    surfaces = []
    for piece in pieces:
        if piece["id"].endswith("_header"):
            continue
        surfaces.extend(_bottom_support(piece))
    return surfaces


def _jitter(rng, center: float, spread: float) -> float:
    if spread == 0.0:
        return json_number(center)
    return json_number(center + float(rng.uniform(-spread, spread)))


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return json_number(value)
    if isinstance(value, (np.floating, np.integer)):
        return json_number(float(value))
    return value

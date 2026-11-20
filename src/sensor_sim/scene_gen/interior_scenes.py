"""Six furnished interiors for visual acceptance.

Rooms are new specifications. Existing qualification bundles are not rewritten.
RGB review images come from the supported renderer. Stokes channels are not the
review product.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import canonical_json_bytes, create_package, verify_package
from sensor_sim.scene_gen.indoor_contact import measure_support, unintended_intersection, world_vertices
from sensor_sim.scene_gen.indoor_mitsuba import render_stokes, require_variant, write_rgb_png
from sensor_sim.scene_gen.indoor_transform import apply_points, compose, deserialize_matrix, identity_matrix, rotation_matrix, serialize_matrix, translation_matrix
from sensor_sim.scene_gen.interior_assets import BUILDERS, build_asset
from sensor_sim.scene_gen.narrow import mesh_clearance
from sensor_sim.scene_gen.room_generator import _daylight_emitters, _emitter_record, _shell_meshes, package_profile
from sensor_sim.scene_gen.scene_quality import _fit_origin, _peak_ndc
from sensor_sim.scene_gen.room_schema import make_specification
from sensor_sim.scene_gen.scene_context import require_scene_context
from sensor_sim.scene_gen.semantic_assets import build_asset as semantic_build
from sensor_sim.scene_gen.mesh_io import write_obj
from sensor_sim.scene_gen.semantic_composition import (
    _ceiling_emitter,
    _largest_free_rectangle,
    _object_slots,
    _place_object,
    _place_workbench,
    _slot_matrix,
    _yaw_about_footprint,
    frame_support_camera,
)
from sensor_sim.scene_gen.structural import support_quad
from sensor_sim.scene_gen.tabletop_layout import load_mesh
from sensor_sim.scene_gen.tabletop_render import SCENE_AC_BSDF, _scene, mitsuba_bsdf

logger = logging.getLogger("sensor_sim.scene_gen.interior_scenes")

ROOT = Path("dataset/qualification/diverse_interior_review")
SEMANTIC = Path("dataset/qualification/semantic_scenes/catalog")
CANDIDATES = Path("assets/candidates")
VERSION = "1.0.0"
RENDER_SEED = 17501
SPP = 16
RESOLUTION = (1280, 720)
SEPARATION = 0.004
DIMENSION_NOTES = {
    "kitchen": "Depth is 4.6 m so the west cabinet run, oven, and north-window keepout fit. The sketch depth was 4.0 m.",
    "bedroom": "Size is 4.4 x 3.8 x 2.5 m so the bed, two nightstands, and wardrobe fit. The sketch was 3.8 x 3.4 x 2.5 m.",
    "workshop": "Depth is 4.0 m so the reused 1.0.2 workbench clears the north-window keepout. The sketch depth was 3.6 m.",
}
REQUIRED = {
    "kitchen": {"furn_base_cabinet", "furn_upper_cabinet", "furn_sink", "furn_cooktop", "furn_oven", "furn_microwave", "furn_fridge", "furn_island"},
    "dining": {"furn_dining_table", "furn_sideboard", "prop_chair"},
    "living": {"furn_sofa", "furn_armchair", "furn_coffee", "furn_media", "furn_tv", "furn_bookcase", "furn_floor_lamp"},
    "bedroom": {"furn_bed", "furn_nightstand", "furn_wardrobe", "furn_desk_lamp"},
    "office": {"furn_office_desk", "prop_chair", "furn_monitor", "furn_bookcase", "furn_desk_lamp", "prop_keyboard", "furn_mouse"},
    "workshop": {"furniture_workbench", "prop_work_panel", "prop_fixture_block", "prop_stock_plate"},
}

ROOMS = {
    "kitchen": {"width": 5.4, "depth": 4.6, "height": 2.6, "seed": 17511, "floor": [0.72, 0.70, 0.66], "wall": [0.93, 0.91, 0.86], "ceiling": [0.96, 0.96, 0.94], "openings": [
        {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.40, "width_m": 0.90, "height_m": 2.10, "sill_m": 0.0},
        {"id": "window_north", "kind": "window", "wall": "north", "offset_m": 0.80, "width_m": 1.60, "height_m": 1.10, "sill_m": 0.90},
        {"id": "window_east", "kind": "window", "wall": "east", "offset_m": 0.80, "width_m": 1.20, "height_m": 1.00, "sill_m": 1.00},
    ]},
    "dining": {"width": 4.6, "depth": 3.6, "height": 2.7, "seed": 17512, "floor": [0.42, 0.28, 0.16], "wall": [0.78, 0.74, 0.68], "ceiling": [0.94, 0.93, 0.90], "openings": [
        {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.50, "width_m": 0.90, "height_m": 2.10, "sill_m": 0.0},
        {"id": "window_north", "kind": "window", "wall": "north", "offset_m": 0.70, "width_m": 1.60, "height_m": 1.10, "sill_m": 0.90},
    ]},
    "living": {"width": 6.2, "depth": 4.8, "height": 2.8, "seed": 17513, "floor": [0.55, 0.42, 0.32], "wall": [0.86, 0.84, 0.78], "ceiling": [0.95, 0.95, 0.93], "openings": [
        {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.60, "width_m": 0.90, "height_m": 2.10, "sill_m": 0.0},
        {"id": "window_north", "kind": "window", "wall": "north", "offset_m": 1.00, "width_m": 2.00, "height_m": 1.30, "sill_m": 0.80},
        {"id": "window_west", "kind": "window", "wall": "west", "offset_m": 0.80, "width_m": 1.20, "height_m": 1.20, "sill_m": 0.90},
    ]},
    "bedroom": {"width": 4.4, "depth": 3.8, "height": 2.5, "seed": 17514, "floor": [0.62, 0.50, 0.40], "wall": [0.80, 0.78, 0.84], "ceiling": [0.95, 0.94, 0.96], "openings": [
        {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.30, "width_m": 0.90, "height_m": 2.10, "sill_m": 0.0},
        {"id": "window_east", "kind": "window", "wall": "east", "offset_m": 0.60, "width_m": 1.00, "height_m": 1.00, "sill_m": 0.90},
    ]},
    "office": {"width": 3.4, "depth": 3.0, "height": 2.5, "seed": 17515, "floor": [0.40, 0.40, 0.42], "wall": [0.88, 0.88, 0.86], "ceiling": [0.94, 0.94, 0.92], "openings": [
        {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.25, "width_m": 0.90, "height_m": 2.10, "sill_m": 0.0},
        {"id": "window_north", "kind": "window", "wall": "north", "offset_m": 0.50, "width_m": 1.20, "height_m": 1.00, "sill_m": 0.90},
    ]},
    "workshop": {"width": 5.0, "depth": 4.0, "height": 2.8, "seed": 17516, "floor": [0.35, 0.34, 0.32], "wall": [0.75, 0.74, 0.70], "ceiling": [0.90, 0.90, 0.88], "openings": [
        {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.40, "width_m": 0.90, "height_m": 2.10, "sill_m": 0.0},
        {"id": "window_north", "kind": "window", "wall": "north", "offset_m": 0.70, "width_m": 1.60, "height_m": 1.10, "sill_m": 0.90},
    ]},
}


def room_spec(name: str) -> dict:
    row = ROOMS[name]
    spec = make_specification(room_id=f"room_{name}", width=row["width"], depth=row["depth"], height=row["height"], seed=row["seed"], openings=row["openings"])
    spec["materials"] = {
        "floor": {"type": "diffuse", "reflectance": row["floor"]},
        "wall": {"type": "diffuse", "reflectance": row["wall"]},
        "ceiling": {"type": "diffuse", "reflectance": row["ceiling"]},
    }
    return spec


def _write_furniture(destination: Path, built: dict) -> None:
    payload = {
        "geometry/render.obj": write_obj(built["vertices"], built["polygons"], name=built["asset_id"]),
        "geometry/collision.obj": write_obj(built["vertices"], built["polygons"], name=f"{built['asset_id']}_collision"),
    }
    supports = []
    if built["support"] is not None:
        rect = built["support"]["rectangle_m"]
        surface = support_quad((rect[0], rect[1]), (rect[2], rect[3]), built["support"]["z_m"], (0.0, 0.0, 1.0))
        surface["id"] = "top"
        supports.append(surface)
    material = {key: value for key, value in built["material"].items() if key != "research_role"}
    manifest = {
        "asset_id": built["asset_id"],
        "version": VERSION,
        "units": "m",
        "frame": {"up_axis": "Z", "handedness": "right", "transform_convention": "row_major_column_vector", "matrix": serialize_matrix(identity_matrix())},
        "provenance": {
            "source": "sensor_sim",
            "generator": "interior_assets",
            "generator_version": "1.0.0",
            "seed": RENDER_SEED,
            "parameters": {"asset_id": built["asset_id"]},
            "license": "project-owned procedural geometry",
            "license_status": "permitted",
        },
        "geometry": {
            "profile": built["profile"],
            "units": "m",
            "polygon_count": built["report"]["polygon_count"],
            "triangle_count": built["triangle_count"],
            "bounds_min": [float(value) for value in built["report"]["bounds_min"]],
            "bounds_max": [float(value) for value in built["report"]["bounds_max"]],
            "functional_zones": built["zones"],
            "interior": built["interior"],
            "components": [
                {"id": component["id"], "profile": component["profile"], **({"thickness_axis": component["thickness_axis"]} if "thickness_axis" in component else {})}
                for component in built["components"]
            ],
        },
        "collision": [{"id": "proxy", "path": "geometry/collision.obj", "render_path": "geometry/render.obj", "triangle_count": built["triangle_count"], "qualified": True}],
        "support_surfaces": supports,
        "materials": [{"id": "default", **material}],
    }
    create_package(destination, manifest, payload, {"geometry/render.obj": "render", "geometry/collision.obj": "collision"})


def package_furniture(catalog: Path) -> list[dict]:
    catalog = Path(catalog)
    written = []
    for asset_id in BUILDERS:
        built = build_asset(asset_id)
        destination = catalog / asset_id / VERSION
        if not (destination / "asset.json").is_file():
            _write_furniture(destination, built)
        manifest = verify_package(destination)
        written.append({
            "asset_id": asset_id,
            "version": VERSION,
            "path": f"{asset_id}/{VERSION}",
            "package_sha256": manifest["package_sha256"],
            "status": "technically_validated",
            "visually_reviewed": False,
            "admitted": False,
            "registered": False,
            "components": [component["id"] for component in built["components"]],
        })
        logger.info("interior_scenes event=package asset_id=%s package_sha256=%s", asset_id, manifest["package_sha256"])
    return written


def package_rooms(catalog: Path) -> list[dict]:
    catalog = Path(catalog)
    written = []
    for name in ROOMS:
        spec = room_spec(name)
        floor = catalog / f"{spec['room_id']}_floor" / VERSION / "asset.json"
        if floor.is_file():
            for group in ("floor", "ceiling", "wall_south", "wall_north", "wall_east", "wall_west"):
                asset_id = f"{spec['room_id']}_{group}"
                manifest = verify_package(catalog / asset_id / VERSION)
                written.append({"asset_id": asset_id, "version": VERSION, "path": f"{asset_id}/{VERSION}", "package_sha256": manifest["package_sha256"]})
            continue
        rows = package_profile(spec, catalog)
        written.extend(rows)
        logger.info("interior_scenes event=room name=%s packages=%d", name, len(rows))
    return written


def write_catalog_index(catalog: Path, furniture: list[dict], rooms: list[dict]) -> None:
    payload = {
        "schema_version": "interior_catalog_v1",
        "status": "technically_validated",
        "visually_reviewed": False,
        "admitted": False,
        "registered": False,
        "packages": furniture + rooms,
    }
    (Path(catalog) / "interior_catalog.json").write_bytes(canonical_json_bytes(payload))


def _locate(asset_id: str, version: str) -> tuple[Path, str]:
    for root in (ROOT / "catalog", SEMANTIC, CANDIDATES):
        path = root / asset_id / version
        if (path / "asset.json").is_file():
            return root, version
    if asset_id == "furniture_workbench" and (SEMANTIC / asset_id / "1.0.2" / "asset.json").is_file():
        return SEMANTIC, "1.0.2"
    raise FileNotFoundError(asset_id)


def _load(asset_id: str, version: str = VERSION) -> dict:
    if asset_id in BUILDERS:
        return build_asset(asset_id)
    root, found = _locate(asset_id, version)
    loaded = load_mesh(root, asset_id, found)
    loaded["asset_id"] = asset_id
    loaded["role"] = "furniture"
    loaded["material"] = loaded["manifest"]["materials"][0]
    support = next((item for item in loaded["manifest"].get("support_surfaces") or [] if item.get("id") == "top"), None)
    if support is None:
        loaded["support"] = None
    else:
        polygon = np.asarray(support["polygon"], dtype=np.float64)
        loaded["support"] = {
            "rectangle_m": [float(polygon[:, 0].min()), float(polygon[:, 1].min()), float(polygon[:, 0].max()), float(polygon[:, 1].max())],
            "z_m": float(polygon[0, 2]),
        }
        loaded["support_polygon"] = polygon
    return loaded


def _floor_item(asset_id: str, instance_id: str, center, yaw: float, version: str = VERSION, mounted_z: float | None = None) -> dict:
    built = _load(asset_id, version)
    vertices = built["vertices"]
    z_value = 0.0 if mounted_z is None else float(mounted_z)
    matrix = _yaw_about_footprint(vertices, yaw, np.asarray(center, dtype=np.float64), z_value)
    world = world_vertices(vertices, matrix)
    return {
        "instance_id": instance_id,
        "asset_id": asset_id,
        "version": version if asset_id not in BUILDERS else VERSION,
        "role": "furniture",
        "matrix": matrix,
        "vertices": world,
        "faces": built["faces"],
        "support": built.get("support"),
        "mounted": mounted_z is not None,
        "material": built.get("material"),
    }


def _on_surface(asset_id: str, instance_id: str, parent: dict, xy, yaw: float = 0.0, version: str = VERSION, lay_on_side: bool = False) -> dict:
    built = _load(asset_id, version)
    support = parent["support"]
    local = _slot_matrix(built["vertices"], xy, support["z_m"], yaw, lay_on_side=lay_on_side)
    matrix = parent["matrix"] @ local
    world = world_vertices(built["vertices"], matrix)
    rect = support["rectangle_m"]
    polygon = apply_points(parent["matrix"], np.array([
        [rect[0], rect[1], support["z_m"]],
        [rect[2], rect[1], support["z_m"]],
        [rect[2], rect[3], support["z_m"]],
        [rect[0], rect[3], support["z_m"]],
    ], dtype=np.float64))
    contact = measure_support(world, built["faces"], polygon)
    return {
        "instance_id": instance_id,
        "asset_id": asset_id,
        "version": version if asset_id not in BUILDERS else VERSION,
        "role": "object",
        "matrix": matrix,
        "vertices": world,
        "faces": built["faces"],
        "support_parent": parent["instance_id"],
        "contact": contact,
        "material": built.get("material"),
        "support": built.get("support"),
    }


def _chairs(table: dict, count_prefix: str = "chair") -> list[dict]:
    loaded = load_mesh(CANDIDATES, "prop_chair", VERSION)
    table_low, table_high = table["vertices"].min(0), table["vertices"].max(0)
    chair_size = loaded["vertices"].max(0) - loaded["vertices"].min(0)
    center = 0.5 * (table_low + table_high)
    chairs = []
    offsets = ((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0))
    for index, (sx, sy) in enumerate(offsets, start=1):
        gap = 0.5 * float(chair_size[0 if sy == 0 else 1]) + 0.12
        center_xy = np.array([
            center[0] + sx * (0.5 * (table_high[0] - table_low[0]) + gap),
            center[1] + sy * (0.5 * (table_high[1] - table_low[1]) + gap),
        ])
        desired = center[:2] - center_xy
        yaw = float(np.atan2(desired[0], -desired[1]))
        matrix = _yaw_about_footprint(loaded["vertices"], yaw, center_xy, 0.0)
        world = world_vertices(loaded["vertices"], matrix)
        chairs.append({
            "instance_id": f"{count_prefix}_{index}",
            "asset_id": "prop_chair",
            "version": VERSION,
            "role": "furniture",
            "matrix": matrix,
            "vertices": world,
            "faces": loaded["faces"],
            "support": None,
            "mounted": False,
            "material": loaded["manifest"]["materials"][0],
        })
    return chairs


def _compose_kitchen(spec) -> list[dict]:
    west = -np.pi / 2.0
    cab_a = _floor_item("furn_base_cabinet", "cabinet_a", [0.36, 1.80], west)
    cab_b = _floor_item("furn_base_cabinet", "cabinet_b", [0.36, 3.05], west)
    upper_a = _floor_item("furn_upper_cabinet", "upper_a", [0.22, 1.80], west, mounted_z=1.48)
    upper_b = _floor_item("furn_upper_cabinet", "upper_b", [0.22, 3.05], west, mounted_z=1.48)
    oven = _floor_item("furn_oven", "oven_1", [0.36, 4.05], west)
    cooktop = _on_surface("furn_cooktop", "cooktop_1", oven, (0.30, 0.30))
    cooktop["support"] = build_asset("furn_cooktop")["support"]
    sink = _on_surface("furn_sink", "sink_1", cab_a, (0.32, 0.30))
    microwave = _on_surface("furn_microwave", "microwave_1", cab_b, (0.85, 0.28))
    fridge = _floor_item("furn_fridge", "fridge_1", [4.85, 3.15], np.pi / 2.0)
    island = _floor_item("furn_island", "island_1", [2.90, 2.30], 0.0)
    lamp = _on_surface("furn_desk_lamp", "lamp_island", island, (0.30, 0.20))
    plate = _on_surface("prop_plate", "plate_1", island, (0.70, 0.28), version=VERSION)
    board = _on_surface("prop_cutting_board", "board_1", island, (0.95, 0.45), version=VERSION)
    apple = _on_surface("prop_apple", "apple_1", island, (0.55, 0.48), version=VERSION)
    spoon = _on_surface("prop_spoon", "spoon_1", island, (0.18, 0.50), version=VERSION)
    fork = _on_surface("prop_fork", "fork_1", island, (1.15, 0.22), version=VERSION)
    pot = _on_surface("prop_pot", "pot_1", cooktop, (0.30, 0.26), version=VERSION)
    spatula = _on_surface("prop_spatula", "spatula_1", cooktop, (0.48, 0.40), version=VERSION, lay_on_side=True)
    return [cab_a, cab_b, upper_a, upper_b, oven, cooktop, sink, microwave, fridge, island, lamp, plate, board, apple, spoon, fork, pot, spatula]


def _compose_dining(spec) -> list[dict]:
    table = _floor_item("furn_dining_table", "table_1", [2.30, 1.80], 0.0)
    sideboard = _floor_item("furn_sideboard", "sideboard_1", [3.40, 3.25], np.pi)
    lamp = _on_surface("furn_desk_lamp", "lamp_side", sideboard, (0.30, 0.20))
    items = [table, sideboard, lamp, *_chairs(table)]
    items.append(_on_surface("prop_plate", "plate_1", table, (0.40, 0.25)))
    items.append(_on_surface("prop_plate", "plate_2", table, (1.20, 0.25)))
    items.append(_on_surface("prop_bowl", "bowl_1", table, (0.40, 0.55)))
    items.append(_on_surface("prop_bowl", "bowl_2", table, (1.20, 0.55)))
    items.append(_on_surface("prop_cup", "cup_1", table, (0.70, 0.40)))
    items.append(_on_surface("prop_bottle", "bottle_1", table, (0.85, 0.65)))
    items.append(_on_surface("prop_fork", "fork_1", table, (0.55, 0.18)))
    items.append(_on_surface("prop_serving_dish", "dish_1", table, (1.00, 0.42)))
    return items


def _compose_living(spec) -> list[dict]:
    sofa = _floor_item("furn_sofa", "sofa_1", [3.60, 1.35], 0.0)
    chair = _floor_item("furn_armchair", "armchair_1", [1.70, 2.40], np.pi / 2.0)
    coffee = _floor_item("furn_coffee", "coffee_1", [3.60, 2.30], 0.0)
    media = _floor_item("furn_media", "media_1", [3.40, 4.40], np.pi)
    tv = _on_surface("furn_tv", "tv_1", media, (0.70, 0.18))
    shelf = _floor_item("furn_bookcase", "shelf_1", [0.55, 3.80], np.pi / 2.0)
    lamp = _floor_item("furn_floor_lamp", "lamp_1", [5.40, 1.20], 0.0)
    book = _on_surface("prop_book", "book_1", coffee, (0.40, 0.30))
    cup = _on_surface("prop_cup", "cup_1", coffee, (0.75, 0.35))
    phone = _on_surface("prop_phone", "phone_1", coffee, (0.90, 0.18))
    return [sofa, chair, coffee, media, tv, shelf, lamp, book, cup, phone]


def _compose_bedroom(spec) -> list[dict]:
    bed = _floor_item("furn_bed", "bed_1", [2.15, 1.85], 0.0)
    stand_a = _floor_item("furn_nightstand", "stand_a", [0.55, 2.45], 0.0)
    stand_b = _floor_item("furn_nightstand", "stand_b", [3.75, 2.45], 0.0)
    wardrobe = _floor_item("furn_wardrobe", "wardrobe_1", [3.95, 3.25], np.pi / 2.0)
    lamp = _on_surface("furn_desk_lamp", "lamp_bed", stand_a, (0.22, 0.20))
    book = _on_surface("prop_book", "book_1", stand_b, (0.12, 0.18))
    phone = _on_surface("prop_phone", "phone_1", stand_b, (0.32, 0.28))
    return [bed, stand_a, stand_b, wardrobe, lamp, book, phone]


def _compose_office(spec) -> list[dict]:
    desk = _floor_item("furn_office_desk", "desk_1", [1.80, 2.20], np.pi)
    chair = _chairs(desk, "office_chair")[0]
    shelf = _floor_item("furn_bookcase", "shelf_1", [0.45, 1.60], np.pi / 2.0)
    monitor = _on_surface("furn_monitor", "monitor_1", desk, (0.70, 0.48))
    keyboard = _on_surface("prop_keyboard", "keyboard_1", desk, (0.70, 0.25))
    mouse = _on_surface("furn_mouse", "mouse_1", desk, (1.10, 0.28))
    notebook = _on_surface("prop_notebook", "notebook_1", desk, (0.28, 0.30), version=VERSION)
    pen = _on_surface("prop_pen", "pen_1", desk, (0.40, 0.16), version=VERSION, lay_on_side=True)
    phone = _on_surface("prop_phone", "phone_1", desk, (0.22, 0.48), version=VERSION)
    lamp = _on_surface("furn_desk_lamp", "lamp_desk", desk, (1.15, 0.50))
    return [desk, chair, shelf, monitor, keyboard, mouse, notebook, pen, phone, lamp]


def _compose_workshop(spec) -> list[dict]:
    semantic_assets = {aid: semantic_build(aid) for aid in (
        "furniture_workbench", "prop_work_panel", "prop_fixture_block", "prop_stock_plate", "prop_screwdriver", "prop_pliers", "prop_wrench", "prop_hammer", "prop_clamp", "prop_hex_key_group", "prop_cable", "prop_fastener_tray", "prop_wire_spool", "prop_wood_offcut", "prop_metal_bracket", "prop_parts_bin", "prop_component_block",
    )}
    bench = _place_workbench(semantic_assets["furniture_workbench"], spec)
    bench["version"] = "1.0.2"
    bench["mounted"] = False
    bench["material"] = semantic_assets["furniture_workbench"]["material"]
    polygon = bench["support_polygon"]
    bench["support"] = {"z_m": bench["support_z"], "rectangle_m": [float(polygon[:, 0].min()), float(polygon[:, 1].min()), float(polygon[:, 0].max()), float(polygon[:, 1].max())]}
    items = [bench]
    for instance_id, asset_id, zone_id, slot in _object_slots("workbench_high"):
        placed = _place_object(instance_id, asset_id, zone_id, slot, 0.0, bench, semantic_assets, CANDIDATES)
        placed["version"] = "1.0.0" if asset_id != "furniture_workbench" else "1.0.2"
        placed["role"] = "object"
        placed["support_parent"] = bench["instance_id"]
        placed["mounted"] = False
        items.append(placed)
    lamp = _on_surface("furn_desk_lamp", "lamp_bench", bench, (1.55, 0.25))
    items.append(lamp)
    return items


COMPOSERS = {
    "kitchen": _compose_kitchen,
    "dining": _compose_dining,
    "living": _compose_living,
    "bedroom": _compose_bedroom,
    "office": _compose_office,
    "workshop": _compose_workshop,
}


def _aabb_gap(a_min, a_max, b_min, b_max) -> float:
    delta = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)
    return float(np.linalg.norm(delta))


def _keepout(opening, spec):
    minimum = np.array(opening["clear_aperture_m"]["minimum_m"], dtype=np.float64)
    maximum = np.array(opening["clear_aperture_m"]["maximum_m"], dtype=np.float64)
    wall = opening["wall"]
    if wall == "north":
        minimum[1] -= 0.20
    elif wall == "south":
        maximum[1] += 0.20
    elif wall == "east":
        minimum[0] -= 0.20
    else:
        maximum[0] += 0.20
    return minimum, maximum


def _overlaps(a_min, a_max, b_min, b_max) -> bool:
    return bool(np.all(a_min < b_max) and np.all(a_max > b_min))


def validate_layout(items: list[dict], spec: dict) -> list[dict]:
    failures = []
    floor = np.array([[0.0, 0.0, 0.0], [spec["interior_width_m"], 0.0, 0.0], [spec["interior_width_m"], spec["interior_depth_m"], 0.0], [0.0, spec["interior_depth_m"], 0.0]], dtype=np.float64)
    door = next(item for item in spec["openings"] if item["kind"] == "door")
    corridor_min = np.array([door["offset_m"], 0.0, 0.0])
    corridor_max = np.array([door["offset_m"] + door["width_m"], 0.90, 2.2])
    for item in items:
        low, high = item["vertices"].min(0), item["vertices"].max(0)
        if not item.get("mounted"):
            contact = measure_support(item["vertices"], item["faces"], floor)
            item["floor_contact"] = contact
            if item.get("support_parent") is None and not contact["passed"]:
                failures.append({"name": f"floor:{item['instance_id']}", "passed": False, "reason": contact.get("reason")})
        if item.get("contact") is not None and not item["contact"]["passed"]:
            failures.append({"name": f"support:{item['instance_id']}", "passed": False, "reason": item["contact"].get("reason")})
        if _overlaps(low, high, corridor_min, corridor_max):
            failures.append({"name": f"door:{item['instance_id']}", "passed": False})
        for opening in spec["openings"]:
            if opening["kind"] != "window":
                continue
            kmin, kmax = _keepout(opening, spec)
            if _overlaps(low, high, kmin, kmax):
                failures.append({"name": f"window:{item['instance_id']}:{opening['id']}", "passed": False})
        if float(low[0]) < -0.02 or float(high[0]) > spec["interior_width_m"] + 0.02 or float(low[1]) < -0.02 or float(high[1]) > spec["interior_depth_m"] + 0.02:
            failures.append({"name": f"bounds:{item['instance_id']}", "passed": False})
        if not np.isfinite(item["matrix"]).all():
            failures.append({"name": f"transform:{item['instance_id']}", "passed": False})
    shell = _shell_meshes(spec)
    for item in items:
        for wall_id, wall in shell.items():
            if wall_id == "floor":
                continue
            found = unintended_intersection({"vertices": item["vertices"], "faces": item["faces"]}, wall)
            if found["unintended"] or float(found["clearance_m"]) < SEPARATION:
                failures.append({"name": f"wall:{item['instance_id']}:{wall_id}", "passed": False, "clearance_m": found["clearance_m"]})
    for index, left in enumerate(items):
        left_low, left_high = left["vertices"].min(0), left["vertices"].max(0)
        for right in items[index + 1 :]:
            right_low, right_high = right["vertices"].min(0), right["vertices"].max(0)
            if _aabb_gap(left_low, left_high, right_low, right_high) > 0.02:
                continue
            if left.get("support_parent") == right["instance_id"] or right.get("support_parent") == left["instance_id"]:
                continue
            found = mesh_clearance({"vertices": left["vertices"], "faces": left["faces"]}, {"vertices": right["vertices"], "faces": right["faces"]})
            if float(found["clearance_m"]) < SEPARATION:
                failures.append({"name": f"clearance:{left['instance_id']}:{right['instance_id']}", "passed": False, "clearance_m": found["clearance_m"]})
    from sensor_sim.scene_gen.placement_separation import PlacementSeparationError, require_outside

    allowed = {frozenset((item["instance_id"], item["support_parent"])) for item in items if item.get("support_parent")}
    try:
        require_outside(items, allowed_pairs=allowed, minimum_m=SEPARATION)
    except PlacementSeparationError as exc:
        failures.append({"name": "object_inside", "passed": False, "reason": str(exc)})
    return failures


def _aims_through_opening(origin, target, spec) -> bool:
    start = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(target, dtype=np.float64) - start
    direction = direction / np.linalg.norm(direction)
    for opening in spec["openings"]:
        minimum = np.array(opening["clear_aperture_m"]["minimum_m"], dtype=np.float64)
        maximum = np.array(opening["clear_aperture_m"]["maximum_m"], dtype=np.float64)
        wall = opening["wall"]
        if wall in {"south", "north"}:
            plane = 0.0 if wall == "south" else float(spec["interior_depth_m"])
            if abs(direction[1]) < 1e-8:
                continue
            distance = (plane - start[1]) / direction[1]
            axis = 0
        else:
            plane = 0.0 if wall == "west" else float(spec["interior_width_m"])
            if abs(direction[0]) < 1e-8:
                continue
            distance = (plane - start[0]) / direction[0]
            axis = 1
        target_distance = float(np.linalg.norm(np.asarray(target, dtype=np.float64) - start))
        if distance <= 0.2 or distance > target_distance - 0.05:
            continue
        point = start + direction * distance
        if minimum[axis] - 0.05 <= point[axis] <= maximum[axis] + 0.05 and minimum[2] - 0.05 <= point[2] <= maximum[2] + 0.05:
            return True
    return False


def _inside_room(origin, spec, margin: float = 0.35, z_min: float = 0.55) -> bool:
    x_value, y_value, z_value = origin
    return margin < x_value < spec["interior_width_m"] - margin and margin < y_value < spec["interior_depth_m"] - margin and z_min < z_value < spec["interior_height_m"] - 0.15


def _ray_aabb(origin, direction, low, high):
    tmin, tmax = 1e-3, 1e6
    for axis in range(3):
        if abs(float(direction[axis])) < 1e-9:
            if origin[axis] < low[axis] or origin[axis] > high[axis]:
                return None
            continue
        inv = 1.0 / float(direction[axis])
        near = (low[axis] - origin[axis]) * inv
        far = (high[axis] - origin[axis]) * inv
        if near > far:
            near, far = far, near
        tmin = max(tmin, float(near))
        tmax = min(tmax, float(far))
        if tmax < tmin:
            return None
    return tmin


def _classify_view(origin, target, fov, spec, items) -> dict:
    """Count a 32 by 18 ray grid. This is the orientation probe before the RGB render."""
    start = np.asarray(origin, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - start
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    grid_w, grid_h = 32, 18
    tan_h = float(np.tan(np.deg2rad(fov) * 0.5))
    tan_v = tan_h * (grid_h / grid_w)
    boxes = [(item["instance_id"], item["vertices"].min(0), item["vertices"].max(0)) for item in items if item["role"] == "furniture"]
    counts = {"opening": 0, "floor": 0, "ceiling": 0, "furniture": 0, "wall_south": 0, "wall_north": 0, "wall_east": 0, "wall_west": 0, "miss": 0}
    hit_ids = set()
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    height = float(spec["interior_height_m"])
    for row in range(grid_h):
        for column in range(grid_w):
            x_ndc = ((column + 0.5) / grid_w) * 2.0 - 1.0
            y_ndc = 1.0 - ((row + 0.5) / grid_h) * 2.0
            direction = forward + right * (x_ndc * tan_h) + cam_up * (y_ndc * tan_v)
            direction = direction / np.linalg.norm(direction)
            best_t = 1e6
            kind = "miss"
            hit_id = None
            for instance_id, low, high in boxes:
                found = _ray_aabb(start, direction, low, high)
                if found is not None and found < best_t:
                    best_t = found
                    kind = "furniture"
                    hit_id = instance_id
            if hit_id is not None and kind == "furniture":
                hit_ids.add(hit_id)
            if direction[2] < -1e-8:
                distance = (0.0 - start[2]) / direction[2]
                point = start + direction * distance
                if distance > 1e-3 and distance < best_t and 0.0 <= point[0] <= width and 0.0 <= point[1] <= depth:
                    best_t = distance
                    kind = "floor"
            if direction[2] > 1e-8:
                distance = (height - start[2]) / direction[2]
                point = start + direction * distance
                if distance > 1e-3 and distance < best_t and 0.0 <= point[0] <= width and 0.0 <= point[1] <= depth:
                    best_t = distance
                    kind = "ceiling"
            planes = (
                ("wall_south", 1, 0.0, direction[1] < -1e-8),
                ("wall_north", 1, depth, direction[1] > 1e-8),
                ("wall_west", 0, 0.0, direction[0] < -1e-8),
                ("wall_east", 0, width, direction[0] > 1e-8),
            )
            for name, axis, plane, going in planes:
                if not going:
                    continue
                distance = (plane - start[axis]) / direction[axis]
                if distance <= 1e-3 or distance >= best_t:
                    continue
                point = start + direction * distance
                other = 1 - axis
                limit = depth if axis == 0 else width
                if point[2] < 0.0 or point[2] > height or point[other] < 0.0 or point[other] > limit:
                    continue
                best_t = distance
                kind = name
                for opening in spec["openings"]:
                    if opening["wall"] != name.replace("wall_", ""):
                        continue
                    minimum = np.array(opening["clear_aperture_m"]["minimum_m"], dtype=np.float64)
                    maximum = np.array(opening["clear_aperture_m"]["maximum_m"], dtype=np.float64)
                    if np.all(point >= minimum - 1e-6) and np.all(point <= maximum + 1e-6):
                        kind = "opening"
            counts[kind] += 1
    total = float(grid_w * grid_h)
    fractions = {key: counts[key] / total for key in counts}
    wall_max = max(fractions[key] for key in fractions if key.startswith("wall_"))
    rejected = fractions["opening"] > 0.35 or fractions["floor"] > 0.62 or wall_max > 0.62 or fractions["furniture"] < 0.08
    return {"fractions": fractions, "wall_max": wall_max, "rejected": rejected, "furniture_ids": sorted(hit_ids)}


def _clear_of_furniture(origin, items, gap: float = 0.55) -> bool:
    point = np.asarray(origin, dtype=np.float64)
    for item in items:
        if item["role"] != "furniture":
            continue
        low, high = item["vertices"].min(0), item["vertices"].max(0)
        delta = np.maximum(np.maximum(low - point, point - high), 0.0)
        if float(np.linalg.norm(delta)) < gap:
            return False
    return True


def _opening_is_behind(origin, target, spec) -> bool:
    start = np.asarray(origin, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - start
    forward = forward / np.linalg.norm(forward)
    nearest = None
    for opening in spec["openings"]:
        minimum = np.array(opening["clear_aperture_m"]["minimum_m"], dtype=np.float64)
        maximum = np.array(opening["clear_aperture_m"]["maximum_m"], dtype=np.float64)
        center = 0.5 * (minimum + maximum)
        offset = center - start
        distance = float(np.linalg.norm(offset))
        behind = float(np.dot(forward, offset / distance)) < 0.0
        if nearest is None or distance < nearest[0]:
            nearest = (distance, behind)
    return bool(nearest and nearest[1])


def _support_world(parent: dict) -> np.ndarray:
    rect = parent["support"]["rectangle_m"]
    z_value = parent["support"]["z_m"]
    return apply_points(parent["matrix"], np.array([
        [rect[0], rect[1], z_value],
        [rect[2], rect[1], z_value],
        [rect[2], rect[3], z_value],
        [rect[0], rect[3], z_value],
    ], dtype=np.float64))


def _task_camera(polygon: np.ndarray, spec: dict) -> dict:
    """Close view of the task surface from the side of the room that has space for it."""
    polygon = np.asarray(polygon, dtype=np.float64)
    center = polygon.mean(axis=0)
    target = np.array([center[0], center[1], center[2] + 0.04], dtype=np.float64)
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    options = [
        np.array([0.0, -1.0, 0.55]),
        np.array([0.0, 1.0, 0.55]),
        np.array([-1.0, 0.0, 0.55]),
        np.array([1.0, 0.0, 0.55]),
    ]
    best = None
    for back in options:
        back = back / np.linalg.norm(back)
        for fov in (50.0, 62.0, 74.0):
            origin = _fit_origin(polygon, target, back, np.array([0.0, 0.0, 1.0]), fov, 1280.0 / 720.0, 0.08)
            clipped = np.array([
                float(np.clip(origin[0], 0.25, width - 0.25)),
                float(np.clip(origin[1], 0.25, depth - 0.25)),
                float(np.clip(origin[2], 0.70, float(spec["interior_height_m"]) - 0.20)),
            ])
            shift = float(np.linalg.norm(clipped - origin))
            peak = float(_peak_ndc(polygon, clipped, target, np.array([0.0, 0.0, 1.0]), fov, 1280.0 / 720.0))
            if _aims_through_opening(clipped, target, spec):
                continue
            distance_xy = float(np.linalg.norm(clipped[:2] - target[:2]))
            score = (shift > 0.12, peak > 0.96, -distance_xy, peak)
            if best is None or score < best[0]:
                best = (score, {
                    "name": "task",
                    "origin": clipped.tolist(),
                    "target": target.tolist(),
                    "up": [0.0, 0.0, 1.0],
                    "fov_deg": fov,
                    "resolution": list(RESOLUTION),
                    "peak_ndc": peak,
                    "clip_shift_m": shift,
                })
    if best is None:
        fallback = frame_support_camera(polygon, spec)
        fallback["name"] = "task"
        fallback["resolution"] = list(RESOLUTION)
        return fallback
    return best[1]


def interior_cameras(spec: dict, items: list[dict], task_parent: dict) -> dict:
    majors = [item for item in items if item["role"] == "furniture"]
    center = np.mean([item["vertices"].mean(0) for item in majors], axis=0)
    target = [float(center[0]), float(center[1]), 1.05]
    width, depth = float(spec["interior_width_m"]), float(spec["interior_depth_m"])
    eye = min(1.55, float(spec["interior_height_m"]) - 0.4)
    corners = []
    for inset, z_value in ((0.90, eye), (1.20, eye)):
        corners.extend([
            [inset, 0.90, z_value],
            [width - inset, 0.90, z_value],
            [width - inset, depth - inset, z_value],
            [inset, depth - inset, z_value],
            [width * 0.5, 0.90, z_value],
            [width - 0.90, depth * 0.5, z_value],
            [0.90, depth * 0.5, z_value],
        ])
    accepted = []
    rejected = []
    for origin in corners:
        if not _inside_room(origin, spec) or _aims_through_opening(origin, target, spec) or not _clear_of_furniture(origin, items, 0.40):
            rejected.append({"origin": origin, "reason": "outside_or_opening_ray"})
            continue
        for fov in (55.0, 80.0, 100.0):
            report = _classify_view(origin, target, fov, spec, items)
            record = {"origin": origin, "fov": fov, "probe": report, "opening_behind": _opening_is_behind(origin, target, spec)}
            if report["rejected"] or report["wall_max"] > 0.45 or report["fractions"]["furniture"] < 0.15 or report["fractions"]["floor"] > 0.45:
                rejected.append({**record, "reason": "probe"})
                continue
            accepted.append(record)
    if not accepted:
        raise RuntimeError(f"interior camera failed for {spec['room_id']} rejected={len(rejected)}")
    def _overview_key(row):
        furn = row["probe"]["fractions"]["furniture"]
        crowded = max(0.0, furn - 0.55)
        return (not row["opening_behind"], -len(row["probe"]["furniture_ids"]), row["probe"]["wall_max"] + crowded, abs(furn - 0.38))

    accepted.sort(key=_overview_key)
    primary_row = accepted[0]

    def _apart(row) -> bool:
        return float(np.linalg.norm(np.asarray(row["origin"][:2]) - np.asarray(primary_row["origin"][:2]))) >= 1.2

    distant = [row for row in accepted if row is not primary_row and _apart(row)]
    if not distant:
        distant = [
            row for row in rejected
            if "probe" in row and _apart(row) and row["probe"]["fractions"]["opening"] <= 0.25 and row["probe"]["wall_max"] <= 0.55 and row["probe"]["fractions"]["furniture"] >= 0.12
        ]
        distant.sort(key=_overview_key)
    if not distant:
        raise RuntimeError(f"secondary camera failed for {spec['room_id']}")
    secondary_row = distant[0]
    primary = {"name": "primary", "origin": primary_row["origin"], "target": target, "up": [0.0, 0.0, 1.0], "fov_deg": primary_row["fov"], "resolution": list(RESOLUTION), "probe": primary_row["probe"]["fractions"], "furniture_ids": primary_row["probe"]["furniture_ids"], "opening_behind": primary_row["opening_behind"]}
    secondary = {"name": "secondary", "origin": secondary_row["origin"], "target": target, "up": [0.0, 0.0, 1.0], "fov_deg": secondary_row["fov"], "resolution": list(RESOLUTION), "probe": secondary_row["probe"]["fractions"], "furniture_ids": secondary_row["probe"]["furniture_ids"], "opening_behind": secondary_row["opening_behind"]}
    task = _task_camera(_support_world(task_parent), spec)
    if not _inside_room(task["origin"], spec, margin=0.20, z_min=0.45) or _aims_through_opening(task["origin"], task["target"], spec):
        raise RuntimeError(f"task camera failed for {spec['room_id']} origin={task['origin']}")
    return {"primary": primary, "secondary": secondary, "task": task, "rejected_candidates": len(rejected)}


def _lighting(spec: dict, lamp_center: list[float]) -> dict:
    width, depth, height = float(spec["interior_width_m"]), float(spec["interior_depth_m"]), float(spec["interior_height_m"])
    ceiling = {
        "id": "ceiling_panel",
        "center_m": [width * 0.5, depth * 0.5, height - 0.08],
        "size_m": [max(0.8, width - 0.8), max(0.8, depth - 0.8)],
        "radiance": 1.8,
        "polarized": False,
        "calibrated": False,
    }
    emitters = []
    for opening in spec["openings"]:
        if opening["kind"] != "window":
            continue
        record = _emitter_record(spec, opening)
        record["radiance"] = 1.6
        emitters.append(record)
    lamp = {"id": "task_lamp", "center_m": lamp_center, "size_m": [0.35, 0.35], "radiance": 12.0}
    return {"ceiling": ceiling, "daylight": {"id": "windows", "emitters": emitters, "radiance": 1.6, "polarized": False, "calibrated": False}, "lamp": lamp}


def _lamp_emitter(lamp: dict) -> dict:
    import mitsuba as mi

    require_variant()
    transform = mi.ScalarTransform4f.translate(lamp["center_m"]) @ mi.ScalarTransform4f.rotate([1.0, 0.0, 0.0], 180.0) @ mi.ScalarTransform4f.scale([lamp["size_m"][0] * 0.5, lamp["size_m"][1] * 0.5, 1.0])
    return {"type": "rectangle", "to_world": transform, "emitter": {"type": "area", "radiance": {"type": "rgb", "value": lamp["radiance"]}}}


def _scene_for(items, camera, spec_lighting) -> dict:
    instances = []
    for item in items:
        root, version = _locate(item["asset_id"], item["version"])
        mesh = root / item["asset_id"] / version / "geometry" / "render.obj"
        material = item.get("material") or verify_package(root / item["asset_id"] / version)["materials"][0]
        if item["asset_id"] in SCENE_AC_BSDF:
            material = dict(SCENE_AC_BSDF[item["asset_id"]])
        instances.append({"id": item["instance_id"], "mesh": mesh, "matrix": item["matrix"], "bsdf": material})
    scene = _scene(instances, camera, tuple(camera.get("resolution", RESOLUTION)), _ceiling_emitter({"lighting": spec_lighting}))
    for emitter in _daylight_emitters({"lighting": spec_lighting}):
        scene[emitter["id"]] = emitter["mitsuba"]
    scene["task_lamp"] = _lamp_emitter(spec_lighting["lamp"])
    return scene


def _public_item(item: dict) -> dict:
    root, version = _locate(item["asset_id"], item["version"])
    digest = verify_package(root / item["asset_id"] / version)["package_sha256"]
    return {
        "instance_id": item["instance_id"],
        "asset_id": item["asset_id"],
        "version": version,
        "role": item["role"],
        "package_sha256": digest,
        "catalog": str(root),
        "matrix": serialize_matrix(item["matrix"]),
        "support_parent": item.get("support_parent"),
        "mounted": bool(item.get("mounted")),
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


def replay_interior(bundle_dir: Path) -> dict:
    """Reload the saved recipe and bundle, check hashes, and rebuild scene.json in a temp directory.

    This does not remesh, retrieve assets, or call task replay or room replay.
    """
    bundle_dir = Path(bundle_dir)
    names = ("bundle.json", "scene.json", "recipe.json")
    before = {name: hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest() for name in names}
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    recipe = json.loads((bundle_dir / "recipe.json").read_text(encoding="utf-8"))
    if recipe != bundle["recipe"]:
        raise RuntimeError("recipe mismatch")
    rebuilt = canonical_json_bytes(scene_document(bundle))
    with tempfile.TemporaryDirectory(prefix="interior-replay-") as temporary:
        target = Path(temporary) / "scene.json"
        target.write_bytes(rebuilt)
        if target.read_bytes() != (bundle_dir / "scene.json").read_bytes():
            raise RuntimeError("replay mismatch")
    for item in bundle["instances"]:
        manifest = verify_package(Path(item["catalog"]) / item["asset_id"] / item["version"])
        if manifest["package_sha256"] != item["package_sha256"]:
            raise RuntimeError(f"hash {item['asset_id']}")
    after = {name: hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest() for name in names}
    if after != before:
        raise RuntimeError("replay modified source")
    logger.info("interior_scenes event=replay task=%s result=ok remesh=false", bundle["task_id"])
    return {"valid": True, "task_id": bundle["task_id"], "remeshed": False, "hashes": before}


def _task_parent(name: str, items: list[dict]) -> dict:
    wanted = {"kitchen": "island_1", "dining": "table_1", "living": "coffee_1", "bedroom": "bed_1", "office": "desk_1", "workshop": "workbench_1"}[name]
    return next(item for item in items if item["instance_id"] == wanted)


def _lamp_center(name: str, items: list[dict], spec: dict) -> list[float]:
    parent = _task_parent(name, items)
    center = parent["vertices"].mean(0)
    return [float(center[0]), float(center[1]), min(float(spec["interior_height_m"]) - 0.35, float(center[2]) + 0.85)]


def build_scene(name: str) -> dict:
    spec = room_spec(name)
    items = COMPOSERS[name](spec)
    failures = validate_layout(items, spec)
    present = {item["asset_id"] for item in items}
    missing = sorted(REQUIRED[name] - present)
    if missing:
        failures.append({"name": "required", "passed": False, "missing": missing})
    return {"name": name, "spec": spec, "items": items, "failures": failures}


def _room_instances(spec: dict, catalog: Path) -> list[dict]:
    rows = []
    for group in ("floor", "ceiling", "wall_south", "wall_north", "wall_east", "wall_west"):
        asset_id = f"{spec['room_id']}_{group}"
        manifest = verify_package(Path(catalog) / asset_id / VERSION)
        rows.append({
            "instance_id": asset_id,
            "asset_id": asset_id,
            "version": VERSION,
            "role": "context",
            "package_sha256": manifest["package_sha256"],
            "catalog": str(Path(catalog)),
            "matrix": serialize_matrix(identity_matrix()),
            "support_parent": None,
            "mounted": False,
        })
    return rows


def _composition_warnings(name: str, spec: dict, items: list[dict], cameras: dict) -> dict:
    parent = _task_parent(name, items)
    rect = parent["support"]["rectangle_m"]
    resolution = 0.02
    nx = max(1, int(np.ceil((rect[2] - rect[0]) / resolution)))
    ny = max(1, int(np.ceil((rect[3] - rect[1]) / resolution)))
    grid = np.zeros((ny, nx), dtype=bool)
    inverse = np.linalg.inv(parent["matrix"])
    for child in items:
        if child.get("support_parent") != parent["instance_id"]:
            continue
        local = apply_points(inverse, child["vertices"])
        low, high = local.min(0), local.max(0)
        x0 = int(np.clip(np.floor((low[0] - rect[0]) / resolution), 0, nx - 1))
        x1 = int(np.clip(np.ceil((high[0] - rect[0]) / resolution), 0, nx))
        y0 = int(np.clip(np.floor((low[1] - rect[1]) / resolution), 0, ny - 1))
        y1 = int(np.clip(np.ceil((high[1] - rect[1]) / resolution), 0, ny))
        if x1 > x0 and y1 > y0:
            grid[y0:y1, x0:x1] = True
    free = _largest_free_rectangle(grid, resolution, float(rect[0]), float(rect[1]), 0.20)
    start = np.asarray(cameras["primary"]["origin"], dtype=np.float64)
    target = np.asarray(cameras["primary"]["target"], dtype=np.float64)
    forward = target - start
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    tan_h = float(np.tan(np.deg2rad(cameras["primary"]["fov_deg"]) * 0.5))
    tan_v = tan_h * (RESOLUTION[1] / RESOLUTION[0])
    visible = []
    for item in items:
        if item["role"] != "furniture":
            continue
        offset = item["vertices"].mean(0) - start
        depth = float(np.dot(offset, forward))
        if depth <= 0.05:
            continue
        if abs(float(np.dot(offset, right)) / (depth * tan_h)) <= 1.0 and abs(float(np.dot(offset, up)) / (depth * tan_v)) <= 1.0:
            visible.append(item["instance_id"])
    return {
        "furniture_count": int(sum(item["role"] == "furniture" for item in items)),
        "object_count": int(sum(item["role"] == "object" for item in items)),
        "room_m": [float(spec["interior_width_m"]), float(spec["interior_depth_m"]), float(spec["interior_height_m"])],
        "task_surface_occupancy": float(grid.mean()),
        "usable_free_rectangle_m": {"origin_m": [float(v) for v in free["origin_m"]], "size_m": [float(v) for v in free["size_m"]], "area_m2": float(free["area_m2"])},
        "primary_furniture_in_frustum": visible,
        "definition": "Occupancy is the fraction of 0.02 m cells on the task support covered by child AABBs. The free rectangle is the largest empty rectangle whose shorter side is at least 0.20 m. Frustum membership is the furniture centroid inside the primary view. These are warnings, not cutoffs. The workbench 10-30 percent band is not applied.",
    }


def write_scene(name: str, catalog: Path) -> Path:
    built = build_scene(name)
    if built["failures"]:
        path = ROOT / name / "failure.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"scene": name, "failed": built["failures"]}, indent=2, default=str) + "\n", encoding="utf-8")
        raise RuntimeError(f"{name} failed {built['failures'][:6]}")
    spec = built["spec"]
    items = built["items"]
    cameras = interior_cameras(spec, items, _task_parent(name, items))
    lighting = _lighting(spec, _lamp_center(name, items, spec))
    instances = _room_instances(spec, catalog) + [_public_item(item) for item in items]
    require_scene_context(instances, room_kind=name)
    relations = []
    for item in items:
        if item.get("support_parent"):
            gap = item.get("contact", {}).get("gap_m")
            if gap is None:
                gap = item.get("support_gap_m")
            relations.append({"type": "supported_on", "instances": [item["instance_id"], item["support_parent"]], "passed": True, "measured_m": float(gap or 0.0)})
        elif not item.get("mounted"):
            relations.append({"type": "supported_on", "instances": [item["instance_id"], f"room_{name}_floor"], "passed": True, "measured_m": float(item.get("floor_contact", {}).get("gap_m") or 0.0)})
    recipe = {
        "scene": name,
        "seed": ROOMS[name]["seed"],
        "room_id": spec["room_id"],
        "interior_m": [float(spec["interior_width_m"]), float(spec["interior_depth_m"]), float(spec["interior_height_m"])],
        "dimension_note": DIMENSION_NOTES.get(name, "Matches the planned size."),
        "placement": "room, then major furniture, then objects on those supports",
    }
    bundle = {
        "task_id": name,
        "seed": ROOMS[name]["seed"],
        "recipe": recipe,
        "room_id": spec["room_id"],
        "instances": instances,
        "relations": relations,
        "cameras": {key: value for key, value in cameras.items() if key != "rejected_candidates"},
        "camera_rejections": cameras["rejected_candidates"],
        "lighting": lighting,
        "renderer": {
            "variant": "cuda_ad_spectral_polarized",
            "spp": SPP,
            "seed": RENDER_SEED,
            "resolution": list(RESOLUTION),
            "review": "rgb",
            "exposure": "write_rgb_png sRGB gamma of the film RGB channels. No additional tone mapper.",
        },
        "validation": {"valid": True, "visual_verification": "pending", "admitted": False, "registered": False, "warnings": _composition_warnings(name, spec, items, cameras)},
        "counts": {"furniture": sum(item["role"] == "furniture" for item in items), "objects": sum(item["role"] == "object" for item in items)},
    }
    run = hashlib.sha256(canonical_json_bytes({"scene": name, "seed": bundle["seed"]})).hexdigest()[:16]
    path = ROOT / name / run
    path.mkdir(parents=True, exist_ok=True)
    bundle = _plain(bundle)
    (path / "room_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    (path / "recipe.json").write_bytes(canonical_json_bytes(bundle["recipe"]))
    (path / "bundle.json").write_bytes(canonical_json_bytes(bundle))
    (path / "scene.json").write_bytes(canonical_json_bytes(scene_document(bundle)))
    replay_interior(path)
    logger.info("interior_scenes event=scene name=%s path=%s", name, path)
    return path


def _linear_stats(exr: Path) -> dict:
    import mitsuba as mi

    image = np.array(mi.Bitmap(str(exr)))[:, :, :3].astype(np.float64)
    luminance = image.mean(axis=2)
    return {
        "median": float(np.median(luminance)),
        "p95": float(np.percentile(luminance, 95)),
        "p99": float(np.percentile(luminance, 99)),
        "dark_fraction": float(np.mean(luminance < 0.02)),
        "bright_fraction": float(np.mean(luminance > 1.2)),
        "mean_rgb": [float(value) for value in image.mean(axis=(0, 1))],
    }


def _scale_lighting(lighting: dict, ceiling: float, window: float, lamp: float) -> None:
    lighting["ceiling"]["radiance"] = round(float(lighting["ceiling"]["radiance"]) * ceiling, 4)
    lighting["daylight"]["radiance"] = round(float(lighting["daylight"]["radiance"]) * window, 4)
    for emitter in lighting["daylight"]["emitters"]:
        emitter["radiance"] = lighting["daylight"]["radiance"]
    lighting["lamp"]["radiance"] = round(float(lighting["lamp"]["radiance"]) * lamp, 4)


def _readable(stats: dict) -> bool:
    return 0.10 <= stats["median"] <= 0.38 and stats["p99"] <= 1.15 and stats["bright_fraction"] < 0.04 and stats["dark_fraction"] < 0.35


def _tune_lighting(items: list[dict], camera: dict, lighting: dict) -> dict:
    """Pick radiance from a 160 by 90, 4 spp probe. Exposure stays the existing sRGB conversion."""
    chosen = None
    with tempfile.TemporaryDirectory(prefix="interior-light-") as temporary:
        for attempt in range(3):
            exr = Path(temporary) / f"probe_{attempt}.exr"
            scene = _scene_for(items, {**camera, "resolution": [160, 90]}, lighting)
            render_stokes(scene, exr, 4, RENDER_SEED)
            stats = _linear_stats(exr)
            chosen = stats
            logger.info(
                "interior_scenes event=light_probe attempt=%d median=%.4f p99=%.4f dark=%.3f bright=%.3f ceiling=%.3f window=%.3f lamp=%.3f",
                attempt,
                stats["median"],
                stats["p99"],
                stats["dark_fraction"],
                stats["bright_fraction"],
                lighting["ceiling"]["radiance"],
                lighting["daylight"]["radiance"],
                lighting["lamp"]["radiance"],
            )
            if _readable(stats):
                break
            if stats["p99"] > 1.15 or stats["bright_fraction"] >= 0.04:
                _scale_lighting(lighting, 0.85, 0.55, 0.85)
            elif stats["median"] < 0.10 or stats["dark_fraction"] >= 0.35:
                _scale_lighting(lighting, 1.30, 1.0, 1.25)
            else:
                _scale_lighting(lighting, 0.80, 0.70, 0.80)
    lighting["probe"] = chosen
    lighting["exposure"] = "sRGB gamma in write_rgb_png. No separate exposure control or tone mapper."
    return lighting


def render_scene(path: Path) -> None:
    path = Path(path)
    bundle = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    items = []
    for record in bundle["instances"]:
        root, version = Path(record["catalog"]), record["version"]
        loaded = load_mesh(root, record["asset_id"], version)
        matrix = deserialize_matrix(record["matrix"])
        items.append({**record, "matrix": matrix, "material": loaded["manifest"]["materials"][0]})
    output = path / "render_rgb_v1"
    output.mkdir(parents=True, exist_ok=True)
    bundle["lighting"] = _tune_lighting(items, bundle["cameras"]["primary"], bundle["lighting"])
    images = {}
    for name in ("primary", "secondary", "task"):
        camera = {**bundle["cameras"][name], "resolution": list(RESOLUTION)}
        scene = _scene_for(items, camera, bundle["lighting"])
        exr = output / f"{name}.exr"
        png = output / f"{name}.png"
        render_stokes(scene, exr, SPP, RENDER_SEED)
        write_rgb_png(exr, png)
        stats = _linear_stats(exr)
        images[name] = {"png": str(png), "exr": str(exr), "resolution": list(RESOLUTION), "linear_rgb": stats}
        logger.info("interior_scenes event=rgb scene=%s view=%s median=%.4f dark=%.3f", bundle["task_id"], name, stats["median"], stats["dark_fraction"])
    bundle = _plain(bundle)
    (path / "bundle.json").write_bytes(canonical_json_bytes(bundle))
    replay_interior(path)
    (output / "validation.json").write_text(json.dumps({
        "variant": bundle["renderer"]["variant"],
        "spp": SPP,
        "seed": RENDER_SEED,
        "resolution": list(RESOLUTION),
        "images": images,
        "review": "rgb",
        "exposure": bundle["lighting"]["exposure"],
        "lighting": bundle["lighting"],
        "visual_verification": "pending",
        "admitted": False,
        "registered": False,
    }, indent=2) + "\n", encoding="utf-8")


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_gallery(review: Path) -> None:
    import html
    import os

    review = Path(review)
    review.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Diverse interiors</title>",
        "<style>body{font-family:sans-serif;margin:2rem;background:#111;color:#eee} img{max-width:960px;background:#222} figure{margin:1rem 0} .notice{border:1px solid #c90;padding:1rem}</style>",
        "</head><body><h1>Diverse furnished interiors</h1>",
        "<p class=\"notice\"><strong>RGB visual acceptance. Human visual review, admission, and registration are pending.</strong> Primary views look into the furnished room.</p>",
    ]
    for name in ("kitchen", "dining", "living", "bedroom", "office", "workshop"):
        path = next((ROOT / name).glob("*/bundle.json")).parent
        bundle = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
        spec = json.loads((path / "room_spec.json").read_text(encoding="utf-8"))
        validation = json.loads((path / "render_rgb_v1" / "validation.json").read_text(encoding="utf-8"))
        body.append(f"<h2>{html.escape(name)}</h2>")
        body.append("<p>" + html.escape(
            f"room {spec['interior_width_m']} x {spec['interior_depth_m']} x {spec['interior_height_m']} m "
            f"furniture {bundle['counts']['furniture']} objects {bundle['counts']['objects']} seed {bundle['seed']}"
        ) + "</p>")
        body.append("<p>" + html.escape("inventory " + ", ".join(sorted({item['asset_id'] for item in bundle['instances']}))) + "</p>")
        for view in ("primary", "secondary", "task"):
            image = validation["images"][view]
            href = Path(os.path.relpath(image["png"], review.parent)).as_posix()
            body.append(f"<figure><a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(name + ' ' + view)}\"></a><figcaption>{html.escape(view)} RGB {image['resolution'][0]}x{image['resolution'][1]}</figcaption></figure>")
        camera = bundle["cameras"]["primary"]
        body.append("<p>" + html.escape(f"primary origin {camera['origin']} target {camera['target']} fov {camera['fov_deg']}") + "</p>")
        light = bundle["lighting"]
        body.append("<p>" + html.escape(f"ceiling radiance {light['ceiling']['radiance']} at {light['ceiling']['center_m']} window radiance {light['daylight']['radiance']} lamp {light['lamp']['center_m']} radiance {light['lamp']['radiance']}") + "</p>")
        warnings = bundle["validation"]["warnings"]
        body.append("<p>" + html.escape(
            f"task occupancy {warnings['task_surface_occupancy']:.3f} "
            f"usable free rectangle {warnings['usable_free_rectangle_m']['size_m']} m "
            f"primary furniture in frustum {len(warnings['primary_furniture_in_frustum'])}"
        ) + "</p>")
        body.append("<p>" + html.escape(warnings["definition"]) + "</p>")
        body.append("<p>" + html.escape(bundle["recipe"].get("dimension_note", "")) + "</p>")
        body.append("<p>Geometry, support, wall clearance, and replay passed. Exposure is the existing sRGB conversion. Visual review, admission, and registration are pending. No Stokes image is the main figure.</p>")
    body.append("</body></html>\n")
    review.write_text("\n".join(body), encoding="utf-8")


def pack_zip(zip_path: Path = Path("local_review/diverse_interior_review.zip")) -> None:
    import shutil
    import zipfile

    staging = Path("/tmp/diverse_interior_review")
    if staging.exists():
        shutil.rmtree(staging)
    files = staging / "files"
    files.mkdir(parents=True)
    sections = []
    for name in ("kitchen", "dining", "living", "bedroom", "office", "workshop"):
        path = next((ROOT / name).glob("*/render_rgb_v1"))
        for view in ("primary", "secondary", "task"):
            target = files / f"{name}_{view}.png"
            shutil.copy(path / f"{view}.png", target)
        sections.append(name)
    html = ["<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Diverse interiors</title>",
            "<style>body{font-family:sans-serif;margin:2rem;background:#111;color:#eee} img{max-width:1000px}</style></head><body>",
            "<h1>Diverse furnished interiors</h1>",
            "<p>Open this file after unzipping. Each room starts with the interior overview.</p>"]
    for name in sections:
        html.append(f"<h2>{name}</h2>")
        for view, caption in (("primary", "Primary interior overview"), ("secondary", "Secondary interior"), ("task", "Task close-up")):
            html.append(f"<figure><img src=\"files/{name}_{view}.png\" alt=\"{name} {view}\"><figcaption>{caption}</figcaption></figure>")
    html.append("</body></html>")
    (staging / "index.html").write_text("\n".join(html), encoding="utf-8")
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in staging.rglob("*"):
            if file.is_file():
                archive.write(file, file.relative_to(staging.parent))
    logger.info("interior_scenes event=zip path=%s bytes=%d", zip_path, zip_path.stat().st_size)


def generate_review() -> list[Path]:
    """Package the new catalog, write six scenes, render RGB, and pack the local zip."""
    ROOT.mkdir(parents=True, exist_ok=True)
    catalog = ROOT / "catalog"
    furniture = package_furniture(catalog)
    rooms = package_rooms(catalog)
    write_catalog_index(catalog, furniture, rooms)
    paths = [write_scene(name, catalog) for name in ROOMS]
    for path in paths:
        render_scene(path)
    write_gallery(ROOT / "review" / "index.html")
    pack_zip()
    logger.info("interior_scenes event=done scenes=%d", len(paths))
    return paths

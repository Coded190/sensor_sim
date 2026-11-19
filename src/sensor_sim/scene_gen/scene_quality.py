"""Scene-quality pilots. The old tabletop sampler and its 15 bundles are not rewritten.

Coverage is recomputed from saved meshes. New recipes place named groups inside
explicit table zones. The close camera frames the support polygon. Daylight is
added only after the packaged north wall is shown to be open.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import verify_package
from sensor_sim.scene_gen.asset_validate import triangulate
from sensor_sim.scene_gen.indoor_contact import measure_support, unintended_intersection, world_vertices
from sensor_sim.scene_gen.indoor_mitsuba import render_stokes, require_variant, write_rgb_png
from sensor_sim.scene_gen.indoor_transform import (
    apply_points,
    compose,
    deserialize_matrix,
    identity_matrix,
    rotation_matrix,
    serialize_matrix,
    translation_matrix,
)
from sensor_sim.scene_gen.mesh_io import read_obj
from sensor_sim.scene_gen.narrow import mesh_clearance
from sensor_sim.scene_gen.tabletop_layout import (
    INSET_M,
    _inside_rect,
    _inset_rect,
    _occupancy,
    _relation,
    _union_area,
    _yaw_bounds,
    load_mesh,
)
from sensor_sim.scene_gen.tabletop_render import (
    LIGHT_CENTER,
    LIGHT_RADIANCE,
    LIGHT_SCALE,
    _ceiling_light,
    _instances,
    _project_point,
    _scene,
    instance_visibility,
)
from sensor_sim.scene_gen.tabletop_scene import (
    ROOM_CAMERA,
    SHELL,
    VERSION,
    _bundle,
    _furniture_id,
    _publish,
    replay,
)

logger = logging.getLogger("sensor_sim.scene_gen.scene_quality")

ROOT = Path("dataset/qualification/scene_quality_v2")
OLD_ROOT = Path("dataset/qualification/tabletop_scenes")
CATALOG = Path("assets/candidates")
SEEDS = {"quality_controlled": 17201, "quality_mixed": 17202, "quality_dense": 17203}
RENDER_SEED = 17211
SCENE_SPP = 16
OBSERVATION_SHORT_SIDE_PX = 32
WINDOW_EXPECTATION = {
    "wall_y_m": [3.32, 3.44],
    "opening_x_m": [0.62, 1.82],
    "opening_z_m": [0.95, 1.95],
    "source": "default_room offset 0.50 m, width 1.20 m, sill 0.90 m, height 1.00 m, remeasured on the packaged mesh",
}

RECIPES = {
    "quality_controlled": {
        "description": "Dining group of repeated plates and bowls on one table, with spoon, bottle, apple, and cup.",
        "separation_m": 0.02,
        "zones": {"dining": (0.0, 0.0, 1.0, 1.0)},
        "objects": [
            ("prop_plate__1", "prop_plate", "dining"),
            ("prop_plate__2", "prop_plate", "dining"),
            ("prop_plate__3", "prop_plate", "dining"),
            ("prop_plate__4", "prop_plate", "dining"),
            ("prop_bowl__1", "prop_bowl", "dining"),
            ("prop_bowl__2", "prop_bowl", "dining"),
            ("prop_spoon__1", "prop_spoon", "dining"),
            ("prop_apple__1", "prop_apple", "dining"),
            ("prop_bottle__1", "prop_bottle", "dining"),
            ("prop_cup__1", "prop_cup", "dining"),
        ],
    },
    "quality_mixed": {
        "description": "A dining cluster and a work cluster with open space between them.",
        "separation_m": 0.012,
        "zones": {"dining": (0.0, 0.0, 0.62, 1.0), "work": (0.70, 0.0, 1.0, 1.0)},
        "objects": [
            ("prop_plate__1", "prop_plate", "dining"),
            ("prop_plate__2", "prop_plate", "dining"),
            ("prop_plate__3", "prop_plate", "dining"),
            ("prop_plate__4", "prop_plate", "dining"),
            ("prop_plate__5", "prop_plate", "dining"),
            ("prop_plate__6", "prop_plate", "dining"),
            ("prop_bowl__1", "prop_bowl", "dining"),
            ("prop_bowl__2", "prop_bowl", "dining"),
            ("prop_apple__1", "prop_apple", "dining"),
            ("prop_spoon__1", "prop_spoon", "dining"),
            ("prop_phone__1", "prop_phone", "work"),
            ("prop_marker__1", "prop_marker", "work"),
            ("prop_tape_roll__1", "prop_tape_roll", "work"),
            ("prop_small_box__1", "prop_small_box", "work"),
            ("prop_bottle__1", "prop_bottle", "work"),
        ],
    },
    "quality_dense": {
        "description": "Repeated plates and bowls with a side group of phones, boxes, and tape. Separation stays at 0.008 m.",
        "separation_m": 0.008,
        "zones": {"dining": (0.0, 0.0, 0.70, 1.0), "staging": (0.74, 0.0, 1.0, 1.0)},
        "objects": [
            ("prop_plate__1", "prop_plate", "dining"),
            ("prop_plate__2", "prop_plate", "dining"),
            ("prop_plate__3", "prop_plate", "dining"),
            ("prop_plate__4", "prop_plate", "dining"),
            ("prop_plate__5", "prop_plate", "dining"),
            ("prop_plate__6", "prop_plate", "dining"),
            ("prop_plate__7", "prop_plate", "dining"),
            ("prop_plate__8", "prop_plate", "dining"),
            ("prop_bowl__1", "prop_bowl", "dining"),
            ("prop_bowl__2", "prop_bowl", "dining"),
            ("prop_bowl__3", "prop_bowl", "dining"),
            ("prop_bowl__4", "prop_bowl", "dining"),
            ("prop_apple__1", "prop_apple", "dining"),
            ("prop_apple__2", "prop_apple", "dining"),
            ("prop_spoon__1", "prop_spoon", "dining"),
            ("prop_spoon__2", "prop_spoon", "dining"),
            ("prop_phone__1", "prop_phone", "staging"),
            ("prop_phone__2", "prop_phone", "staging"),
            ("prop_small_box__1", "prop_small_box", "staging"),
            ("prop_small_box__2", "prop_small_box", "staging"),
            ("prop_tape_roll__1", "prop_tape_roll", "staging"),
            ("prop_tape_roll__2", "prop_tape_roll", "staging"),
            ("prop_bottle__1", "prop_bottle", "staging"),
            ("prop_bottle__2", "prop_bottle", "staging"),
        ],
    },
}


def bundle_hashes(root: Path = OLD_ROOT) -> dict[str, str]:
    found = {}
    for path in sorted(Path(root).glob("*/*/bundle.json")):
        found[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def diagnose_saved(catalog: Path = CATALOG, root: Path = OLD_ROOT) -> dict:
    """Per-asset union contribution from the saved poses. Does not rewrite scenes."""
    catalog = Path(catalog)
    rows = []
    for scene_path in sorted(Path(root).glob("*/*/scene.json")):
        if scene_path.parent.parent.name.startswith("comparison_"):
            continue
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        table = next(item for item in scene["instances"] if item["instance_id"] == "table_1")
        table_matrix = deserialize_matrix(table["matrix"])
        support_local = np.asarray(
            next(item["polygon"] for item in verify_package(catalog / "table_rect" / VERSION)["support_surfaces"] if item["id"] == "top"),
            dtype=np.float64,
        )
        support = apply_points(table_matrix, support_local)
        placed = []
        contributions = []
        for instance in scene["instances"]:
            if not instance["instance_id"].startswith("prop_") or instance["asset_id"] in {"prop_chair"}:
                continue
            mesh = load_mesh(catalog, instance["asset_id"], instance["version"])
            world = world_vertices(mesh["vertices"], deserialize_matrix(instance["matrix"]))
            row = {"instance_id": instance["instance_id"], "asset_id": instance["asset_id"], "world": world, "faces": mesh["faces"], "nearest_clearance_m": None}
            placed.append(row)
            alone = _union_area([row], support)
            contributions.append({"instance_id": instance["instance_id"], "asset_id": instance["asset_id"], "union_m2": alone})
        occupancy = _occupancy(placed, support) if placed else {}
        rows.append(
            {
                "task_id": scene["task_id"],
                "seed": scene["seed"],
                "union_fraction": occupancy.get("occupied_fraction"),
                "footprint_sum_m2": occupancy.get("footprint_sum_m2"),
                "union_m2": occupancy.get("union_m2"),
                "table_area_m2": occupancy.get("table_area_m2"),
                "contributions": contributions,
            }
        )
        logger.info(
            "scene_quality event=diagnose task=%s seed=%s fraction=%s",
            scene["task_id"],
            scene["seed"],
            occupancy.get("occupied_fraction"),
        )
    return {"method": "4mm boolean raster of XY-projected triangles over the table support rectangle", "scenes": rows}


def measure_window(catalog: Path = CATALOG) -> dict:
    """Ray through the expected north opening, and a control ray into solid wall."""
    import mitsuba as mi

    require_variant()
    mesh = Path(catalog) / "room_wall_north" / VERSION / "geometry" / "render.obj"
    scene = mi.load_dict(
        {
            "type": "scene",
            "wall": {
                "type": "obj",
                "filename": str(mesh.resolve()),
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]}},
            },
        }
    )
    opening = _ray_hit(scene, [1.22, 4.20, 1.45], [0.0, -1.0, 0.0])
    solid = _ray_hit(scene, [0.30, 4.20, 1.45], [0.0, -1.0, 0.0])
    open_ok = not opening["hit"]
    solid_ok = solid["hit"] and 0.7 < solid["distance_m"] < 0.95
    logger.info(
        "scene_quality event=window_probe opening_hit=%s solid_hit=%s solid_t=%.4f",
        opening["hit"],
        solid["hit"],
        solid["distance_m"],
    )
    return {
        "expectation": WINDOW_EXPECTATION,
        "opening_ray": opening,
        "solid_ray": solid,
        "opening_is_clear": bool(open_ok and solid_ok),
    }


def daylight_emitter(radiance: float) -> dict:
    """Rectangle outside the north wall, aimed south through the measured opening."""
    import mitsuba as mi

    require_variant()
    transform = (
        mi.ScalarTransform4f.translate([1.22, 3.70, 1.45])
        @ mi.ScalarTransform4f.rotate([1.0, 0.0, 0.0], 90.0)
        @ mi.ScalarTransform4f.scale([0.50, 0.40, 1.0])
    )
    return {"type": "rectangle", "to_world": transform, "emitter": {"type": "area", "radiance": {"type": "rgb", "value": float(radiance)}}}


def close_table_camera(
    support_world: np.ndarray,
    *,
    margin: float = 0.08,
    fov_deg: float = 50.0,
    aspect: float = 1600.0 / 900.0,
) -> dict:
    """Front-elevated camera. The support polygon fills the frame with ``margin`` extra."""
    center = np.asarray(support_world, dtype=np.float64).mean(axis=0)
    target = np.array([center[0], center[1], center[2] + 0.04], dtype=np.float64)
    back = np.array([0.0, -1.0, 0.42], dtype=np.float64)
    back /= np.linalg.norm(back)
    origin = _fit_origin(support_world, target, back, [0.0, 0.0, 1.0], fov_deg, aspect, margin)
    origin = np.array(
        [
            float(np.clip(origin[0], 0.45, 3.7)),
            float(np.clip(origin[1], 0.40, 3.0)),
            float(np.clip(origin[2], 0.90, 2.20)),
        ]
    )
    return {
        "name": "tabletop_close",
        "origin": origin.tolist(),
        "target": target.tolist(),
        "up": [0.0, 0.0, 1.0],
        "fov_deg": float(fov_deg),
        "margin": float(margin),
        "aspect": float(aspect),
        "resolution": [1600, 900],
        "framing": "support polygon plus 8 percent, horizontal field of view, no extra room pad",
    }


def overhead_table_camera(
    support_world: np.ndarray,
    *,
    margin: float = 0.08,
    fov_deg: float = 50.0,
    aspect: float = 1600.0 / 900.0,
) -> dict:
    center = np.asarray(support_world, dtype=np.float64).mean(axis=0)
    target = np.array([center[0], center[1], center[2]], dtype=np.float64)
    back = np.array([0.0, -0.12, 1.0], dtype=np.float64)
    back /= np.linalg.norm(back)
    origin = _fit_origin(support_world, target, back, [0.0, 1.0, 0.0], fov_deg, aspect, margin)
    return {
        "name": "tabletop_overhead",
        "origin": origin.tolist(),
        "target": target.tolist(),
        "up": [0.0, 1.0, 0.0],
        "fov_deg": float(fov_deg),
        "margin": float(margin),
        "aspect": float(aspect),
        "resolution": [1600, 900],
        "framing": "overhead of the support polygon plus 8 percent",
    }


def _fit_origin(polygon, target, back, up, fov_deg: float, aspect: float, margin: float) -> np.ndarray:
    """Distance where the projected polygon reaches ``1 / (1 + margin)`` in NDC on its tighter axis."""
    limit = 1.0 / (1.0 + float(margin))
    low, high = 0.25, 8.0
    for _ in range(48):
        distance = 0.5 * (low + high)
        origin = np.asarray(target, dtype=np.float64) + np.asarray(back, dtype=np.float64) * distance
        peak = _peak_ndc(polygon, origin, target, up, fov_deg, aspect)
        if peak > limit:
            low = distance
        else:
            high = distance
    return np.asarray(target, dtype=np.float64) + np.asarray(back, dtype=np.float64) * high


def _peak_ndc(polygon, origin, target, up, fov_deg: float, aspect: float) -> float:
    camera = {"origin": np.asarray(origin).tolist(), "target": np.asarray(target).tolist(), "up": list(up), "fov_deg": float(fov_deg)}
    width, height = 1600, int(round(1600 / float(aspect)))
    peak = 0.0
    for point in np.asarray(polygon, dtype=np.float64):
        pixel = _project_point(camera, width, height, point)
        if pixel is None:
            return 2.0
        ndc_x = (pixel[0] + 0.5) / width * 2.0 - 1.0
        ndc_y = (pixel[1] + 0.5) / height * 2.0 - 1.0
        peak = max(peak, abs(ndc_x), abs(ndc_y))
    return peak


def quality_room(catalog: Path = CATALOG) -> dict:
    """One arrangement for every quality pilot. The chair faces the table."""
    catalog = Path(catalog)
    poses = {asset_id: identity_matrix() for asset_id in SHELL}
    poses["table_rect"] = translation_matrix([1.55, 1.70, 0.05])
    poses["shelf_two"] = translation_matrix([3.20, 2.30, 0.05])
    chair_vertices, _polygons = read_obj(catalog / "prop_chair" / VERSION / "geometry" / "render.obj")
    poses["prop_chair"] = _yaw_about_footprint(chair_vertices, np.pi, [2.15, 1.15], 0.05)
    return _validate_room(catalog, "quality_south", poses)


def place_recipe(catalog: Path, recipe_id: str, room: dict, seed: int) -> dict:
    recipe = RECIPES[recipe_id]
    support = room["table_support_world"]
    inset = _inset_rect(support, INSET_M)
    zones = _zone_rects(inset, recipe["zones"])
    rng = np.random.default_rng(int(seed))
    items = []
    for instance_id, asset_id, zone_name in recipe["objects"]:
        mesh = load_mesh(catalog, asset_id, VERSION)
        items.append(
            {
                "instance_id": instance_id,
                "asset_id": asset_id,
                "version": VERSION,
                "mesh": mesh,
                "zone": zones[zone_name],
                "zone_name": zone_name,
                "area": float((mesh["vertices"][:, 0].max() - mesh["vertices"][:, 0].min()) * (mesh["vertices"][:, 1].max() - mesh["vertices"][:, 1].min())),
                "support_parent": "table_1",
                "support_surface": "top",
            }
        )
    items.sort(key=lambda item: (-item["area"], item["instance_id"]))
    placed = []
    relations = []
    attempts = []
    support_z = float(support[0, 2])
    for item in items:
        found = _place_in_zone(item, placed, inset, support, support_z, rng, recipe["separation_m"])
        attempts.append(
            {
                "instance_id": item["instance_id"],
                "zone": item["zone_name"],
                "tries": found["tries"],
                "placed": found["pose"] is not None,
            }
        )
        if found["pose"] is None:
            logger.info("scene_quality event=placement_failed recipe=%s instance=%s", recipe_id, item["instance_id"])
            return {
                "valid": False,
                "collision_log": {"termination": "placement_failed", "failed_instance": item["instance_id"], "attempts": attempts, "seed": int(seed)},
            }
        placed.append(found["pose"])
        relations.append(found["relation"])
    occupancy = _occupancy(placed, support)
    logger.info(
        "scene_quality event=placed recipe=%s seed=%d instances=%d fraction=%.4f",
        recipe_id,
        seed,
        len(placed),
        occupancy["occupied_fraction"],
    )
    return {
        "valid": True,
        "poses": {row["instance_id"]: row["matrix"] for row in placed},
        "relations": relations,
        "collision_log": {
            "termination": "placed",
            "attempts": attempts,
            "minimum_separation_m": recipe["separation_m"],
            "inset_m": INSET_M,
            "seed": int(seed),
            "zones": {key: [float(value) for value in bounds] for key, bounds in recipe["zones"].items()},
            "grouping": "named_table_zones",
            "occupancy": occupancy,
        },
    }


def generate_quality_pilot(catalog: Path = CATALOG, output: Path = ROOT) -> list[dict]:
    catalog = Path(catalog)
    output = Path(output)
    room = quality_room(catalog)
    if not room["valid"]:
        raise RuntimeError(f"quality room invalid: {room['failure']}")
    written = []
    for recipe_id, seed in SEEDS.items():
        placed = place_recipe(catalog, recipe_id, room, seed)
        if not placed["valid"]:
            path = output / recipe_id / f"failed_{seed}"
            path.mkdir(parents=True, exist_ok=True)
            (path / "failure.json").write_text(json.dumps(placed["collision_log"], indent=2) + "\n", encoding="utf-8")
            written.append({"valid": False, "task_id": recipe_id, "seed": seed, "path": str(path)})
            raise RuntimeError(f"{recipe_id} failed placement; batch stopped")
        cameras = {
            "room": ROOM_CAMERA,
            "tabletop": close_table_camera(room["table_support_world"]),
            "overhead": overhead_table_camera(room["table_support_world"]),
        }
        bundle = _bundle(
            catalog,
            recipe_id,
            RECIPES[recipe_id]["description"],
            seed,
            room,
            placed["poses"],
            placed["relations"],
            placed["collision_log"],
            cameras=cameras,
        )
        path = _publish(output, recipe_id, bundle)
        replay(path, catalog)
        written.append({"valid": True, "task_id": recipe_id, "seed": seed, "path": str(path), "fraction": placed["collision_log"]["occupancy"]["occupied_fraction"]})
        logger.info("scene_quality event=bundle_saved task=%s path=%s", recipe_id, path)
    return written


def render_quality_bundle(
    bundle_dir: Path,
    catalog: Path,
    output: Path,
    *,
    resolution: tuple[int, int],
    include_overhead: bool,
    daylight: bool,
    daylight_radiance: float,
) -> dict:
    bundle_dir = Path(bundle_dir)
    output = Path(output)
    if (output / "validation.json").exists():
        raise RuntimeError(f"output_exists path={output}")
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    instances = _instances(catalog, bundle)
    cameras = bundle["resolved"]["collision_log"]["cameras"]
    support = _table_support_world(catalog, bundle)
    output.mkdir(parents=True, exist_ok=True)
    views = [("room", cameras["room"], (512, 384))]
    close = dict(cameras["tabletop"])
    close["resolution"] = list(resolution)
    views.append(("tabletop", close, resolution))
    if include_overhead:
        overhead = dict(cameras["overhead"])
        overhead["resolution"] = list(resolution)
        views.append(("overhead", overhead, resolution))
    images = {}
    visibility = {}
    started = time.perf_counter()
    for name, camera, view_resolution in views:
        scene = _scene(instances, camera, view_resolution, _ceiling_light(LIGHT_RADIANCE))
        if daylight:
            scene["daylight"] = daylight_emitter(daylight_radiance)
        exr = output / f"{name}.exr"
        png = output / f"{name}.png"
        stokes = render_stokes(scene, exr, SCENE_SPP, RENDER_SEED)
        write_rgb_png(exr, png)
        polygon = support if name != "room" else None
        report = instance_visibility(instances, camera, view_resolution, support_polygon=polygon, return_image=True)
        image = report.pop("id_image")
        report.pop("roi_mask", None)
        np.save(output / f"{name}_instance_ids.npy", image.astype(np.int32))
        preview = output / f"{name}_ids.png"
        _write_id_preview(image, preview)
        if name == "tabletop":
            _write_crops(exr, image, report["instances"], output / "crops")
        images[name] = {
            "png": str(png),
            "exr": str(exr),
            "ids": str(output / f"{name}_instance_ids.npy"),
            "shape": stokes["shape"],
            "channel_names": stokes["channel_names"],
            "dtype": stokes["dtype"],
            "finite": stokes["finite"],
            "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
            "exr_sha256": hashlib.sha256(exr.read_bytes()).hexdigest(),
            "ids_preview": str(output / f"{name}_ids.png"),
        }
        if list(image.shape) != [stokes["shape"][0], stokes["shape"][1]]:
            raise RuntimeError(f"mask_shape image={image.shape} exr={stokes['shape']}")
        visibility[name] = report
    elapsed = time.perf_counter() - started
    by_instance = {asset["instance_id"]: asset for asset in bundle["resolved"]["assets"]}
    metadata = []
    for row in visibility["tabletop"]["instances"]:
        asset = by_instance[row["instance_id"]]
        manifest = verify_package(Path(catalog) / asset["asset_id"] / asset["version"])
        metadata.append(
            {
                **row,
                "version": asset["version"],
                "package_sha256": asset["package_sha256"],
                "category": (manifest.get("geometry") or {}).get("category"),
                "material": (manifest.get("materials") or [{}])[0],
                "matrix": next(item["matrix"] for item in bundle["resolved"]["poses"] if item["instance_id"] == row["instance_id"]),
            }
        )
    (output / "instances.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    validation = {
        "task_id": bundle["task"]["task_id"],
        "seed": bundle["task"]["seed"],
        "rendered": True,
        "lighting_id": "ceiling_plus_daylight_v1" if daylight else "ceiling_v1",
        "daylight_radiance": daylight_radiance if daylight else None,
        "render_seed": RENDER_SEED,
        "spp": SCENE_SPP,
        "elapsed_s": elapsed,
        "images": images,
        "visibility": {key: value for key, value in visibility.items()},
        "occupancy": bundle["resolved"]["collision_log"].get("occupancy"),
        "cameras": {"room": cameras["room"], "tabletop": close, "overhead": cameras.get("overhead")},
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    logger.info("scene_quality event=rendered task=%s lighting=%s seconds=%.2f", bundle["task"]["task_id"], validation["lighting_id"], elapsed)
    return validation


def _place_in_zone(item, placed, inset, support, support_z, rng, separation: float) -> dict:
    vertices = item["mesh"]["vertices"]
    region = item["zone"]
    for attempt in range(1, 161):
        yaw = float(rng.uniform(0.0, 2.0 * np.pi))
        bounds = _yaw_bounds(vertices, yaw)
        lo_x = region[0] - bounds[0]
        hi_x = region[2] - bounds[2]
        lo_y = region[1] - bounds[1]
        hi_y = region[3] - bounds[3]
        if hi_x < lo_x or hi_y < lo_y:
            continue
        translation = [float(rng.uniform(lo_x, hi_x)), float(rng.uniform(lo_y, hi_y)), support_z]
        matrix = compose(translation_matrix(translation), rotation_matrix("z", yaw))
        world = world_vertices(vertices, matrix)
        if not _inside_rect(world[:, :2], region) or not _inside_rect(world[:, :2], inset):
            continue
        contact = measure_support(world, item["mesh"]["faces"], support)
        if not contact["passed"]:
            continue
        nearest = float("inf")
        blocked = False
        candidate = {"vertices": world, "faces": item["mesh"]["faces"]}
        for other in placed:
            clearance = float(mesh_clearance(candidate, {"vertices": other["world"], "faces": other["faces"]})["clearance_m"])
            nearest = min(nearest, clearance)
            if clearance < separation:
                blocked = True
                break
        if blocked:
            continue
        return {
            "tries": attempt,
            "pose": {
                "instance_id": item["instance_id"],
                "asset_id": item["asset_id"],
                "matrix": serialize_matrix(matrix),
                "world": world,
                "faces": item["mesh"]["faces"],
                "nearest_clearance_m": None if nearest == float("inf") else nearest,
                "zone": item["zone_name"],
            },
            "relation": _relation(item["instance_id"], item["support_parent"], item["support_surface"], contact),
        }
    return {"tries": 160, "pose": None, "relation": None}


def _zone_rects(inset, fractions: dict) -> dict:
    x0, y0, x1, y1 = inset
    width, height = x1 - x0, y1 - y0
    zones = {}
    for name, (fx0, fy0, fx1, fy1) in fractions.items():
        zones[name] = (x0 + fx0 * width, y0 + fy0 * height, x0 + fx1 * width, y0 + fy1 * height)
    return zones


def _yaw_about_footprint(vertices: np.ndarray, yaw: float, world_xy: list[float], floor_z: float) -> np.ndarray:
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = translation_matrix([-0.5 * (low[0] + high[0]), -0.5 * (low[1] + high[1]), 0.0])
    spun = compose(rotation_matrix("z", float(yaw)), center)
    return compose(translation_matrix([world_xy[0], world_xy[1], floor_z]), spun)


def _validate_room(catalog: Path, name: str, poses: dict) -> dict:
    floor = verify_package(catalog / "room_floor" / VERSION)
    floor_support = np.asarray(floor["support_surfaces"][0]["polygon"], dtype=np.float64)
    table = verify_package(catalog / "table_rect" / VERSION)
    table_support = np.asarray(next(item["polygon"] for item in table["support_surfaces"] if item["id"] == "top"), dtype=np.float64)
    contacts = []
    valid = True
    failure = ""
    for asset_id in ("table_rect", "shelf_two", "prop_chair"):
        mesh = load_mesh(catalog, asset_id)
        world = world_vertices(mesh["vertices"], poses[asset_id])
        contact = measure_support(world, mesh["faces"], floor_support)
        contacts.append({"asset_id": asset_id, "passed": contact["passed"], "gap_m": contact["gap_m"], "reason": contact["reason"]})
        if not contact["passed"]:
            valid = False
            failure = f"floor_support:{asset_id}"
    bodies = []
    for asset_id, matrix in poses.items():
        mesh = load_mesh(catalog, asset_id)
        bodies.append((asset_id, {"vertices": world_vertices(mesh["vertices"], matrix), "faces": triangulate(mesh["polygons"])}))
    intersections = []
    shell = set(SHELL)
    for index, (left_id, left) in enumerate(bodies):
        for right_id, right in bodies[index + 1 :]:
            pair = {left_id, right_id}
            if pair <= shell or ("room_floor" in pair and pair & {"table_rect", "shelf_two", "prop_chair"}):
                continue
            found = unintended_intersection(left, right)
            if found["unintended"]:
                intersections.append({"pair": [left_id, right_id], "clearance_m": found["clearance_m"]})
                valid = False
                failure = failure or f"intersection:{left_id}:{right_id}"
    table_world = apply_points(poses["table_rect"], table_support)
    logger.info("scene_quality event=room name=%s valid=%s intersections=%d", name, valid, len(intersections))
    return {
        "valid": valid,
        "failure": failure,
        "arrangement": name,
        "poses": {key: serialize_matrix(value) for key, value in poses.items()},
        "matrices": poses,
        "table_matrix": poses["table_rect"],
        "table_support_local": table_support,
        "table_support_world": table_world,
        "contacts": contacts,
        "intersections": intersections,
    }


def _table_support_world(catalog: Path, bundle: dict) -> np.ndarray:
    table = next(item for item in bundle["resolved"]["poses"] if item["instance_id"] == "table_1")
    manifest = verify_package(Path(catalog) / "table_rect" / VERSION)
    local = np.asarray(next(item["polygon"] for item in manifest["support_surfaces"] if item["id"] == "top"), dtype=np.float64)
    return apply_points(deserialize_matrix(table["matrix"]), local)


def _ray_hit(scene, origin, direction) -> dict:
    import mitsuba as mi

    ray = mi.Ray3f(
        o=mi.Point3f(np.array(origin, dtype=np.float32).reshape(3, 1)),
        d=mi.Vector3f(np.array(direction, dtype=np.float32).reshape(3, 1)),
    )
    hit = scene.ray_intersect(ray)
    valid = bool(np.array(hit.is_valid()).reshape(-1)[0])
    distance = float(np.array(hit.t).reshape(-1)[0]) if valid else float("inf")
    return {"origin_m": origin, "direction": direction, "hit": valid, "distance_m": distance}


def _write_id_preview(image: np.ndarray, path: Path) -> None:
    import mitsuba as mi

    palette = np.array(
        [
            [0.05, 0.05, 0.05],
            [0.90, 0.20, 0.15],
            [0.15, 0.55, 0.90],
            [0.95, 0.75, 0.15],
            [0.20, 0.75, 0.35],
            [0.70, 0.30, 0.85],
            [0.95, 0.45, 0.10],
            [0.30, 0.85, 0.80],
            [0.85, 0.85, 0.85],
        ],
        dtype=np.float32,
    )
    color = palette[image % len(palette)]
    color[image == 0] = palette[0]
    bitmap = mi.Bitmap(color)
    bitmap = bitmap.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    bitmap.write(str(path))


def _write_crops(exr: Path, ids: np.ndarray, rows: list[dict], output: Path) -> None:
    import mitsuba as mi

    output.mkdir(parents=True, exist_ok=True)
    image = np.array(mi.Bitmap(str(exr)))
    for row in rows:
        box = row.get("pixel_bbox_xyxy")
        if not box or row["visible_pixels"] == 0:
            continue
        x0, y0, x1, y1 = box
        crop = image[y0 : y1 + 1, x0 : x1 + 1]
        safe = row["instance_id"].replace("/", "_")
        np.save(output / f"{safe}_stokes.npy", crop.astype(np.float32))
        rgb = mi.Bitmap(np.ascontiguousarray(crop[:, :, :3]))
        rgb = rgb.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
        rgb.write(str(output / f"{safe}_rgb.png"))


def write_gallery(review_path: Path, quality_root: Path, old_root: Path = OLD_ROOT) -> None:
    review_path = Path(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Scene quality v2</title>",
        "<style>body{font-family:sans-serif;margin:2rem;background:#111;color:#eee} a{color:#9cf} img{max-width:480px;background:#222} figure{display:inline-block;margin:1rem;vertical-align:top} .notice{border:1px solid #c90;padding:1rem}</style>",
        "</head><body><h1>Scene quality refinement</h1>",
        "<p class=\"notice\"><strong>These outputs are qualification candidates. Human visual review, admission, and registration are pending.</strong></p>",
    ]
    validations = []
    for path in sorted(Path(quality_root).glob("**/render_*/validation.json")):
        validations.append(json.loads(path.read_text(encoding="utf-8")))
    ceiling = [item for item in validations if item["lighting_id"] == "ceiling_v1"]
    daylight = [item for item in validations if item["lighting_id"] == "ceiling_plus_daylight_v1"]
    body.append("<h2>A. Room overview</h2>")
    for item in ceiling:
        _figure(body, review_path.parent, item, "room")
        body.append("<p>" + html.escape(f"{item['task_id']} seed {item['seed']} lighting {item['lighting_id']}") + "</p>")
    body.append("<h2>B. Tabletop density</h2>")
    for item in ceiling:
        occ = item.get("occupancy") or {}
        body.append("<p>" + html.escape(
            f"{item['task_id']} instances shown by occupancy fraction {occ.get('occupied_fraction')} "
            f"footprint sum {occ.get('footprint_sum_m2')} nearest {occ.get('nearest_clearance_m')}"
        ) + "</p>")
        _figure(body, review_path.parent, item, "tabletop")
    body.append("<h2>C. Close tabletop views and masks</h2>")
    for item in ceiling:
        _figure(body, review_path.parent, item, "tabletop")
        if "overhead" in item["images"]:
            _figure(body, review_path.parent, item, "overhead")
        ids = item["images"]["tabletop"].get("ids_preview")
        if ids:
            href = _rel(review_path.parent, Path(ids))
            body.append(f"<figure><img src=\"{html.escape(href)}\" alt=\"ids\"><figcaption>instance ids</figcaption></figure>")
        for row in item["visibility"]["tabletop"]["instances"]:
            if not str(row["instance_id"]).startswith("prop_") or row["asset_id"] == "prop_chair":
                continue
            body.append("<p>" + html.escape(
                f"{row['instance_id']} visible {row['visible_pixels']} roi_fraction {row.get('roi_fraction')} "
                f"shorter_px {row['shorter_pixel_side']} low_spatial {row['low_spatial_support']}"
            ) + "</p>")
    body.append("<h2>D. Lighting</h2>")
    if daylight:
        body.append("<p>Ceiling only is above. Ceiling plus daylight for the controlled scene is below. Camera and materials match.</p>")
        for item in daylight:
            _figure(body, review_path.parent, item, "room")
            _figure(body, review_path.parent, item, "tabletop")
    else:
        body.append("<p>Daylight was not rendered.</p>")
    body.append("<h2>E. Existing controlled comparisons</h2>")
    body.append("<p>These files are the original comparisons. Camera, light, and the other transforms were held fixed. Only the named material or mesh changed. They were not re-rendered.</p>")
    for task, note in (
        ("comparison_bowl_diffuse", "diffuse bowl"),
        ("comparison_bowl_glossy", "pplastic bowl, same geometry"),
        ("comparison_can_polished", "smooth aluminum can"),
        ("comparison_can_rough", "rough aluminum can, same geometry"),
        ("comparison_geometry_plate", "plate, shared diffuse RGB"),
        ("comparison_geometry_apple", "apple, shared diffuse RGB"),
    ):
        png = next((old_root / task).glob("*/render_17111/tabletop.png"), None)
        if png is None:
            body.append(f"<p>Missing {html.escape(task)}</p>")
            continue
        href = _rel(review_path.parent, png)
        body.append(f"<figure><a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(task)}\"></a><figcaption>{html.escape(task + ' ' + note)}</figcaption></figure>")
    body.append("</body></html>\n")
    review_path.write_text("\n".join(body), encoding="utf-8")


def _figure(body, origin: Path, item: dict, view: str) -> None:
    image = item["images"][view]
    href = _rel(origin, Path(image["png"]))
    exr = _rel(origin, Path(image["exr"]))
    body.append(f"<figure><a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(item['task_id'] + ' ' + view)}\"></a><figcaption>{html.escape(view)}</figcaption></figure>")
    body.append(f"<p><a href=\"{html.escape(exr)}\">Stokes EXR {html.escape(view)}</a></p>")


def _rel(origin: Path, target: Path) -> str:
    return Path(os.path.relpath(target, origin)).as_posix()

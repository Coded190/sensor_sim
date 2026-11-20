"""Place curated kitchen, dining, desk, and workbench scenes on saved rooms.

Room shells are loaded from the room-diversity packages. This module does not
rebuild them. Replay reads the saved bundle and does not place again.
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
from sensor_sim.scene_gen.asset_validate import validate_mesh
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
from sensor_sim.scene_gen.mesh_io import write_obj
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES
from sensor_sim.scene_gen.narrow import mesh_clearance
from sensor_sim.scene_gen.room_generator import _daylight_emitters, _emitter_record
from sensor_sim.scene_gen.scene_context import require_scene_context
from sensor_sim.scene_gen.scene_quality import _fit_origin, _peak_ndc
from sensor_sim.scene_gen.semantic_assets import BUILDERS, build_asset, interior_is_open
from sensor_sim.scene_gen.semantic_schema import SCHEMA_VERSION, RecipeError, require_recipe
from sensor_sim.scene_gen.structural import support_quad
from sensor_sim.scene_gen.tabletop_layout import _stamp_triangle, load_mesh
from sensor_sim.scene_gen.tabletop_render import (
    SCENE_AC_BSDF,
    _project_point,
    _scene,
    instance_visibility,
    mitsuba_bsdf,
)

logger = logging.getLogger("sensor_sim.scene_gen.semantic_composition")

ROOT = Path("dataset/qualification/semantic_scenes")
ASSET_CATALOG = Path("assets/candidates")
ROOM_ROOT = Path("dataset/qualification/room_diversity")
ROOM_RUNS = {
    "room_compact": "7f774e080dde15d6",
    "room_standard": "0ea3fd7596033729",
    "room_spacious": "dd985a60befaf87d",
}
VERSION = "1.0.0"
PACKAGE_VERSIONS = {
    "furniture_workbench": "1.0.2",
    "furniture_counter": "1.0.1",
}
VERSION_REASONS = {
    "furniture_workbench": "zones resized so a workpiece, fixture, and stock plate fit, and the right column stays open. The 1.0.0 mesh is unchanged.",
    "furniture_counter": "storage allows manipulation_target; the pot sits in that zone. The 1.0.0 mesh is unchanged.",
}


def package_version(asset_id: str) -> str:
    return PACKAGE_VERSIONS.get(asset_id, VERSION)
RENDER_SEED = 17421
SCENE_SPP = 16
CEILING_INSET_M = 0.20
CEILING_DROP_M = 0.08
CEILING_RADIANCE = 2.2
WINDOW_RADIANCE = 5.0
MIN_SEPARATION_M = 0.004
YAW_SPREAD = {"low": 0.08, "moderate": 0.30, "high": 0.55}
FIXED_YAW = {"prop_hammer__1", "prop_pliers__2", "prop_work_panel__1", "prop_fixture_block__1", "prop_stock_plate__1"}
LAY_ON_SIDE = {"prop_screwdriver", "prop_pen", "prop_spatula"}
LOW_IDS = (
    "prop_work_panel__1",
    "prop_screwdriver__1",
    "prop_pliers__1",
    "prop_wrench__1",
    "prop_hammer__1",
    "prop_fastener_tray__1",
    "prop_wire_spool__1",
    "prop_wood_offcut__1",
    "prop_metal_bracket__1",
)
MODERATE_IDS = LOW_IDS + ("prop_fixture_block__1", "prop_clamp__1", "prop_hex_key_group__1", "prop_cable__1", "prop_tape_roll__1")
HIGH_IDS = MODERATE_IDS + ("prop_stock_plate__1", "prop_parts_bin__1", "prop_component_block__1", "prop_screwdriver__2", "prop_pliers__2")
SLOTS = {
    "prop_work_panel__1": ("assembly", (0.34, 0.22)),
    "prop_wood_offcut__1": ("assembly", (0.20, 0.44)),
    "prop_metal_bracket__1": ("assembly", (0.46, 0.44)),
    "prop_cable__1": ("assembly", (0.62, 0.44)),
    "prop_component_block__1": ("assembly", (0.66, 0.28)),
    "prop_screwdriver__1": ("tools", (0.90, 0.12)),
    "prop_pliers__1": ("tools", (1.14, 0.12)),
    "prop_wrench__1": ("tools", (1.22, 0.30)),
    "prop_hammer__1": ("tools", (0.82, 0.30)),
    "prop_clamp__1": ("tools", (1.00, 0.30)),
    "prop_hex_key_group__1": ("tools", (1.16, 0.22)),
    "prop_fixture_block__1": ("staging", (0.22, 0.64)),
    "prop_stock_plate__1": ("staging", (0.56, 0.64)),
    "prop_fastener_tray__1": ("small_parts", (0.94, 0.68)),
    "prop_wire_spool__1": ("small_parts", (1.16, 0.68)),
    "prop_tape_roll__1": ("tools", (1.30, 0.16)),
    "prop_parts_bin__1": ("small_parts", (1.28, 0.68)),
    "prop_screwdriver__2": ("small_parts", (0.98, 0.52)),
    "prop_pliers__2": ("small_parts", (1.24, 0.52)),
}
EXISTING_ROLES = {
    "prop_plate": "manipulation_target",
    "prop_bowl": "manipulation_target",
    "prop_apple": "material",
    "prop_bottle": "material",
    "prop_spoon": "tool",
    "prop_cup": "manipulation_target",
    "prop_phone": "manipulation_target",
    "prop_marker": "context",
    "prop_tape_roll": "tool",
    "prop_chair": "furniture",
    "table_rect": "furniture",
}


def load_room(profile: str) -> dict:
    path = ROOM_ROOT / profile / ROOM_RUNS[profile] / "room_spec.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    return {"spec": spec, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": str(path)}


def preflight(asset_catalog: Path = ASSET_CATALOG) -> dict:
    """Measure proposed meshes and dry-run every recipe before any package hash."""
    assets = {asset_id: build_asset(asset_id) for asset_id in BUILDERS}
    east = [item for item in load_room("room_spacious")["spec"]["openings"] if item["wall"] == "east"]
    if east:
        raise RecipeError("reason=east_wall_not_free")
    reports = []
    for name in ("kitchen", "dining", "desk", "workbench_low", "workbench_moderate", "workbench_high"):
        placed = place_recipe(name, assets, asset_catalog)
        reports.append({"recipe": name, "valid": placed["valid"], "failed": placed["failed"], "clutter": placed.get("clutter")})
        if not placed["valid"]:
            write_failure(ROOT / name / "failure.json", placed)
            raise RecipeError(f"reason=preflight_failed recipe={name} failed={placed['failed']}")
    bands = _freeze_bands(reports)
    logger.info("semantic_composition event=preflight recipes=%d bands=%s", len(reports), bands)
    return {"assets": [_public_asset(item) for item in assets.values()], "placements": reports, "occupancy_bands": bands, "east_wall_openings": 0}


def place_recipe(name: str, assets: dict | None = None, asset_catalog: Path = ASSET_CATALOG, seed: int | None = None) -> dict:
    recipe = _recipe(name)
    if seed is not None:
        recipe["seed"] = int(seed)
    room = load_room(recipe["room_profile"])
    spec = room["spec"]
    assets = assets or {asset_id: build_asset(asset_id) for asset_id in BUILDERS}
    furniture = []
    if name.startswith("workbench"):
        furniture.append(_place_workbench(assets["furniture_workbench"], spec))
    elif name == "kitchen":
        furniture.append(_place_counter(assets["furniture_counter"], spec))
    elif name == "desk":
        furniture.append(_place_desk(assets["furniture_desk"], spec))
        furniture.append(_place_chair(asset_catalog, spec, furniture[0], "chair_1", side=-1.0))
    elif name == "dining":
        furniture.append(_place_table(asset_catalog, spec))
        furniture.append(_place_chair(asset_catalog, spec, furniture[0], "chair_1", side=-1.0))
        furniture.append(_place_chair(asset_catalog, spec, furniture[0], "chair_2", side=1.0))
    support_body = next(item for item in furniture if item["asset_id"] != "prop_chair")
    objects = []
    failures = []
    rng = np.random.default_rng(int(recipe["seed"]))
    for instance_id, asset_id, zone_id, slot in _object_slots(name):
        yaw = 0.0 if instance_id in FIXED_YAW else float(rng.uniform(-YAW_SPREAD[recipe["clutter_level"]], YAW_SPREAD[recipe["clutter_level"]]))
        placed = _place_object(instance_id, asset_id, zone_id, slot, yaw, support_body, assets, asset_catalog)
        if not placed["inside_zone"]:
            yaw = 0.0
            placed = _place_object(instance_id, asset_id, zone_id, slot, yaw, support_body, assets, asset_catalog)
        objects.append(placed)
        if not placed["inside_zone"] or not placed["on_support"]:
            failures.append({"name": f"support:{instance_id}", "passed": False})
    from sensor_sim.scene_gen.placement_separation import PlacementSeparationError, require_outside

    try:
        require_outside(objects, minimum_m=MIN_SEPARATION_M)
    except PlacementSeparationError as exc:
        failures.append({"name": "object_inside", "passed": False, "reason": str(exc)})
    clutter = _clutter(objects, support_body, recipe["clutter_level"])
    if name.startswith("workbench") and recipe["clutter_level"] == "high" and clutter["free_rectangle_m"][0] < 0.25:
        failures.append({"name": "free_rectangle", "passed": False, "free_rectangle_m": clutter["free_rectangle_m"]})
    room_clearance = _room_checks(furniture, spec, failures)
    return {
        "valid": not failures,
        "failed": failures,
        "recipe": recipe,
        "room": room,
        "furniture": furniture,
        "objects": objects,
        "clutter": clutter,
        "room_clearance": room_clearance,
    }


def write_failure(path: Path, placed: dict) -> None:
    """Stop a recipe with the checks that failed. This does not publish a bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"recipe": placed["recipe"]["scene_type"], "seed": placed["recipe"]["seed"], "failed": placed["failed"]}, indent=2) + "\n", encoding="utf-8")
    logger.info("semantic_composition event=failure path=%s checks=%d", path, len(placed["failed"]))


def scene_document(bundle: dict) -> dict:
    instances = [*bundle["room_components"], *bundle["furniture_instances"], *bundle["object_instances"]]
    return {
        "schema_version": 1,
        "task_id": bundle["task_id"],
        "seed": bundle["seed"],
        "qualification": True,
        "visual_review": "pending",
        "admitted": False,
        "valid": True,
        "instances": instances,
        "relations": bundle["relations"],
    }


def replay_semantic(bundle_dir: Path, semantic_catalog: Path, room_catalog: Path) -> dict:
    bundle_dir = Path(bundle_dir)
    sources = [bundle_dir / "bundle.json", bundle_dir / "recipe.json", bundle_dir / "scene.json"]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    if hashlib.sha256((bundle_dir / "recipe.json").read_bytes()).hexdigest() != bundle["recipe_sha256"]:
        raise RecipeError("reason=recipe_hash")
    require_recipe(bundle["recipe"])
    for package in bundle["room_packages"]:
        manifest = verify_package(room_catalog / package["path"])
        if manifest["package_sha256"] != package["package_sha256"]:
            raise RecipeError(f"reason=room_hash asset={package['asset_id']}")
    for package in bundle["semantic_packages"]:
        manifest = verify_package(semantic_catalog / package["path"])
        if manifest["package_sha256"] != package["package_sha256"]:
            raise RecipeError(f"reason=semantic_hash asset={package['asset_id']}")
    if canonical_json_bytes(scene_document(bundle)) != (bundle_dir / "scene.json").read_bytes():
        raise RecipeError("reason=replay_mismatch")
    if {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources} != before:
        raise RecipeError("reason=replay_modified_source")
    logger.info("semantic_composition event=replay task=%s result=ok", bundle["task_id"])
    return {"valid": True, "task_id": bundle["task_id"]}


def package_new_assets(catalog: Path) -> list[dict]:
    catalog = Path(catalog)
    written = []
    for asset_id in BUILDERS:
        built = build_asset(asset_id)
        version = package_version(asset_id)
        destination = catalog / asset_id / version
        _write_package(destination, built)
        manifest = verify_package(destination)
        geometry_hash = hashlib.sha256((destination / "geometry" / "render.obj").read_bytes()).hexdigest()
        written.append(
            {
                "asset_id": asset_id,
                "version": version,
                "path": f"{asset_id}/{version}",
                "package_sha256": manifest["package_sha256"],
                "geometry_sha256": geometry_hash,
                "status": "technically_validated",
                "triangle_count": built["triangle_count"],
                "measured_m": built["measured_m"],
                "role": built["role"],
                "material": built["material"],
                "renderer_validation": "pending",
                "visible_pixels": 0,
            }
        )
        logger.info("semantic_composition event=package asset_id=%s package_sha256=%s", asset_id, manifest["package_sha256"])
    return written


def render_scene(bundle_dir: Path, semantic_catalog: Path, room_catalog: Path, asset_catalog: Path = ASSET_CATALOG) -> dict:
    bundle_dir = Path(bundle_dir)
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    output = bundle_dir / "render_ceiling_v1"
    if (output / "validation.json").exists():
        raise RecipeError(f"reason=output_exists path={output}")
    output.mkdir(parents=True)
    instances = _render_instances(bundle, semantic_catalog, room_catalog, asset_catalog)
    images = {}
    visibility = {}
    views = [
        ("room", bundle["cameras"]["overview"], (512, 384)),
        ("tabletop", bundle["cameras"]["close"], (1600, 900)),
    ]
    for extra, resolution in (("grouping", (800, 450)), ("workspace", (800, 450))):
        if extra in bundle["cameras"]:
            views.append((extra, bundle["cameras"][extra], resolution))
    for name, camera, resolution in views:
        scene = _scene(instances, camera, resolution, _ceiling_emitter(bundle))
        for emitter in _daylight_emitters(bundle):
            scene[emitter["id"]] = emitter["mitsuba"]
        exr = output / f"{name}.exr"
        png = output / f"{name}.png"
        stokes = render_stokes(scene, exr, SCENE_SPP, RENDER_SEED)
        write_rgb_png(exr, png)
        support = _support_world(bundle) if name == "tabletop" else None
        report = instance_visibility(instances, camera, resolution, support_polygon=support, return_image=name == "tabletop")
        image = report.pop("id_image", None)
        report.pop("roi_mask", None)
        if image is not None:
            np.save(output / "instance_ids.npy", image.astype(np.int32))
            if list(image.shape) != stokes["shape"][:2]:
                raise RecipeError("reason=mask_shape")
        images[name] = {"png": str(png), "exr": str(exr), "shape": stokes["shape"], "channel_names": stokes["channel_names"], "finite": stokes["finite"], "dtype": stokes["dtype"]}
        visibility[name] = report
    rows = []
    by_id = {item["instance_id"]: item for item in [*bundle["furniture_instances"], *bundle["object_instances"]]}
    for row in visibility["tabletop"]["instances"]:
        meta = by_id.get(row["instance_id"], {})
        rows.append({**row, "role": meta.get("role"), "zone_id": meta.get("zone_id"), "image_support_flag": row["shorter_pixel_side"] >= 32, "low_spatial_support_note": row["low_spatial_support_note"]})
    (output / "instances.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    validation = {"task_id": bundle["task_id"], "lighting_id": "ceiling_v1", "images": images, "render_seed": RENDER_SEED, "spp": SCENE_SPP, "variant": "cuda_ad_spectral_polarized"}
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    logger.info("semantic_composition event=rendered task=%s", bundle["task_id"])
    return validation


def write_gallery(review_path: Path, root: Path = ROOT) -> None:
    review_path = Path(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    bundles = []
    for path in sorted(Path(root).glob("*/*/bundle.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        bundles.append((path, bundle))
    work = [item for item in bundles if item[1]["recipe"]["scene_type"] == "workbench"]
    other = [item for item in bundles if item[1]["recipe"]["scene_type"] != "workbench"]
    body = [
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Semantic scenes</title>",
        "<style>body{font-family:sans-serif;margin:2rem;background:#111;color:#eee} a{color:#9cf} img{max-width:520px;background:#222} figure{display:inline-block;margin:0.6rem;vertical-align:top} table{border-collapse:collapse;margin:0.5rem 0} td,th{border:1px solid #444;padding:0.2rem 0.45rem;text-align:left} .notice{border:1px solid #c90;padding:1rem}</style>",
        "</head><body><h1>Semantic scene composition</h1>",
        "<p class=\"notice\"><strong>These scenes are qualification candidates. Human visual review, admission, and registration are pending.</strong></p>",
        "<p>Union fraction is the 4 mm XY raster of object triangles divided by that work surface's support area. "
        "The free-rectangle value is 0.25 m by 0.25 m when an empty window of that size exists, and 0 otherwise. "
        "A shorter image side under 32 px is an observation flag. It does not change the semantic role.</p>",
    ]
    if work:
        close = work[0][1]["cameras"]["close"]["origin"]
        light = work[0][1]["lighting"]["ceiling"]
        body.append("<h2>Workbench clutter</h2>")
        body.append("<p>" + html.escape(
            f"Low, moderate, and high share room_spacious, one bench pose, close camera origin {close}, "
            f"a {light['size_m'][0]:.2f} by {light['size_m'][1]:.2f} m ceiling panel of radiance {light['radiance']} "
            f"at {light['center_m']}, window daylight, and cuda_ad_spectral_polarized."
        ) + "</p>")
        for path, bundle in sorted(work, key=lambda item: item[1]["recipe"]["seed"]):
            body.extend(_gallery_scene(path, bundle, review_path))
    for path, bundle in other:
        body.extend(_gallery_scene(path, bundle, review_path))
    body.append("</body></html>\n")
    review_path.write_text("\n".join(body), encoding="utf-8")


def _gallery_scene(path: Path, bundle: dict, review_path: Path) -> list[str]:
    validation = json.loads((path.parent / "render_ceiling_v1" / "validation.json").read_text(encoding="utf-8"))
    rows = json.loads((path.parent / "render_ceiling_v1" / "instances.json").read_text(encoding="utf-8"))
    clutter = bundle["clutter"]
    zones = {item["zone_id"]: [] for item in bundle["object_instances"]}
    for item in bundle["object_instances"]:
        zones.setdefault(item["zone_id"], []).append(item["instance_id"])
    partial = [row["instance_id"] for row in rows if row.get("role") not in {None, "context", "furniture"} and row["visible_to_projected_ratio"] < 1 and row["visible_pixels"] > 0]
    flagged = [row["instance_id"] for row in rows if row.get("image_support_flag") is False and str(row["instance_id"]).startswith("prop_")]
    parts = [
        "<h2>" + html.escape(f"{bundle['task_id']} seed {bundle['seed']} {bundle['recipe']['room_profile']}") + "</h2>",
        "<p>" + html.escape(
            f"objects {clutter['instance_count']} unique assets {clutter['unique_assets']} "
            f"union {clutter['union_fraction']:.4f} footprint sum {clutter['footprint_area_sum_m2']:.4f} m2 "
            f"free area {clutter['free_area_m2']:.4f} m2 free rectangle {clutter['free_rectangle_m']} "
            f"yaw std {clutter['yaw_std_rad']:.3f} rad nearest clearance {clutter['nearest_clearance_m']:.4f} m "
            f"repeats {clutter['repeated_assets']}"
        ) + "</p>",
        "<p>" + html.escape(f"zones {zones}") + "</p>",
        "<p>" + html.escape(f"partial visibility {partial or ['none']} image-support flag false {flagged or ['none']}") + "</p>",
    ]
    for name, image in validation["images"].items():
        href = Path(os.path.relpath(image["png"], review_path.parent)).as_posix()
        exr = Path(os.path.relpath(image["exr"], review_path.parent)).as_posix()
        label = {"room": "room overview", "tabletop": "work surface", "grouping": "object grouping", "workspace": "usable workspace"}.get(name, name)
        parts.append(f"<figure><a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(label)}\"></a><figcaption>{html.escape(label)} RGB</figcaption></figure>")
        parts.append(f"<p><a href=\"{html.escape(exr)}\">Stokes EXR {html.escape(label)}</a></p>")
    mask = path.parent / "render_ceiling_v1" / "instance_ids.npy"
    href = Path(os.path.relpath(mask, review_path.parent)).as_posix()
    parts.append(f"<p><a href=\"{html.escape(href)}\">Instance id mask</a></p>")
    return parts


def record_renderer_evidence(root: Path = ROOT) -> dict:
    """Mark a package renderer_validated only when a rendered view shows it and the EXR contract holds."""
    import mitsuba as mi

    require_variant()
    root = Path(root)
    catalog_path = root / "catalog" / "semantic_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    best: dict[str, dict] = {}
    exr_reports = []
    for bundle_path in sorted(root.glob("*/*/bundle.json")):
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle["renderer"]["variant"] != "cuda_ad_spectral_polarized":
            raise RecipeError(f"reason=variant task={bundle['task_id']}")
        rows = json.loads((bundle_path.parent / "render_ceiling_v1" / "instances.json").read_text(encoding="utf-8"))
        for name in ("room", "tabletop"):
            exr = bundle_path.parent / "render_ceiling_v1" / f"{name}.exr"
            bitmap = mi.Bitmap(str(exr))
            image = np.array(bitmap)
            names = _bitmap_names(bitmap)
            if names != list(CHANNEL_NAMES) or image.dtype != np.float32 or image.shape[2] != 15 or not np.isfinite(image).all():
                raise RecipeError(f"reason=exr_contract path={exr}")
            mask = np.load(bundle_path.parent / "render_ceiling_v1" / "instance_ids.npy")
            if name == "tabletop" and list(mask.shape) != list(image.shape[:2]):
                raise RecipeError(f"reason=mask_shape path={exr}")
            exr_reports.append({"path": str(exr), "shape": list(image.shape), "variant": "cuda_ad_spectral_polarized"})
        by_id = {row["instance_id"]: row for row in rows}
        for item in [*bundle["furniture_instances"], *bundle["object_instances"]]:
            row = by_id.get(item["instance_id"])
            if row is None or int(row["visible_pixels"]) <= 0:
                continue
            current = best.get(item["asset_id"])
            if current is None or int(row["visible_pixels"]) > int(current["visible_pixels"]):
                best[item["asset_id"]] = {
                    "asset_id": item["asset_id"],
                    "instance_id": item["instance_id"],
                    "visible_pixels": int(row["visible_pixels"]),
                    "scene": str(bundle_path.parent),
                    "exr": str(bundle_path.parent / "render_ceiling_v1" / "tabletop.exr"),
                    "package_sha256": item["package_sha256"],
                }
    missing = []
    for package in catalog["packages"]:
        evidence = best.get(package["asset_id"])
        if evidence is None or evidence["package_sha256"] != package["package_sha256"]:
            missing.append(package["asset_id"])
            continue
        package["status"] = "renderer_validated"
        package["renderer_validation"] = "renderer_validated"
        package["visible_pixels"] = evidence["visible_pixels"]
        package["evidence"] = evidence
    if missing:
        raise RecipeError(f"reason=renderer_evidence_missing assets={missing}")
    catalog_path.write_bytes(canonical_json_bytes(catalog))
    (root / "catalog" / "renderer_evidence.json").write_bytes(canonical_json_bytes({"packages": list(best.values()), "exr": exr_reports}))
    logger.info("semantic_composition event=renderer_evidence packages=%d exr=%d", len(best), len(exr_reports))
    return {"packages": len(best), "missing": missing}


def _recipe(name: str) -> dict:
    level = "moderate"
    profile = {"kitchen": "room_standard", "dining": "room_spacious", "desk": "room_compact"}.get(name, "room_spacious")
    scene_type = name
    seed = {"kitchen": 17401, "dining": 17402, "desk": 17403}.get(name, 17411)
    if name.startswith("workbench"):
        scene_type = "workbench"
        level = name.split("_", 1)[1]
        seed = {"low": 17411, "moderate": 17412, "high": 17413}[level]
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "scene_type": scene_type,
        "room_profile": profile,
        "clutter_level": level,
        "seed": seed,
        "furniture_instances": [],
        "object_instances": [],
        "functional_zones": [],
    }
    return recipe


def _object_slots(name: str) -> list[tuple]:
    if name == "kitchen":
        return [
            ("prop_cutting_board__1", "prop_cutting_board", "preparation", (0.28, 0.28)),
            ("prop_spatula__1", "prop_spatula", "preparation", (0.55, 0.22)),
            ("prop_plate__1", "prop_plate", "preparation", (0.28, 0.48)),
            ("prop_apple__1", "prop_apple", "ingredients", (0.90, 0.22)),
            ("prop_bowl__1", "prop_bowl", "ingredients", (1.05, 0.40)),
            ("prop_bottle__1", "prop_bottle", "storage", (1.40, 0.24)),
            ("prop_spoon__1", "prop_spoon", "preparation", (0.55, 0.42)),
            ("prop_pot__1", "prop_pot", "storage", (1.38, 0.42)),
        ]
    if name == "dining":
        return [
            ("prop_plate__1", "prop_plate", "place_a", (0.36, 0.18)),
            ("prop_bowl__1", "prop_bowl", "place_a", (0.20, 0.18)),
            ("prop_spoon__1", "prop_spoon", "place_a", (0.48, 0.30)),
            ("prop_plate__2", "prop_plate", "place_b", (0.92, 0.18)),
            ("prop_bowl__2", "prop_bowl", "place_b", (0.76, 0.18)),
            ("prop_spoon__2", "prop_spoon", "place_b", (1.06, 0.30)),
            ("prop_serving_dish__1", "prop_serving_dish", "serving", (0.48, 0.54)),
            ("prop_folded_cloth__1", "prop_folded_cloth", "serving", (0.76, 0.54)),
            ("prop_bottle__1", "prop_bottle", "drinks", (0.14, 0.50)),
            ("prop_cup__1", "prop_cup", "drinks", (0.14, 0.62)),
            ("prop_fork__1", "prop_fork", "place_a", (0.22, 0.32)),
        ]
    if name == "desk":
        return [
            ("prop_notebook__1", "prop_notebook", "main_work", (0.28, 0.32)),
            ("prop_cable_group__1", "prop_cable_group", "main_work", (0.30, 0.52)),
            ("prop_keyboard__1", "prop_keyboard", "input_devices", (0.82, 0.28)),
            ("prop_phone__1", "prop_phone", "input_devices", (0.82, 0.50)),
            ("prop_pen__1", "prop_pen", "writing", (1.18, 0.24)),
            ("prop_marker__1", "prop_marker", "writing", (1.22, 0.42)),
        ]
    ids = {"workbench_low": LOW_IDS, "workbench_moderate": MODERATE_IDS, "workbench_high": HIGH_IDS}[name]
    slots = []
    for instance_id in ids:
        asset_id = instance_id.rsplit("__", 1)[0]
        zone, xy = SLOTS[instance_id]
        slots.append((instance_id, asset_id, zone, xy))
    return slots


def _mesh(asset_id: str, assets: dict, catalog: Path):
    if asset_id in assets:
        built = assets[asset_id]
        return built["vertices"], built["faces"], built
    if asset_id in BUILDERS:
        built = build_asset(asset_id)
        return built["vertices"], built["faces"], built
    loaded = load_mesh(catalog, asset_id, VERSION)
    return loaded["vertices"], loaded["faces"], None


def _place_object(instance_id, asset_id, zone_id, slot, yaw, support_body, assets, catalog) -> dict:
    vertices, faces, built = _mesh(asset_id, assets, catalog)
    zones = {zone["id"]: zone for zone in support_body["zones"]}
    zone = zones[zone_id]
    local = _slot_matrix(vertices, slot, support_body["support_z"], yaw, lay_on_side=asset_id in LAY_ON_SIDE)
    local_world = world_vertices(vertices, local)
    low, high = local_world.min(axis=0), local_world.max(axis=0)
    rect = zone["rectangle_m"]
    inside = float(low[0]) >= rect[0] - 1e-4 and float(high[0]) <= rect[2] + 1e-4 and float(low[1]) >= rect[1] - 1e-4 and float(high[1]) <= rect[3] + 1e-4
    world_matrix = support_body["matrix"] @ local
    world = world_vertices(vertices, world_matrix)
    polygon = apply_points(support_body["matrix"], support_body["support_polygon"])
    contact = measure_support(world, faces, polygon)
    role = built["role"] if built is not None else EXISTING_ROLES[asset_id]
    if built is not None:
        geometry_category = built["profile"]
        material_category = built["material"]["type"]
    else:
        geometry_category = "reused"
        material_category = SCENE_AC_BSDF.get(asset_id, verify_package(catalog / asset_id / VERSION)["materials"][0])["type"]
    return {
        "instance_id": instance_id,
        "asset_id": asset_id,
        "zone_id": zone_id,
        "role": role,
        "matrix": world_matrix,
        "local_matrix": local,
        "vertices": world,
        "faces": faces,
        "yaw": yaw,
        "inside_zone": inside,
        "on_support": bool(contact["passed"]),
        "support_gap_m": contact["gap_m"],
        "material": None if built is None else built["material"],
        "geometry_category": geometry_category,
        "material_category": material_category,
    }


def _slot_matrix(vertices, slot, support_z, yaw, lay_on_side: bool = False) -> np.ndarray:
    """Drop the mesh onto the support. A side rest turns a vertical shaft onto local +X."""
    rest = rotation_matrix("y", np.pi / 2.0) if lay_on_side else identity_matrix()
    rested = apply_points(rest, vertices)
    low, high = rested.min(axis=0), rested.max(axis=0)
    center = translation_matrix([-0.5 * (low[0] + high[0]), -0.5 * (low[1] + high[1]), -float(low[2])])
    spun = compose(rotation_matrix("z", float(yaw)), compose(center, rest))
    return compose(translation_matrix([float(slot[0]), float(slot[1]), float(support_z)]), spun)


def _place_workbench(built, spec) -> dict:
    return _against_wall(built, spec, "east", 2.40, "workbench_1")


def _place_counter(built, spec) -> dict:
    return _against_wall(built, spec, "west", 1.90, "counter_1")


def _place_desk(built, spec) -> dict:
    return _against_wall(built, spec, "east", 1.80, "desk_1")


def _against_wall(built, spec, wall: str, along: float, instance_id: str) -> dict:
    yaw = {"east": np.pi / 2.0, "west": -np.pi / 2.0}[wall]
    parked = _yaw_about_footprint(built["vertices"], yaw, np.zeros(2), 0.0)
    world = world_vertices(built["vertices"], parked)
    low, high = world.min(axis=0), world.max(axis=0)
    shift = [0.0, 0.0, 0.0]
    if wall == "east":
        shift[0] = float(spec["interior_width_m"]) - 0.05 - float(high[0])
    else:
        shift[0] = 0.05 - float(low[0])
    shift[1] = along - 0.5 * float(low[1] + high[1])
    matrix = compose(translation_matrix(shift), parked)
    return _furniture_record(instance_id, built["asset_id"], built, matrix)


def _place_table(catalog: Path, spec: dict) -> dict:
    loaded = load_mesh(catalog, "table_rect", VERSION)
    low, high = loaded["vertices"].min(axis=0), loaded["vertices"].max(axis=0)
    size = high - low
    origin = [float(spec["interior_width_m"]) * 0.5 - float(size[0]) * 0.5, float(spec["interior_depth_m"]) * 0.5 - float(size[1]) * 0.5, 0.0]
    matrix = translation_matrix(origin)
    support = next(item for item in loaded["manifest"]["support_surfaces"] if item["id"] == "top")
    polygon = np.asarray(support["polygon"], dtype=np.float64)
    world = world_vertices(loaded["vertices"], matrix)
    return {
        "instance_id": "table_1",
        "asset_id": "table_rect",
        "role": "furniture",
        "matrix": matrix,
        "vertices": world,
        "faces": loaded["faces"],
        "support_z": float(polygon[0, 2]),
        "support_polygon": polygon,
        "zones": [
            {"id": "place_a", "support_surface": "top", "rectangle_m": [0.05, 0.02, 0.55, 0.38], "purpose": "first place setting", "roles": ["manipulation_target", "tool", "context"]},
            {"id": "place_b", "support_surface": "top", "rectangle_m": [0.65, 0.02, 1.15, 0.38], "purpose": "second place setting", "roles": ["manipulation_target", "tool"]},
            {"id": "serving", "support_surface": "top", "rectangle_m": [0.30, 0.40, 0.90, 0.68], "purpose": "shared serving", "roles": ["manipulation_target", "context"]},
            {"id": "drinks", "support_surface": "top", "rectangle_m": [0.02, 0.40, 0.28, 0.68], "purpose": "drinks", "roles": ["material", "manipulation_target"]},
        ],
        "zone_id": None,
    }


def _place_chair(catalog, spec, table_body, instance_id, side: float) -> dict:
    loaded = load_mesh(catalog, "prop_chair", VERSION)
    table_low, table_high = table_body["vertices"].min(axis=0), table_body["vertices"].max(axis=0)
    chair_size = loaded["vertices"].max(axis=0) - loaded["vertices"].min(axis=0)
    center_x = 0.5 * float(table_low[0] + table_high[0])
    center_y = float(table_low[1]) - 0.12 - 0.5 * float(chair_size[1]) if side < 0 else float(table_high[1]) + 0.12 + 0.5 * float(chair_size[1])
    desired = np.array([center_x, 0.5 * float(table_low[1] + table_high[1])]) - np.array([center_x, center_y])
    yaw = float(np.atan2(desired[0], -desired[1]))
    matrix = _yaw_about_footprint(loaded["vertices"], yaw, np.array([center_x, center_y]), 0.0)
    world = world_vertices(loaded["vertices"], matrix)
    return {
        "instance_id": instance_id,
        "asset_id": "prop_chair",
        "role": "furniture",
        "matrix": matrix,
        "vertices": world,
        "faces": loaded["faces"],
        "support_z": None,
        "support_polygon": None,
        "zones": [],
        "zone_id": None,
        "seating_direction": (matrix[:3, :3] @ np.array([0.0, -1.0, 0.0])).tolist(),
    }


def _furniture_record(instance_id, asset_id, built, matrix) -> dict:
    world = world_vertices(built["vertices"], matrix)
    support = built["support"]
    polygon = np.array(
        [
            [support["rectangle_m"][0], support["rectangle_m"][1], support["z_m"]],
            [support["rectangle_m"][2], support["rectangle_m"][1], support["z_m"]],
            [support["rectangle_m"][2], support["rectangle_m"][3], support["z_m"]],
            [support["rectangle_m"][0], support["rectangle_m"][3], support["z_m"]],
        ],
        dtype=np.float64,
    )
    return {
        "instance_id": instance_id,
        "asset_id": asset_id,
        "role": "furniture",
        "matrix": matrix,
        "vertices": world,
        "faces": built["faces"],
        "support_z": support["z_m"],
        "support_polygon": polygon,
        "zones": built["zones"],
        "zone_id": None,
    }


def _yaw_about_footprint(vertices, yaw, world_xy, floor_z) -> np.ndarray:
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = translation_matrix([-0.5 * (low[0] + high[0]), -0.5 * (low[1] + high[1]), -float(low[2])])
    spun = compose(rotation_matrix("z", float(yaw)), center)
    return compose(translation_matrix([float(world_xy[0]), float(world_xy[1]), floor_z]), spun)


def _room_checks(furniture, spec, failures) -> list[dict]:
    from sensor_sim.scene_gen.room_generator import _keepout_mesh

    floor = np.array([[0.0, 0.0, 0.0], [spec["interior_width_m"], 0.0, 0.0], [spec["interior_width_m"], spec["interior_depth_m"], 0.0], [0.0, spec["interior_depth_m"], 0.0]], dtype=np.float64)
    door = next(item for item in spec["openings"] if item["kind"] == "door")
    corridor = [float(door["offset_m"]), 0.0, float(door["offset_m"]) + float(door["width_m"]), 0.90]
    records = []
    for item in furniture:
        contact = measure_support(item["vertices"], item["faces"], floor)
        low, high = item["vertices"].min(axis=0), item["vertices"].max(axis=0)
        door_overlap = _rects_overlap([float(low[0]), float(low[1]), float(high[0]), float(high[1])], corridor)
        if not contact["passed"]:
            failures.append({"name": f"floor:{item['instance_id']}", "passed": False, "reason": contact["reason"]})
        if door_overlap:
            failures.append({"name": f"door_corridor:{item['instance_id']}", "passed": False})
        windows = []
        for opening in spec["openings"]:
            if opening["kind"] != "window":
                continue
            found = unintended_intersection(item, _keepout_mesh(opening, spec))
            windows.append({"id": opening["id"], "wall": opening["wall"], "clearance_m": found["clearance_m"], "intersects": bool(found["unintended"])})
            if found["unintended"]:
                failures.append({"name": f"window_keepout:{item['instance_id']}:{opening['id']}", "passed": False, "clearance_m": found["clearance_m"]})
        facing = float(low[0]) if float(low[0]) > 0.2 else float(spec["interior_width_m"]) - float(high[0])
        records.append({
            "instance_id": item["instance_id"],
            "floor_gap_m": contact["gap_m"],
            "floor_passed": bool(contact["passed"]),
            "door_corridor_clear": not door_overlap,
            "window_clearance": windows,
            "room_facing_gap_m": facing,
            "aabb_min_m": [float(value) for value in low],
            "aabb_max_m": [float(value) for value in high],
        })
    return records


def _rects_overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _clutter(objects, support_body, level: str) -> dict:
    rect = support_body["support_polygon"]
    x0, y0 = float(rect[:, 0].min()), float(rect[:, 1].min())
    x1, y1 = float(rect[:, 0].max()), float(rect[:, 1].max())
    resolution = 0.004
    width = max(1, int(np.ceil((x1 - x0) / resolution)))
    height = max(1, int(np.ceil((y1 - y0) / resolution)))
    grid = np.zeros((height, width), dtype=bool)
    local_vertices = []
    for item in objects:
        local = world_vertices(item["vertices"], np.linalg.inv(support_body["matrix"]))
        local_vertices.append(local)
        for face in item["faces"]:
            _stamp_triangle(grid, local[face][:, :2], x0, y0, resolution, width, height)
    area = float((x1 - x0) * (y1 - y0))
    union = float(grid.sum() * resolution * resolution)
    yaws = [float(item["yaw"]) for item in objects]
    free = _largest_free_rectangle(grid, resolution, x0, y0)
    usable = _largest_free_rectangle(grid, resolution, x0, y0, min_side_m=0.20)
    counts = _counts(objects)
    footprints = []
    for local in local_vertices:
        low, high = local.min(axis=0), local.max(axis=0)
        footprints.append(float((high[0] - low[0]) * (high[1] - low[1])))
    return {
        "level": level,
        "instance_count": len(objects),
        "unique_assets": len(counts),
        "unique_geometry_categories": sorted({item["geometry_category"] for item in objects}),
        "material_categories": sorted({item["material_category"] for item in objects}),
        "union_m2": union,
        "union_fraction": union / area if area else 0.0,
        "support_area_m2": area,
        "footprint_area_sum_m2": float(sum(footprints)),
        "free_area_m2": area - union,
        "yaw_std_rad": float(np.std(yaws)) if yaws else 0.0,
        "repeated_assets": sorted(asset for asset, count in counts.items() if count > 1),
        "repeat_count": int(sum(count - 1 for count in counts.values() if count > 1)),
        "largest_free_rectangle": free,
        "usable_free_rectangle": usable,
        "free_rectangle_m": usable["size_m"],
        "nearest_clearance_m": _nearest(objects),
        "union_definition": "4 mm XY raster of support-frame triangles divided by the support polygon area",
        "footprint_definition": "sum of per-object XY axis-aligned bounding boxes in the support frame",
        "free_rectangle_definition": "largest empty axis-aligned rectangle on the 4 mm raster, origin and size in the support frame",
    }


def _counts(objects) -> dict:
    found = {}
    for item in objects:
        found[item["asset_id"]] = found.get(item["asset_id"], 0) + 1
    return found


def _nearest(objects) -> float:
    if len(objects) < 2:
        return 0.0
    best = 1e9
    for index, left in enumerate(objects):
        for right in objects[index + 1 :]:
            best = min(best, float(mesh_clearance(left, right)["clearance_m"]))
    return best


def _largest_free_rectangle(grid: np.ndarray, resolution: float, x0: float, y0: float, min_side_m: float = 0.0) -> dict:
    """Largest empty axis-aligned rectangle on the occupancy raster.

    ``min_side_m`` keeps only rectangles whose shorter side is at least that
    long, so a thin empty margin is not reported as the usable workspace.
    """
    height, width = grid.shape
    min_cells = int(np.ceil(min_side_m / resolution)) if min_side_m else 0
    histogram = np.zeros(width, dtype=np.int32)
    best = (0, 0, 0, 0, 0)
    for row in range(height):
        histogram = np.where(grid[row], 0, histogram + 1)
        area, left, rect_width, rect_height = _histogram_rectangle(histogram, min_cells)
        if area > best[0]:
            best = (area, left, row - rect_height + 1, rect_width, rect_height)
    _area, left, top, rect_width, rect_height = best
    return {
        "origin_m": [float(x0 + left * resolution), float(y0 + top * resolution)],
        "size_m": [float(rect_width * resolution), float(rect_height * resolution)],
        "area_m2": float(rect_width * rect_height * resolution * resolution),
    }


def _histogram_rectangle(histogram: np.ndarray, min_cells: int = 0) -> tuple[int, int, int, int]:
    stack = [-1]
    best = (0, 0, 0, 0)
    values = list(histogram) + [0]
    for index, height in enumerate(values):
        while stack[-1] != -1 and values[stack[-1]] >= height:
            bar = values[stack.pop()]
            left = stack[-1] + 1
            width = index - left
            if bar >= min_cells and width >= min_cells and bar * width > best[0]:
                best = (bar * width, left, width, bar)
        stack.append(index)
    return best


def _freeze_bands(reports) -> dict:
    work = {item["recipe"]: item["clutter"]["union_fraction"] for item in reports if item["recipe"].startswith("workbench")}
    return {"low_max": work["workbench_low"], "moderate": work["workbench_moderate"], "high_min": work["workbench_high"]}


def _public_asset(built) -> dict:
    return {
        "asset_id": built["asset_id"],
        "expected_m": built["expected_m"],
        "measured_m": built["measured_m"],
        "origin_m": built["origin_m"],
        "role": built["role"],
        "material": built["material"],
        "triangle_count": built["triangle_count"],
        "footprint_m2": built["footprint_m2"],
        "interior": built["interior"],
        "profile": built["profile"],
    }


def _write_package(destination: Path, built: dict) -> None:
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
    version = package_version(built["asset_id"])
    parameters = {"asset_id": built["asset_id"]}
    if built["asset_id"] in VERSION_REASONS:
        parameters["revision_reason"] = VERSION_REASONS[built["asset_id"]]
    manifest = {
        "asset_id": built["asset_id"],
        "version": version,
        "units": "m",
        "frame": {"up_axis": "Z", "handedness": "right", "transform_convention": "row_major_column_vector", "matrix": serialize_matrix(identity_matrix())},
        "provenance": {
            "source": "sensor_sim",
            "generator": "semantic_assets",
            "generator_version": "1.0.0",
            "seed": 17401,
            "parameters": parameters,
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
            "components": [{"id": component["id"], "profile": component["profile"], **({"thickness_axis": component["thickness_axis"]} if "thickness_axis" in component else {})} for component in built["components"]],
        },
        "collision": [{"id": "proxy", "path": "geometry/collision.obj", "render_path": "geometry/render.obj", "triangle_count": built["triangle_count"], "qualified": True}],
        "support_surfaces": supports,
        "materials": [{"id": "default", **built["material"]}],
    }
    create_package(destination, manifest, payload, {"geometry/render.obj": "render", "geometry/collision.obj": "collision"})


def _support_world(bundle: dict) -> np.ndarray:
    support = next(item for item in bundle["furniture_instances"] if item.get("support_polygon") is not None)
    return apply_points(deserialize_matrix(support["matrix"]), np.asarray(support["support_polygon"], dtype=np.float64))


def _overview_camera(support_world: np.ndarray, spec: dict) -> dict:
    """Room view. Retarget only when the saved corner misses the work surface."""
    polygon = np.asarray(support_world, dtype=np.float64)
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    height = float(spec["interior_height_m"])
    up = [0.0, 0.0, 1.0]
    fov = 58.0
    aspect = 512.0 / 384.0
    origin = [0.45, 0.55, min(1.5, height - 0.3)]
    target = [width * 0.62, depth * 0.62, 1.0]
    peak = float(_peak_ndc(polygon, origin, target, up, fov, aspect))
    if peak > 0.98:
        center = polygon.mean(axis=0)
        choices = []
        for xy in ((width - 0.55, 0.55), (width - 0.55, depth - 0.55), (0.55, depth - 0.55), (0.55, 0.55)):
            trial_origin = [float(xy[0]), float(xy[1]), min(1.5, height - 0.3)]
            trial_target = [float(center[0]), float(center[1]), 1.0]
            trial_peak = float(_peak_ndc(polygon, trial_origin, trial_target, up, fov, aspect))
            distance = float(np.hypot(trial_origin[0] - center[0], trial_origin[1] - center[1]))
            choices.append((trial_peak > 1.0, trial_peak, -distance, trial_origin, trial_target))
        choices.sort()
        _outside, peak, _distance, origin, target = choices[0]
        logger.info("semantic_composition event=overview_retarget peak_ndc=%.4f origin=%s", peak, origin)
    return {"name": "room_overview", "origin": list(origin), "target": list(target), "up": up, "fov_deg": fov, "resolution": [512, 384], "peak_ndc": peak}


def _zone_world(support: dict, zone_id: str) -> np.ndarray:
    zone = next(item for item in support["zones"] if item["id"] == zone_id)
    x0, y0, x1, y1 = zone["rectangle_m"]
    z_value = float(support["support_z"])
    local = np.array([[x0, y0, z_value], [x1, y0, z_value], [x1, y1, z_value], [x0, y1, z_value]], dtype=np.float64)
    return apply_points(support["matrix"], local)


def _rect_world(support: dict, origin: list[float], size: list[float]) -> np.ndarray:
    x0, y0 = origin
    width, height = size
    z_value = float(support["support_z"])
    local = np.array(
        [[x0, y0, z_value], [x0 + width, y0, z_value], [x0 + width, y0 + height, z_value], [x0, y0 + height, z_value]],
        dtype=np.float64,
    )
    return apply_points(support["matrix"], local)


def frame_support_camera(support_world: np.ndarray, spec: dict) -> dict:
    """Frame any support polygon with the table-camera projection.

    The approach is perpendicular to the support's long axis and comes from the
    room interior, so a wall bench is not viewed down its length. The clip is
    that room's interior, not the old 3.7 m table-room clip.
    """
    polygon = np.asarray(support_world, dtype=np.float64)
    center = polygon.mean(axis=0)
    target = np.array([center[0], center[1], center[2] + 0.04], dtype=np.float64)
    extent = polygon.max(axis=0) - polygon.min(axis=0)
    if float(extent[0]) >= float(extent[1]):
        back = np.array([0.0, -1.0, 0.42], dtype=np.float64)
    else:
        side = -1.0 if float(center[0]) > 0.5 * float(spec["interior_width_m"]) else 1.0
        back = np.array([side, 0.0, 0.42], dtype=np.float64)
    back /= np.linalg.norm(back)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    origin = _fit_origin(polygon, target, back, up, 50.0, 1600.0 / 900.0, 0.08)
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    height = float(spec["interior_height_m"])
    clipped = np.array([
        float(np.clip(origin[0], 0.35, width - 0.35)),
        float(np.clip(origin[1], 0.35, depth - 0.35)),
        float(np.clip(origin[2], 0.90, height - 0.25)),
    ])
    shift = float(np.linalg.norm(clipped - origin))
    peak = float(_peak_ndc(polygon, clipped, target, up, 50.0, 1600.0 / 900.0))
    if shift > 0.02 or peak > 0.94:
        logger.info("semantic_composition event=camera_fit shift_m=%.4f peak_ndc=%.4f", shift, peak)
    return {
        "name": "work_surface_close",
        "origin": clipped.tolist(),
        "target": target.tolist(),
        "up": [0.0, 0.0, 1.0],
        "fov_deg": 50.0,
        "margin": 0.08,
        "resolution": [1600, 900],
        "peak_ndc": peak,
        "clip_shift_m": shift,
    }


def _room_lighting(spec: dict) -> dict:
    """One ceiling panel plus an unpolarized rectangle outside each window.

    The panel is inset from the walls and faces down. Its area is most of the
    ceiling, so the floor is lit across the room instead of under one small
    hotspot. Window emitters sit outside the openings and shine inward.
    """
    width = float(spec["interior_width_m"])
    depth = float(spec["interior_depth_m"])
    height = float(spec["interior_height_m"])
    ceiling = {
        "id": "ceiling_panel_v1",
        "center_m": [width * 0.5, depth * 0.5, height - CEILING_DROP_M],
        "size_m": [width - 2.0 * CEILING_INSET_M, depth - 2.0 * CEILING_INSET_M],
        "radiance": CEILING_RADIANCE,
        "normal": [0.0, 0.0, -1.0],
        "polarized": False,
        "calibrated": False,
    }
    emitters = []
    for opening in spec["openings"]:
        if opening["kind"] != "window":
            continue
        record = _emitter_record(spec, opening)
        record["radiance"] = WINDOW_RADIANCE
        emitters.append(record)
    return {
        "ceiling": ceiling,
        "daylight": {"id": "window_daylight_v1", "emitters": emitters, "radiance": WINDOW_RADIANCE, "polarized": False, "calibrated": False},
    }


def _ceiling_emitter(bundle: dict) -> dict:
    import mitsuba as mi

    require_variant()
    light = bundle["lighting"]["ceiling"]
    size = light.get("size_m", [0.80, 0.50])
    transform = mi.ScalarTransform4f.translate(light["center_m"]) @ mi.ScalarTransform4f.rotate([1.0, 0.0, 0.0], 180.0) @ mi.ScalarTransform4f.scale([size[0] * 0.5, size[1] * 0.5, 1.0])
    return {"type": "rectangle", "to_world": transform, "emitter": {"type": "area", "radiance": {"type": "rgb", "value": light["radiance"]}}}


def _render_instances(bundle, semantic_catalog, room_catalog, asset_catalog) -> list[dict]:
    instances = []
    for group, root in ((bundle["room_components"], room_catalog), (bundle["furniture_instances"], None), (bundle["object_instances"], None)):
        for item in group:
            asset_id = item["asset_id"]
            version = item.get("version", VERSION)
            if asset_id.startswith("room_"):
                mesh_root = room_catalog / asset_id / version
                material = verify_package(mesh_root)["materials"][0]
            elif (semantic_catalog / asset_id / version).is_dir():
                mesh_root = semantic_catalog / asset_id / version
                material = verify_package(mesh_root)["materials"][0]
            else:
                mesh_root = asset_catalog / asset_id / version
                material = dict(SCENE_AC_BSDF.get(asset_id, verify_package(mesh_root)["materials"][0]))
            instances.append({"id": item["instance_id"], "asset_id": asset_id, "role": item.get("role", "required"), "mesh": mesh_root / "geometry" / "render.obj", "matrix": deserialize_matrix(item["matrix"]), "bsdf": material})
    return instances


def write_pilot(output: Path = ROOT, asset_catalog: Path = ASSET_CATALOG) -> list[dict]:
    """Package frozen assets and write the six bundles. Rendering is separate."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = preflight(asset_catalog)
    (output / "preflight.json").write_text(json.dumps({"placements": [{k: v for k, v in item.items() if k != "clutter"} | {"clutter": item["clutter"]} for item in report["placements"]], "occupancy_bands": report["occupancy_bands"], "assets": report["assets"]}, indent=2) + "\n", encoding="utf-8")
    catalog = output / "catalog"
    packages = {item["asset_id"]: item for item in package_new_assets(catalog)}
    (catalog / "semantic_catalog.json").write_bytes(canonical_json_bytes({"schema_version": "semantic_catalog_v1", "packages": list(packages.values())}))
    written = []
    assets = {asset_id: build_asset(asset_id) for asset_id in BUILDERS}
    for name in ("kitchen", "dining", "desk", "workbench_low", "workbench_moderate", "workbench_high"):
        placed = place_recipe(name, assets, asset_catalog)
        bundle = _bundle(placed, packages, asset_catalog)
        run = hashlib.sha256(canonical_json_bytes({"task": bundle["task_id"], "seed": bundle["seed"]})).hexdigest()[:16]
        path = output / bundle["recipe"]["scene_type"] / run
        path.mkdir(parents=True, exist_ok=False)
        (path / "recipe.json").write_bytes(canonical_json_bytes(bundle["recipe"]))
        (path / "bundle.json").write_bytes(canonical_json_bytes(bundle))
        (path / "scene.json").write_bytes(canonical_json_bytes(scene_document(bundle)))
        replay_semantic(path, catalog, ROOM_ROOT / "catalog")
        written.append({"path": str(path), "task_id": bundle["task_id"]})
        logger.info("semantic_composition event=scene task=%s path=%s", bundle["task_id"], path)
    return written


def _bundle(placed, packages, asset_catalog: Path) -> dict:
    recipe = placed["recipe"]
    room = placed["room"]
    spec = room["spec"]
    room_packages = []
    room_components = []
    for group in ("floor", "ceiling", "wall_south", "wall_north", "wall_east", "wall_west"):
        asset_id = f"{spec['room_id']}_{group}"
        manifest = verify_package(ROOM_ROOT / "catalog" / asset_id / VERSION)
        room_packages.append({"asset_id": asset_id, "version": VERSION, "path": f"{asset_id}/{VERSION}", "package_sha256": manifest["package_sha256"]})
        room_components.append({"instance_id": asset_id, "asset_id": asset_id, "version": VERSION, "package_sha256": manifest["package_sha256"], "matrix": serialize_matrix(identity_matrix()), "role": "context"})
    furniture_instances = []
    object_instances = []
    relations = []
    semantic_packages = []
    seen_packages = set()
    for item in placed["furniture"] + placed["objects"]:
        record = _saved_instance(item, packages, asset_catalog)
        if item in placed["furniture"]:
            furniture_instances.append(record)
            relations.append(_floor_relation(record["instance_id"], f"{spec['room_id']}_floor"))
        else:
            object_instances.append(record)
            support = next(entry for entry in placed["furniture"] if entry.get("support_polygon") is not None)
            relations.append(_surface_relation(record["instance_id"], support["instance_id"], item["support_gap_m"]))
        if item["asset_id"] in packages and item["asset_id"] not in seen_packages:
            semantic_packages.append(packages[item["asset_id"]])
            seen_packages.add(item["asset_id"])
    support = next(item for item in placed["furniture"] if item.get("support_polygon") is not None)
    zones = support["zones"]
    recipe = {
        **recipe,
        "furniture_instances": furniture_instances,
        "object_instances": object_instances,
        "functional_zones": zones,
    }
    require_recipe(recipe)
    support_world = apply_points(support["matrix"], support["support_polygon"])
    close = frame_support_camera(support_world, spec)
    overview = _overview_camera(support_world, spec)
    primary_zone = {"workbench": "assembly", "kitchen": "preparation", "dining": "serving", "desk": "main_work"}[recipe["scene_type"]]
    grouping = frame_support_camera(_zone_world(support, primary_zone), spec)
    if recipe["scene_type"] == "workbench":
        workspace = frame_support_camera(_zone_world(support, "free_workspace"), spec)
    else:
        usable = placed["clutter"]["usable_free_rectangle"]
        workspace = frame_support_camera(_rect_world(support, usable["origin_m"], usable["size_m"]), spec) if usable["area_m2"] > 0 else close
    level = recipe["clutter_level"]
    bundle = {
        "schema_version": "semantic_bundle_v1",
        "task_id": f"{recipe['scene_type']}_{level}" if recipe["scene_type"] == "workbench" else recipe["scene_type"],
        "seed": recipe["seed"],
        "recipe": recipe,
        "recipe_sha256": hashlib.sha256(canonical_json_bytes(recipe)).hexdigest(),
        "room_spec_sha256": room["sha256"],
        "room_spec_path": room["path"],
        "room_packages": room_packages,
        "semantic_packages": semantic_packages,
        "room_components": room_components,
        "furniture_instances": furniture_instances,
        "object_instances": object_instances,
        "relations": relations,
        "clutter": placed["clutter"],
        "cameras": {"overview": overview, "close": close, "grouping": grouping, "workspace": workspace},
        "lighting": _room_lighting(spec),
        "renderer": {"variant": "cuda_ad_spectral_polarized", "spp": SCENE_SPP, "seed": RENDER_SEED},
        "room_clearance": placed["room_clearance"],
        "validation": {"valid": True, "visual_verification": "pending", "admitted": False},
    }
    # Workbench clutter comparisons share one camera. Freeze the low camera onto the later levels.
    require_scene_context([*room_components, *furniture_instances, *object_instances], room_kind=recipe["scene_type"])
    return bundle


def _saved_instance(item, packages, asset_catalog: Path) -> dict:
    asset_id = item["asset_id"]
    if asset_id in packages:
        digest = packages[asset_id]["package_sha256"]
        version = packages[asset_id]["version"]
    else:
        version = VERSION
        digest = verify_package(asset_catalog / asset_id / version)["package_sha256"]
    saved = {
        "instance_id": item["instance_id"],
        "asset_id": asset_id,
        "version": version,
        "package_sha256": digest,
        "matrix": serialize_matrix(item["matrix"]),
        "role": item["role"],
        "zone_id": item.get("zone_id"),
    }
    if item.get("support_polygon") is not None:
        saved["support_polygon"] = np.asarray(item["support_polygon"], dtype=np.float64).round(6).tolist()
        saved["support_z"] = item["support_z"]
        saved["zones"] = item["zones"]
    return saved


def _floor_relation(instance_id: str, floor_id: str) -> dict:
    return {"type": "supported_on", "instances": [instance_id, floor_id], "frame": "support_surface_local", "frame_instance": floor_id, "surface": "interior", "priority": "required", "constraints": {}, "passed": True, "reason": "contact", "measured_m": 0.0}


def _surface_relation(instance_id: str, support_id: str, gap) -> dict:
    return {"type": "supported_on", "instances": [instance_id, support_id], "frame": "support_surface_local", "frame_instance": support_id, "surface": "top", "priority": "required", "constraints": {}, "passed": True, "reason": "contact", "measured_m": float(gap or 0.0), "gap_limit_m": 0.001, "penetration_limit_m": 0.0001}


def validate_bsdfs() -> None:
    import mitsuba as mi

    require_variant()
    samples = [
        {"type": "diffuse", "reflectance": [0.5, 0.5, 0.5]},
        {"type": "pplastic", "int_ior": 1.5, "ext_ior": 1.0, "diffuse_reflectance": [0.4, 0.4, 0.4], "alpha": 0.1, "distribution": "beckmann"},
        {"type": "conductor", "material": "Al"},
        {"type": "roughconductor", "material": "Al", "distribution": "beckmann", "alpha": 0.1},
    ]
    for material in samples:
        mi.load_dict({"type": "scene", "shape": {"type": "sphere", "bsdf": mitsuba_bsdf(material)}})
    try:
        mitsuba_bsdf({"type": "plastic", "diffuse_reflectance": [0.2, 0.2, 0.2]})
    except Exception:
        logger.info("semantic_composition event=bsdf_reject type=plastic")
        return
    raise RecipeError("reason=unsupported_material_accepted")

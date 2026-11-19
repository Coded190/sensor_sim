"""Render saved task bundles and asset inspection views.

This module does not generate tasks or retrieve assets. It calls the existing
Mitsuba Stokes writer. Materials and cameras are render settings. They do not
change saved object transforms.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import PackageError, read_index, verify_package
from sensor_sim.scene_gen.asset_validate import triangulate
from sensor_sim.scene_gen.indoor_contact import measure_support, unintended_intersection
from sensor_sim.scene_gen.indoor_mitsuba import load_shapes, render_stokes, require_variant, write_rgb_png
from sensor_sim.scene_gen.indoor_transform import deserialize_matrix, inspect_linear
from sensor_sim.scene_gen.mesh_io import read_obj
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES
from sensor_sim.scene_gen.task_replay import replay
from sensor_sim.scene_gen.task_schema import TaskError

logger = logging.getLogger("sensor_sim.scene_gen.task_render")

VARIANT = "cuda_ad_spectral_polarized"
RESOLUTION = (256, 192)
SPP = 16
SEED = 17011
FOV_DEG = 45.0
FRAME_MARGIN = 1.25
PRODUCTION_ASSETS = (
    "room_floor",
    "room_ceiling",
    "room_wall_south",
    "room_wall_north",
    "room_wall_west",
    "room_wall_east",
    "table_rect",
    "shelf_two",
    "prop_crate",
    "prop_crate_open",
    "prop_cup",
    "prop_book",
)
FIXTURE_ASSETS = ("fixture_sphere", "fixture_cone")
SAVED_RUNS = {
    "book_on_table": "e54a5e18636df8e7",
    "cup_on_shelf": "8a0a6b98b048ef83",
    "book_inside_crate": "e547b5c803391dbe",
    "crate_and_book_on_table": "44ac50f76b66afcb",
    "cup_near_book": "95d8b3f496b78836",
    "objects_in_workspace": "3cf07be40ab68252",
}
# Qualified Scene A–C render colors. Package materials are recorded separately.
SCENE_AC_BSDF = {
    "room_floor": {"type": "diffuse", "reflectance": [0.55, 0.52, 0.46]},
    "room_ceiling": {"type": "diffuse", "reflectance": [0.9, 0.9, 0.88]},
    "room_wall_south": {"type": "diffuse", "reflectance": [0.82, 0.82, 0.78]},
    "room_wall_north": {"type": "diffuse", "reflectance": [0.82, 0.82, 0.78]},
    "room_wall_west": {"type": "diffuse", "reflectance": [0.82, 0.82, 0.78]},
    "room_wall_east": {"type": "diffuse", "reflectance": [0.82, 0.82, 0.78]},
    "table_rect": {"type": "diffuse", "reflectance": [0.45, 0.28, 0.16]},
    "shelf_two": {"type": "diffuse", "reflectance": [0.36, 0.24, 0.14]},
    "prop_crate": {"type": "diffuse", "reflectance": [0.62, 0.24, 0.14]},
    "prop_book": {"type": "diffuse", "reflectance": [0.15, 0.35, 0.22]},
    "prop_cup": {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0},
}


def render_asset_inspections(catalog: Path, output: Path, run_id: str = "inspection_v1") -> list[dict]:
    catalog = Path(catalog)
    written = []
    for asset_id in (*PRODUCTION_ASSETS, *FIXTURE_ASSETS):
        destination = Path(output) / asset_id / run_id
        refuse_output(destination)
        if destination.exists():
            raise TaskError(f"reason=output_exists path={destination}")
        written.append(render_one_asset(catalog, asset_id, destination))
    return written


def render_one_asset(catalog: Path, asset_id: str, destination: Path) -> dict:
    catalog = Path(catalog)
    destination = Path(destination)
    refuse_output(destination)
    record = _catalog_record(catalog, asset_id, "1.0.0")
    manifest = record["manifest"]
    mesh_file = _mesh_file(catalog, asset_id, "1.0.0")
    vertices, polygons = read_obj(mesh_file)
    bounds = _bounds(vertices)
    bsdf, provenance = _bsdf(asset_id, manifest)
    destination.mkdir(parents=True)
    images = {}
    for view in ("front", "side", "elevated"):
        camera = _camera_for_bounds(bounds, view)
        scene = _scene([{"id": asset_id, "mesh": mesh_file, "matrix": np.eye(4), "bsdf": bsdf}], camera, bounds)
        exr = destination / f"{view}.exr"
        png = destination / f"{view}.png"
        render_stokes(scene, exr, spp=8, seed=SEED)
        write_rgb_png(exr, png)
        images[view] = {"png": str(png), "exr": str(exr), "png_sha256": _sha256(png), "exr_sha256": _sha256(exr)}
    collision = _collision_hash(catalog / asset_id / "1.0.0")
    summary = {
        "asset_id": asset_id,
        "profile": (manifest.get("geometry") or {}).get("profile"),
        "package_sha256": manifest["package_sha256"],
        "version": "1.0.0",
        "catalog_status": record["status"],
        "license_status": manifest["provenance"]["license_status"],
        "fixture": asset_id in FIXTURE_ASSETS,
        "bounds_min_m": bounds["minimum_m"],
        "bounds_max_m": bounds["maximum_m"],
        "size_m": bounds["size_m"],
        "vertex_count": int(len(vertices)),
        "polygon_count": int(len(polygons)),
        "triangle_count": int(len(triangulate(polygons))),
        "package_materials": manifest.get("materials"),
        "applied_bsdf": bsdf,
        "material_provenance": provenance,
        "components": (manifest.get("geometry") or {}).get("components"),
        "interior": (manifest.get("geometry") or {}).get("interior"),
        "collision_geometry": collision,
        "render": {"variant": VARIANT, "resolution": [256, 192], "spp": 8, "seed": SEED, "fov_deg": FOV_DEG},
        "images": images,
    }
    _write_json(destination / "summary.json", summary)
    logger.info("task_render event=asset asset_id=%s status=%s", asset_id, record["status"])
    return summary


def render_saved_bundle(
    bundle_dir: Path,
    catalog: Path,
    output: Path,
    *,
    resolution: tuple[int, int] = RESOLUTION,
    spp: int = SPP,
) -> dict:
    bundle_dir = Path(bundle_dir)
    catalog = Path(catalog)
    output = Path(output)
    refuse_output(output)
    if output.exists():
        raise TaskError(f"reason=output_exists path={output}")
    before = _file_hashes(bundle_dir)
    prepared = prepare_bundle(bundle_dir, catalog)
    output.mkdir(parents=True)
    camera = _camera_for_bounds(prepared["bounds"], "elevated")
    camera["reason"] = "frame the combined instance bounds with the shared front-elevated rule"
    scene = _scene(prepared["instances"], camera, prepared["bounds"], resolution)
    exr = output / "overview.exr"
    png = output / "overview.png"
    stokes = render_stokes(scene, exr, spp=int(spp), seed=SEED)
    write_rgb_png(exr, png)
    detail_camera = _camera_for_bounds(prepared["focus_bounds"], "elevated")
    detail_camera["reason"] = "closer elevated view of the smallest task object; object transforms unchanged"
    detail_scene = _scene(prepared["instances"], detail_camera, prepared["focus_bounds"], resolution)
    detail_exr = output / "detail.exr"
    detail_png = output / "detail.png"
    render_stokes(detail_scene, detail_exr, spp=int(spp), seed=SEED)
    write_rgb_png(detail_exr, detail_png)
    optical = load_shapes({"type": "scene", **{item["id"]: scene[item["id"]] for item in prepared["instances"]}})
    identity = _identity_loads(prepared["instances"])
    validation = {
        "rendered": True,
        "task_id": prepared["task_id"],
        "source_bundle": str(bundle_dir),
        "bundle_sha256": before["bundle.json"],
        "scene_sha256": before["scene.json"],
        "replay_byte_match": prepared["replay_byte_match"],
        "assets": prepared["assets"],
        "relations": prepared["relation_checks"],
        "camera": camera,
        "detail_camera": detail_camera,
        "lights": "one area light beside the camera, scaled to the scene bounds, radiance 40",
        "variant": VARIANT,
        "resolution": list(resolution),
        "spp": int(spp),
        "seed": SEED,
        "integrator": "stokes around path, max_depth 5",
        "optical_shapes": optical,
        "identity_loads": identity,
        "exr_names": stokes["exr_names"],
        "exr_shape": stokes["shape"],
        "exr_dtype": stokes["dtype"],
        "channel_contract": list(CHANNEL_NAMES),
        "overview_png": str(png),
        "overview_exr": str(exr),
        "detail_png": str(detail_png),
        "detail_exr": str(detail_exr),
        "source_unchanged": _file_hashes(bundle_dir) == before,
        "visual_review": "pending",
        "admitted": False,
    }
    if stokes["exr_names"] != list(CHANNEL_NAMES) or not validation["source_unchanged"]:
        validation["rendered"] = False
        validation["failure"] = "exr_contract_or_source_mutation"
    _write_json(output / "validation.json", validation)
    _write_json(output / "scene_instances.json", {"instances": prepared["assets"], "task": prepared["task"]})
    logger.info("task_render event=bundle task_id=%s rendered=%s", prepared["task_id"], validation["rendered"])
    return validation


def render_saved_tasks(catalog: Path, bundles: Path, output: Path) -> list[dict]:
    results = []
    for task_id, run_id in SAVED_RUNS.items():
        source = Path(bundles) / task_id / run_id
        destination = Path(output) / task_id / "render_17011_lit"
        try:
            results.append(render_saved_bundle(source, catalog, destination))
        except (TaskError, PackageError, OSError, RuntimeError) as exc:
            refuse_output(destination)
            destination.mkdir(parents=True, exist_ok=False)
            failure = {"rendered": False, "task_id": task_id, "failure": str(exc), "visual_review": "pending"}
            _write_json(destination / "validation.json", failure)
            results.append(failure)
            logger.error("task_render event=bundle_failed task_id=%s detail=%s", task_id, exc)
    return results


def prepare_bundle(bundle_dir: Path, catalog: Path) -> dict:
    bundle_dir = Path(bundle_dir)
    bundle_path = bundle_dir / "bundle.json"
    scene_path = bundle_dir / "scene.json"
    if not bundle_path.is_file() or not scene_path.is_file():
        raise TaskError("reason=missing_bundle")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    replayed = replay(bundle_dir, catalog)
    match = json.dumps(replayed["scene"], sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n" == scene_path.read_bytes()
    instances = []
    assets = []
    points = []
    for instance in scene["instances"]:
        record = _catalog_record(catalog, instance["asset_id"], instance["version"])
        require_eligible(record["manifest"], record["status"])
        if record["manifest"]["package_sha256"] != instance["package_sha256"]:
            raise TaskError(f"reason=package_hash asset={instance['asset_id']}")
        matrix = deserialize_matrix(instance["matrix"])
        inspect_linear(matrix)
        mesh_file = _mesh_file(catalog, instance["asset_id"], instance["version"])
        vertices, polygons = read_obj(mesh_file)
        bsdf, provenance = _bsdf(instance["asset_id"], record["manifest"])
        world = _apply(vertices, matrix)
        points.append(world)
        instances.append({"id": instance["instance_id"], "mesh": mesh_file, "matrix": matrix, "bsdf": bsdf})
        assets.append(
            {
                "instance_id": instance["instance_id"],
                "asset_id": instance["asset_id"],
                "version": instance["version"],
                "package_sha256": instance["package_sha256"],
                "matrix": instance["matrix"],
                "catalog_status": record["status"],
                "license_status": record["manifest"]["provenance"]["license_status"],
                "applied_bsdf": bsdf,
                "material_provenance": provenance,
                "package_materials": record["manifest"].get("materials"),
            }
        )
    stacked = np.vstack(points)
    bounds = _bounds(stacked)
    focus = _bounds(min(points, key=lambda item: np.prod(item.max(0) - item.min(0))))
    return {
        "task_id": scene["task_id"],
        "task": bundle.get("task"),
        "instances": instances,
        "assets": assets,
        "bounds": bounds,
        "focus_bounds": focus,
        "replay_byte_match": match,
        "relation_checks": _recheck_relations(scene, catalog),
    }


def require_eligible(manifest: dict, status: str) -> None:
    if status != "renderer_validated":
        raise TaskError(f"reason=renderer_status status={status}")
    if manifest["provenance"]["license_status"] != "permitted":
        raise TaskError("reason=license")


def refuse_output(path: Path) -> None:
    resolved = Path(path).resolve()
    for relative in (
        "dataset/qualification/indoor_scenes",
        "dataset/qualification/task_scenes",
        "dataset/dataset_v1",
    ):
        root = Path(relative).resolve()
        if resolved == root or root in resolved.parents:
            raise TaskError(f"reason=output_forbidden path={resolved}")


def write_gallery(review_path: Path, asset_root: Path, task_root: Path, scene_root: Path) -> None:
    review_path = Path(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "<!DOCTYPE html>",
        "<html><head><meta charset=\"utf-8\"><title>Task and asset review</title>",
        "<style>body{font-family:sans-serif;margin:2rem;background:#111;color:#eee} a{color:#9cf} img{max-width:360px;height:auto;background:#222} figure{display:inline-block;margin:1rem;vertical-align:top} .fail{border:1px solid #c66;padding:1rem}</style>",
        "</head><body>",
        "<h1>Visual review gallery</h1>",
        "<p><strong>Visual review, admission, and registration are pending human approval.</strong> A successful render is not human approval.</p>",
        "<h2>A. Qualified scenes A–C</h2>",
    ]
    for scene, names in {
        "scene_a": ("overview", "door", "window"),
        "scene_b": ("overview", "table_contact", "shelf_contact"),
        "scene_c": ("overview", "prop_crate_contact", "prop_cup_contact", "prop_book_contact", "table_contact", "shelf_contact"),
    }.items():
        body.append(f"<h3>{html.escape(scene)}</h3>")
        for name in names:
            href = _relative(review_path.parent, Path(scene_root) / scene / f"{name}.png")
            _exists(Path(scene_root) / scene / f"{name}.png")
            body.append(f"<figure><a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(scene + ' ' + name)}\"></a><figcaption>{html.escape(name)}</figcaption></figure>")
    body.append("<h2>B. Six saved task scenes</h2>")
    for task_id in SAVED_RUNS:
        card = Path(task_root) / task_id / "render_17011_lit" / "validation.json"
        if not card.is_file():
            body.append(f"<section class=\"fail\"><h3>{html.escape(task_id)}</h3><p>No validation file.</p></section>")
            continue
        data = json.loads(card.read_text(encoding="utf-8"))
        body.append("<section>")
        body.append(f"<h3>{html.escape(task_id)}</h3>")
        description = ""
        if isinstance(data.get("task"), dict):
            description = str(data["task"].get("description", ""))
        body.append(f"<p>{html.escape(description)}</p>")
        body.append(f"<p>Rendered: {html.escape(str(data.get('rendered')))}</p>")
        if data.get("rendered"):
            for label, key in (("Overview", "overview_png"), ("Detail", "detail_png")):
                href = _relative(review_path.parent, Path(data[key]))
                _exists(Path(data[key]))
                body.append(f"<figure><a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(task_id + ' ' + label)}\"></a><figcaption>{html.escape(label)}</figcaption></figure>")
            exr = _relative(review_path.parent, Path(data["overview_exr"]))
            report = _relative(review_path.parent, Path(task_root) / task_id / "render_17011_lit" / "validation.json")
            body.append(f"<p><a href=\"{html.escape(exr)}\">Stokes EXR</a> <a href=\"{html.escape(report)}\">validation.json</a></p>")
        else:
            body.append(f"<p class=\"fail\">Failed: {html.escape(str(data.get('failure')))}</p>")
        for asset in data.get("assets") or []:
            body.append("<p>" + html.escape(f"{asset['instance_id']} {asset['asset_id']} {asset['package_sha256']}") + "</p>")
        body.append("</section>")
    body.append("<h2>C. Asset inspection</h2>")
    body.append("<h3>Production assets</h3>")
    _asset_figures(body, review_path.parent, asset_root, PRODUCTION_ASSETS)
    body.append("<h3>Technical Fixtures — Not Production Assets</h3>")
    _asset_figures(body, review_path.parent, asset_root, FIXTURE_ASSETS)
    body.append("</body></html>\n")
    review_path.write_text("\n".join(body), encoding="utf-8")


def _asset_figures(body: list[str], gallery_dir: Path, asset_root: Path, names: tuple[str, ...]) -> None:
    for asset_id in names:
        summary_path = Path(asset_root) / asset_id / "inspection_v2" / "summary.json"
        if not summary_path.is_file():
            body.append(f"<p>Missing inspection for {html.escape(asset_id)}</p>")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        body.append(f"<h4>{html.escape(asset_id)} status={html.escape(str(summary['catalog_status']))}</h4>")
        body.append("<p>" + html.escape(summary["package_sha256"]) + "</p>")
        for view, image in summary["images"].items():
            href = _relative(gallery_dir, Path(image["png"]))
            _exists(Path(image["png"]))
            body.append(f"<figure><a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(asset_id + ' ' + view)}\"></a><figcaption>{html.escape(view)}</figcaption></figure>")


def _scene(instances: list[dict], camera: dict, bounds: dict, resolution: tuple[int, int] = RESOLUTION) -> dict:
    import mitsuba as mi

    require_variant()
    width, height = resolution
    center = bounds["center_m"]
    extent = max(bounds["size_m"])
    light_distance = max(extent, 0.2)
    scene = {
        "type": "scene",
        "integrator": {"type": "stokes", "integrator": {"type": "path", "max_depth": 5}},
        "sensor": {
            "type": "perspective",
            "fov": FOV_DEG,
            "to_world": mi.ScalarTransform4f.look_at(origin=camera["origin"], target=camera["target"], up=[0.0, 0.0, 1.0]),
            "sampler": {"type": "independent"},
            "film": {"type": "hdrfilm", "width": int(width), "height": int(height), "pixel_format": "rgb", "rfilter": {"type": "box"}},
        },
        "light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[float(camera["origin"][0]), float(camera["origin"][1]), float(camera["origin"][2]) + light_distance * 0.25],
                target=center,
                up=[0.0, 0.0, 1.0],
            )
            @ mi.ScalarTransform4f.scale([light_distance * 0.8, light_distance * 0.8, 1.0]),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 40.0}},
        },
    }
    for instance in instances:
        scene[instance["id"]] = {
            "type": "obj",
            "filename": str(instance["mesh"].resolve()),
            "to_world": mi.ScalarTransform4f(np.asarray(instance["matrix"], dtype=np.float64)),
            "bsdf": _mitsuba_bsdf(instance["bsdf"]),
        }
    return scene


def _camera_for_bounds(bounds: dict, view: str) -> dict:
    center = np.asarray(bounds["center_m"], dtype=np.float64)
    radius = max(float(np.max(bounds["size_m"])), 0.05)
    distance = radius / np.tan(np.deg2rad(FOV_DEG * 0.5)) * FRAME_MARGIN
    if view == "front":
        offset = np.array([0.0, -distance, radius * 0.15])
    elif view == "side":
        offset = np.array([distance, 0.0, radius * 0.15])
    else:
        offset = np.array([distance * 0.65, -distance * 0.65, distance * 0.55])
    origin = center + offset
    return {"origin": origin.tolist(), "target": center.tolist(), "fov_deg": FOV_DEG, "view": view}


def _identity_loads(instances: list[dict]) -> list[dict]:
    loads = []
    for instance in instances:
        alone = {
            "type": "scene",
            instance["id"]: {
                "type": "obj",
                "filename": str(instance["mesh"].resolve()),
                "to_world": __import__("mitsuba").ScalarTransform4f(np.asarray(instance["matrix"], dtype=np.float64)),
                "bsdf": _mitsuba_bsdf(instance["bsdf"]),
            },
        }
        loaded = load_shapes(alone)
        loads.append({"instance_id": instance["id"], "loaded_count": len(loaded), "loaded": loaded})
    return loads


def _recheck_relations(scene: dict, catalog: Path) -> list[dict]:
    checks = []
    instances = {item["instance_id"]: item for item in scene["instances"]}
    meshes = {}
    for instance_id, instance in instances.items():
        vertices, polygons = read_obj(_mesh_file(catalog, instance["asset_id"], instance["version"]))
        meshes[instance_id] = {"vertices": vertices, "polygons": polygons, "matrix": deserialize_matrix(instance["matrix"]), "manifest": verify_package(catalog / instance["asset_id"] / instance["version"])}
    for relation in scene.get("relations") or []:
        kind = relation.get("type")
        if kind == "supported_on":
            child, parent = relation["instances"]
            polygon = _world_surface(meshes[parent], relation["surface"])
            world = _apply(meshes[child]["vertices"], meshes[child]["matrix"])
            contact = measure_support(world, meshes[child]["polygons"], polygon)
            checks.append({"type": kind, "instances": relation["instances"], "passed": contact["passed"], "gap_m": contact["gap_m"], "penetration_m": contact["penetration_m"]})
        elif kind in {"near", "keep_clear_of", "adjacent_to"}:
            left, right = relation["instances"]
            result = unintended_intersection(
                {"vertices": _apply(meshes[left]["vertices"], meshes[left]["matrix"]), "faces": triangulate(meshes[left]["polygons"])},
                {"vertices": _apply(meshes[right]["vertices"], meshes[right]["matrix"]), "faces": triangulate(meshes[right]["polygons"])},
            )
            checks.append({"type": kind, "instances": relation["instances"], "clearance_m": result["clearance_m"], "unintended": result["unintended"]})
        else:
            checks.append({"type": kind, "instances": relation.get("instances"), "passed": relation.get("passed"), "note": "recorded from the saved bundle; no new validator was added"})
    return checks


def _world_surface(mesh: dict, surface_id: str) -> np.ndarray:
    for surface in mesh["manifest"].get("support_surfaces") or []:
        if surface["id"] == surface_id:
            return _apply(np.asarray(surface["polygon"], dtype=np.float64), mesh["matrix"])
    raise TaskError(f"reason=missing_surface surface={surface_id}")


def _bsdf(asset_id: str, manifest: dict) -> tuple[dict, str]:
    if asset_id in SCENE_AC_BSDF:
        return dict(SCENE_AC_BSDF[asset_id]), "scene_a_c_qualified_render_config"
    materials = manifest.get("materials") or []
    if materials and "reflectance" in materials[0]:
        return {"type": materials[0].get("type", "diffuse"), "reflectance": materials[0]["reflectance"]}, "package_materials"
    raise TaskError(f"reason=missing_material asset={asset_id}")


def _mitsuba_bsdf(bsdf: dict) -> dict:
    if bsdf.get("type") == "dielectric":
        return {"type": "dielectric", "int_ior": float(bsdf["int_ior"]), "ext_ior": float(bsdf["ext_ior"])}
    value = bsdf["reflectance"]
    return {"type": "diffuse", "reflectance": {"type": "rgb", "value": [float(channel) for channel in value]}}
    if asset_id in SCENE_AC_BSDF:
        return dict(SCENE_AC_BSDF[asset_id]), "scene_a_c_qualified_render_config"
    materials = manifest.get("materials") or []
    if materials and "reflectance" in materials[0]:
        return {"type": materials[0].get("type", "diffuse"), "reflectance": materials[0]["reflectance"]}, "package_materials"
    raise TaskError(f"reason=missing_material asset={asset_id}")


def _catalog_record(catalog: Path, asset_id: str, version: str) -> dict:
    index = read_index(Path(catalog) / "index.json")
    try:
        status = index["assets"][asset_id][version]["status"]
    except KeyError as exc:
        raise TaskError(f"reason=missing_asset asset={asset_id}") from exc
    manifest = verify_package(Path(catalog) / asset_id / version)
    return {"status": status, "manifest": manifest}


def _mesh_file(catalog: Path, asset_id: str, version: str) -> Path:
    if asset_id.startswith(".") or ".." in asset_id.split("/"):
        raise TaskError("reason=path_escape")
    root = (Path(catalog) / asset_id / version).resolve()
    candidate = (root / "geometry" / "render.obj").resolve()
    if root not in candidate.parents or not candidate.is_file() or candidate.is_symlink():
        raise TaskError(f"reason=missing_mesh asset={asset_id}")
    return candidate


def _collision_hash(package: Path) -> dict:
    path = package / "geometry" / "collision.obj"
    if not path.is_file() or path.is_symlink():
        return {"present": False, "sha256": None}
    return {"present": True, "sha256": _sha256(path)}


def _bounds(vertices: np.ndarray) -> dict:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    return {"minimum_m": low.tolist(), "maximum_m": high.tolist(), "size_m": (high - low).tolist(), "center_m": ((low + high) * 0.5).tolist()}


def _apply(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T[:, :3]


def _file_hashes(directory: Path) -> dict[str, str]:
    return {name: _sha256(directory / name) for name in ("bundle.json", "scene.json")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _relative(start: Path, target: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), Path(start).resolve())).as_posix()


def _exists(path: Path) -> None:
    if not Path(path).is_file():
        raise TaskError(f"reason=missing_gallery_file path={path}")

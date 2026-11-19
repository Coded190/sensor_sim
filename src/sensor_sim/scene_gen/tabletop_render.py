"""Stokes renders for tabletop assets and room-context scenes.

BSDF conversion lives here. ``task_render._mitsuba_bsdf`` is not used, because
it only emits diffuse and dielectric. An unknown material type is an error.
The room light is one ceiling rectangle. Emitters on this path are unpolarized.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import set_candidate_status, verify_package
from sensor_sim.scene_gen.indoor_mitsuba import render_stokes, require_variant, write_rgb_png
from sensor_sim.scene_gen.indoor_transform import deserialize_matrix
from sensor_sim.scene_gen.mesh_io import read_obj
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES
from sensor_sim.scene_gen.tabletop_assets import ASSET_IDS, FURNITURE_IDS, VERSION, build_asset
from sensor_sim.scene_gen.task_render import SCENE_AC_BSDF
from sensor_sim.scene_gen.task_schema import TaskError

logger = logging.getLogger("sensor_sim.scene_gen.tabletop_render")

VARIANT = "cuda_ad_spectral_polarized"
RENDER_SEED = 17111
INSPECTION_RESOLUTION = (256, 192)
INSPECTION_SPP = 8
ROOM_RESOLUTION = (512, 384)
TABLETOP_RESOLUTION = (1024, 768)
SCENE_SPP = 16
FOV_INSPECTION = 45.0
# Ceiling rectangle in the existing room. Scale is applied to Mitsuba's [-1, 1] rectangle,
# so the luminaire is 0.80 m by 0.50 m. Radiance is frozen after the room probe.
LIGHT_CENTER = [2.12, 1.72, 2.40]
LIGHT_SCALE = [0.40, 0.25, 1.0]
# Probe on the controlled room at spp 4: radiance 8 had RGB p99 0.16 (too dark),
# radiance 40 had mean 0.22 and p99 0.79, radiance 120 clipped (p99 2.38).
# 40 is frozen for every pilot and comparison render.
LIGHT_RADIANCE = 40.0
INSPECTION_RADIANCE = 4.0
TIE_M = 1e-6
LOW_SUPPORT_FRAME_FRACTION = 0.01
ASSET_ROOT = Path("dataset/qualification/tabletop_assets")
SCENE_ROOT = Path("dataset/qualification/tabletop_scenes")


def mitsuba_bsdf(material: dict) -> dict:
    """Convert one package or qualified material dict into a Mitsuba BSDF."""
    kind = material.get("type")
    if kind == "diffuse":
        return {"type": "diffuse", "reflectance": {"type": "rgb", "value": [float(channel) for channel in material["reflectance"]]}}
    if kind == "dielectric":
        return {"type": "dielectric", "int_ior": float(material["int_ior"]), "ext_ior": float(material["ext_ior"])}
    if kind == "pplastic":
        mapped = {
            "type": "pplastic",
            "int_ior": float(material["int_ior"]),
            "ext_ior": float(material["ext_ior"]),
            "diffuse_reflectance": {"type": "rgb", "value": [float(channel) for channel in material["diffuse_reflectance"]]},
            "alpha": float(material["alpha"]),
        }
        if "distribution" in material:
            mapped["distribution"] = material["distribution"]
        return mapped
    if kind == "conductor":
        return {"type": "conductor", "material": str(material["material"])}
    if kind == "roughconductor":
        return {
            "type": "roughconductor",
            "material": str(material["material"]),
            "distribution": str(material["distribution"]),
            "alpha": float(material["alpha"]),
        }
    raise TaskError(f"reason=unsupported_bsdf type={kind}")


def render_asset_inspections(catalog: Path, output: Path | None = None) -> list[dict]:
    catalog = Path(catalog)
    output = Path(output or ASSET_ROOT)
    written = []
    for asset_id in ASSET_IDS:
        written.append(render_asset(catalog, asset_id, output / asset_id / "inspection_v1"))
    return written


def render_asset(catalog: Path, asset_id: str, destination: Path) -> dict:
    destination = Path(destination)
    if destination.exists():
        raise TaskError(f"reason=output_exists path={destination}")
    manifest = verify_package(Path(catalog) / asset_id / VERSION)
    vertices, _polygons = read_obj(Path(catalog) / asset_id / VERSION / "geometry" / "render.obj")
    profile = manifest["geometry"]["profile"]
    mesh_report = build_asset(asset_id)["report"]
    bounds = _bounds(vertices)
    material = manifest["materials"][0]
    images = {}
    for view in ("front", "side", "elevated"):
        camera = _frame_camera(bounds, view, FOV_INSPECTION, 1.25)
        scene = _scene(
            [{"id": asset_id, "mesh": Path(catalog) / asset_id / VERSION / "geometry" / "render.obj", "matrix": np.eye(4), "bsdf": material}],
            camera,
            INSPECTION_RESOLUTION,
            _inspection_light(bounds, camera),
        )
        exr = destination / f"{view}.exr"
        png = destination / f"{view}.png"
        stokes = render_stokes(scene, exr, INSPECTION_SPP, RENDER_SEED)
        write_rgb_png(exr, png)
        _assert_png(png)
        images[view] = {"png": str(png), "exr": str(exr), "png_sha256": _sha(png), "exr_sha256": _sha(exr), "stokes": stokes}
        logger.info("tabletop_render event=asset_view asset=%s view=%s exr=%s", asset_id, view, exr)
    set_candidate_status(catalog, asset_id, VERSION, "renderer_validated")
    summary = {
        "asset_id": asset_id,
        "version": VERSION,
        "profile": profile,
        "provenance": manifest["provenance"],
        "package_sha256": manifest["package_sha256"],
        "geometry_sha256": manifest["geometry"]["geometry_sha256"],
        "shared_geometry_sha256": manifest["geometry"]["geometry_sha256"] if asset_id in {"prop_bowl_glossy", "prop_can_rough", "prop_bowl", "prop_can"} else None,
        "bounds_min_m": manifest["geometry"]["bounds_min"],
        "bounds_max_m": manifest["geometry"]["bounds_max"],
        "dimensions_m": [float(high - low) for low, high in zip(manifest["geometry"]["bounds_min"], manifest["geometry"]["bounds_max"], strict=True)],
        "vertex_count": int(len(vertices)),
        "polygon_count": manifest["geometry"]["polygon_count"],
        "triangle_count": manifest["geometry"]["triangle_count"],
        "components": manifest["geometry"].get("components", []),
        "category": manifest["geometry"].get("category"),
        "material_category": material.get("research_role"),
        "bsdf": material,
        "material_provenance": "package_materials",
        "render": {
            "variant": VARIANT,
            "resolution": list(INSPECTION_RESOLUTION),
            "spp": INSPECTION_SPP,
            "seed": RENDER_SEED,
            "fov_deg": FOV_INSPECTION,
            "light": "asset-relative area rectangle",
            "radiance": INSPECTION_RADIANCE,
        },
        "images": images,
        "package_verification": "ok",
        "mesh_validation": {"passed": mesh_report["passed"], "issues": mesh_report["issues"]},
        "renderer_validation": "renderer_validated",
        "catalog_status": "renderer_validated",
        "role": "room_furniture" if asset_id in FURNITURE_IDS else "tabletop",
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("tabletop_render event=asset_renderer_validated asset=%s package_sha256=%s", asset_id, manifest["package_sha256"])
    return summary


def render_saved_scene(bundle_dir: Path, catalog: Path, output: Path, *, radiance: float = LIGHT_RADIANCE) -> dict:
    bundle_dir = Path(bundle_dir)
    output = Path(output)
    if output.exists():
        raise TaskError(f"reason=output_exists path={output}")
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    instances = _instances(catalog, bundle)
    cameras = bundle["resolved"]["collision_log"]["cameras"]
    output.mkdir(parents=True, exist_ok=True)
    images = {}
    visibility = {}
    for name, resolution in (("room", ROOM_RESOLUTION), ("tabletop", TABLETOP_RESOLUTION)):
        camera = cameras[name]
        scene = _scene(instances, camera, resolution, _ceiling_light(radiance))
        exr = output / f"{name}.exr"
        png = output / f"{name}.png"
        stokes = render_stokes(scene, exr, SCENE_SPP, RENDER_SEED)
        write_rgb_png(exr, png)
        _assert_png(png)
        images[name] = {"png": str(png), "exr": str(exr), "png_sha256": _sha(png), "exr_sha256": _sha(exr), "stokes": _public_stokes(stokes)}
        visibility[name] = instance_visibility(instances, camera, resolution)
        logger.info("tabletop_render event=scene_view task=%s camera=%s shape=%s", bundle["task"]["task_id"], name, stokes["shape"])
    validation = {
        "task_id": bundle["task"]["task_id"],
        "seed": bundle["task"]["seed"],
        "rendered": True,
        "variant": VARIANT,
        "spp": SCENE_SPP,
        "render_seed": RENDER_SEED,
        "radiance": radiance,
        "light": _light_record(radiance),
        "cameras": cameras,
        "images": images,
        "visibility": visibility,
        "collision_log": bundle["resolved"]["collision_log"],
        "assets": [
            {
                "instance_id": item["id"],
                "asset_id": item["asset_id"],
                "package_sha256": item["package_sha256"],
                "bsdf": item["bsdf"],
                "material_provenance": item["material_provenance"],
            }
            for item in instances
        ],
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def instance_visibility(
    instances: list[dict],
    camera: dict,
    resolution: tuple[int, int],
    *,
    support_polygon=None,
    return_image: bool = False,
) -> dict:
    """Nearest-hit ids on the real OBJ meshes. Background is 0. A tie keeps the smaller id."""
    import mitsuba as mi

    require_variant()
    width, height = resolution
    origins, directions = _rays(camera["origin"], camera["target"], camera.get("up", [0.0, 0.0, 1.0]), camera["fov_deg"], width, height)
    best_t = np.full(origins.shape[0], np.inf, dtype=np.float64)
    best_id = np.zeros(origins.shape[0], dtype=np.int32)
    alone = []
    for index, instance in enumerate(instances, start=1):
        scene = mi.load_dict(
            {
                "type": "scene",
                "mesh": {
                    "type": "obj",
                    "filename": str(instance["mesh"]),
                    "to_world": mi.ScalarTransform4f(np.asarray(instance["matrix"], dtype=np.float64)),
                    "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]}},
                },
            }
        )
        hit = scene.ray_intersect(_ray(origins, directions))
        valid = np.array(hit.is_valid(), dtype=bool).reshape(-1)
        distance = np.array(hit.t, dtype=np.float64).reshape(-1)
        alone.append(valid.copy())
        closer = valid & np.isfinite(distance) & (distance < best_t - TIE_M)
        comparable = valid & np.isfinite(distance) & np.isfinite(best_t)
        tie = np.zeros(best_id.shape, dtype=bool)
        if np.any(comparable):
            tie[comparable] = np.abs(distance[comparable] - best_t[comparable]) <= TIE_M
        tie &= (index < best_id) & (best_id != 0)
        best_t[closer] = distance[closer]
        best_id[closer | tie] = index
    frame = width * height
    rows = []
    for index, instance in enumerate(instances, start=1):
        visible_mask = best_id.reshape(height, width) == index
        visible = int(np.count_nonzero(visible_mask))
        projected = int(np.count_nonzero(alone[index - 1]))
        fraction = visible / frame
        ratio = visible / projected if projected else 0.0
        if visible == 0 and projected > 0:
            occlusion = "full"
        elif projected > 0 and visible < projected:
            occlusion = "partial"
        elif visible == 0:
            occlusion = "not_projected"
        else:
            occlusion = "none"
        ys, xs = np.nonzero(visible_mask)
        if len(xs):
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            pixel_width = int(xs.max() - xs.min() + 1)
            pixel_height = int(ys.max() - ys.min() + 1)
        else:
            box = None
            pixel_width = 0
            pixel_height = 0
        shorter = int(min(pixel_width, pixel_height))
        low_support = fraction < LOW_SUPPORT_FRAME_FRACTION or (projected > 0 and ratio < 0.5 and occlusion == "partial")
        rows.append(
            {
                "instance_id": instance["id"],
                "asset_id": instance["asset_id"],
                "label": index,
                "visible_pixels": visible,
                "frame_fraction": fraction,
                "unoccluded_projected_pixels": projected,
                "visible_to_projected_ratio": ratio,
                "occlusion": occlusion,
                "pixel_bbox_xyxy": box,
                "pixel_width": pixel_width,
                "pixel_height": pixel_height,
                "shorter_pixel_side": shorter,
                "low_spatial_support": shorter < 32,
                "low_spatial_support_note": "32 px is an observation flag for a later experiment, not a polarization cutoff",
                "low_support": bool(low_support and instance.get("role") == "required"),
                "polarization_ready": False,
            }
        )
    image = best_id.reshape(height, width)
    roi_mask = None
    if support_polygon is not None:
        roi_mask = _raster_support(camera, resolution, np.asarray(support_polygon, dtype=np.float64))
        roi_count = int(np.count_nonzero(roi_mask))
        for index, row in enumerate(rows, start=1):
            inside = int(np.count_nonzero((image == index) & roi_mask))
            row["roi_pixels"] = roi_count
            row["roi_visible_pixels"] = inside
            row["roi_fraction"] = inside / roi_count if roi_count else 0.0
    logger.info(
        "tabletop_render event=visibility camera=%s width=%d height=%d instances=%d low_support=%d low_spatial=%d",
        camera.get("name", "camera"),
        width,
        height,
        len(instances),
        sum(row["low_support"] for row in rows),
        sum(row["low_spatial_support"] for row in rows),
    )
    result = {
        "width": width,
        "height": height,
        "tie_m": TIE_M,
        "background_id": 0,
        "low_support_rule": "frame fraction below 0.01, or visible count below half of the unoccluded projection when partially occluded",
        "low_spatial_support_px": 32,
        "instances": rows,
    }
    if roi_mask is not None:
        result["roi_pixels"] = int(np.count_nonzero(roi_mask))
        result["roi_source"] = "projected_table_support_polygon"
    if return_image:
        result["id_image"] = image
        if roi_mask is not None:
            result["roi_mask"] = roi_mask
    return result


def probe_radiance(bundle_dir: Path, catalog: Path, candidates: tuple[float, ...] = (2.0, 4.0, 8.0, 15.0, 30.0)) -> dict:
    """Pick one ceiling radiance from a low-resolution room view. The caller freezes it."""
    bundle = json.loads((Path(bundle_dir) / "bundle.json").read_text(encoding="utf-8"))
    instances = _instances(catalog, bundle)
    camera = dict(bundle["resolved"]["collision_log"]["cameras"]["room"])
    camera["resolution"] = [128, 96]
    chosen = candidates[-1]
    report = []
    for radiance in candidates:
        scene = _scene(instances, camera, (128, 96), _ceiling_light(radiance))
        import mitsuba as mi

        require_variant()
        image = np.array(mi.render(mi.load_dict(scene), spp=4, seed=RENDER_SEED))
        rgb = image[:, :, :3]
        mean = float(rgb.mean())
        peak = float(np.quantile(rgb, 0.99))
        report.append({"radiance": radiance, "rgb_mean": mean, "rgb_p99": peak})
        logger.info("tabletop_render event=light_probe radiance=%.3f mean=%.4f p99=%.4f", radiance, mean, peak)
        if 0.04 <= mean <= 0.45 and peak <= 0.95:
            chosen = radiance
            break
    return {"chosen_radiance": chosen, "samples": report, "note": "unpolarized area emitter; not calibrated radiometry"}


def write_gallery(review_path: Path, asset_root: Path, scene_root: Path) -> None:
    review_path = Path(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "<!DOCTYPE html>",
        "<html><head><meta charset=\"utf-8\"><title>Tabletop room review</title>",
        "<style>body{font-family:sans-serif;margin:2rem;background:#111;color:#eee} a{color:#9cf} img{max-width:420px;height:auto;background:#222} figure{display:inline-block;margin:1rem;vertical-align:top} .notice{border:1px solid #c90;padding:1rem}</style>",
        "</head><body>",
        "<h1>Tabletop room qualification</h1>",
        "<p class=\"notice\"><strong>These assets and scenes are qualification candidates. Human visual review, admission, and registration are pending.</strong></p>",
        "<h2>A. Individual assets</h2>",
    ]
    for asset_id in ASSET_IDS:
        summary_path = Path(asset_root) / asset_id / "inspection_v1" / "summary.json"
        if not summary_path.is_file():
            body.append(f"<p>Missing {html.escape(asset_id)}</p>")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        body.append(f"<h3>{html.escape(asset_id)} ({html.escape(str(summary['role']))})</h3>")
        body.append("<p>" + html.escape(
            f"category {summary.get('category')}  {summary.get('material_category')}  {_bsdf_type(summary['bsdf'])}  "
            f"dimensions {summary['dimensions_m']}  package {summary['package_sha256']}  geometry {summary['geometry_sha256']}  status {summary['catalog_status']}"
        ) + "</p>")
        for view, image in summary["images"].items():
            href = _relative(review_path.parent, Path(image["png"]))
            body.append(f"<figure><a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(asset_id + ' ' + view)}\"></a><figcaption>{html.escape(view)}</figcaption></figure>")
    pilots = []
    comparisons = []
    for validation_path in sorted(Path(scene_root).glob("**/render_17111/validation.json")):
        data = json.loads(validation_path.read_text(encoding="utf-8"))
        (comparisons if data["task_id"].startswith("comparison_") else pilots).append((validation_path, data))
    body.append("<h2>B. Room-context scenes</h2>")
    _scene_section(body, review_path.parent, pilots, camera_name="room")
    body.append("<h2>C. Tabletop polarization views</h2>")
    _scene_section(body, review_path.parent, pilots, camera_name="tabletop")
    body.append("<h2>D. Controlled comparisons</h2>")
    body.append("<p>Each pair keeps the room, furniture, camera, ceiling light, and object support point fixed. The gallery states the one variable that changes.</p>")
    _comparison_section(body, review_path.parent, comparisons)
    body.append("</body></html>\n")
    review_path.write_text("\n".join(body), encoding="utf-8")
    logger.info("tabletop_render event=gallery path=%s", review_path)


def _scene_section(body, gallery_dir, rows, camera_name: str) -> None:
    for _path, data in rows:
        log = data["collision_log"]
        occupancy = log.get("occupancy", {})
        body.append("<section>")
        body.append(f"<h3>{html.escape(data['task_id'])} seed {html.escape(str(data['seed']))}</h3>")
        counts = log.get("geometry_counts", {})
        body.append("<p>" + html.escape(
            f"arrangement {log.get('room_arrangement')}  tabletop instances {counts.get('tabletop_instances')}  "
            f"geometries {counts.get('unique_geometry_ids')}  occupied fraction {occupancy.get('occupied_fraction')}  "
            f"nearest clearance m {occupancy.get('nearest_clearance_m')}"
        ) + "</p>")
        image = data["images"][camera_name]
        href = _relative(gallery_dir, Path(image["png"]))
        exr = _relative(gallery_dir, Path(image["exr"]))
        body.append(f"<figure><a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(data['task_id'] + ' ' + camera_name)}\"></a><figcaption>{html.escape(camera_name)}</figcaption></figure>")
        body.append(f"<p><a href=\"{html.escape(exr)}\">Stokes EXR {html.escape(camera_name)}</a></p>")
        if camera_name == "tabletop":
            for row in data["visibility"]["tabletop"]["instances"]:
                if row["instance_id"] in {"table_1"} or row["asset_id"].startswith("room_") or row["asset_id"] in {"shelf_two", "prop_chair", "table_rect"}:
                    continue
                body.append("<p>" + html.escape(
                    f"{row['instance_id']} visible {row['visible_pixels']} projected {row['unoccluded_projected_pixels']} "
                    f"occlusion {row['occlusion']} low_support {row['low_support']}"
                ) + "</p>")
        body.append("</section>")


def _comparison_section(body, gallery_dir, rows) -> None:
    pairs = [
        ("comparison_bowl_diffuse", "comparison_bowl_glossy", "BSDF: diffuse versus pplastic. Geometry hash, transform, room, camera, and light stay fixed."),
        ("comparison_can_polished", "comparison_can_rough", "BSDF: conductor Al versus roughconductor Al, Beckmann alpha 0.1. Geometry hash and transform stay fixed."),
        ("comparison_geometry_plate", "comparison_geometry_apple", "Geometry: plate versus apple. Both use the same diffuse reflectance. The support point, room, camera, and light stay fixed."),
    ]
    by_id = {data["task_id"]: data for _path, data in rows}
    for left, right, note in pairs:
        body.append(f"<h3>{html.escape(left)} / {html.escape(right)}</h3>")
        body.append(f"<p>{html.escape(note)}</p>")
        for task_id in (left, right):
            data = by_id.get(task_id)
            if data is None:
                body.append(f"<p>Missing {html.escape(task_id)}</p>")
                continue
            image = data["images"]["tabletop"]
            href = _relative(gallery_dir, Path(image["png"]))
            body.append(f"<figure><a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(task_id)}\"></a><figcaption>{html.escape(task_id)}</figcaption></figure>")


def _instances(catalog: Path, bundle: dict) -> list[dict]:
    instances = []
    for asset in bundle["resolved"]["assets"]:
        manifest = verify_package(Path(catalog) / asset["asset_id"] / asset["version"])
        if asset["asset_id"] in SCENE_AC_BSDF:
            material = dict(SCENE_AC_BSDF[asset["asset_id"]])
            provenance = "scene_a_c_qualified_render_config"
        else:
            material = dict(manifest["materials"][0])
            provenance = "package_materials"
        if bundle["task"]["task_id"].startswith("comparison_geometry") and material.get("type") == "diffuse":
            material["reflectance"] = [0.82, 0.82, 0.78]
            material["research_role"] = "depolarizing_reference"
            provenance = "comparison_shared_diffuse_reflectance"
        matrix = next(item["matrix"] for item in bundle["resolved"]["poses"] if item["instance_id"] == asset["instance_id"])
        instances.append(
            {
                "id": asset["instance_id"],
                "asset_id": asset["asset_id"],
                "role": asset["role"],
                "package_sha256": asset["package_sha256"],
                "mesh": Path(catalog) / asset["asset_id"] / asset["version"] / "geometry" / "render.obj",
                "matrix": deserialize_matrix(matrix),
                "bsdf": material,
                "material_provenance": provenance,
            }
        )
    return instances


def _scene(instances: list[dict], camera: dict, resolution: tuple[int, int], light: dict) -> dict:
    import mitsuba as mi

    require_variant()
    width, height = resolution
    scene = {
        "type": "scene",
        "integrator": {"type": "stokes", "integrator": {"type": "path", "max_depth": 5}},
        "sensor": {
            "type": "perspective",
            "fov": float(camera["fov_deg"]),
            "to_world": mi.ScalarTransform4f.look_at(origin=list(camera["origin"]), target=list(camera["target"]), up=list(camera.get("up", [0.0, 0.0, 1.0]))),
            "sampler": {"type": "independent"},
            "film": {"type": "hdrfilm", "width": int(width), "height": int(height), "pixel_format": "rgb", "rfilter": {"type": "box"}},
        },
        "light": light,
    }
    for instance in instances:
        scene[instance["id"]] = {
            "type": "obj",
            "filename": str(Path(instance["mesh"]).resolve()),
            "to_world": mi.ScalarTransform4f(np.asarray(instance["matrix"], dtype=np.float64)),
            "bsdf": mitsuba_bsdf(instance["bsdf"]),
        }
    return scene


def _ceiling_light(radiance: float) -> dict:
    import mitsuba as mi

    require_variant()
    transform = (
        mi.ScalarTransform4f.translate(LIGHT_CENTER)
        @ mi.ScalarTransform4f.rotate([1.0, 0.0, 0.0], 180.0)
        @ mi.ScalarTransform4f.scale(LIGHT_SCALE)
    )
    return {"type": "rectangle", "to_world": transform, "emitter": {"type": "area", "radiance": {"type": "rgb", "value": float(radiance)}}}


def _inspection_light(bounds: dict, camera: dict) -> dict:
    """One area light above the asset, offset from the camera so it does not cover the view."""
    import mitsuba as mi

    require_variant()
    center = np.asarray(bounds["center_m"], dtype=np.float64)
    distance = max(float(np.max(bounds["size_m"])), 0.05)
    origin = (center + np.array([0.35 * distance, -0.55 * distance, 1.15 * distance])).tolist()
    transform = mi.ScalarTransform4f.look_at(origin=origin, target=center.tolist(), up=[0.0, 0.0, 1.0]) @ mi.ScalarTransform4f.scale([0.45 * distance, 0.45 * distance, 1.0])
    return {"type": "rectangle", "to_world": transform, "emitter": {"type": "area", "radiance": {"type": "rgb", "value": INSPECTION_RADIANCE}}}


def _light_record(radiance: float) -> dict:
    return {
        "type": "area",
        "shape": "rectangle",
        "center_m": LIGHT_CENTER,
        "scale_xy_m": [LIGHT_SCALE[0] * 2.0, LIGHT_SCALE[1] * 2.0],
        "size_m": [0.80, 0.50],
        "orientation": "rotate_x_180_normal_down",
        "radiance_rgb": float(radiance),
        "polarized_emission": False,
        "daylight": "omitted; north wall opening is geometric only",
    }


def _frame_camera(bounds: dict, view: str, fov_deg: float, margin: float) -> dict:
    center = np.asarray(bounds["center_m"], dtype=np.float64)
    radius = max(float(np.max(bounds["size_m"])), 0.02)
    distance = radius / np.tan(np.deg2rad(fov_deg * 0.5)) * margin
    if view == "front":
        offset = np.array([0.0, -distance, radius * 0.15])
    elif view == "side":
        offset = np.array([distance, 0.0, radius * 0.15])
    else:
        offset = np.array([distance * 0.65, -distance * 0.65, distance * 0.55])
    origin = center + offset
    return {"name": view, "origin": origin.tolist(), "target": center.tolist(), "up": [0.0, 0.0, 1.0], "fov_deg": fov_deg}


def _bounds(vertices: np.ndarray) -> dict:
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    return {"minimum_m": low.tolist(), "maximum_m": high.tolist(), "size_m": (high - low).tolist(), "center_m": ((low + high) * 0.5).tolist()}


def _raster_support(camera: dict, resolution: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    """Pixels whose centers lie inside the projected support polygon."""
    width, height = resolution
    projected = []
    for point in polygon:
        pixel = _project_point(camera, width, height, point)
        if pixel is None:
            continue
        projected.append(pixel)
    mask = np.zeros((height, width), dtype=bool)
    if len(projected) < 3:
        return mask
    corners = np.asarray(projected, dtype=np.float64)
    columns = np.arange(width) + 0.5
    rows = np.arange(height) + 0.5
    xx, yy = np.meshgrid(columns, rows)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = _point_in_polygon(points, corners)
    return inside.reshape(height, width)


def _project_point(camera: dict, width: int, height: int, point: np.ndarray):
    origin = np.asarray(camera["origin"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    up = np.asarray(camera.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)
    forward = target - origin
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)
    vector = np.asarray(point, dtype=np.float64) - origin
    depth = float(np.dot(vector, forward))
    if depth <= 1e-6:
        return None
    scale = np.tan(np.deg2rad(float(camera["fov_deg"])) * 0.5)
    aspect = float(width) / float(height)
    camera_x = float(np.dot(vector, right)) / depth
    camera_y = float(np.dot(vector, true_up)) / depth
    # Mitsuba's perspective ``fov`` is the horizontal field (x_fov).
    column = ((camera_x / scale + 1.0) * width / 2.0) - 0.5
    row = ((1.0 - camera_y * aspect / scale) * height / 2.0) - 0.5
    return [column, row]


def _point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    x_values, y_values = points[:, 0], points[:, 1]
    count = len(polygon)
    for index in range(count):
        x0, y0 = polygon[index]
        x1, y1 = polygon[(index + 1) % count]
        crosses = ((y0 > y_values) != (y1 > y_values)) & (
            x_values < (x1 - x0) * (y_values - y0) / (y1 - y0 + 1e-15) + x0
        )
        inside ^= crosses
    return inside


def _rays(origin, target, up, fov_deg, width, height):
    origin = np.asarray(origin, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - origin
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)
    scale = np.tan(np.deg2rad(float(fov_deg)) * 0.5)
    aspect = float(width) / float(height)
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    # Horizontal field of view, matching Mitsuba perspective x_fov.
    camera_x = (2.0 * (columns + 0.5) / width - 1.0) * scale
    camera_y = (1.0 - 2.0 * (rows + 0.5) / height) * (scale / aspect)
    directions = forward + right * camera_x[..., None] + true_up * camera_y[..., None]
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(origin, directions.shape).copy()
    return origins.reshape(-1, 3), directions.reshape(-1, 3)


def _ray(origins, directions):
    import mitsuba as mi

    return mi.Ray3f(o=mi.Point3f(np.ascontiguousarray(origins.T, dtype=np.float32)), d=mi.Vector3f(np.ascontiguousarray(directions.T, dtype=np.float32)))


def _public_stokes(report: dict) -> dict:
    if list(report["channel_names"]) != list(CHANNEL_NAMES) or list(report["exr_names"]) != list(CHANNEL_NAMES):
        raise TaskError(f"reason=channel_contract names={report.get('channel_names')}")
    return {key: report[key] for key in ("variant", "channel_names", "shape", "dtype", "finite", "exr_names")}


def _assert_png(path: Path) -> None:
    payload = Path(path).read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise TaskError(f"reason=png path={path}")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(origin: Path, target: Path) -> str:
    return Path(os.path.relpath(target, origin)).as_posix()


def _bsdf_type(material: dict) -> str:
    return str(material.get("type"))

"""Render the qualified LivingRoom seed 21 package.

Writes linear Stokes EXR and sRGB PNG separately. Environment radiance 1.2 is
a labeled replacement for the unconverted world shader. The frozen
scene_manifest.json is not modified.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.indoor_mitsuba import render_stokes, require_variant, write_rgb_png
from sensor_sim.scene_gen.material_spec import to_mitsuba
from sensor_sim.scene_gen.mitsuba_scene_package import build_scene, load_manifest

PACKAGE = Path("dataset/qualification/asset_realism/exported/LivingRoom/21/mitsuba")
OUT = Path("dataset/qualification/asset_realism/gallery/LivingRoom/21/qualified")
REPORTS = Path("dataset/qualification/asset_realism/reports")


def _camera_from_matrix(template: dict, name: str, matrix: list) -> dict:
    camera = dict(template)
    origin = [float(matrix[row][3]) for row in range(3)]
    forward = np.array([-matrix[row][2] for row in range(3)], dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    camera["id"] = name
    camera["origin"] = origin
    camera["target"] = (np.asarray(origin) + forward).tolist()
    camera["matrix_world"] = matrix
    return camera


def _subset(manifest: dict, predicate) -> dict:
    copied = json.loads(json.dumps(manifest))
    copied["instances"] = [item for item in copied["instances"] if predicate(item)]
    return copied


def _render(manifest: dict, stem: Path, spp: int, width: int, height: int) -> dict:
    manifest = json.loads(json.dumps(manifest))
    manifest["camera"]["width"] = width
    manifest["camera"]["height"] = height
    started = time.perf_counter()
    scene = build_scene(manifest, PACKAGE)
    print(f"event=scene_built stem={stem.name} instances={len(manifest['instances'])} seconds={time.perf_counter()-started:.1f}", flush=True)
    exr = stem.with_suffix(".exr")
    png = stem.with_suffix(".png")
    report = render_stokes(scene, exr, spp, 7)
    write_rgb_png(exr, png)
    image = np.array(__import__("mitsuba").Bitmap(str(exr)))
    payload = {
        "exr": str(exr),
        "png": str(png),
        "spp": spp,
        "width": width,
        "height": height,
        "instances": len(manifest["instances"]),
        "seconds": time.perf_counter() - started,
        "exr_rgb_mean": [float(value) for value in image[:, :, :3].mean(axis=(0, 1))],
        "exr_dtype": "float32",
        "png_encoding": "uint8 sRGB gamma via write_rgb_png",
        "exposure": 1.0,
        "view_transform": "sRGB gamma on RGB only; the EXR is scene-referred and is not tone-mapped",
        "environment_radiance": manifest["environment"]["radiance"],
        "environment_note": "Labeled replacement. This is not the Blender world shader.",
        "stokes_channels": report.get("channels") or report.get("names"),
        "variant": "cuda_ad_spectral_polarized",
    }
    print(f"event=rendered stem={stem.name} seconds={payload['seconds']:.1f} mean={payload['exr_rgb_mean']}", flush=True)
    return payload


def _swatches(manifest: dict) -> list[dict]:
    wanted = {
        "wood": "shader_wood",
        "plaster": "shader_plaster",
        "fabric": "shader_rug",
        "metal": "shader_metal",
        "glass": "shader_glass",
        "plastic": "shader_black",
    }
    ledger = json.loads((REPORTS / "livingroom_21_material_ledger.json").read_text(encoding="utf-8"))
    chosen = {}
    for row in ledger["rows"]:
        slot = str(row.get("source_slot") or "")
        for label, prefix in wanted.items():
            if label not in chosen and slot.startswith(prefix):
                chosen[label] = row
    import mitsuba as mi

    plane = OUT / "swatch_plane.obj"
    plane.write_text("v -0.5 -0.5 0\nv -0.5 0.5 0\nv 0.5 0.5 0\nv 0.5 -0.5 0\nvt 0 0\nvt 0 1\nvt 1 1\nvt 1 0\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n", encoding="utf-8")
    results = []
    by_region = {}
    for instance in manifest["instances"]:
        for region in instance["regions"]:
            by_region[(instance["id"], region["slot"])] = region["spec"]
    for label, row in chosen.items():
        spec = by_region[(row["instance"], row["package_slot"])]
        bsdf = to_mitsuba(spec, package_root=PACKAGE)
        scene = {
            "type": "scene",
            "integrator": {"type": "stokes", "integrator": {"type": "path", "max_depth": 3}},
            "sensor": {
                "type": "perspective",
                "fov": 35,
                "to_world": mi.ScalarTransform4f.look_at(origin=[0.0, -1.2, 0.9], target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0]),
                "sampler": {"type": "independent"},
                "film": {"type": "hdrfilm", "width": 96, "height": 96, "pixel_format": "rgb", "rfilter": {"type": "box"}},
            },
            "light": {"type": "point", "position": [-0.4, -0.6, 1.0], "intensity": {"type": "rgb", "value": 6.0}},
            "plane": {"type": "obj", "filename": str(plane), "bsdf": bsdf},
        }
        exr = OUT / f"swatch_{label}.exr"
        png = OUT / f"swatch_{label}.png"
        render_stokes(scene, exr, 16, 5)
        write_rgb_png(exr, png)
        results.append({"label": label, "family": spec["family"], "slot": row.get("source_slot"), "png": str(png), "exr": str(exr)})
        print(f"event=swatch label={label} family={spec['family']}", flush=True)
    return results


def main() -> None:
    require_variant()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = PACKAGE / "scene_manifest_render.json"
    if not manifest_path.is_file():
        manifest_path = PACKAGE / "scene_manifest_qualified.json"
    manifest = load_manifest(manifest_path)
    manifest["environment"]["radiance"] = 1.2
    manifest["environment"]["reason"] = "labeled_replacement_for_unconverted_world_shader"
    graph = json.loads(Path("dataset/qualification/asset_realism/exported/LivingRoom/21/blend_graph.json").read_text(encoding="utf-8"))
    secondary = next(item for item in graph["objects"] if item["name"] == "camera_0_1")
    manifest["secondary_camera"] = _camera_from_matrix(manifest["camera"], "camera_0_1", secondary["matrix_world"])
    full_only = "--full-only" in sys.argv
    swatches = [] if full_only else _swatches(manifest)
    groups = {
        "architecture": lambda item: any(token in item["id"].lower() for token in ("floor", "wall", "ceiling", "door", "window", "skirting", "exterior")),
        "furniture": lambda item: item["category"] in {"Sofa", "SimpleDesk", "TVStand", "LargeShelf", "SimpleBookcase", "KitchenCabinet", "CellShelf", "Rug"},
        "small_objects": lambda item: item["category"] in {"Monitor", "TV", "DeskLamp", "LargePlantContainer", "WallArt", "BookStack", "BookColumn"},
        "scatter": lambda item: item["category"] == "NatureShelfTrinkets",
    }
    reports = []
    if full_only:
        groups = {}
    for name, predicate in groups.items():
        group = _subset(manifest, predicate)
        try:
            reports.append({"group": name, **_render(group, OUT / f"layer_{name}", 4, 160, 90)})
        except Exception as exc:
            reports.append({"group": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"event=layer_failed group={name} error={type(exc).__name__}: {exc}", flush=True)
    full = {"group": "full_primary"}
    try:
        full.update(_render(manifest, OUT / "full_primary", 16, 640, 360))
        secondary_manifest = json.loads(json.dumps(manifest))
        secondary_manifest["camera"] = manifest["secondary_camera"]
        secondary_report = _render(secondary_manifest, OUT / "full_secondary", 16, 640, 360)
        full["secondary"] = secondary_report
    except Exception as exc:
        full["error"] = f"{type(exc).__name__}: {exc}"
        print(f"event=full_failed error={type(exc).__name__}: {exc}", flush=True)
    payload = {
        "swatches": swatches,
        "layers": reports,
        "full": full,
        "encoding": {
            "exr": "float32 scene-referred Stokes film from render_stokes",
            "png": "uint8 sRGB gamma of the RGB channels only",
            "exposure": 1.0,
            "environment_radiance": 1.2,
            "environment_role": "labeled replacement, not the world shader and not a radiometric calibration",
        },
    }
    (REPORTS / "livingroom_21_render_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("event=render_livingroom_21_done", flush=True)


if __name__ == "__main__":
    main()

"""Primary BSDF traversal, one patched-control render, and EXR integrity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.dataset import exr_schema, load_sample, sha256, training_views
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES, polarization_from_stokes, scalar_stokes, summarize
from sensor_sim.scene_gen.pipeline_converge import prepare_scenes

OUT = Path("dataset/qualification/convergence_v2/reports")
VARIANT = "cuda_ad_spectral_polarized"


def _traverse(xml: Path) -> dict:
    import mitsuba as mi

    mi.set_variant(VARIANT)
    scene = mi.load_file(str(xml))
    params = mi.traverse(scene)
    rows = {}
    for key in params.keys():
        if any(token in key for token in ("eta", "alpha", "material", "reflectance", "int_ior", "ext_ior")):
            value = params[key]
            try:
                rows[key] = np.array(value).reshape(-1)[:8].tolist()
            except Exception:
                rows[key] = str(value)
    shapes = [str(shape.id()) for shape in scene.shapes()]
    return {
        "xml": str(xml),
        "xml_sha256": sha256(xml),
        "variant": VARIANT,
        "status": "mitsuba_load_integration_tested",
        "shape_ids": shapes,
        "parameters": rows,
    }


def _path_text(text: str) -> str:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(text)
    for child in list(root):
        if child.tag.rsplit("}", 1)[-1] == "integrator" and child.get("type") == "stokes":
            inner = next(item for item in list(child) if item.tag.rsplit("}", 1)[-1] == "integrator")
            index = list(root).index(child)
            root.remove(child)
            root.insert(index, inner)
    return ET.tostring(root, encoding="unicode")


def _path_rgb(xml: Path, spp: int, seed: int) -> np.ndarray:
    import mitsuba as mi
    import os
    from sensor_sim.scene_gen.convergence_v2 import prepare_xml_text

    mi.set_variant(VARIANT)
    text = _path_text(prepare_xml_text(xml.read_text(), spp, float32_film=True))
    previous = Path.cwd()
    try:
        os.chdir(xml.parent)
        scene = mi.load_string(text)
        image = np.array(mi.render(scene, seed=int(seed)))
    finally:
        os.chdir(previous)
    return image[:, :, :3]


def _exr(path: Path) -> dict:
    import mitsuba as mi

    mi.set_variant(VARIANT)
    before = path.read_bytes()
    schema = exr_schema(path)
    sample_dir = path.parent
    if (sample_dir / "metadata.json").is_file():
        sample = load_sample(sample_dir)
        views = training_views(sample)
        after = path.read_bytes()
        view_shapes = {key: list(value.shape) for key, value in views.items()}
        mutated = before != after
    else:
        view_shapes = None
        mutated = None
    channels, info = __import__("sensor_sim.scene_gen.convergence_v2", fromlist=["named_exr"]).named_exr(path)
    missing = [name for name in CHANNEL_NAMES if name not in channels]
    scalar = None if missing else scalar_stokes(channels)
    return {
        "status": "audit_tested",
        "schema": schema,
        "named_channels_missing": missing,
        "component_format": info["component_format"],
        "finite": info["finite"],
        "has_alpha": info["has_alpha"],
        "s1_min": None if scalar is None else float(np.min(scalar["s1"])),
        "s1_max": None if scalar is None else float(np.max(scalar["s1"])),
        "training_view_shapes": view_shapes,
        "exr_bytes_changed_by_training_views": mutated,
        "reopened_sha256": sha256(path),
    }


def _compare(stokes_rgb: np.ndarray, path_rgb: np.ndarray, s0: np.ndarray) -> dict:
    mask = s0 > 1e-4
    rows = {}
    for index, name in enumerate(("R", "G", "B")):
        error = np.abs(stokes_rgb[:, :, index] - path_rgb[:, :, index])
        relative = np.zeros(s0.shape, dtype=np.float64)
        np.divide(error, np.maximum(stokes_rgb[:, :, index], 1e-4), out=relative, where=mask)
        absolute = summarize(error[mask])
        absolute["relative"] = summarize(relative[mask])
        rows[name] = absolute
    return {"mask": "stokes S0 > 1e-4", "n": int(mask.sum()), "channels": rows, "note": "a nonzero difference is not by itself a loader bug"}


def main() -> None:
    import mitsuba as mi
    from sensor_sim.scene_gen.convergence_v2 import named_exr

    scenes = prepare_scenes()
    traversed = {name: _traverse(path) for name, path in scenes.items() if name != "multi_object"}
    multi = scenes["multi_object"]
    stokes_path = Path("dataset/qualification/convergence_v2/candidate/multi_object/spp256_seed2101.exr")
    mi.set_variant(VARIANT)
    channels, _info = named_exr(stokes_path)
    scalar = scalar_stokes(channels)
    path = _path_rgb(multi, 256, 2101)
    comparison = _compare(scalar["rgb"], path, scalar["s0"])
    rerun = Path("dataset/qualification/final/rerun/rerun_269837089/polarization.exr")
    report = {
        "bsdf_traverse": traversed,
        "rough_au": "weak stratum retained; promotion was not rerun",
        "patch_render_comparison": "not claimed from DoLP alone; see reference uncertainty before any noise-floor pass",
        "exr_candidate": _exr(stokes_path),
        "exr_historical_rerun": _exr(rerun),
        "polarized_vs_path_rgb": comparison,
        "dolp_from_candidate": summarize(polarization_from_stokes(scalar["s0"], scalar["s1"], scalar["s2"])["dolp"][scalar["s0"] > 1e-4]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "materials_exr.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("materials_exr wrote", len(traversed), "path_n", comparison["n"], flush=True)


if __name__ == "__main__":
    main()

"""Matched-seed patch comparisons. Tolerances are fixed before the renders."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# Frozen before the renders. A larger difference is a behavior change.
ABSOLUTE_TOLERANCE = 1e-5
DOLP_TOLERANCE = 1e-5
AOLP_TOLERANCE_DEG = 0.01

OUT = Path("dataset/qualification/dataset_v1_readiness/material_patch")


def _render(text: str, xml_dir: Path, label: str) -> dict:
    import mitsuba as mi

    from sensor_sim.scene_gen.convergence_v2 import named_exr, prepare_xml_text
    from sensor_sim.scene_gen.metric_audit import angular_difference_deg, polarization_from_stokes, scalar_stokes, summarize

    mi.set_variant("cuda_ad_spectral_polarized")
    previous = Path.cwd()
    try:
        os.chdir(xml_dir)
        scene = mi.load_string(prepare_xml_text(text, 256, float32_film=True))
        image = np.array(mi.render(scene, seed=11))
        names = list(scene.integrator().aov_names())
        if image.shape[2] == 3 + len(names):
            names = ["R", "G", "B"] + names
        shapes = [str(shape.id()) for shape in scene.shapes()]
        params = mi.traverse(scene)
    finally:
        os.chdir(previous)
    path = OUT / f"{label}.exr"
    path.parent.mkdir(parents=True, exist_ok=True)
    mi.Bitmap(np.ascontiguousarray(image, dtype=np.float32), channel_names=names).write(str(path))
    channels, info = named_exr(path)
    scalar = scalar_stokes(channels)
    return {"scalar": scalar, "channels": channels, "info": info, "shapes": shapes, "path": str(path), "params": {key: np.array(params[key]).reshape(-1)[:4].tolist() for key in params.keys() if "eta" in key or "alpha" in key or "opacity" in key}}


def _pair(left: dict, right: dict) -> dict:
    from sensor_sim.scene_gen.metric_audit import angular_difference_deg, polarization_from_stokes, summarize

    mask = left["scalar"]["s0"] > 1e-4
    rows = {}
    for index, name in enumerate(("R", "G", "B")):
        error = np.abs(left["channels"][name] - right["channels"][name])
        rows[name] = summarize(error[mask])
    for name in ("s0", "s1", "s2", "s3"):
        error = np.abs(left["scalar"][name] - right["scalar"][name])
        relative = np.zeros_like(left["scalar"]["s0"])
        np.divide(error, left["scalar"]["s0"], out=relative, where=mask)
        rows[name] = {"absolute": summarize(error[mask]), "relative_to_s0": summarize(relative[mask])}
        for band in ("R", "G", "B"):
            channel = f"{name.upper()}.{band}"
            channel_error = np.abs(left["channels"][channel] - right["channels"][channel])
            rows[channel] = summarize(channel_error[mask])
    left_pol = polarization_from_stokes(left["scalar"]["s0"], left["scalar"]["s1"], left["scalar"]["s2"])
    right_pol = polarization_from_stokes(right["scalar"]["s0"], right["scalar"]["s1"], right["scalar"]["s2"])
    rows["dolp"] = summarize(np.abs(left_pol["dolp"] - right_pol["dolp"])[mask])
    aolp_mask = mask & (left_pol["dolp"] >= 0.05) & (right_pol["dolp"] >= 0.05)
    if aolp_mask.any():
        rows["aolp_deg"] = summarize(angular_difference_deg(left_pol["aolp"], right_pol["aolp"])[aolp_mask])
    else:
        rows["aolp_deg"] = {"median": None, "p95": None, "max": None, "n": 0}
    passed = rows["s0"]["absolute"]["max"] <= ABSOLUTE_TOLERANCE and rows["dolp"]["max"] <= DOLP_TOLERANCE
    if rows["aolp_deg"]["n"]:
        passed = passed and rows["aolp_deg"]["max"] <= AOLP_TOLERANCE_DEG
    return {
        "pass": bool(passed),
        "tolerances": {"absolute": ABSOLUTE_TOLERANCE, "dolp": DOLP_TOLERANCE, "aolp_deg": AOLP_TOLERANCE_DEG},
        "shapes_equal": left["shapes"] == right["shapes"],
        "shapes": [left["shapes"], right["shapes"]],
        "n": int(mask.sum()),
        "metrics": rows,
        "comparison": "matched seed 11, 256 spp, float32; not a noise-floor test",
    }


def main() -> None:
    root = Path("dataset/qualification/pipeline_closure/convergence_xml")
    dielectric = (root / "dielectric_1_50" / "scene.xml").read_text()
    masked = dielectric.replace(
        '<bsdf type="dielectric" id="mat-obj_ball" name="mat-obj_ball"><float name="int_ior" value="1.5" /><float name="ext_ior" value="1.0" /></bsdf>',
        '<bsdf type="mask" id="mat-obj_ball" name="mat-obj_ball"><float name="opacity" value="1.0" /><bsdf type="dielectric"><float name="int_ior" value="1.5" /><float name="ext_ior" value="1.0" /></bsdf></bsdf>',
    )
    rough = (root / "rough_copper" / "scene.xml").read_text()
    anisotropic = rough.replace(
        '<float name="alpha" value="0.1" />',
        '<float name="alpha_u" value="0.1" /><float name="alpha_v" value="0.1" />',
    )
    report = {
        "opacity_1_mask": _pair(_render(dielectric, root / "dielectric_1_50", "dielectric_unpatched"), _render(masked, root / "dielectric_1_50", "dielectric_opacity_mask")),
        "alpha_u_alpha_v": _pair(_render(rough, root / "rough_copper", "rough_alpha"), _render(anisotropic, root / "rough_copper", "rough_alpha_uv")),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print({key: value["pass"] for key, value in report.items()}, flush=True)


if __name__ == "__main__":
    main()

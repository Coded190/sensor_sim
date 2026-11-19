"""High-spp reference and candidate comparison. Does not change the 0.15 cap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.pipeline_closure import OUT, _report, _save, stokes_of

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_SEEDS = (1101, 1102, 1103, 1104)
CANDIDATE_SEEDS = (2101, 2102, 2103)
LEVELS = (256, 512, 1024)
# Frozen before rendering. 25 percent of each candidate cap.
REFERENCE_LIMITS = {"s0_p95": 0.15 * 0.25, "s1_p95": 0.08 * 0.25, "dolp_p95": 0.08 * 0.25, "aolp_p95_deg": 10.0 * 0.25}
CANDIDATE_LIMITS = {
    "s0_median": 0.05,
    "s0_p95": 0.15,
    "component_median": 0.02,
    "component_p95": 0.08,
    "dolp_median": 0.02,
    "dolp_p95": 0.08,
    "aolp_median_deg": 2.0,
    "aolp_p95_deg": 10.0,
}


def _avg(value):
    array = np.asarray(value, dtype=np.float64)
    return array.mean(axis=2) if array.ndim == 3 else array


def _relative(candidate, reference, illuminated):
    denom = np.maximum(reference, 1e-4)
    error = np.abs(candidate - reference) / denom
    return {
        "median": float(np.median(error[illuminated])),
        "p95": float(np.percentile(error[illuminated], 95)),
        "max": float(np.max(error[illuminated])),
        "n": int(illuminated.sum()),
    }


def _aolp(candidate, reference, s1c, s2c, s1r, s2r, illuminated):
    from sensor_sim.stokes import polarization_maps

    left = polarization_maps(reference, s1r, s2r, np.zeros_like(reference))
    right = polarization_maps(candidate, s1c, s2c, np.zeros_like(candidate))
    mask = illuminated & (left["dolp"] >= 0.05) & (right["dolp"] >= 0.05)
    delta = np.abs(left["aolp"] - right["aolp"])
    delta = np.minimum(delta, np.pi - np.minimum(delta, np.pi))
    delta = np.minimum(delta, np.pi - delta)
    if not mask.any():
        return {"median_deg": None, "p95_deg": None, "n": 0}
    degrees = np.degrees(delta[mask])
    return {"median_deg": float(np.median(degrees)), "p95_deg": float(np.percentile(degrees, 95)), "n": int(mask.sum())}


def evaluate(xml: Path, reference_seed_stack, candidate_spp: int, candidate_seed: int, illuminated: np.ndarray) -> dict:
    rendered = [_avg(item) for item in stokes_of(xml, candidate_spp, candidate_seed)]
    reference = reference_seed_stack
    out = {}
    for index, name in enumerate(("s0", "s1", "s2", "s3")):
        out[name] = _relative(rendered[index], reference[index], illuminated)
    abs_dolp_num = np.sqrt(rendered[1] ** 2 + rendered[2] ** 2)
    abs_dolp_den = np.sqrt(reference[1] ** 2 + reference[2] ** 2)
    dolp_c = np.zeros_like(rendered[0])
    dolp_r = np.zeros_like(reference[0])
    np.divide(abs_dolp_num, rendered[0], out=dolp_c, where=rendered[0] > 1e-4)
    np.divide(abs_dolp_den, reference[0], out=dolp_r, where=reference[0] > 1e-4)
    dolp_error = np.abs(dolp_c - dolp_r)
    out["dolp"] = {
        "median": float(np.median(dolp_error[illuminated])),
        "p95": float(np.percentile(dolp_error[illuminated], 95)),
        "max": float(np.max(dolp_error[illuminated])),
        "n": int(illuminated.sum()),
    }
    out["aolp"] = _aolp(rendered[0], reference[0], rendered[1], rendered[2], reference[1], reference[2], illuminated)
    limits = CANDIDATE_LIMITS
    out["pass"] = bool(
        out["s0"]["median"] <= limits["s0_median"]
        and out["s0"]["p95"] <= limits["s0_p95"]
        and all(out[name]["median"] <= limits["component_median"] and out[name]["p95"] <= limits["component_p95"] for name in ("s1", "s2", "s3"))
        and out["dolp"]["median"] <= limits["dolp_median"]
        and out["dolp"]["p95"] <= limits["dolp_p95"]
        and out["aolp"]["median_deg"] is not None
        and out["aolp"]["median_deg"] <= limits["aolp_median_deg"]
        and out["aolp"]["p95_deg"] <= limits["aolp_p95_deg"]
    )
    return out


def reference_stack(xml: Path, seeds: tuple[int, ...]) -> tuple[list[np.ndarray], dict]:
    renders = []
    for seed in seeds:
        renders.append([_avg(item) for item in stokes_of(xml, 1024, seed)])
        print("reference", xml.parent.name, seed, flush=True)
    mean = [np.mean(np.stack([item[index] for item in renders]), axis=0) for index in range(4)]
    illuminated = mean[0] > 1e-4
    uncertainty = {}
    for index, name in enumerate(("s0", "s1", "s2", "s3")):
        errors = []
        for rendered in renders:
            errors.append(_relative(rendered[index], mean[index], illuminated)["p95"])
        uncertainty[name] = {"p95_vs_mean_max": float(max(errors)), "limit": REFERENCE_LIMITS["s0_p95"] if name == "s0" else REFERENCE_LIMITS["s1_p95"]}
    stable = uncertainty["s0"]["p95_vs_mean_max"] <= REFERENCE_LIMITS["s0_p95"] and all(
        uncertainty[name]["p95_vs_mean_max"] <= REFERENCE_LIMITS["s1_p95"] for name in ("s1", "s2", "s3")
    )
    return mean, {"stable": bool(stable), "uncertainty": uncertainty, "seeds": list(seeds), "n_illuminated": int(illuminated.sum())}


def prepare_scenes() -> dict[str, Path]:
    import shutil
    import xml.etree.ElementTree as ET
    from sensor_sim.scene_gen.materials import fixed_bsdf
    from sensor_sim.scene_gen.xml_patch import bsdf_element

    base = ROOT / "dataset/qualification/final/convergence/conv_sphere/scene.xml"
    presets = {
        "dielectric_1_33": {"type": "dielectric", "int_ior": 1.33, "ext_ior": 1.0},
        "dielectric_1_40": {"type": "dielectric", "int_ior": 1.40, "ext_ior": 1.0},
        "dielectric_1_50": {"type": "dielectric", "int_ior": 1.50, "ext_ior": 1.0},
        "dielectric_1_60": {"type": "dielectric", "int_ior": 1.60, "ext_ior": 1.0},
        "dielectric_1_70": {"type": "dielectric", "int_ior": 1.70, "ext_ior": 1.0},
        "pplastic": fixed_bsdf("pplastic"),
        "copper": {"type": "conductor", "material": "Cu"},
        "rough_copper": {"type": "roughconductor", "material": "Cu", "distribution": "beckmann", "alpha": 0.1},
    }
    scenes = {"multi_object": ROOT / "dataset/qualification/final/rerun/rerun_269837089/scene.xml"}
    if not base.is_file():
        return scenes
    for name, bsdf in presets.items():
        folder = OUT / "convergence_xml" / name
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / "scene.xml"
        if not dest.is_file():
            tree = ET.parse(base)
            root = tree.getroot()
            for node in list(root.iter()):
                if node.tag.rsplit("}", 1)[-1] == "bsdf" and node.get("id") == "mat-obj_ball":
                    parent = next(parent for parent in root.iter() if node in list(parent))
                    index = list(parent).index(node)
                    parent.remove(node)
                    parent.insert(index, bsdf_element(bsdf, "mat-obj_ball"))
            text = ET.tostring(root, encoding="unicode")
            text = text.replace("meshes/ball.ply", "../../../final/convergence/conv_sphere/meshes/ball.ply")
            text = text.replace("meshes/ground.ply", "../../../final/convergence/conv_sphere/meshes/ground.ply")
            dest.write_text(text)
        scenes[name] = dest
    return scenes


def main() -> None:
    report = _report()
    scenes = prepare_scenes()
    report["convergence_scene_paths"] = {name: str(path) for name, path in scenes.items()}
    _save(report)
    missing = [name for name, path in scenes.items() if not path.is_file()]
    report["convergence"] = {"missing_scenes": missing, "reference_limit_rule": "uncertainty <= 25 percent of each cap, frozen before rendering", "caps_not_loosened": True}
    results = {}
    any_unstable = False
    for name, xml in scenes.items():
        if not xml.is_file():
            results[name] = {"status": "blocked", "reason": "xml missing"}
            continue
        mean, info = reference_stack(xml, REFERENCE_SEEDS)
        illuminated = mean[0] > 1e-4
        candidates = {}
        for spp in LEVELS:
            candidates[str(spp)] = []
            for seed in CANDIDATE_SEEDS:
                row = evaluate(xml, mean, spp, seed, illuminated)
                row["seed"] = seed
                row["spp"] = spp
                candidates[str(spp)].append(row)
                print(name, spp, seed, row["pass"], flush=True)
        passed = {str(spp): all(item["pass"] for item in rows) for spp, rows in candidates.items()}
        results[name] = {"reference": info, "candidates_pass": passed, "status": "mitsuba_render_tested"}
        any_unstable = any_unstable or not info["stable"]
        report["convergence"]["scenes"] = results
        _save(report)
    chosen = None
    if not any_unstable and not missing:
        for spp in LEVELS:
            if all(item.get("candidates_pass", {}).get(str(spp)) for item in results.values() if "candidates_pass" in item):
                chosen = spp
                break
    report["convergence"]["chosen_spp"] = chosen
    report["convergence"]["reference_unstable"] = any_unstable
    _save(report)
    print("chosen", chosen, "unstable", any_unstable, flush=True)


if __name__ == "__main__":
    main()

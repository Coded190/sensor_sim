"""Eight-seed reference qualification and candidate spp. Not Dataset v1."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.convergence_v2 import (
    OUT,
    VARIANT,
    _jsonable,
    _log,
    _scalar_from_exr,
    environment,
    load_scene,
)
from sensor_sim.scene_gen.dataset import sha256
from sensor_sim.scene_gen.metric_audit import (
    candidate_metrics,
    corrected_object_mask,
    corrected_relative,
    derived_uncertainty,
    historical_reference_decision,
    reference_target_decision,
    stokes_consistency,
    stokes_uncertainty,
    stratify,
    weak_illumination,
)
from sensor_sim.scene_gen.pipeline_converge import prepare_scenes

REFERENCE_ROOT = OUT / "reference" / "scenes"
CANDIDATE_ROOT = OUT / "candidate"
MATERIAL_SEEDS = tuple(range(4101, 4109))
CANDIDATE_SEEDS = (2101, 2102, 2103)
CANDIDATE_LEVELS = (256, 512, 1024)


def _ground_from_aov(xml: Path) -> dict:
    from sensor_sim.scene_gen.render_sample import render_aovs

    _scene, aovs, _names = render_aovs(xml, VARIANT)
    index = np.rint(np.asarray(aovs["idx.I"], dtype=np.float64)).astype(np.int32)
    geo = np.stack([np.asarray(aovs["geo.X"]), np.asarray(aovs["geo.Y"]), np.asarray(aovs["geo.Z"])], axis=-1)
    depth = np.asarray(aovs["depth.T"], dtype=np.float64)
    lengths = np.linalg.norm(geo, axis=-1)
    valid = np.isfinite(geo).all(axis=-1) & (np.abs(lengths - 1.0) < 1e-2) & (depth > 0.0)
    return {"object_ids": index, "valid": valid, "source": "aov shape_index and finite unit geometric normal"}


def _multi_ground(shape: tuple[int, int]) -> dict:
    from sensor_sim.scene_gen.convergence_v2 import _ground

    ground = _ground(shape)
    ground["source"] = "saved rerun ground truth, not the shape-index voter"
    return ground


def render_reference(xml: Path, spp: int, seed: int, destination: Path) -> dict:
    import mitsuba as mi

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(".json")
    if destination.is_file() and sidecar.is_file():
        record = json.loads(sidecar.read_text())
        if int(record.get("sampler_sample_count", -1)) == int(spp):
            return record
    scene = load_scene(spp, float32_film=True, xml=xml)
    started = time.perf_counter()
    image = np.array(mi.render(scene, seed=int(seed)))
    elapsed = time.perf_counter() - started
    names = list(scene.integrator().aov_names())
    if image.shape[2] == 3 + len(names):
        names = ["R", "G", "B"] + names
    mi.Bitmap(np.ascontiguousarray(image, dtype=np.float32), channel_names=names).write(str(destination))
    record = {
        "scene": xml.parent.name,
        "xml": str(xml),
        "xml_sha256": sha256(xml),
        "spp": int(spp),
        "seed": int(seed),
        "variant": VARIANT,
        "integrator": "stokes/path",
        "sampler": "independent",
        "sampler_sample_count": int(scene.sensors()[0].sampler().sample_count()),
        "film_component_format": "float32",
        "resolution": [int(image.shape[1]), int(image.shape[0])],
        "filter": "box",
        "seconds": elapsed,
        "exr": str(destination),
        "exr_sha256": sha256(destination),
        "channel_names": names,
    }
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    _log(f"scene={record['scene']} spp={spp} seed={seed} seconds={elapsed:.3f} status=reference")
    return record


def evaluate_scene(name: str, xml: Path, spp: int, seeds: tuple[int, ...], ground: dict) -> dict:
    folder = REFERENCE_ROOT / name
    renders = []
    files = []
    for seed in seeds:
        path = folder / f"spp{spp}_seed{seed}.exr"
        render_reference(xml, spp, seed, path)
        scalar, info = _scalar_from_exr(path)
        info["seed"] = int(seed)
        renders.append(scalar)
        files.append({key: value for key, value in info.items() if key != "finite" or True})
    historical = historical_reference_decision(renders)
    mean = {key: historical["mean"][key] for key in ("s0", "s1", "s2", "s3")}
    if ground["object_ids"].shape != mean["s0"].shape:
        raise RuntimeError(f"{name} ground truth shape {ground['object_ids'].shape} != render {mean['s0'].shape}")
    mask = corrected_object_mask(mean["s0"], ground["object_ids"], ground["valid"])
    uncertainty = {}
    for key in ("s0", "s1", "s2", "s3"):
        stack = np.stack([item[key] for item in renders], axis=0)
        uncertainty[key] = stokes_uncertainty(stack, mean["s0"], mask)
    derived = derived_uncertainty(renders, mask)
    decision = reference_target_decision(uncertainty, derived)
    decision["qualified"] = bool(decision["pass"] and len(seeds) >= 8)
    decision["n_seeds"] = len(seeds)
    worst = int(np.argmax([corrected_relative(item["s1"], mean["s1"], mean["s0"], mask)["p95"] or -1 for item in renders]))
    error = np.zeros(mean["s0"].shape, dtype=np.float64)
    np.divide(np.abs(renders[worst]["s1"] - mean["s1"]), mean["s0"], out=error, where=mask)
    return {
        "scene": name,
        "xml": str(xml),
        "xml_sha256": sha256(xml),
        "spp": int(spp),
        "seeds": list(seeds),
        "n_mask": int(mask.sum()),
        "mask": "object id > 0, valid geometry, reference S0 > 1e-4",
        "ground": ground.get("source"),
        "historical_component_denominator_stable": historical["stable"],
        "historical_s0_p95_vs_mean_max": historical["uncertainty"]["s0"]["p95_vs_mean_max"],
        "historical_s1_p95_vs_mean_max": historical["uncertainty"]["s1"]["p95_vs_mean_max"],
        "uncertainty": uncertainty,
        "derived_uncertainty": derived,
        "decision": decision,
        "s1_strata_worst_seed": stratify(error, mask, ground["object_ids"], mean["s0"]),
        "consistency": stokes_consistency(mean["s0"], mean["s1"], mean["s2"], mean["s3"], mask),
        "weak_illumination": weak_illumination(mean["s0"], ground["object_ids"], ground["valid"]),
        "files": [{key: value for key, value in item.items() if not isinstance(value, np.ndarray)} for item in files],
        "reference_stokes": mean,
        "mask_array": mask,
        "object_ids": ground["object_ids"],
    }


def _public(result: dict) -> dict:
    hidden = {"reference_stokes", "mask_array", "object_ids"}
    return {key: value for key, value in result.items() if key not in hidden}


def _limiting(result: dict) -> str:
    failed = [name for name, row in result["decision"]["checks"].items() if not row["pass"]]
    return ",".join(failed) if failed else "none"


def qualify() -> dict:
    scenes = prepare_scenes()
    multi_xml = scenes["multi_object"]
    seeds = tuple(range(1101, 1109))
    probe = evaluate_scene("multi_object", multi_xml, 1024, seeds, _multi_ground((128, 128)))
    spp = 1024
    results = {"multi_object_1024": _public(probe)}
    if not probe["decision"]["qualified"]:
        _log(f"scene=multi_object spp=1024 qualified=false limiting={_limiting(probe)}")
        if probe["decision"]["checks"]["s0"]["p95_upper"] > 0.0375:
            seeds = tuple(range(3101, 3109))
            spp = 2048
            probe = evaluate_scene("multi_object", multi_xml, spp, seeds, _multi_ground((128, 128)))
            results["multi_object_2048"] = _public(probe)
            _log(f"scene=multi_object spp=2048 qualified={probe['decision']['qualified']} limiting={_limiting(probe)}")
    if probe["decision"]["qualified"]:
        results["multi_object"] = _public(probe)
    if not probe["decision"]["qualified"]:
        _write(results, None, blocked="multi_object reference uncertainty")
        return results
    control_name = "dielectric_1_50"
    control = evaluate_scene(control_name, scenes[control_name], spp, MATERIAL_SEEDS, _ground_from_aov(scenes[control_name]))
    results[control_name] = _public(control)
    _log(f"scene={control_name} spp={spp} qualified={control['decision']['qualified']} limiting={_limiting(control)}")
    if not control["decision"]["qualified"]:
        _write(results, None, blocked=f"{control_name} reference uncertainty")
        return results
    for name, xml in scenes.items():
        if name in results:
            continue
        ground = _ground_from_aov(xml)
        row = evaluate_scene(name, xml, spp, MATERIAL_SEEDS, ground)
        results[name] = _public(row)
        _log(f"scene={name} spp={spp} qualified={row['decision']['qualified']} limiting={_limiting(row)}")
        if not row["decision"]["qualified"]:
            _write(results, None, blocked=f"{name} reference uncertainty")
            return results
    chosen = _candidates(scenes, spp, results)
    _write(results, chosen, blocked=None if chosen else "no candidate spp passed every scene")
    return results


def _candidates(scenes: dict[str, Path], reference_spp: int, references: dict) -> int | None:
    """Lowest spp at which every seed passes every scene. 2048 is a separate tier."""
    passing = []
    report = {}
    for spp in CANDIDATE_LEVELS:
        scene_pass = {}
        for name, xml in scenes.items():
            reference = _load_reference(name, reference_spp)
            rows = []
            for seed in CANDIDATE_SEEDS:
                if seed in set(reference["seeds"]):
                    raise RuntimeError(f"candidate seed {seed} is in the reference")
                destination = CANDIDATE_ROOT / name / f"spp{spp}_seed{seed}.exr"
                record = render_reference(xml, spp, seed, destination)
                scalar, _info = _scalar_from_exr(destination)
                metrics = candidate_metrics(scalar, reference["stokes"], reference["mask"])
                metrics["seed"] = seed
                metrics["spp"] = spp
                metrics["seconds"] = record["seconds"]
                metrics["exr_sha256"] = record["exr_sha256"]
                rows.append(_jsonable(metrics))
                _log(f"scene={name} candidate_spp={spp} seed={seed} pass={metrics['pass']}")
            scene_pass[name] = {"renders": rows, "pass": all(item["pass"] for item in rows)}
        report[str(spp)] = scene_pass
        if all(item["pass"] for item in scene_pass.values()):
            passing.append(spp)
    chosen = passing[0] if passing else None
    if chosen is None:
        cost = _candidate_2048_cost(report)
        (OUT / "reports" / "candidate_2048_tier.json").write_text(json.dumps(cost, indent=2) + "\n")
        if cost["projection_could_pass"]:
            _log("candidate_2048 status=not_run reason=projection_recorded_before_any_2048_candidate")
    (OUT / "candidate" / "summary.json").write_text(json.dumps({"levels": report, "chosen_spp": chosen}, indent=2) + "\n")
    return chosen


def _load_reference(name: str, spp: int) -> dict:
    folder = REFERENCE_ROOT / name
    seeds = tuple(sorted(int(path.name.split("_seed")[1].split(".")[0]) for path in folder.glob(f"spp{spp}_seed*.exr")))
    renders = []
    for seed in seeds:
        scalar, _info = _scalar_from_exr(folder / f"spp{spp}_seed{seed}.exr")
        renders.append(scalar)
    historical = historical_reference_decision(renders)
    mean = historical["mean"]
    if name == "multi_object":
        ground = _multi_ground(mean["s0"].shape)
    else:
        from sensor_sim.scene_gen.pipeline_converge import prepare_scenes

        ground = _ground_from_aov(prepare_scenes()[name])
    mask = corrected_object_mask(mean["s0"], ground["object_ids"], ground["valid"])
    return {"stokes": mean, "mask": mask, "seeds": seeds}


def _candidate_2048_cost(report: dict) -> dict:
    """Project 1024 errors by 1/sqrt(2). Do not run 2048 until this file exists."""
    level = report.get("1024", {})
    projections = {}
    could = True
    seconds = []
    for name, row in level.items():
        for item in row["renders"]:
            seconds.append(float(item.get("seconds") or 0.0))
            for metric, limit_name, scale in (
                ("s0", "s0_relative", "p95"),
                ("s1", "s1_relative", "p95"),
                ("dolp", "dolp_absolute", "p95"),
            ):
                value = item[metric][scale]
                if value is None:
                    could = False
                    continue
                from sensor_sim.scene_gen.metric_audit import CANDIDATE_LIMITS

                projected = float(value) / np.sqrt(2.0)
                projections.setdefault(name, {})[metric] = projected
                if projected > CANDIDATE_LIMITS[limit_name]["p95"]:
                    could = False
    mean_seconds = float(np.mean(seconds)) if seconds else None
    n_renders = 9 * 3
    return {
        "spp": 2048,
        "assumption": "candidate error scales as 1/sqrt(spp) from the 1024 measurements",
        "projected_p95": projections,
        "projection_could_pass": bool(could and projections),
        "measured_mean_seconds_at_1024": mean_seconds,
        "estimated_seconds_for_2048_matrix": None if mean_seconds is None else mean_seconds * 2.0 * n_renders,
        "n_renders": n_renders,
        "ran": False,
    }


def _write(results: dict, chosen: int | None, blocked: str | None) -> None:
    payload = {
        "reference_spp_scenes": results,
        "chosen_spp": chosen,
        "spp_frozen": chosen is not None,
        "blocked": blocked,
        "dataset_v1_authorized": False,
        "candidate_seeds": list(CANDIDATE_SEEDS),
        "candidate_seeds_not_in_reference": True,
    }
    path = OUT / "reports" / "reference_qualification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    evidence_path = OUT / "evidence.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.is_file() else {}
    evidence["reference_qualification"] = _jsonable(payload)
    evidence["spp_frozen"] = chosen is not None
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")


def main() -> None:
    _log(f"environment={json.dumps(environment())}")
    qualify()


if __name__ == "__main__":
    main()

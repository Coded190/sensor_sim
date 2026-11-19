"""Historical convergence reconstruction and the corrected S0-normalized audit.

This does not generate Dataset v1. It does not call pipeline_converge._relative.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.dataset import channels_to_array, sha256
from sensor_sim.scene_gen.metric_audit import (
    RECONSTRUCTION_ABSOLUTE_TOLERANCE,
    absolute_error,
    corrected_object_mask,
    corrected_relative,
    derived_uncertainty,
    historical_reference_decision,
    sampling_ratio,
    scalar_stokes,
    stokes_consistency,
    stokes_uncertainty,
    stratify,
    summaries_match,
    weak_illumination,
    write_protocol,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "dataset" / "qualification" / "convergence_v2"
HISTORICAL_XML = ROOT / "dataset" / "qualification" / "final" / "rerun" / "rerun_269837089" / "scene.xml"
GROUND = HISTORICAL_XML.parent / "ground_truth"
EVIDENCE = ROOT / "dataset" / "qualification" / "pipeline_closure" / "evidence.json"
VARIANT = "cuda_ad_spectral_polarized"
JOBS = (
    {"label": "multi_object", "spp": 1024, "seeds": (1101, 1102, 1103, 1104)},
    {"label": "multi_object", "spp": 2048, "seeds": (3101, 3102, 3103, 3104)},
)
HISTORICAL_SUMMARIES = {
    1024: {"s0": 0.060615128579098766, "s1": 1.041381605318754, "s2": 0.5206311646195383, "s3": 0.0, "n_illuminated": 5945},
    2048: {"s0": 0.044129988798494196, "s1": 0.7115930784493624},
}


def _log(message: str) -> None:
    print(f"convergence_v2 {message}", flush=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key not in {"mean", "illuminated"}}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def environment() -> dict:
    import mitsuba as mi

    versions = {"python": platform.python_version(), "platform": platform.platform(), "mitsuba": mi.__version__}
    try:
        import drjit

        versions["drjit"] = drjit.__version__
    except ImportError as exc:
        versions["drjit_error"] = str(exc)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    versions["nvidia_smi"] = gpu.stdout.strip()
    from sensor_sim.film.blender_build import find_blender
    from sensor_sim.scene_gen.blender_export import addon_commit

    binary = find_blender()
    versions["blender"] = subprocess.check_output([str(binary), "--version"], text=True).splitlines()[0] if binary else "missing"
    versions["addon_commit"] = addon_commit()
    versions["variant_required"] = VARIANT
    versions["variant_installed"] = VARIANT in list(mi.variants())
    return versions


def named_exr(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    import mitsuba as mi

    bitmap = mi.Bitmap(str(path))
    names = [field.name for field in bitmap.struct_()]
    raw = np.array(bitmap)
    if raw.ndim == 2:
        raw = raw[:, :, None]
    if raw.shape[-1] != len(names):
        raise RuntimeError(f"{path} has {raw.shape[-1]} planes and names {names}")
    channels = {names[index]: np.ascontiguousarray(raw[:, :, index], dtype=np.float64) for index in range(len(names))}
    info = {
        "path": str(path),
        "sha256": sha256(path),
        "height": int(bitmap.height()),
        "width": int(bitmap.width()),
        "component_format": str(bitmap.component_format()),
        "pixel_format": str(bitmap.pixel_format()),
        "file_channels": names,
        "dtype_on_disk": "float32" if "float32" in str(bitmap.component_format()).lower() or "Float32" in str(bitmap.component_format()) else str(bitmap.component_format()),
        "finite": bool(all(np.isfinite(values).all() for values in channels.values())),
        "has_alpha": any(name.lower() == "a" or name.endswith(".A") for name in names),
    }
    return channels, info


def _ground(shape: tuple[int, int]) -> dict:
    from sensor_sim.film.images import read_exr

    def one(name: str) -> np.ndarray:
        array = channels_to_array(read_exr(GROUND / name))
        return np.asarray(array, dtype=np.float64).reshape(array.shape[0], array.shape[1]) if array.ndim >= 2 else array

    object_ids = np.rint(one("object_ids.exr")).astype(np.int32)
    valid = one("valid_mask.exr") > 0.5
    if object_ids.shape != shape or valid.shape != shape:
        raise RuntimeError(f"ground truth shape {object_ids.shape} does not match render {shape}")
    return {"object_ids": object_ids, "valid": valid, "object_ids_sha256": sha256(GROUND / "object_ids.exr"), "valid_sha256": sha256(GROUND / "valid_mask.exr")}


def prepare_xml_text(text: str, spp: int, *, float32_film: bool) -> str:
    """Set the spp default and float32 film without writing the source XML."""
    if '<default name="spp" value="' not in text:
        raise RuntimeError("the scene XML has no spp default")
    start = text.index('<default name="spp" value="')
    end = text.index("/>", start)
    text = text[:start] + f'<default name="spp" value="{int(spp)}" ' + text[end:]
    if float32_film and 'name="component_format"' not in text:
        text = text.replace(
            '<film type="hdrfilm" name="film">',
            '<film type="hdrfilm" name="film"><string name="component_format" value="float32"/>',
            1,
        )
    return text


def load_scene(spp: int, *, float32_film: bool, xml: Path = HISTORICAL_XML):
    """Load a saved XML with an explicit sample count.

    ``mi.render(..., spp=N)`` updates the sampler only when it is passed, and a
    scene constructed at the XML default of 16 stays at 16 if that argument is
    omitted. The on-disk XML is not modified.
    """
    import mitsuba as mi

    if VARIANT not in mi.variants():
        raise RuntimeError(f"{VARIANT} is not installed; refusing to describe an LLVM render as CUDA")
    mi.set_variant(VARIANT)
    text = prepare_xml_text(Path(xml).read_text(), spp, float32_film=float32_film)
    previous = Path.cwd()
    try:
        os.chdir(Path(xml).parent)
        scene = mi.load_string(text)
    finally:
        os.chdir(previous)
    count = int(scene.sensors()[0].sampler().sample_count())
    if count != int(spp):
        raise RuntimeError(f"sampler sample_count is {count}, requested {spp}")
    return scene


def render_job(spp: int, seed: int, destination: Path, log_path: Path) -> dict:
    import mitsuba as mi

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(".json")
    if destination.is_file() and sidecar.is_file():
        record = json.loads(sidecar.read_text())
        if int(record.get("sampler_sample_count", -1)) == int(spp) and record.get("film_component_format") == "float32":
            record["reused_existing_exr"] = True
            _log(f"scene=multi_object spp={spp} seed={seed} status=reuse path={destination}")
            return record
    scene = load_scene(spp, float32_film=True)
    started = time.perf_counter()
    image = mi.render(scene, seed=int(seed))
    elapsed = time.perf_counter() - started
    array = np.array(image)
    names = list(scene.integrator().aov_names())
    if array.shape[2] == 3 + len(names):
        names = ["R", "G", "B"] + names
    bitmap = mi.Bitmap(np.ascontiguousarray(array, dtype=np.float32), channel_names=names)
    bitmap.write(str(destination))
    film_format = str(scene.sensors()[0].film())
    record = {
        "scene": "multi_object",
        "xml": str(HISTORICAL_XML),
        "xml_sha256": sha256(HISTORICAL_XML),
        "spp": int(spp),
        "seed": int(seed),
        "variant": VARIANT,
        "integrator": "stokes/path",
        "sampler": "independent",
        "resolution": [int(array.shape[1]), int(array.shape[0])],
        "filter": "box",
        "seconds": elapsed,
        "exr": str(destination),
        "exr_sha256": sha256(destination),
        "channel_names": names,
        "sampler_sample_count": int(scene.sensors()[0].sampler().sample_count()),
        "film_component_format": "float32" if "float32" in film_format else film_format,
        "spp_argument_to_mi_render": "not used; Mitsuba 3.9.1 left sample_count at the XML default when spp was passed to mi.render",
        "reused_existing_exr": False,
    }
    record.update(environment())
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    _log(f"scene=multi_object spp={spp} seed={seed} seconds={elapsed:.3f} sha256={record['exr_sha256']} status=wrote")
    return record


def _scalar_from_exr(path: Path) -> tuple[dict, dict]:
    channels, info = named_exr(path)
    scalar = scalar_stokes(channels)
    info["scalar_reduction"] = "mean of component.R, component.G, component.B; RGB-converted, not wavelength-resolved"
    info["s1_negative_fraction"] = float(np.mean(scalar["s1"] < 0.0))
    info["s1_min"] = float(np.min(scalar["s1"]))
    info["s1_max"] = float(np.max(scalar["s1"]))
    return scalar, info


def analyze_level(spp: int, seeds: tuple[int, ...], folder: Path) -> dict:
    renders = []
    files = []
    for seed in seeds:
        path = folder / f"spp{spp}_seed{seed}.exr"
        scalar, info = _scalar_from_exr(path)
        info["seed"] = int(seed)
        info["spp"] = int(spp)
        renders.append(scalar)
        files.append(info)
    historical = historical_reference_decision(renders)
    mean = historical["mean"]
    ground = _ground(mean["s0"].shape)
    mask = corrected_object_mask(mean["s0"], ground["object_ids"], ground["valid"])
    per_seed = []
    pooled = {name: [] for name in ("s0", "s1", "s2", "s3")}
    for render in renders:
        row = {}
        for name in ("s0", "s1", "s2", "s3"):
            row[name] = corrected_relative(render[name], mean[name], mean["s0"], mask)
            row[name + "_absolute"] = absolute_error(render[name], mean[name], mask)
            pooled[name].append(np.abs(render[name] - mean[name])[mask] / mean["s0"][mask])
        per_seed.append(row)
    corrected = {}
    for name in ("s0", "s1", "s2", "s3"):
        values = np.concatenate(pooled[name]) if pooled[name] else np.array([])
        from sensor_sim.scene_gen.metric_audit import summarize

        corrected[name] = summarize(values)
        corrected[name]["denominator"] = "reference_S0"
        corrected[name]["aggregation"] = "pixels pooled across seeds for the error distribution; this is not a seed replicate"
        corrected[name]["per_seed_p95"] = [row[name]["p95"] for row in per_seed]
        corrected[name]["per_seed_median"] = [row[name]["median"] for row in per_seed]
    stacks = {name: np.stack([item[name] for item in renders], axis=0) for name in ("s0", "s1", "s2", "s3")}
    uncertainty = {name: stokes_uncertainty(stacks[name], mean["s0"], mask) for name in ("s0", "s1", "s2", "s3")}
    derived = derived_uncertainty(renders, mask)
    worst = int(np.argmax([row["s1"]["p95"] or -1.0 for row in per_seed]))
    worst_error = np.zeros(mean["s0"].shape, dtype=np.float64)
    np.divide(np.abs(renders[worst]["s1"] - mean["s1"]), mean["s0"], out=worst_error, where=mask)
    reference_rgb = np.mean(np.stack([item["s1_rgb"] for item in renders], axis=0), axis=0)
    channel_error = np.zeros(reference_rgb.shape, dtype=np.float64)
    np.divide(np.abs(renders[worst]["s1_rgb"] - reference_rgb), mean["s0"][:, :, None], out=channel_error, where=mask[:, :, None])
    strata = stratify(worst_error, mask, ground["object_ids"], mean["s0"], channel_error)
    consistency = stokes_consistency(mean["s0"], mean["s1"], mean["s2"], mean["s3"], mask)
    expected = HISTORICAL_SUMMARIES[spp]
    comparison = {}
    for name in ("s0", "s1", "s2", "s3"):
        if name not in expected:
            continue
        reconstructed = historical["uncertainty"][name]["p95_vs_mean_max"]
        comparison[name] = {
            "historical": expected[name],
            "reconstructed": reconstructed,
            "absolute_difference": abs(reconstructed - expected[name]),
            "matches_tolerance": summaries_match(reconstructed, expected[name]),
            "tolerance": RECONSTRUCTION_ABSOLUTE_TOLERANCE,
        }
    if "n_illuminated" in expected:
        comparison["n_illuminated"] = {
            "historical": expected["n_illuminated"],
            "reconstructed": historical["n_illuminated"],
            "matches": historical["n_illuminated"] == expected["n_illuminated"],
        }
    recovered = all(item.get("matches_tolerance", item.get("matches", False)) for item in comparison.values())
    return {
        "spp": spp,
        "seeds": list(seeds),
        "files": files,
        "historical": _jsonable(historical),
        "corrected_vs_mean": corrected,
        "uncertainty": uncertainty,
        "derived_uncertainty": derived,
        "s1_strata_worst_seed": strata,
        "stokes_consistency_on_mean": consistency,
        "weak_illumination": weak_illumination(mean["s0"], ground["object_ids"], ground["valid"]),
        "n_corrected_mask": int(mask.sum()),
        "mask": "object id > 0 and valid ground truth and reference S0 > 1e-4",
        "historical_mask": "mean S0 > 1e-4, no object mask",
        "comparison_to_saved_json": comparison,
        "reconstruction_status": "reconstructed" if recovered else "new_reconstruction_not_exact_recovery",
        "historical_stable_recomputed": historical["stable"],
        "historical_stable_kept_false": False,
    }


def _save_figure(folder: Path, spp: int, seeds: tuple[int, ...]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    renders = []
    for seed in seeds:
        scalar, _info = _scalar_from_exr(folder / f"spp{spp}_seed{seed}.exr")
        renders.append(scalar)
    historical = historical_reference_decision(renders)
    mean = historical["mean"]
    ground = _ground(mean["s0"].shape)
    mask = corrected_object_mask(mean["s0"], ground["object_ids"], ground["valid"])
    pieces = []
    for item in renders:
        piece = np.zeros(mean["s0"].shape, dtype=np.float64)
        np.divide(np.abs(item["s1"] - mean["s1"]), mean["s0"], out=piece, where=mask)
        pieces.append(piece)
    error = np.mean(pieces, axis=0)
    shown = np.where(mask, error, np.nan)
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    image = axes[0].imshow(shown, cmap="magma")
    axes[0].set_title(f"S1 |error| / S0, mean of seeds, {spp} spp")
    axes[0].axis("off")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].hist(error[mask], bins=40, color="0.2")
    axes[1].set_xlabel("|S1 - mean S1| / reference S0")
    axes[1].set_ylabel("pixels")
    axes[1].set_title(f"n={int(mask.sum())}")
    figure.tight_layout()
    path = OUT / "reports" / f"s1_relative_s0_spp{spp}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return str(path)


def diagnose(levels: dict[int, dict]) -> dict:
    low = levels[1024]["corrected_vs_mean"]["s1"]["p95"]
    high = levels[2048]["corrected_vs_mean"]["s1"]["p95"]
    ratio = sampling_ratio(low, high, 1024, 2048)
    historical_low = levels[1024]["historical"]["uncertainty"]["s1"]["p95_vs_mean_max"]
    ratio_to_historical = None if low in (None, 0) else float(historical_low / low)
    denominator_explains = bool(ratio_to_historical is not None and ratio_to_historical > 5.0)
    return {
        "corrected_s1_p95_1024": low,
        "corrected_s1_p95_2048": high,
        "historical_s1_p95_1024": historical_low,
        "historical_over_corrected_1024": ratio_to_historical,
        "denominator_explains_historical_s1": denominator_explains,
        "sampling": ratio,
        "stop_before_more_spp": bool(ratio["decreases_with_spp"] is False),
        "note": "Historical S1 uses np.maximum(S1_mean, 1e-4). Corrected S1 uses reference S0. The historical stable flag stays false.",
    }


def reconstruct() -> dict:
    if not HISTORICAL_XML.is_file():
        raise RuntimeError(f"missing historical xml {HISTORICAL_XML}")
    write_protocol(OUT / "protocol.md")
    env = environment()
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    if not env["variant_installed"]:
        raise RuntimeError(env)
    import mitsuba as mi

    mi.set_variant(VARIANT)
    folder = OUT / "reference" / "historical_reconstruction"
    log_path = OUT / "reports" / "render_log.jsonl"
    rendered = []
    for job in JOBS:
        for seed in job["seeds"]:
            destination = folder / f"spp{job['spp']}_seed{seed}.exr"
            rendered.append(render_job(job["spp"], seed, destination, log_path))
    levels = {}
    figures = {}
    for job in JOBS:
        levels[job["spp"]] = analyze_level(job["spp"], job["seeds"], folder)
        figures[job["spp"]] = _save_figure(folder, job["spp"], job["seeds"])
    diagnosis = diagnose(levels)
    report = {
        "stage": "historical_reconstruction",
        "environment": env,
        "tolerance": RECONSTRUCTION_ABSOLUTE_TOLERANCE,
        "tolerance_fixed_before_comparison": True,
        "candidate_seeds_excluded": [2101, 2102, 2103],
        "candidate_results": "historical and superseded; not used to qualify spp",
        "spp_loading": "The on-disk XML default remains 16. mi.render(spp=) left IndependentSampler.sample_count at 16, so those files are not the claimed spp. This reconstruction sets the default before load_string and checks sample_count. The film component format of the new EXRs is float32.",
        "renders": rendered,
        "levels": {str(key): value for key, value in levels.items()},
        "diagnosis": diagnosis,
        "figures": figures,
        "spp_frozen": False,
        "dataset_v1_authorized": False,
    }
    (OUT / "evidence.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    _write_metric_audit(report)
    _log(
        "status=diagnosed "
        f"s1_historical_1024={diagnosis['historical_s1_p95_1024']} "
        f"s1_corrected_1024={diagnosis['corrected_s1_p95_1024']} "
        f"s1_corrected_2048={diagnosis['corrected_s1_p95_2048']} "
        f"denominator_explains={diagnosis['denominator_explains_historical_s1']} "
        f"decreases={diagnosis['sampling']['decreases_with_spp']}"
    )
    return report


def _write_metric_audit(report: dict) -> None:
    diagnosis = report["diagnosis"]
    lines = [
        "# Metric audit",
        "",
        "The historical denominator is `np.maximum(reference, 1e-4)`, where `reference` is the mean of the same Stokes component. The corrected denominator is reference S0. Both are reported. The historical `stable: false` result is unchanged.",
        "",
        f"Reconstruction tolerance, fixed before the comparison: absolute difference at most {RECONSTRUCTION_ABSOLUTE_TOLERANCE} on `p95_vs_mean_max`.",
        "",
        f"Denominator explains the historical S1 p95: {diagnosis['denominator_explains_historical_s1']}.",
        f"Corrected S1 p95 at 1024 spp: {diagnosis['corrected_s1_p95_1024']}.",
        f"Corrected S1 p95 at 2048 spp: {diagnosis['corrected_s1_p95_2048']}.",
        f"Sampling check: {json.dumps(diagnosis['sampling'])}.",
        "",
        "Candidate passes from the previous experiment used the component denominator for S1, S2, and S3. They are superseded and do not qualify an spp.",
        "",
    ]
    for spp, level in report["levels"].items():
        lines.append(f"## {spp} spp")
        lines.append("")
        lines.append(f"Reconstruction status: {level['reconstruction_status']}.")
        lines.append(f"Historical recomputed stable: {level['historical_stable_recomputed']}.")
        lines.append(f"Corrected mask count: {level['n_corrected_mask']}. Mask: {level['mask']}.")
        lines.append("")
        lines.append("Comparison with the saved JSON summaries:")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps(level["comparison_to_saved_json"], indent=2))
        lines.append("```")
        lines.append("")
    path = OUT / "metric_audit.md"
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    reconstruct()


if __name__ == "__main__":
    main()

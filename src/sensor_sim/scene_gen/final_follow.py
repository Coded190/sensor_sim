"""The rest of the qualification measurements. Dataset v1 is not generated here."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "dataset" / "qualification" / "final"
PILOT = ROOT / "dataset" / "qualification" / "pilot"


def _load() -> dict:
    return json.loads((OUT / "evidence.json").read_text())


def _save(report: dict) -> None:
    (OUT / "evidence.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


def failure_breakdown() -> dict:
    rows = []
    for path in sorted((PILOT / "table" / "failures").glob("*.json")):
        payload = json.loads(path.read_text())
        text = payload.get("error", "")
        if "cannot_fit" in text:
            kind = "cannot_fit"
        elif "placement_failed" in text:
            kind = "placement_failed"
        elif "do not include" in text:
            kind = "empty_or_missing_shape_id"
        else:
            kind = "other"
        rows.append({"file": path.name, "kind": kind, "error": text})
    counts = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    return {"n": len(rows), "by_kind": counts, "rows": rows}


def packing_stress() -> dict:
    from sensor_sim.scene_gen.assets import build_asset
    from sensor_sim.scene_gen.placement import place_objects

    camera = {"position": [0.0, -0.55, 0.32], "target": [0.0, 0.0, 0.18], "up": [0.0, 0.0, 1.0], "fov_deg": 55.0}
    report = {}
    for count in (1, 2, 3, 4):
        ok = 0
        reasons = {}
        for seed in range(8):
            rng = np.random.default_rng(2000 + count * 20 + seed)
            meshes = []
            for index in range(count):
                family = ("sphere", "capsule", "ellipsoid", "cone")[index % 4]
                meshes.append(build_asset(family, rng, radius=float(rng.uniform(0.03, 0.05)))[0])
            try:
                placed = place_objects(meshes, rng, camera, seed=seed, max_attempts=30)
            except RuntimeError as exc:
                text = str(exc)
                kind = "cannot_fit" if "cannot_fit" in text else "placement_failed"
                reasons[kind] = reasons.get(kind, 0) + 1
                continue
            if len(placed) == count:
                ok += 1
        report[str(count)] = {"successes": ok, "attempts": 8, "failures": reasons}
    return report


def visibility() -> dict:
    from sensor_sim.scene_gen.dataset import load_sample

    rows = []
    for folder in sorted((PILOT / "one" / "samples").glob("scene_*")) + sorted((PILOT / "table" / "samples").glob("scene_*")):
        meta = json.loads((folder / "metadata.json").read_text())
        loaded = load_sample(folder)
        for item in meta["objects"]:
            if item.get("asset_id") == "plane":
                continue
            visible = int(np.count_nonzero(loaded["object_ids"] == item["object_index"]))
            rows.append({"scene_id": meta["scene_id"], "name": item["name"], "family": item.get("asset_id"), "visible_pixels": visible})
    counts = np.array([row["visible_pixels"] for row in rows], dtype=np.float64)
    # The cut is below the bulk of this 128x128 pilot and above a near-empty sliver.
    training_min = 80
    for row in rows:
        pixels = row["visible_pixels"]
        if pixels == 0:
            row["class"] = "fully_occluded"
        elif pixels < training_min:
            row["class"] = "low_visibility"
        else:
            row["class"] = "training_eligible"
    return {
        "n_objects": len(rows),
        "visible_min": float(counts.min()) if len(counts) else None,
        "visible_median": float(np.median(counts)) if len(counts) else None,
        "visible_p10": float(np.percentile(counts, 10)) if len(counts) else None,
        "training_min_pixels_at_128": training_min,
        "rule": "At 128x128, training-eligible means at least 80 object-id pixels. Fewer than 80 is low visibility. Zero is fully occluded. Low-visibility objects stay in the sample.",
        "rows": rows,
    }


def stokes_table(reference, other, mask) -> dict:
    from sensor_sim.stokes import polarization_maps

    def avg(value):
        array = np.asarray(value, dtype=np.float64)
        return array.mean(axis=2) if array.ndim == 3 else array

    ref = [avg(component) for component in reference]
    cur = [avg(component) for component in other]
    use = np.asarray(mask, dtype=bool) & np.isfinite(ref[0]) & (ref[0] > 1e-4)
    denom = np.maximum(ref[0], 1e-4)
    out = {"n": int(use.sum())}
    for index, name in enumerate(("s0", "s1", "s2", "s3")):
        if name == "s0":
            error = np.abs(cur[0] - ref[0]) / denom
        else:
            error = np.abs(cur[index] - ref[index]) / denom
        out[name] = {"median": float(np.median(error[use])), "p95": float(np.percentile(error[use], 95))}
    ref_maps = polarization_maps(*ref)
    cur_maps = polarization_maps(*cur)
    dolp = np.abs(cur_maps["dolp"] - ref_maps["dolp"])
    out["dolp"] = {"median": float(np.median(dolp[use])), "p95": float(np.percentile(dolp[use], 95))}
    aolp_mask = use & (ref_maps["dolp"] >= 0.05) & (cur_maps["dolp"] >= 0.05)
    delta = np.abs(cur_maps["aolp"] - ref_maps["aolp"])
    delta = np.minimum(delta, np.pi - np.minimum(delta, np.pi))
    delta = np.minimum(delta, np.pi - delta)
    out["aolp_deg"] = {
        "median": float(np.degrees(np.median(delta[aolp_mask]))) if aolp_mask.any() else None,
        "p95": float(np.degrees(np.percentile(delta[aolp_mask], 95))) if aolp_mask.any() else None,
        "n": int(aolp_mask.sum()),
    }
    return out


def passes(row: dict) -> bool:
    if row["s0"]["median"] > 0.05 or row["s0"]["p95"] > 0.15:
        return False
    for name in ("s1", "s2", "s3"):
        if row[name]["median"] > 0.02 or row[name]["p95"] > 0.08:
            return False
    if row["dolp"]["median"] > 0.02 or row["dolp"]["p95"] > 0.08:
        return False
    if row["aolp_deg"]["median"] is None or row["aolp_deg"]["median"] > 2.0 or row["aolp_deg"]["p95"] > 10.0:
        return False
    return True


def geometry_compare(sample: Path) -> dict:
    import mitsuba as mi

    mi.set_variant("cuda_ad_spectral_polarized")
    scene = mi.load_file(str(sample / "scene.xml"))
    rows = {}
    for shape in scene.shapes():
        ident = str(shape.id())
        if not ident.startswith("mesh-") or ident == "mesh-ground":
            continue
        name = ident.removeprefix("mesh-")
        obj = sample / "source" / f"{name}.obj"
        if not obj.is_file():
            rows[name] = {"pass": False, "error": "missing source obj"}
            continue
        source = []
        for line in obj.read_text().splitlines():
            if line.startswith("v "):
                source.append([float(value) for value in line.split()[1:4]])
        source = np.asarray(source, dtype=np.float64)
        count = int(shape.vertex_count())
        loaded = np.stack([np.array(shape.vertex_position(index)).reshape(-1)[:3] for index in range(count)])
        if source.shape == loaded.shape:
            error = np.linalg.norm(source - loaded, axis=1)
        else:
            error = np.array([np.linalg.norm(loaded - point, axis=1).min() for point in source[:: max(1, len(source) // 200)]])
        rows[name] = {
            "pass": bool(error.max() <= 1e-4),
            "max_m": float(error.max()),
            "median_m": float(np.median(error)),
            "source_vertices": int(source.shape[0]),
            "mitsuba_vertices": int(count),
        }
    return rows


def convergence(sample: Path) -> dict:
    import mitsuba as mi

    from sensor_sim.scene_gen.dataset import load_sample
    from sensor_sim.scene_gen.render_sample import render_stokes, stack_stokes

    mi.set_variant("cuda_ad_spectral_polarized")
    loaded = load_sample(sample)
    mask = loaded["measurement_mask"]
    stacks = {}
    for spp in (16, 64, 256):
        _scene, channels, _names, _bitmap = render_stokes(sample / "scene.xml", "cuda_ad_spectral_polarized", spp, 0)
        stacks[spp] = [stack_stokes(channels, name) for name in ("S0", "S1", "S2", "S3")]
        print("spp", spp, sample.name, flush=True)
    reference = stacks[256]
    table = {}
    for spp, components in stacks.items():
        table[str(spp)] = stokes_table(reference, components, mask)
        table[str(spp)]["pass"] = passes(table[str(spp)]) if spp != 256 else True
    return table


def rough_gold() -> dict:
    import mitsuba as mi

    from sensor_sim.scene_gen.characterize import measure_bsdf
    from sensor_sim.scene_gen.dataset import channels_to_array
    from sensor_sim.film.images import read_exr

    mi.set_variant("cuda_ad_spectral_polarized")
    sample = ROOT / "dataset" / "samples" / "scene_000000"
    ids = np.squeeze(channels_to_array(read_exr(sample / "ground_truth" / "object_ids.exr")))
    stats = measure_bsdf(
        sample / "scene.xml",
        {"type": "roughconductor", "material": "Au", "distribution": "beckmann", "alpha": 0.1},
        np.rint(ids) == 1,
        "cuda_ad_spectral_polarized",
        256,
        OUT / "gold",
    )
    promote = stats["p95_dolp"] > 0.2 and stats["fraction_above_0_2"] >= 0.05
    stats["decision"] = "primary" if promote else "weak_polarization_stratum"
    stats["spp"] = 256
    stats["denominator"] = "object id 1 and S0 > 1e-4"
    return {key: value for key, value in stats.items() if key != "maps"}


def main() -> None:
    report = _load()
    report["failure_breakdown"] = failure_breakdown()
    report["packing_stress"] = packing_stress()
    report["visibility"] = visibility()
    _save(report)
    sample = OUT / "rerun" / "rerun_269837089"
    report["geometry"] = geometry_compare(sample)
    _save(report)
    print("geometry", report["geometry"], flush=True)
    report["convergence_multi"] = convergence(sample)
    _save(report)
    print("convergence", {key: row.get("pass") for key, row in report["convergence_multi"].items()}, flush=True)
    report["rough_au_256"] = rough_gold()
    _save(report)
    print("gold", report["rough_au_256"].get("decision"), report["rough_au_256"].get("p95_dolp"), flush=True)


if __name__ == "__main__":
    main()

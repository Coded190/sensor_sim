"""Recompute saved convergence decisions and record the asset catalog."""

from __future__ import annotations

import json
from pathlib import Path

from sensor_sim.scene_gen.assets import ASSET_IDS, DRAGON_ID, FAMILIES
from sensor_sim.scene_gen.materials import primary_catalog
from sensor_sim.scene_gen.metric_audit import CANDIDATE_LIMITS, PROSPECTIVE_LIMITS

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "dataset/qualification/convergence_v2/candidate/summary.json"
REFERENCE = ROOT / "dataset/qualification/convergence_v2/reports/reference_qualification.json"
REQUIRED_SCENES = (
    "multi_object",
    "dielectric_1_33",
    "dielectric_1_40",
    "dielectric_1_50",
    "dielectric_1_60",
    "dielectric_1_70",
    "pplastic",
    "copper",
    "rough_copper",
)
PRIMARY_PRESET_COUNT = 8


def _candidate_failures(row: dict) -> list[str]:
    failures = []
    if row.get("noise_floor_subtracted"):
        failures.append("noise floor was subtracted")
    for name in ("s0", "s1", "s2", "s3"):
        limit = CANDIDATE_LIMITS[f"{name}_relative"]
        metric = row[name]
        if metric.get("denominator") != "reference_S0":
            failures.append(f"{name} denominator {metric.get('denominator')}")
        if metric["n"] <= 0 or metric["median"] > limit["median"] or metric["p95"] > limit["p95"]:
            failures.append(f"{name} median {metric['median']} p95 {metric['p95']} n {metric['n']}")
    dolp = row["dolp"]
    limit = CANDIDATE_LIMITS["dolp_absolute"]
    if dolp["n"] <= 0 or dolp["median"] > limit["median"] or dolp["p95"] > limit["p95"]:
        failures.append(f"dolp {dolp}")
    aolp = row["aolp"]
    limit = CANDIDATE_LIMITS["aolp_angular"]
    if row.get("aolp_status") != "measured" or aolp["n"] <= 0 or aolp["median"] > limit["median"] or aolp["p95"] > limit["p95"]:
        failures.append(f"aolp {aolp}")
    return failures


def audit_candidate(document: dict) -> dict:
    failures = []
    levels = document["levels"]
    for spp in ("256", "512", "1024"):
        scenes = levels.get(spp, {})
        for scene in REQUIRED_SCENES:
            renders = scenes.get(scene, {}).get("renders", [])
            seeds = [item.get("seed") for item in renders]
            if seeds != [2101, 2102, 2103]:
                failures.append(f"{scene} spp {spp} seeds {seeds}")
            for item in renders:
                for reason in _candidate_failures(item):
                    failures.append(f"{scene} spp {spp} seed {item.get('seed')}: {reason}")
    return {"pass": not failures, "failures": failures, "scenes": list(REQUIRED_SCENES)}


def audit_reference(document: dict) -> dict:
    failures = []
    scenes = document["reference_spp_scenes"]
    for scene in REQUIRED_SCENES:
        row = scenes.get(scene)
        if row is None:
            failures.append(f"missing reference {scene}")
            continue
        decision = row["decision"]
        if int(decision.get("n_seeds", 0)) < 8 or int(row.get("spp", 0)) != 1024:
            failures.append(f"{scene} seeds {decision.get('n_seeds')} spp {row.get('spp')}")
        for name, check in decision["checks"].items():
            if name in ("s0", "s1", "s2", "s3"):
                limit = PROSPECTIVE_LIMITS[f"{name}_relative"]
            elif name == "dolp":
                limit = PROSPECTIVE_LIMITS["dolp_absolute"]
            else:
                limit = PROSPECTIVE_LIMITS["aolp_angular"]
            if check["median_upper"] > limit["median"] or check["p95_upper"] > limit["p95"] or not check["pass"]:
                failures.append(f"{scene} {name} upper {check}")
    return {"pass": not failures, "failures": failures}


def compatibility_table() -> dict:
    presets = []
    for name, preset in primary_catalog():
        presets.append({"family": name, "preset": preset, "coverage": preset.get("material") != "Au"})
    coverage = [item for item in presets if item["coverage"]]
    pairs = []
    unsupported = [
        {
            "family": DRAGON_ID,
            "reason": "dragon is in ASSET_IDS but not FAMILIES, and Dataset v1 does not request it",
        },
        {
            "preset": {"type": "roughconductor", "material": "Au", "distribution": "beckmann", "alpha": 0.1},
            "reason": "gold is in primary_catalog but is excluded from this coverage matrix and from the production quotas",
        },
    ]
    for family in FAMILIES:
        for item in coverage:
            pairs.append({"asset_family": family, "preset": item["preset"], "supported": True})
    return {
        "asset_families": list(FAMILIES),
        "asset_ids_including_dragon": list(ASSET_IDS),
        "primary_presets": presets,
        "coverage_preset_count": len(coverage),
        "expected_coverage_presets": PRIMARY_PRESET_COUNT,
        "supported_pairs": pairs,
        "unsupported": unsupported,
        "object_counts": [1, 2, 3, 4],
        "placement": "largest first, 5 mm gap, cannot_fit when footprint sum exceeds the floor, no silent drop",
        "assumption": "the generator does not restrict a FAMILIES mesh from using a non-gold primary preset",
    }


def main() -> None:
    candidate = audit_candidate(json.loads(CANDIDATE.read_text()))
    reference = audit_reference(json.loads(REFERENCE.read_text()))
    catalog = compatibility_table()
    out = ROOT / "dataset/qualification/dataset_v1_readiness"
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix_audit.json").write_text(json.dumps({"candidate": candidate, "reference": reference, "catalog": catalog}, indent=2) + "\n")
    print("candidate", candidate["pass"], "failures", len(candidate["failures"]))
    print("reference", reference["pass"], "failures", len(reference["failures"]))
    print("coverage presets", catalog["coverage_preset_count"])
    if not candidate["pass"] or not reference["pass"] or catalog["coverage_preset_count"] != PRIMARY_PRESET_COUNT:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

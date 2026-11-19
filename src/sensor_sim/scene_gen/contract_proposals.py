"""Scene-slot tables for the three unauthorized Dataset v1 proposals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.assets import FAMILIES
from sensor_sim.scene_gen.materials import primary_catalog

OUT = Path("dataset/qualification/dataset_v1_readiness/dataset_v1_contract_proposals")
PRESETS = [preset for _name, preset in primary_catalog() if preset.get("material") != "Au"]


def _split(slots: list[dict], seed: int) -> dict:
    """Within each material, put a deterministic tail into validation and test."""
    rng = np.random.default_rng(seed)
    by_material: dict[str, list[dict]] = {}
    for slot in slots:
        by_material.setdefault(json.dumps(slot["preset"], sort_keys=True), []).append(slot)
    assigned = {"train": [], "validation": [], "test": []}
    for key in sorted(by_material):
        group = list(by_material[key])
        order = rng.permutation(len(group))
        ordered = [group[int(index)] for index in order]
        if len(ordered) < 3:
            assigned["train"].extend(item["slot_id"] for item in ordered)
            continue
        test_n = max(1, round(0.1 * len(ordered)))
        val_n = max(1, round(0.1 * len(ordered)))
        assigned["test"].extend(item["slot_id"] for item in ordered[:test_n])
        assigned["validation"].extend(item["slot_id"] for item in ordered[test_n : test_n + val_n])
        assigned["train"].extend(item["slot_id"] for item in ordered[test_n + val_n :])
    seen = assigned["train"] + assigned["validation"] + assigned["test"]
    if len(seen) != len(set(seen)) or set(seen) != {slot["slot_id"] for slot in slots}:
        raise RuntimeError("scene split is not a partition")
    return assigned


def _minimum() -> list[dict]:
    slots = []
    for index, (family, preset) in enumerate(zip(FAMILIES, PRESETS, strict=True)):
        slots.append(
            {
                "slot_id": f"min_{index:02d}",
                "family": family,
                "preset": preset,
                "object_count": 1,
                "partial_occlusion": False,
                "repeated_asset": False,
                "resolution": [128, 128],
                "visibility_min_pixels": 80,
            }
        )
    slots.append({**slots[0], "slot_id": "min_occlusion", "object_count": 2, "partial_occlusion": True, "family": "sphere"})
    slots.append({**slots[1], "slot_id": "min_repeated", "object_count": 2, "repeated_asset": True, "family": "sphere"})
    slots.append({**slots[2], "slot_id": "min_count3", "object_count": 3, "family": "capsule"})
    slots.append({**slots[3], "slot_id": "min_count4", "object_count": 4, "family": "ellipsoid"})
    return slots


def _recommended() -> list[dict]:
    slots = []
    counts = [1] * 32 + [2] * 16 + [3] * 8 + [4] * 8
    pair_index = 0
    for family in FAMILIES:
        for preset in PRESETS:
            count = counts[pair_index]
            repeated = count == 2 and pair_index < 40
            occlusion = count == 2 and not repeated
            slots.append(
                {
                    "slot_id": f"rec_{pair_index:02d}",
                    "family": family,
                    "preset": preset,
                    "object_count": count,
                    "partial_occlusion": occlusion,
                    "repeated_asset": repeated,
                    "resolution": [128, 128],
                    "visibility_min_pixels": 80,
                }
            )
            pair_index += 1
    return slots


def _larger() -> list[dict]:
    slots = []
    index = 0
    two_and_three = []
    for family in FAMILIES:
        for preset in PRESETS:
            for count in (1, 2, 3, 4):
                slot = {
                    "slot_id": f"large_{index:03d}",
                    "family": family,
                    "preset": preset,
                    "object_count": count,
                    "partial_occlusion": False,
                    "repeated_asset": False,
                    "resolution": [128, 128],
                    "visibility_min_pixels": 80,
                }
                slots.append(slot)
                if count in (2, 3):
                    two_and_three.append(index)
                index += 1
    for offset, slot_index in enumerate(two_and_three):
        if offset < 32:
            slots[slot_index]["partial_occlusion"] = True
        elif offset < 64:
            slots[slot_index]["repeated_asset"] = True
    return slots


def _margins(slots: list[dict]) -> dict:
    materials: dict[str, int] = {}
    families: dict[str, int] = {}
    counts: dict[str, int] = {}
    for slot in slots:
        key = json.dumps(slot["preset"], sort_keys=True)
        materials[key] = materials.get(key, 0) + 1
        families[slot["family"]] = families.get(slot["family"], 0) + 1
        counts[str(slot["object_count"])] = counts.get(str(slot["object_count"]), 0) + 1
    return {
        "total": len(slots),
        "materials_sum": sum(materials.values()),
        "families_sum": sum(families.values()),
        "counts_sum": sum(counts.values()),
        "materials": materials,
        "families": families,
        "counts": counts,
        "partial_occlusion": sum(1 for slot in slots if slot["partial_occlusion"]),
        "repeated_asset": sum(1 for slot in slots if slot["repeated_asset"]),
        "unique_slots": len({slot["slot_id"] for slot in slots}) == len(slots),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proposals = {"minimum_12": _minimum(), "recommended_64": _recommended(), "larger_256": _larger()}
    for name, slots in proposals.items():
        margins = _margins(slots)
        if not (margins["materials_sum"] == margins["families_sum"] == margins["counts_sum"] == margins["total"] and margins["unique_slots"]):
            raise RuntimeError(f"{name} quotas do not sum to the total {margins}")
        split = _split(slots, seed=17011)
        (OUT / f"{name}.json").write_text(json.dumps({"slots": slots, "margins": margins, "split": split, "split_seed": 17011, "resolution": [128, 128]}, indent=2) + "\n")
        print(name, margins["total"], margins["counts"], "occlusion", margins["partial_occlusion"], "repeated", margins["repeated_asset"], flush=True)


if __name__ == "__main__":
    main()

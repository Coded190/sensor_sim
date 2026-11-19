"""Reproducible splits. A held-out key never appears in a training sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _parts(keys: list[str], rng, fractions: tuple[float, float, float]) -> dict[str, set[str]]:
    ordered = list(keys)
    rng.shuffle(ordered)
    count = len(ordered)
    train_end = int(round(fractions[0] * count))
    val_end = train_end + int(round(fractions[1] * count))
    train_end = min(train_end, count)
    val_end = min(max(val_end, train_end), count)
    if count >= 3:
        train_end = min(train_end, count - 2)
        val_end = min(max(val_end, train_end + 1), count - 1)
    return {
        "train": set(ordered[:train_end]),
        "validation": set(ordered[train_end:val_end]),
        "test": set(ordered[val_end:]),
    }


def _keys(record: dict, mode: str) -> set[str]:
    if mode == "scene":
        return {str(record["scene_id"])}
    if mode == "asset_instance":
        return {str(item.get("asset_instance_id", item["instance_id"])) for item in record["objects"]}
    if mode == "asset_family":
        return {str(item["family"]) for item in record["objects"]}
    if mode == "material":
        return {str(item["bsdf"]["type"]) for item in record["objects"]}
    if mode == "material_range":
        found = set()
        for item in record["objects"]:
            ior = item["bsdf"].get("int_ior")
            if ior is None:
                found.add(f"{item['bsdf']['type']}:fixed")
            elif float(ior) < 1.5:
                found.add(f"{item['bsdf']['type']}:below_1_5")
            else:
                found.add(f"{item['bsdf']['type']}:at_least_1_5")
        return found
    raise ValueError(f"unknown split mode {mode}")


def assign_splits(
    records: list[dict],
    mode: str,
    seed: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Put each scene in one split. Holdout keys do not leak into train."""
    rng = np.random.default_rng(int(seed))
    universe = sorted({key for record in records for key in _keys(record, mode)})
    groups = _parts(universe, rng, fractions)
    assigned = {"train": [], "validation": [], "test": []}
    for record in records:
        keys = _keys(record, mode)
        if keys & groups["test"]:
            assigned["test"].append(record["scene_id"])
        elif keys & groups["validation"]:
            assigned["validation"].append(record["scene_id"])
        else:
            assigned["train"].append(record["scene_id"])
    seen = assigned["train"] + assigned["validation"] + assigned["test"]
    if len(seen) != len(set(seen)) or len(seen) != len(records):
        raise RuntimeError("a scene was assigned twice or dropped")
    for record in records:
        if record["scene_id"] not in assigned["train"]:
            continue
        leaked = _keys(record, mode) & groups["test"]
        if leaked:
            raise RuntimeError(f"{record['scene_id']} trains on held-out keys {sorted(leaked)}")
    return assigned


def write_splits(records: list[dict], mode: str, seed: int, directory: Path) -> dict[str, list[str]]:
    """Write ``train.txt``, ``validation.txt``, and ``test.txt``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assigned = assign_splits(records, mode, seed)
    for name, scene_ids in assigned.items():
        (directory / f"{name}.txt").write_text("\n".join(scene_ids) + ("\n" if scene_ids else ""))
    return assigned

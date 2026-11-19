"""The 64-slot Dataset v1 schedule consumed by ``expand_schedule``.

The split is an explicit table. Seed 17011 is recorded with it. The scene
splitter in ``splits.py`` is not used, because an 80/10/10 shuffle of 64
scene keys does not produce 48/8/8 or the material and count margins.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sensor_sim.scene_gen.assets import FAMILIES
from sensor_sim.scene_gen.materials import primary_catalog, require_primary_preset
from sensor_sim.scene_gen.run import expand_schedule

SPLIT_SEED = 17011
SEED_RULE = "stream_seed(base, slot, attempt, name) and child_seeds(scene_seed)"
PLACEMENT = {"gap_m": 0.005, "floor_z": 0.001, "max_objects": 4, "drop_objects": False, "yaw_axis": "Z"}
VISIBILITY = {"resolution": [128, 128], "training_min_object_pixels": 80, "policy_version": 1}


def coverage_presets() -> list[tuple[str, dict]]:
    """Eight non-gold primary presets. Gold stays in the catalog and out of this schedule."""
    rows = []
    for name, preset in primary_catalog():
        if preset.get("material") == "Au":
            continue
        rows.append((name, dict(preset)))
    if len(rows) != 8:
        raise RuntimeError(f"expected 8 coverage presets, found {len(rows)}")
    return rows


def schedule_hash(slots: list[dict]) -> str:
    """Hash every nested generation-affecting field, not only the top-level keys."""
    payload = json.dumps(slots, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _objects(family: str, count: int, kind: str) -> list[dict]:
    if count == 1:
        radii = [0.05]
    elif kind == "occlusion":
        radii = [0.055, 0.05]
    elif kind == "repeated":
        radii = [0.055, 0.048]
    elif count == 3:
        radii = [0.05, 0.046, 0.042]
    elif count == 4:
        radii = [0.05, 0.045, 0.042, 0.04]
    else:
        raise RuntimeError(f"object count {count} is outside 1-4")
    return [{"family": family, "radius": radius} for radius in radii]


def _fill(rows: list[dict], pattern: list[tuple[int, int, str]]) -> None:
    cursor = 0
    for count, number, kind in pattern:
        for _item in range(number):
            row = rows[cursor]
            row["count_objects"] = count
            row["placement_kind"] = kind
            row["require_partial_occlusion"] = kind == "occlusion"
            row["repeated_asset"] = kind == "repeated"
            row["objects"] = _objects(row["family"], count, kind)
            cursor += 1
    if cursor != len(rows):
        raise RuntimeError(f"filled {cursor} rows of {len(rows)}")


def recommended_slots() -> list[dict]:
    """One scene for each supported family and non-gold preset.

    Validation is the diagonal and test is the next column. That puts one
    validation scene and one test scene in every family and every preset.
    Counts are then assigned so the split margins are exact.
    """
    presets = coverage_presets()
    slots = []
    for family_index, family in enumerate(FAMILIES):
        for preset_index, (material, preset) in enumerate(presets):
            if preset_index == family_index:
                split = "validation"
            elif preset_index == (family_index + 1) % 8:
                split = "test"
            else:
                split = "train"
            slots.append(
                {
                    "slot_id": f"slot_{family_index * 8 + preset_index:02d}",
                    "count": 1,
                    "family": family,
                    "material": material,
                    "preset": preset,
                    "split": split,
                    "split_seed": SPLIT_SEED,
                    "placement": dict(PLACEMENT),
                    "visibility": dict(VISIBILITY),
                    "seed_rule": SEED_RULE,
                }
            )
    grouped = {name: [slot for slot in slots if slot["split"] == name] for name in ("validation", "test", "train")}
    _fill(grouped["validation"], [(1, 4, "plain"), (2, 1, "occlusion"), (2, 1, "repeated"), (3, 1, "plain"), (4, 1, "plain")])
    _fill(grouped["test"], [(1, 4, "plain"), (2, 1, "occlusion"), (2, 1, "repeated"), (3, 1, "plain"), (4, 1, "plain")])
    _fill(
        grouped["train"],
        [(1, 24, "plain"), (2, 6, "occlusion"), (2, 6, "repeated"), (3, 6, "plain"), (4, 6, "plain")],
    )
    validate_slots(slots)
    return slots


def validate_slots(slots: list[dict]) -> dict:
    """Reject a schedule the catalogs or the CLI cannot execute."""
    if len(slots) != 64:
        raise RuntimeError(f"schedule has {len(slots)} slots")
    seen = set()
    families: dict[str, int] = {}
    presets: dict[str, int] = {}
    counts: dict[int, int] = {}
    splits = {"train": 0, "validation": 0, "test": 0}
    split_family: dict[tuple[str, str], int] = {}
    split_preset: dict[tuple[str, str], int] = {}
    split_count: dict[tuple[str, int], int] = {}
    occlusion = {"train": 0, "validation": 0, "test": 0}
    repeated = {"train": 0, "validation": 0, "test": 0}
    for slot in slots:
        slot_id = str(slot["slot_id"])
        if slot_id in seen:
            raise RuntimeError(f"duplicate slot id {slot_id}")
        seen.add(slot_id)
        family = str(slot["family"])
        if family not in FAMILIES:
            raise RuntimeError(f"unsupported asset family {family}")
        preset = slot.get("preset")
        _name, row = require_primary_preset(preset)
        if row.get("material") == "Au":
            raise RuntimeError("gold is excluded from Dataset v1")
        if slot.get("material") != _name:
            raise RuntimeError(f"{slot_id} material {slot.get('material')} does not match preset {_name}")
        count = int(slot["count_objects"])
        if count < 1 or count > 4 or int(slot["count"]) != 1:
            raise RuntimeError(f"{slot_id} count {count} is outside 1-4")
        objects = slot.get("objects") or []
        if len(objects) != count or any(item.get("family") != family for item in objects):
            raise RuntimeError(f"{slot_id} per-object families do not match the slot")
        if slot["require_partial_occlusion"] and slot["repeated_asset"]:
            raise RuntimeError(f"{slot_id} is both occlusion and repeated")
        if count == 2 and not (slot["require_partial_occlusion"] or slot["repeated_asset"]):
            raise RuntimeError(f"{slot_id} two-object scene has no measured special condition")
        if count != 2 and (slot["require_partial_occlusion"] or slot["repeated_asset"]):
            raise RuntimeError(f"{slot_id} special condition is not on a two-object scene")
        if slot.get("seed_rule") != SEED_RULE:
            raise RuntimeError(f"{slot_id} seed rule changed")
        families[family] = families.get(family, 0) + 1
        key = json.dumps(row, sort_keys=True)
        presets[key] = presets.get(key, 0) + 1
        counts[count] = counts.get(count, 0) + 1
        split = slot["split"]
        splits[split] += 1
        split_family[(split, family)] = split_family.get((split, family), 0) + 1
        split_preset[(split, key)] = split_preset.get((split, key), 0) + 1
        split_count[(split, count)] = split_count.get((split, count), 0) + 1
        if slot["require_partial_occlusion"]:
            occlusion[split] += 1
        if slot["repeated_asset"]:
            repeated[split] += 1
    if splits != {"train": 48, "validation": 8, "test": 8}:
        raise RuntimeError(f"split totals {splits}")
    if counts != {1: 32, 2: 16, 3: 8, 4: 8}:
        raise RuntimeError(f"count totals {counts}")
    if any(value != 8 for value in families.values()) or any(value != 8 for value in presets.values()):
        raise RuntimeError("family or material totals are not 8")
    for family in FAMILIES:
        if (split_family[("train", family)], split_family[("validation", family)], split_family[("test", family)]) != (6, 1, 1):
            raise RuntimeError(f"family {family} split is not 6/1/1")
    for key in presets:
        if (split_preset[("train", key)], split_preset[("validation", key)], split_preset[("test", key)]) != (6, 1, 1):
            raise RuntimeError("a preset split is not 6/1/1")
    expected_counts = {1: (24, 4, 4), 2: (12, 2, 2), 3: (6, 1, 1), 4: (6, 1, 1)}
    for count, expected in expected_counts.items():
        actual = tuple(split_count[(name, count)] for name in ("train", "validation", "test"))
        if actual != expected:
            raise RuntimeError(f"count {count} split {actual} is not {expected}")
    if tuple(occlusion[name] for name in ("train", "validation", "test")) != (6, 1, 1):
        raise RuntimeError(f"occlusion split {occlusion}")
    if tuple(repeated[name] for name in ("train", "validation", "test")) != (6, 1, 1):
        raise RuntimeError(f"repeated split {repeated}")
    return {
        "scenes": 64,
        "splits": splits,
        "counts": {str(key): value for key, value in counts.items()},
        "families": families,
        "presets": len(presets),
        "partial_occlusion": occlusion,
        "repeated_asset": repeated,
        "occlusion_slot_ids": [slot["slot_id"] for slot in slots if slot["require_partial_occlusion"]],
        "repeated_slot_ids": [slot["slot_id"] for slot in slots if slot["repeated_asset"]],
        "generalization": "scene-disjoint; materials and families appear in train, validation, and test",
    }


def bind_schedule(spec: dict) -> str | None:
    """Check a recorded schedule hash without launching Blender or Mitsuba.

    A missing hash is allowed on a qualification file. Production must carry
    the hash of the expanded nested slots. This function does not write the
    hash back into the spec, so a resume fingerprint stays stable.
    """
    strata = spec.get("strata")
    if not strata:
        return None
    requested = sum(int(item["count"]) for item in strata)
    expanded = expand_schedule(strata, requested)
    digest = schedule_hash(expanded)
    recorded = spec.get("schedule_hash")
    if recorded not in (None, digest):
        raise RuntimeError("schedule hash mismatch; refusing to resume or expand a changed schedule")
    if spec.get("dataset_id") == "dataset_v1_spec" and recorded != digest:
        raise RuntimeError("production schedule hash is missing")
    return digest


def write_proposal(directory: Path) -> dict:
    """Write the candidate table. This does not edit the production YAML."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    slots = recommended_slots()
    expanded = expand_schedule(slots, 64)
    digest = schedule_hash(expanded)
    margins = validate_slots(slots)
    margins["schedule_hash"] = digest
    margins["split_seed"] = SPLIT_SEED
    (directory / "slots.json").write_text(json.dumps(slots, indent=2) + "\n")
    (directory / "expanded_schedule.json").write_text(json.dumps(expanded, indent=2) + "\n")
    (directory / "margins.json").write_text(json.dumps(margins, indent=2) + "\n")
    (directory / "schedule_hash.txt").write_text(digest + "\n")
    return margins


def split_membership(slots: list[dict], completed_ids: list[str]) -> dict[str, list[str]]:
    """Join the prevalidated table to one run's completed ids, in slot order."""
    if len(completed_ids) != len(slots):
        raise RuntimeError(f"completed {len(completed_ids)} ids for {len(slots)} slots")
    groups = {"train": [], "validation": [], "test": []}
    for slot, scene_id in zip(slots, completed_ids):
        suffix = str(scene_id).rsplit("_", 1)[-1]
        if suffix != f"{slots.index(slot):04d}" and not str(scene_id).endswith(f"_{slots.index(slot):04d}"):
            raise RuntimeError(f"{scene_id} is not slot {slots.index(slot)}")
        groups[slot["split"]].append(scene_id)
    combined = groups["train"] + groups["validation"] + groups["test"]
    if len(combined) != len(set(combined)) or set(combined) != set(completed_ids):
        raise RuntimeError("split membership dropped or repeated a scene")
    return groups


def write_split_files(groups: dict[str, list[str]], directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, scenes in groups.items():
        (directory / f"{name}.txt").write_text("".join(f"{scene}\n" for scene in scenes))
    (directory / "README.txt").write_text(
        "Scene-disjoint Dataset v1 split from the prevalidated slot table, seed 17011. "
        "Materials and families are shared across splits. This is not a held-out material or family split.\n"
    )

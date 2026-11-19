"""Validation-only check of the Dataset v1 contract. It does not render or publish."""

from __future__ import annotations

import json

PRODUCTION_DATASET_ID = "dataset_v1_spec"

REQUIRED_FIELDS = (
    "mode",
    "dataset_id",
    "variant",
    "resolution",
    "spp",
    "primary_normal",
    "gap_m",
    "visibility",
    "materials",
    "assets",
    "object_counts",
    "camera",
    "light",
    "integrator",
    "sampler",
    "rfilter",
    "successful_scenes",
    "quotas",
    "seeds",
    "retry",
    "output_schema",
    "ground_truth",
    "splits",
    "run",
    "generation_authorized",
)


def _schema_blockers(spec: dict) -> list[str]:
    blockers = []
    if not isinstance(spec, dict):
        return ["spec is not a mapping"]
    for field in REQUIRED_FIELDS:
        if field not in spec:
            blockers.append(f"missing {field}")
    if spec.get("variant") not in (None, "cuda_ad_spectral_polarized"):
        blockers.append("variant is not cuda_ad_spectral_polarized")
    visibility = spec.get("visibility") or {}
    if "visibility" in spec and visibility.get("training_min_object_pixels") != 80:
        blockers.append("training visibility minimum is not 80 pixels")
    if "gap_m" in spec and spec.get("gap_m") != 0.005:
        blockers.append("gap is not 5 mm")
    if "primary_normal" in spec and spec.get("primary_normal") != "camera_space_shading":
        blockers.append("primary normal is not the camera-space shading normal")
    counts = spec.get("object_counts")
    if counts is not None and [int(value) for value in counts] != [1, 2, 3, 4]:
        blockers.append(f"object counts are {counts}")
    resolution = spec.get("resolution")
    if resolution is not None and list(resolution) != [128, 128]:
        blockers.append("resolution is not 128x128")
    if "generation_authorized" in spec and not isinstance(spec.get("generation_authorized"), bool):
        blockers.append("generation_authorized is not a boolean")
    primary = ((spec.get("materials") or {}).get("primary")) or []
    if primary:
        iors = sorted(float(item["int_ior"]) for item in primary if item.get("type") == "dielectric")
        if iors != [1.33, 1.4, 1.5, 1.6, 1.7]:
            blockers.append(f"dielectric IORs are {iors}")
    splits = spec.get("splits") or {}
    if "splits" in spec:
        for mode in ("scene", "asset_instance", "asset_family", "material", "material_range"):
            if mode not in (splits.get("modes") or []):
                blockers.append(f"split mode {mode} is missing")
    return blockers


def _contract_blockers(spec: dict) -> list[str]:
    blockers = []
    if spec.get("spp") is None:
        blockers.append("spp is null")
    scenes = spec.get("successful_scenes")
    if scenes is None:
        blockers.append("successful scene target is not frozen")
    elif not isinstance(scenes, int) or isinstance(scenes, bool) or scenes <= 0:
        blockers.append("successful_scenes is not a positive integer")
    quotas = spec.get("quotas") or {}
    if scenes and isinstance(quotas, dict) and quotas.get("status") == "not_approved":
        blockers.append("quotas are not approved")
    if isinstance(quotas, dict) and quotas.get("status") == "approved":
        blockers.extend(_approved_quota_blockers(spec, quotas, scenes))
    return blockers


def _approved_quota_blockers(spec: dict, quotas: dict, scenes) -> list[str]:
    """Approved means the expanded strata match the numeric quotas."""
    blockers = []
    strata = spec.get("strata")
    if not isinstance(strata, list) or not strata:
        return ["approved quotas have no strata schedule"]
    try:
        from sensor_sim.scene_gen.run import expand_schedule

        requested = int(scenes) if isinstance(scenes, int) and not isinstance(scenes, bool) else sum(int(item["count"]) for item in strata)
        slots = expand_schedule(strata, requested)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return [f"approved schedule does not expand: {exc}"]
    if len(slots) != 64 or scenes != 64:
        blockers.append(f"approved schedule has {len(slots)} slots, expected 64")
    families: dict[str, int] = {}
    presets: dict[str, int] = {}
    counts: dict[int, int] = {}
    occlusion = 0
    repeated = 0
    slot_ids = []
    for item in slots:
        if not isinstance(item, dict):
            blockers.append("schedule slot is empty")
            continue
        family = str(item.get("family"))
        families[family] = families.get(family, 0) + 1
        preset = json.dumps(item.get("preset"), sort_keys=True, default=str)
        presets[preset] = presets.get(preset, 0) + 1
        count = int(item.get("count_objects") or 0)
        counts[count] = counts.get(count, 0) + 1
        occlusion += int(bool(item.get("require_partial_occlusion")))
        repeated += int(bool(item.get("repeated_asset")))
        slot_ids.append(item.get("slot_id"))
        if count < 1 or count > 4:
            blockers.append(f"slot {item.get('slot_id')} object count {count} is outside 1-4")
    if len(slot_ids) != len(set(slot_ids)):
        blockers.append("schedule slot ids are not unique")
    if any(value != 8 for value in families.values()) or len(families) != 8:
        blockers.append(f"family quotas are {families}")
    if any(value != 8 for value in presets.values()) or len(presets) != 8:
        blockers.append(f"material quotas are {presets}")
    if counts != {1: 32, 2: 16, 3: 8, 4: 8}:
        blockers.append(f"object-count quotas are {counts}")
    if occlusion != 8 or repeated != 8:
        blockers.append(f"occlusion {occlusion} repeated {repeated}")
    recorded = {
        "per_family": quotas.get("per_family"),
        "per_material": quotas.get("per_material"),
        "partial_occlusion": quotas.get("partial_occlusion"),
        "repeated_asset": quotas.get("repeated_asset"),
    }
    if recorded["per_family"] not in (None, 8) or recorded["per_material"] not in (None, 8):
        blockers.append(f"quota record {recorded} does not match 8")
    if quotas.get("partial_occlusion") not in (None, 8) or quotas.get("repeated_asset") not in (None, 8):
        blockers.append("special-slot quota record is not 8")
    if spec.get("schedule_hash") in (None, ""):
        blockers.append("approved schedule hash is missing")
    return blockers


def validate_dataset_contract(spec: dict) -> dict:
    """Classify a spec. This function does not render, publish, or write files."""
    schema = _schema_blockers(spec if isinstance(spec, dict) else {})
    contract = _contract_blockers(spec if isinstance(spec, dict) else {})
    authorized = bool(isinstance(spec, dict) and spec.get("generation_authorized") is True)
    schema_valid = not schema
    contract_complete = schema_valid and not contract
    return {
        "schema_valid": schema_valid,
        "contract_complete": contract_complete,
        "generation_authorized": authorized,
        "ready_to_generate": bool(schema_valid and contract_complete and authorized),
        "schema_blockers": schema,
        "contract_blockers": contract,
        "blockers": schema + contract + ([] if authorized or not isinstance(spec, dict) else []),
        "creates_samples": False,
        "mutates_manifests": False,
    }


def production_generation_blockers(spec: dict) -> list[str]:
    """Reasons a production Dataset v1 spec must not reach Blender or Mitsuba."""
    if not isinstance(spec, dict) or spec.get("dataset_id") != PRODUCTION_DATASET_ID:
        return []
    contract = spec.get("dataset_contract") if isinstance(spec.get("dataset_contract"), dict) else spec
    if contract.get("dataset_id") != PRODUCTION_DATASET_ID:
        return ["dataset contract id is not dataset_v1_spec"]
    report = validate_dataset_contract(contract)
    blockers = []
    if not report["schema_valid"]:
        blockers.extend(report["schema_blockers"])
    if not report["contract_complete"]:
        blockers.extend(report["contract_blockers"])
    if not report["generation_authorized"]:
        blockers.append("generation is not authorized")
    return blockers


def refuse_production_generation(spec: dict) -> None:
    """Stop before export when the production contract is unauthorized or incomplete."""
    blockers = production_generation_blockers(spec)
    if blockers:
        raise RuntimeError("refusing Dataset v1 generation before export: " + "; ".join(blockers))

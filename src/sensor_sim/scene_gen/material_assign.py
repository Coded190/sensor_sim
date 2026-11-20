"""Seeded material assignment. Geometry, placement, material, and wrap use separate seeds."""

from __future__ import annotations

import logging

import numpy as np

from sensor_sim.scene_gen.material_spec import SAMPLING, SCHEMA_VERSION, make_spec, preset, validate_spec

logger = logging.getLogger("sensor_sim.scene_gen.material_assign")

STREAMS = ("geometry", "placement", "material", "wrap")


def rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed)]))


def sample_family(family: str, seed: int, material_id: str) -> dict:
    """Draw one family from its own ranges. The distribution is uniform on those ranges."""
    if family not in SAMPLING and family not in {"conductor"}:
        raise ValueError(f"reason=no_sampling_range family={family}")
    generator = rng_for(seed)
    if family == "conductor":
        parameters = {"material": "Al"}
    elif family == "diffuse":
        low, high = SAMPLING["diffuse"]["reflectance"]
        channels = generator.uniform(low, high, size=3)
        parameters = {"reflectance": [float(channel) for channel in channels]}
    elif family == "pplastic":
        color_low, color_high = SAMPLING["pplastic"]["diffuse_reflectance"]
        alpha_low, alpha_high = SAMPLING["pplastic"]["alpha"]
        ior_low, ior_high = SAMPLING["pplastic"]["int_ior"]
        parameters = {
            "diffuse_reflectance": [float(channel) for channel in generator.uniform(color_low, color_high, size=3)],
            "int_ior": float(generator.uniform(ior_low, ior_high)),
            "ext_ior": 1.0,
            "alpha": float(generator.uniform(alpha_low, alpha_high)),
            "distribution": "beckmann",
        }
    elif family == "dielectric":
        low, high = SAMPLING["dielectric"]["int_ior"]
        parameters = {"int_ior": float(generator.uniform(low, high)), "ext_ior": 1.0}
    elif family == "roughdielectric":
        alpha = SAMPLING["roughdielectric"]["alpha"]
        ior = SAMPLING["roughdielectric"]["int_ior"]
        parameters = {"int_ior": float(generator.uniform(*ior)), "ext_ior": 1.0, "alpha": float(generator.uniform(*alpha)), "distribution": "beckmann"}
    elif family == "roughconductor":
        low, high = SAMPLING["roughconductor"]["alpha"]
        parameters = {"material": "Al", "alpha": float(generator.uniform(low, high)), "distribution": "beckmann"}
    else:
        row = SAMPLING["principled"]
        parameters = {
            "base_color": [float(channel) for channel in generator.uniform(0.15, 0.80, size=3)],
            "roughness": float(generator.uniform(*row["roughness"])),
            "metallic": float(generator.uniform(*row["metallic"])),
            "anisotropic": float(generator.uniform(*row["anisotropic"])),
            "spec_trans": float(generator.uniform(*row["spec_trans"])),
            "ior": float(generator.uniform(*row["ior"])),
        }
    spec = make_spec(material_id, family, parameters, seed=int(seed), sampled={"distribution": "uniform", "ranges": SAMPLING.get(family, {}), "units": "linear_rgb_and_mitsuba_alpha"})
    logger.info("material_assign event=sample material_id=%s family=%s seed=%s", material_id, family, seed)
    return spec


def assign_fixed(material_id: str, preset_name: str) -> dict:
    spec = preset(preset_name)
    spec["material_id"] = material_id
    return validate_spec(spec)


def assignment_record(instance_id: str, component_id: str, spec: dict, seeds: dict) -> dict:
    if set(seeds) != set(STREAMS):
        raise ValueError(f"reason=seeds keys={sorted(seeds)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance_id,
        "component_id": component_id,
        "material_id": spec["material_id"],
        "spec": spec,
        "seeds": {name: int(seeds[name]) for name in STREAMS},
    }


def assign_components(components: list[dict], material_seed: int, texture_sets: dict | None = None) -> list[dict]:
    """Assign one material per component. Only the material seed affects the choice.

    ``components`` entries have ``instance_id``, ``component_id``, and ``families``.
    A texture set, when named, is copied onto that component's specification.
    """
    texture_sets = texture_sets or {}
    records = []
    for index, component in enumerate(components):
        families = list(component["families"])
        if not families:
            raise ValueError(f"reason=no_families component={component['component_id']}")
        component_seed = int(rng_for(material_seed).integers(0, 2**31 - 1))
        choice = int(rng_for(component_seed + index + 1).integers(0, len(families)))
        family = families[choice]
        spec = sample_family(family, int(component_seed + index + 17), f"{component['component_id']}_{family}")
        options = list(component.get("texture_sets") or [])
        texture_set = component.get("texture_set")
        if options:
            texture_set = options[int(rng_for(component_seed + index + 5).integers(0, len(options)))]
        if texture_set:
            textures = [dict(item) for item in texture_sets[texture_set]]
            transforms = component.get("uv_transforms") or []
            if transforms:
                chosen = list(transforms[int(rng_for(component_seed + index + 9).integers(0, len(transforms)))])
                for item in textures:
                    item["uv_transform"] = chosen
            spec = dict(spec)
            spec["textures"] = textures
            spec = validate_spec(spec)
            sampled = dict(spec.get("sampled") or {})
            sampled["texture_set"] = texture_set
            sampled["uv_transform"] = list(textures[0].get("uv_transform", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
            sampled["normal_strength"] = float(textures[0].get("normal_strength", 1.0))
            spec["sampled"] = sampled
        seeds = {
            "geometry": int(component.get("geometry_seed", 1)),
            "placement": int(component.get("placement_seed", 2)),
            "material": int(material_seed),
            "wrap": int(component.get("wrap_seed", 3)),
        }
        record = assignment_record(component["instance_id"], component["component_id"], spec, seeds)
        records.append(record)
        logger.info(
            "material_assign event=component instance=%s component=%s family=%s material_seed=%s",
            component["instance_id"],
            component["component_id"],
            family,
            material_seed,
        )
    return records


def replay_assignment(record: dict) -> dict:
    """Restore one saved component assignment without drawing the seed again."""
    spec = validate_spec(record["spec"])
    if spec["parameters"] != record["spec"]["parameters"] or spec.get("textures") != record["spec"].get("textures"):
        raise ValueError("reason=replay_mismatch")
    logger.info("material_assign event=replay_component component=%s result=ok", record["component_id"])
    return spec


def replay_spec(saved: dict) -> dict:
    """Rebuild the specification from the saved sample. The seed is not drawn again."""
    spec = validate_spec(saved)
    if saved.get("sampled") is not None and spec["parameters"] != saved["parameters"]:
        raise ValueError("reason=replay_mismatch")
    logger.info("material_assign event=replay material_id=%s result=ok", spec["material_id"])
    return spec

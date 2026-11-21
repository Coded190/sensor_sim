"""Choose a Mitsuba BSDF from an audited source material.

``pplastic`` is a polarized rough-specular model plus an attenuated diffuse
lobe. Mitsuba documents that it is not energy conserving. It is not the
default for every non-glass surface.

Polarization-aware families on this pin are ``pplastic``, ``dielectric``,
``conductor``, and ``roughconductor``. ``thindielectric`` and
``roughdielectric`` are not polarization-aware here. ``diffuse`` depolarizes.
``measured_polarized`` is used only when a measured pBRDF for that material
is actually available.
"""

from __future__ import annotations


def choose_material(audit: dict) -> dict:
    """Return the Mitsuba family and what the choice preserves or drops.

    ``audit`` keys: ``name``, ``metallic``, ``roughness``, ``transmission``,
    ``ior``, ``thin``, ``emission``, ``base_color_linked``, ``roughness_linked``,
    ``normal_linked``, ``bump_linked``, ``displacement_linked``,
    ``measured_pbrdf``.
    """
    name = str(audit.get("name") or "")
    metallic = float(audit.get("metallic") or 0.0)
    roughness = float(audit.get("roughness") or 0.0)
    transmission = float(audit.get("transmission") or 0.0)
    thin = bool(audit.get("thin"))
    measured = bool(audit.get("measured_pbrdf"))
    glass_name = "glass" in name.lower() or "shader_lamp_bulb" in name.lower()
    lost = []
    approximated = []
    preserved = ["base_color" if audit.get("base_color_linked") or audit.get("base_color") else "base_color_constant"]
    if audit.get("roughness_linked") or "roughness" in audit:
        preserved.append("roughness")
    if audit.get("bump_linked") and not audit.get("normal_linked"):
        lost.append("bump_is_not_a_geometric_displacement")
    if audit.get("displacement_linked"):
        lost.append("shader_displacement_not_in_mesh")
    if measured:
        family = "measured_polarized"
        polarization = "polarization_aware"
        note = "measured pBRDF supplied for this source material"
    elif glass_name or transmission > 0.2:
        if thin:
            family = "thindielectric"
            polarization = "depolarizing"
            note = "thin sheet. thindielectric is not polarization-aware on this pin"
        elif roughness > 0.15:
            family = "roughdielectric"
            polarization = "depolarizing"
            note = "rough dielectric interface. roughdielectric is not polarized on this pin"
            lost.append("polarized_rough_transmission")
        else:
            family = "dielectric"
            polarization = "polarization_aware"
            note = "smooth solid dielectric interface"
        if audit.get("base_color_linked"):
            lost.append("base_color_bitmap_not_applied_to_dielectric")
    elif metallic > 0.5:
        family = "diffuse"
        polarization = "depolarizing"
        note = "metalness is not an optical constant. conductor preset was not assumed"
        lost.append("metallic_optical_constants")
        approximated.append("appearance_as_depolarizing_diffuse")
    elif roughness >= 0.85 and not audit.get("coat"):
        family = "diffuse"
        polarization = "depolarizing"
        note = "matte surface. diffuse depolarizes"
    else:
        family = "pplastic"
        polarization = "polarization_aware"
        note = "coated dielectric approximation. pplastic is not energy conserving and is not a radiometric claim"
        approximated.append("pplastic_not_energy_conserving")
    if float(audit.get("emission") or 0.0) > 0.0:
        lost.append("emission")
    quantitative = family in {"dielectric", "conductor", "roughconductor", "measured_polarized"} and not approximated
    return {
        "family": family,
        "polarization": polarization,
        "qualitative_ok": True,
        "quantitative_polarization": quantitative,
        "preserved": preserved,
        "approximated": approximated,
        "lost": lost,
        "note": note,
        "ior": float(audit.get("ior") or 1.5),
        "roughness": roughness,
        "metallic": metallic,
    }


def respec(spec: dict, choice: dict) -> dict:
    """Rebuild one packaged spec with the audited BSDF. Textures the family cannot use are dropped."""
    from sensor_sim.scene_gen.light_units import polarization_role
    from sensor_sim.scene_gen.material_spec import make_spec

    family = choice["family"]
    parameters = spec.get("parameters") or {}
    reflectance = parameters.get("diffuse_reflectance") or parameters.get("reflectance") or [0.5, 0.5, 0.5]
    alpha = float(parameters.get("alpha", max(float(choice.get("roughness") or 0.0) ** 2, 1e-4)))
    ior = max(float(choice.get("ior") or parameters.get("int_ior") or 1.5), 1.0)
    allowed = {
        "diffuse": {"base_color", "normal_opengl"},
        "pplastic": {"base_color", "alpha_mitsuba", "roughness_blender", "normal_opengl"},
        "roughdielectric": {"alpha_mitsuba", "roughness_blender", "normal_opengl"},
        "roughconductor": {"alpha_mitsuba", "roughness_blender", "normal_opengl"},
        "dielectric": {"normal_opengl"},
        "thindielectric": set(),
        "conductor": set(),
        "measured_polarized": {"base_color", "normal_opengl"},
    }.get(family, set())
    textures = [item for item in (spec.get("textures") or []) if item.get("role") in allowed]
    if family == "diffuse":
        built = {"reflectance": reflectance}
    elif family == "pplastic":
        built = {"diffuse_reflectance": reflectance, "int_ior": 1.5, "ext_ior": 1.0, "alpha": alpha, "distribution": "beckmann"}
    elif family == "roughdielectric":
        built = {"int_ior": ior, "ext_ior": 1.0, "alpha": max(alpha, 1e-4), "distribution": "beckmann"}
    elif family in {"dielectric", "thindielectric"}:
        built = {"int_ior": ior, "ext_ior": 1.0}
    elif family == "conductor":
        built = {"material": choice.get("conductor") or "Al"}
    elif family == "roughconductor":
        built = {"material": choice.get("conductor") or "Al", "alpha": max(alpha, 1e-4), "distribution": "beckmann"}
    else:
        built = {"reflectance": reflectance}
        family = "diffuse"
        textures = [item for item in textures if item.get("role") in {"base_color", "normal_opengl"}]
    rebuilt = make_spec(spec.get("material_id", "region"), family, built, textures=textures)
    rebuilt["binding_status"] = "approximated" if choice.get("approximated") else "bound"
    rebuilt["polarization"] = polarization_role(family)
    rebuilt["displacement_mode"] = spec.get("displacement_mode", "NONE")
    rebuilt["metallic_status"] = spec.get("metallic_status", "constant_zero")
    rebuilt["choice"] = {
        "family": choice["family"],
        "note": choice["note"],
        "preserved": choice["preserved"],
        "approximated": choice["approximated"],
        "lost": choice["lost"],
        "quantitative_polarization": choice["quantitative_polarization"],
        "qualitative_ok": choice["qualitative_ok"],
    }
    return rebuilt

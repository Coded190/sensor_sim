"""Canonical material specification for Blender preview and Mitsuba renders.

Reflectance is linear RGB. It is not a measured spectrum. The scientific
renderer builds BSDFs from this specification. The Blender add-on is a
separate transfer path.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sensor_sim.scene_gen.tabletop_render import mitsuba_bsdf
from sensor_sim.scene_gen.task_schema import TaskError

logger = logging.getLogger("sensor_sim.scene_gen.material_spec")

SCHEMA_VERSION = "material_spec_v1"
COLOR_SPACE = "linear_rgb"
FLOAT_TOLERANCE = 1e-4
ROUGHNESS_NOTE = (
    "Blender Principled roughness stays in the native 0-1 range. "
    "For Beckmann plastic, glossy, and conductor preview mapping, "
    "Mitsuba alpha = Blender roughness squared, and Blender roughness = sqrt(Mitsuba alpha). "
    "A source roughness map is converted once. A map that already stores alpha is not converted again."
)
TEXTURE_ROLES = frozenset({"base_color", "roughness_blender", "alpha_mitsuba", "normal_opengl"})
TEXTURE_COLOR_SPACES = frozenset({"srgb", "non_color"})
INPUT_CLASSES = frozenset({"authored_appearance", "procedural_parameter", "measured_property", "assumed_preset"})

_COLOR_FIELDS = {"reflectance", "diffuse_reflectance", "base_color"}
_FAMILY_FIELDS = {
    "diffuse": {"reflectance"},
    "pplastic": {"diffuse_reflectance", "int_ior", "ext_ior", "alpha", "distribution"},
    "dielectric": {"int_ior", "ext_ior"},
    "roughdielectric": {"int_ior", "ext_ior", "alpha", "distribution"},
    "thindielectric": {"int_ior", "ext_ior"},
    "conductor": {"material"},
    "roughconductor": {"material", "alpha", "distribution"},
    "principled": {"base_color", "roughness", "metallic", "anisotropic", "spec_trans", "ior"},
}
# Ranges stay away from 0/1 reflectance and from alpha so large it hides the family.
SAMPLING = {
    "diffuse": {"reflectance": (0.15, 0.80)},
    "pplastic": {"diffuse_reflectance": (0.20, 0.70), "alpha": (0.03, 0.35), "int_ior": (1.40, 1.60)},
    "dielectric": {"int_ior": (1.33, 1.70)},
    "roughdielectric": {"alpha": (0.03, 0.25), "int_ior": (1.40, 1.60)},
    "roughconductor": {"alpha": (0.03, 0.30)},
    "principled": {"roughness": (0.08, 0.55), "metallic": (0.0, 1.0), "anisotropic": (0.0, 0.4), "spec_trans": (0.0, 0.6), "ior": (1.33, 1.60)},
}
CONDUCTOR_PRESETS = ("Al", "Cu", "Au")
_DIRECT = {"diffuse", "pplastic", "dielectric", "conductor", "roughconductor"}


class MaterialSpecError(TaskError):
    """A material specification or BSDF mapping is not usable."""


def _color(value, name: str) -> list[float]:
    channels = [float(channel) for channel in value]
    if len(channels) != 3 or any(channel < 0.0 or channel > 1.0 or not np.isfinite(channel) for channel in channels):
        raise MaterialSpecError(f"reason=reflectance_range field={name}")
    return channels


def validate_spec(spec: dict) -> dict:
    """Reject fields that do not belong to the family and values outside that family's range."""
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise MaterialSpecError("reason=schema_version")
    family = spec.get("family")
    if family not in _FAMILY_FIELDS:
        raise MaterialSpecError(f"reason=family value={family}")
    if spec.get("color_space") not in {None, COLOR_SPACE}:
        raise MaterialSpecError("reason=color_space")
    parameters = spec.get("parameters")
    if not isinstance(parameters, dict):
        raise MaterialSpecError("reason=parameters")
    unknown = set(parameters) - _FAMILY_FIELDS[family]
    if unknown or "metallic" in parameters and family in {"dielectric", "roughdielectric", "thindielectric", "diffuse", "pplastic"}:
        raise MaterialSpecError(f"reason=family_field family={family} fields={sorted(unknown or {'metallic'})}")
    missing = _FAMILY_FIELDS[family] - set(parameters)
    if missing:
        raise MaterialSpecError(f"reason=missing_fields family={family} fields={sorted(missing)}")
    cleaned = dict(parameters)
    for name in _COLOR_FIELDS & set(cleaned):
        cleaned[name] = _color(cleaned[name], name)
    for name in ("int_ior", "ext_ior", "ior"):
        if name in cleaned and (not np.isfinite(cleaned[name]) or float(cleaned[name]) < 1.0):
            raise MaterialSpecError(f"reason=ior field={name}")
    for name in ("alpha", "roughness", "anisotropic"):
        if name in cleaned and (not np.isfinite(cleaned[name]) or float(cleaned[name]) < 0.0):
            raise MaterialSpecError(f"reason=roughness field={name}")
    if "metallic" in cleaned and not 0.0 <= float(cleaned["metallic"]) <= 1.0:
        raise MaterialSpecError("reason=metallic")
    if "spec_trans" in cleaned and not 0.0 <= float(cleaned["spec_trans"]) <= 1.0:
        raise MaterialSpecError("reason=transmission")
    if "distribution" in cleaned and cleaned["distribution"] not in {"beckmann", "ggx"}:
        raise MaterialSpecError("reason=distribution")
    if "material" in cleaned and cleaned["material"] not in CONDUCTOR_PRESETS:
        raise MaterialSpecError(f"reason=conductor value={cleaned['material']}")
    if "ext_ior" in cleaned and family != "principled" and abs(float(cleaned["ext_ior"]) - 1.0) > FLOAT_TOLERANCE:
        raise MaterialSpecError("reason=ext_ior_not_air")
    spec = dict(spec)
    spec["parameters"] = cleaned
    spec["color_space"] = COLOR_SPACE
    spec["roughness_convention"] = ROUGHNESS_NOTE
    spec["textures"] = _textures(spec.get("textures") or [])
    return spec


def blender_roughness_to_alpha(roughness: float) -> float:
    """Forward conversion. ``alpha = roughness²``. The input is Blender's 0-1 roughness."""
    value = float(roughness)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise MaterialSpecError(f"reason=roughness_out_of_range value={roughness}")
    return value * value


def alpha_to_blender_roughness(alpha: float) -> float:
    """Inverse preview conversion. ``roughness = sqrt(alpha)``."""
    value = float(alpha)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise MaterialSpecError(f"reason=alpha_out_of_range value={alpha}")
    return float(np.sqrt(value))


def roughness_image_to_alpha(image: np.ndarray) -> np.ndarray:
    """Convert one source roughness image to Mitsuba alpha. Each pixel is converted once."""
    array = np.asarray(image, dtype=np.float64)
    channel = array[..., 0] if array.ndim == 3 else array
    if not np.isfinite(channel).all() or np.any(channel < 0.0) or np.any(channel > 1.0):
        raise MaterialSpecError("reason=roughness_image_out_of_range")
    return channel * channel


def _textures(records) -> list[dict]:
    if not isinstance(records, list):
        raise MaterialSpecError("reason=textures")
    cleaned = []
    for record in records:
        if not isinstance(record, dict):
            raise MaterialSpecError("reason=texture_record")
        role = record.get("role")
        color_space = record.get("color_space")
        parameter_class = record.get("parameter_class")
        path = record.get("path")
        if role not in TEXTURE_ROLES or color_space not in TEXTURE_COLOR_SPACES or parameter_class not in INPUT_CLASSES:
            raise MaterialSpecError(f"reason=texture_field role={role} color_space={color_space}")
        if not isinstance(path, str) or path == "" or path.startswith("/") or path.startswith("local_review/") or ".." in Path(path).parts:
            raise MaterialSpecError(f"reason=texture_path path={path}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise MaterialSpecError("reason=texture_hash")
        resolution = record.get("resolution")
        if not isinstance(resolution, list) or len(resolution) != 2 or min(int(resolution[0]), int(resolution[1])) <= 0:
            raise MaterialSpecError("reason=texture_resolution")
        cleaned.append({
            "role": role,
            "path": path,
            "color_space": color_space,
            "channel": record.get("channel", "rgb" if role != "roughness_blender" else "r"),
            "resolution": [int(resolution[0]), int(resolution[1])],
            "sha256": digest,
            "parameter_class": parameter_class,
            "uv_transform": record.get("uv_transform", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "normal_strength": float(record.get("normal_strength", 1.0)),
            "normal_orientation": record.get("normal_orientation", "opengl_tangent"),
            "filter_type": record.get("filter_type", "bilinear"),
            "wrap_mode": record.get("wrap_mode", "repeat"),
            "raw": bool(record.get("raw", color_space == "non_color")),
        })
    return cleaned


def make_spec(material_id: str, family: str, parameters: dict, *, preset: str | None = None, seed: int | None = None, sampled: dict | None = None, textures: list | None = None) -> dict:
    spec = validate_spec({
        "schema_version": SCHEMA_VERSION,
        "material_id": material_id,
        "family": family,
        "preset": preset,
        "color_space": COLOR_SPACE,
        "parameters": parameters,
        "textures": textures or [],
        "seed": seed,
        "sampled": sampled,
    })
    logger.info("material_spec event=validate material_id=%s family=%s", material_id, family)
    return spec


def preset(name: str) -> dict:
    """Named presets used by the qualification comparisons."""
    table = {
        "matte_gray": ("diffuse", {"reflectance": [0.50, 0.50, 0.50]}),
        "matte_red": ("diffuse", {"reflectance": [0.70, 0.12, 0.10]}),
        "matte_blue": ("diffuse", {"reflectance": [0.12, 0.22, 0.65]}),
        "plastic_smooth": ("pplastic", {"diffuse_reflectance": [0.45, 0.45, 0.42], "int_ior": 1.5, "ext_ior": 1.0, "alpha": 0.04, "distribution": "beckmann"}),
        "plastic_rough": ("pplastic", {"diffuse_reflectance": [0.45, 0.45, 0.42], "int_ior": 1.5, "ext_ior": 1.0, "alpha": 0.28, "distribution": "beckmann"}),
        "glass_145": ("dielectric", {"int_ior": 1.45, "ext_ior": 1.0}),
        "glass_160": ("dielectric", {"int_ior": 1.60, "ext_ior": 1.0}),
        "rough_glass": ("roughdielectric", {"int_ior": 1.5, "ext_ior": 1.0, "alpha": 0.08, "distribution": "beckmann"}),
        "thin_sheet": ("thindielectric", {"int_ior": 1.5, "ext_ior": 1.0}),
        "aluminum": ("conductor", {"material": "Al"}),
        "aluminum_rough": ("roughconductor", {"material": "Al", "alpha": 0.12, "distribution": "beckmann"}),
        "principled_plastic": ("principled", {"base_color": [0.20, 0.45, 0.70], "roughness": 0.35, "metallic": 0.0, "anisotropic": 0.25, "spec_trans": 0.0, "ior": 1.46}),
        "principled_metal": ("principled", {"base_color": [0.85, 0.55, 0.20], "roughness": 0.22, "metallic": 0.9, "anisotropic": 0.15, "spec_trans": 0.0, "ior": 1.5}),
        "principled_glass": ("principled", {"base_color": [0.90, 0.95, 0.90], "roughness": 0.05, "metallic": 0.0, "anisotropic": 0.0, "spec_trans": 0.85, "ior": 1.45}),
    }
    if name not in table:
        raise MaterialSpecError(f"reason=preset value={name}")
    family, parameters = table[name]
    return make_spec(f"preset_{name}", family, parameters, preset=name)


def to_mitsuba(spec: dict, *, package_root: Path | None = None) -> dict:
    """Build one Mitsuba BSDF dict. An unsupported family raises. Nothing is substituted.

    Texture bitmaps are resolved relative to ``package_root``. A missing file
    raises. A color bitmap is not marked raw. A non-color bitmap is raw.
    ``filter_type`` is bilinear because spectral variants fall back from
    trilinear and anisotropic filtering for color textures.
    """
    spec = validate_spec(spec)
    family = spec["family"]
    parameters = dict(spec["parameters"])
    textures = _resolved_textures(spec.get("textures") or [], package_root)
    _reject_illegal_textures(family, textures)
    if family in _DIRECT:
        try:
            mapped = mitsuba_bsdf({"type": family, **parameters})
        except TaskError as exc:
            raise MaterialSpecError(str(exc)) from exc
    elif family == "roughdielectric":
        mapped = {"type": "roughdielectric", "int_ior": float(parameters["int_ior"]), "ext_ior": float(parameters["ext_ior"]), "distribution": parameters["distribution"], "alpha": float(parameters["alpha"])}
    elif family == "thindielectric":
        mapped = {"type": "thindielectric", "int_ior": float(parameters["int_ior"]), "ext_ior": float(parameters["ext_ior"])}
    elif family == "principled":
        mapped = {
            "type": "principled",
            "base_color": {"type": "rgb", "value": parameters["base_color"]},
            "roughness": float(parameters["roughness"]),
            "metallic": float(parameters["metallic"]),
            "anisotropic": float(parameters["anisotropic"]),
            "spec_trans": float(parameters["spec_trans"]),
            "eta": float(parameters["ior"]),
        }
    else:
        raise MaterialSpecError(f"reason=unsupported_bsdf type={family}")
    if mapped.get("type") != family and not (family in _DIRECT and mapped.get("type") == family):
        raise MaterialSpecError(f"reason=substituted requested={family} got={mapped.get('type')}")
    mapped = _apply_textures(mapped, family, textures)
    logger.info("material_spec event=mitsuba material_id=%s family=%s textures=%d", spec["material_id"], family, len(textures))
    return mapped


FILTER_NOTE = (
    "bilinear is requested. Mitsuba documents that trilinear and anisotropic "
    "filtering fall back to bilinear for color textures in spectral variants. "
    "The qualification probe records the filter that this pin actually accepts."
)


def bitmap_settings(record: dict, filename: Path) -> dict:
    """Bitmap plugin settings. Color images are linearized. Data images are raw.

    Only Mitsuba bitmap properties are included. Filter notes stay outside the
    plugin dictionary so ``load_dict`` does not see unknown keys.
    """
    raw = record["color_space"] == "non_color"
    plugin = {
        "type": "bitmap",
        "filename": str(filename),
        "raw": raw,
        "filter_type": "bilinear",
        "wrap_mode": str(record.get("wrap_mode", "repeat")),
    }
    transform = [float(value) for value in record.get("uv_transform", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])]
    if transform != [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]:
        plugin["to_uv"] = _uv_transform(transform)
    strength = float(record.get("normal_strength", 1.0))
    if record["role"] == "normal_opengl" and abs(strength - 1.0) > 1e-8:
        raise MaterialSpecError(f"reason=normal_strength_not_applied value={strength}")
    return plugin


def _uv_transform(transform: list[float]):
    """Affine UV map ``u' = a u + b v + c``, ``v' = d u + e v + f``."""
    import mitsuba as mi

    from sensor_sim.scene_gen.indoor_mitsuba import require_variant

    require_variant()
    a_value, b_value, c_value, d_value, e_value, f_value = transform
    matrix = np.array(
        [
            [a_value, b_value, 0.0, c_value],
            [d_value, e_value, 0.0, f_value],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return mi.ScalarTransform4f(matrix)


def _resolved_textures(records: list[dict], package_root: Path | None) -> list[dict]:
    if not records:
        return []
    if package_root is None:
        raise MaterialSpecError("reason=package_root")
    found = []
    for record in records:
        path = Path(package_root) / record["path"]
        if not path.is_file():
            raise MaterialSpecError(f"reason=missing_texture path={record['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise MaterialSpecError(f"reason=texture_hash path={record['path']}")
        found.append({"record": record, "path": path})
    return found


def _reject_illegal_textures(family: str, textures: list[dict]) -> None:
    roles = {item["record"]["role"] for item in textures}
    if "base_color" in roles and family in {"conductor", "roughconductor", "dielectric", "roughdielectric", "thindielectric"}:
        raise MaterialSpecError(f"reason=base_color_unsupported family={family}")
    if roles & {"roughness_blender", "alpha_mitsuba"} and family in {"diffuse", "dielectric", "conductor", "thindielectric"}:
        raise MaterialSpecError(f"reason=roughness_unsupported family={family}")


def _apply_textures(mapped: dict, family: str, textures: list[dict]) -> dict:
    by_role = {item["record"]["role"]: item for item in textures}
    if "base_color" in by_role:
        bitmap = bitmap_settings(by_role["base_color"]["record"], by_role["base_color"]["path"])
        if family == "diffuse":
            mapped["reflectance"] = bitmap
        elif family == "pplastic":
            mapped["diffuse_reflectance"] = bitmap
        else:
            raise MaterialSpecError(f"reason=base_color_unsupported family={family}")
    if "alpha_mitsuba" in by_role:
        if family not in {"pplastic", "roughconductor", "roughdielectric"}:
            raise MaterialSpecError(f"reason=roughness_unsupported family={family}")
        mapped["alpha"] = bitmap_settings(by_role["alpha_mitsuba"]["record"], by_role["alpha_mitsuba"]["path"])
    elif "roughness_blender" in by_role:
        raise MaterialSpecError("reason=unresolved_roughness_map")
    if "normal_opengl" in by_role:
        normal = by_role["normal_opengl"]
        mapped = {"type": "normalmap", "normalmap": bitmap_settings(normal["record"], normal["path"]), "bsdf": mapped}
    return mapped


def blender_recipe(spec: dict) -> dict:
    """Node sockets that the transfer probe assigns. Values are the requested ones, not exporter defaults."""
    spec = validate_spec(spec)
    family = spec["family"]
    parameters = spec["parameters"]
    if family == "diffuse":
        return {"node": "BSDF_DIFFUSE", "sockets": {"Color": parameters["reflectance"], "Roughness": 0.0}}
    if family == "dielectric":
        return {"node": "BSDF_GLASS", "sockets": {"Color": [1.0, 1.0, 1.0], "Roughness": 0.0, "IOR": float(parameters["int_ior"])}}
    if family == "roughdielectric":
        roughness = alpha_to_blender_roughness(parameters["alpha"])
        return {"node": "BSDF_GLASS", "sockets": {"Color": [1.0, 1.0, 1.0], "Roughness": roughness, "IOR": float(parameters["int_ior"])}, "note": "Blender roughness is the square root of Mitsuba alpha"}
    if family == "thindielectric":
        return {"node": "BSDF_GLASS", "sockets": {"Color": [1.0, 1.0, 1.0], "Roughness": 0.0, "IOR": 1.0}, "note": "IOR 1 and roughness 0 is the exporter path to thindielectric"}
    if family in {"conductor", "roughconductor"}:
        roughness = 0.0 if family == "conductor" else alpha_to_blender_roughness(parameters["alpha"])
        return {"node": "BSDF_GLOSSY", "sockets": {"Color": [0.9, 0.9, 0.9], "Roughness": roughness, "Anisotropy": 0.0}, "note": "Glossy color is not the Mitsuba conductor preset name"}
    if family == "pplastic":
        return {"node": "BSDF_PRINCIPLED", "sockets": {"Base Color": parameters["diffuse_reflectance"], "Roughness": alpha_to_blender_roughness(parameters["alpha"]), "Metallic": 0.0, "Anisotropic": 0.0, "Transmission Weight": 0.0, "IOR": float(parameters["int_ior"])}}
    sockets = {
        "Base Color": parameters["base_color"],
        "Roughness": float(parameters["roughness"]),
        "Metallic": float(parameters["metallic"]),
        "Anisotropic": float(parameters["anisotropic"]),
        "Transmission Weight": float(parameters["spec_trans"]),
        "IOR": float(parameters["ior"]),
    }
    return {"node": "BSDF_PRINCIPLED", "sockets": sockets}


def preview_nodes(spec: dict) -> dict:
    """Blender node description generated from the same MaterialSpec as Mitsuba.

    Color images are sRGB. Roughness and normal images are Non-Color. A normal
    image goes through a tangent-space Normal Map node. This is not a Mitsuba export.
    """
    spec = validate_spec(spec)
    recipe = blender_recipe(spec)
    images = []
    for texture in spec.get("textures") or []:
        socket = {"base_color": "Base Color", "roughness_blender": "Roughness", "alpha_mitsuba": None, "normal_opengl": "Normal"}[texture["role"]]
        images.append({
            "role": texture["role"],
            "path": texture["path"],
            "color_space": "sRGB" if texture["color_space"] == "srgb" else "Non-Color",
            "socket": socket,
            "normal_map_node": texture["role"] == "normal_opengl",
            "normal_space": "TANGENT" if texture["role"] == "normal_opengl" else None,
            "parameter_class": texture["parameter_class"],
        })
    recipe["images"] = images
    recipe["renderer"] = "Blender Cycles"
    recipe["uv"] = "Texture Coordinate UV"
    return recipe


def transfer_cases() -> list[dict]:
    """Cases the pinned add-on must export. Expected Mitsuba values follow the exporter source until the XML confirms them."""
    specs = [
        preset("principled_plastic"),
        preset("principled_metal"),
        preset("principled_glass"),
        preset("matte_red"),
        preset("aluminum"),
        preset("aluminum_rough"),
        preset("glass_145"),
        preset("rough_glass"),
    ]
    cases = []
    for spec in specs:
        recipe = blender_recipe(spec)
        cases.append({"case_id": spec["material_id"], "spec": spec, "recipe": recipe, "expected_family": spec["family"]})
    cases.append({"case_id": "image_texture", "kind": "image_texture", "expected_family": "diffuse"})
    cases.append({"case_id": "normal_map", "kind": "normal_map", "expected_family": "principled"})
    cases.append({"case_id": "two_slots", "kind": "two_slots", "expected_family": "diffuse"})
    return cases


def values_close(found: Any, expected: Any, tolerance: float = FLOAT_TOLERANCE) -> bool:
    if isinstance(expected, (list, tuple)):
        if not isinstance(found, (list, tuple)) or len(found) < 3:
            return False
        return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(found[:3], expected[:3], strict=True))
    if isinstance(expected, (int, float)):
        try:
            return abs(float(found) - float(expected)) <= tolerance
        except (TypeError, ValueError):
            return False
    return found == expected


def content_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

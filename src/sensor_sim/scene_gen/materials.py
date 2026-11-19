"""BSDF dictionaries whose keys match the Mitsuba plugins in this repo.

``alpha`` is Mitsuba's Beckmann RMS slope, not a Disney roughness in ``[0, 1]``.
Collapsed ranges reproduce ``material_bsdf`` for the five Phase 2 plugins that
overlap this sampler. ``thindielectric`` is not sampled.
"""

from __future__ import annotations

from sensor_sim.scenes import (
    PHASE2_ALPHA,
    PHASE2_EXT_IOR,
    PHASE2_INT_IOR,
    PHASE2_REFLECTANCE,
    material_bsdf,
)

BSDF_NAMES = (
    "diffuse",
    "dielectric",
    "roughdielectric",
    "pplastic",
    "plastic",
    "conductor",
    "roughconductor",
)
METALS = ("Al", "Ag", "Au", "Cu")
PRIMARY_BSDFS = ("dielectric", "pplastic")
# Phase 2 measured these on one sphere. Conductor is the copper dragon, not that table.
REFERENCE_RESPONSE = {
    "dielectric": "keeps",
    "pplastic": "keeps",
    "roughdielectric": "does_not",
    "plastic": "does_not",
    "diffuse": "does_not",
    "thindielectric": "does_not",
    "conductor": "keeps",
    "roughconductor": "untested",
}

DEFAULT_RANGES = {
    "int_ior": (1.33, 1.70),
    "alpha": (0.02, 0.5),
    "reflectance": (0.05, 0.85),
}


def _rgb(rng, low: float, high: float) -> dict:
    return {
        "type": "rgb",
        "value": [float(rng.uniform(low, high)) for _channel in range(3)],
    }


def _ior(rng, ranges: dict) -> float:
    low, high = ranges["int_ior"]
    value = float(rng.uniform(low, high))
    if value <= float(PHASE2_EXT_IOR):
        raise RuntimeError(f"sampled int_ior {value} is not above ext_ior {PHASE2_EXT_IOR}")
    return value


def _alpha(rng, ranges: dict) -> float:
    low, high = ranges["alpha"]
    return float(rng.uniform(low, high))


def fixed_bsdf(name: str) -> dict:
    """The Phase 2 constants, or smooth copper for ``conductor``."""
    if name in ("diffuse", "dielectric", "roughdielectric", "pplastic", "plastic"):
        return material_bsdf(name)
    if name == "conductor":
        return {"type": "conductor", "material": "Cu"}
    if name == "roughconductor":
        return {
            "type": "roughconductor",
            "material": "Cu",
            "distribution": "beckmann",
            "alpha": PHASE2_ALPHA,
        }
    raise ValueError(f"unknown material {name!r}; expected one of {BSDF_NAMES}")


def sample_bsdf(name: str, rng, ranges: dict | None = None, metals: tuple[str, ...] | None = None) -> dict:
    """One homogeneous BSDF. ``plastic`` does not receive ``ext_ior``."""
    if name not in BSDF_NAMES:
        raise ValueError(f"unknown material {name!r}; expected one of {BSDF_NAMES}")
    ranges = dict(DEFAULT_RANGES if ranges is None else ranges)
    low, high = ranges["reflectance"]
    if name == "diffuse":
        return {"type": "diffuse", "reflectance": _rgb(rng, low, high)}
    if name == "dielectric":
        return {"type": "dielectric", "int_ior": _ior(rng, ranges), "ext_ior": float(PHASE2_EXT_IOR)}
    if name == "roughdielectric":
        return {
            "type": "roughdielectric",
            "int_ior": _ior(rng, ranges),
            "ext_ior": float(PHASE2_EXT_IOR),
            "distribution": "beckmann",
            "alpha": _alpha(rng, ranges),
        }
    if name == "pplastic":
        return {
            "type": "pplastic",
            "int_ior": _ior(rng, ranges),
            "ext_ior": float(PHASE2_EXT_IOR),
            "diffuse_reflectance": _rgb(rng, low, high),
            "alpha": _alpha(rng, ranges),
        }
    if name == "plastic":
        return {
            "type": "plastic",
            "int_ior": _ior(rng, ranges),
            "diffuse_reflectance": _rgb(rng, low, high),
        }
    choices = METALS if metals is None else tuple(metals)
    if not choices:
        raise RuntimeError(f"no measured metals are available for {name}")
    metal = choices[int(rng.integers(0, len(choices)))]
    if name == "conductor":
        return {"type": "conductor", "material": metal}
    return {
        "type": "roughconductor",
        "material": metal,
        "distribution": "beckmann",
        "alpha": _alpha(rng, ranges),
    }


DIELECTRIC_IORS = (1.33, 1.40, 1.50, 1.60, 1.70)


def primary_catalog() -> list[tuple[str, dict]]:
    """The only BSDFs the primary sampler draws. Other plugins stay in the library."""
    rows = [("dielectric", {"type": "dielectric", "int_ior": float(ior), "ext_ior": 1.0}) for ior in DIELECTRIC_IORS]
    rows.append(("pplastic", fixed_bsdf("pplastic")))
    rows.append(("conductor", {"type": "conductor", "material": "Cu"}))
    rows.append(("roughconductor", {"type": "roughconductor", "material": "Cu", "distribution": "beckmann", "alpha": 0.1}))
    rows.append(("roughconductor", {"type": "roughconductor", "material": "Au", "distribution": "beckmann", "alpha": 0.1}))
    return rows


def require_primary_preset(preset: dict) -> tuple[str, dict]:
    """Return the catalog row equal to ``preset``. Unknown dictionaries fail."""
    if not isinstance(preset, dict):
        raise RuntimeError(f"primary preset must be a BSDF dictionary, got {type(preset).__name__}")
    matches = [(name, dict(row)) for name, row in primary_catalog() if row == preset]
    if not matches:
        raise RuntimeError(f"unknown primary preset {preset}")
    return matches[0]


def sample_primary(rng, families: list[str] | None = None, preset: dict | None = None) -> tuple[str, dict]:
    if preset is not None:
        name, row = require_primary_preset(preset)
        if families is not None and name not in families:
            raise RuntimeError(f"preset family {name} is not in {families}")
        return name, row
    rows = [row for row in primary_catalog() if families is None or row[0] in families]
    if not rows:
        raise RuntimeError("no primary presets match the allowed families")
    return rows[int(rng.integers(0, len(rows)))]


def response_class(max_dolp: float, threshold: float = 0.2) -> str:
    """``keeps`` when the brightest valid pixel clears the Phase 2 DoLP gate."""
    return "keeps" if float(max_dolp) > float(threshold) else "does_not"


def parameters_match_reference(bsdf: dict) -> bool:
    """True when this dictionary is a measured primary preset or a fixed library preset."""
    if any(bsdf == preset for _name, preset in primary_catalog()):
        return True
    name = bsdf.get("type")
    if name not in BSDF_NAMES:
        return False
    return bsdf == fixed_bsdf(name)


def phase2_numbers() -> dict:
    """The constants a collapsed sampler has to reproduce."""
    return {
        "int_ior": PHASE2_INT_IOR,
        "ext_ior": PHASE2_EXT_IOR,
        "alpha": PHASE2_ALPHA,
        "reflectance": PHASE2_REFLECTANCE,
    }

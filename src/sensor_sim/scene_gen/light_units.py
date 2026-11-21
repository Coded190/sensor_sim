"""Convert Blender light power into Mitsuba emitter units.

Blender point, spot, and area lights store radiant power in watts. A Mitsuba
point emitter takes radiant intensity in watts per steradian. A Mitsuba area
emitter takes radiance in watts per steradian per square metre.

The point conversion assumes an isotropic source: ``intensity = power / (4 pi)``.
The area conversion assumes a one-sided Lambertian emitter:
``radiance = power / (area * pi)``, because the cosine-weighted hemisphere
integrates to pi.

These formulas are the unit conversion. They are not a claim that an arbitrary
room render is radiometrically calibrated. Environment radiance used as a
stand-in for an unconverted world shader stays a labeled replacement.
"""

from __future__ import annotations

import math

import numpy as np

FOUR_PI = 4.0 * math.pi
HEMISPHERE_LAMBERT = math.pi


class LightUnitError(ValueError):
    """A light power cannot be converted."""


def point_intensity_from_watts(power_w: float) -> float:
    """Isotropic radiant intensity, W/sr, from total radiant power, W."""
    power = float(power_w)
    if not np.isfinite(power) or power < 0.0:
        raise LightUnitError(f"reason=power value={power_w}")
    return power / FOUR_PI


def area_radiance_from_watts(power_w: float, area_m2: float) -> float:
    """Lambertian radiance, W/sr/m^2, from total power and emitting area."""
    power = float(power_w)
    area = float(area_m2)
    if not np.isfinite(power) or power < 0.0:
        raise LightUnitError(f"reason=power value={power_w}")
    if not np.isfinite(area) or area <= 0.0:
        raise LightUnitError(f"reason=area value={area_m2}")
    return power / (area * HEMISPHERE_LAMBERT)


def emitter_area_m2(shape: str, size: float, size_y: float, scale: tuple[float, float, float]) -> float:
    """World-space emitting area after the light object's evaluated scale.

    Blender's ``size`` is the full width. A disk uses it as the diameter.
    A square uses ``size`` on both axes. A rectangle or ellipse uses ``size_y``.
    """
    kind = str(shape or "SQUARE").upper()
    width = abs(float(size) * float(scale[0]))
    height = abs(float((size if kind in {"SQUARE", "DISK"} else size_y) * float(scale[1])))
    if kind in {"SQUARE", "RECTANGLE"}:
        return width * height
    if kind == "DISK":
        return math.pi * (width * 0.5) ** 2
    if kind == "ELLIPSE":
        return math.pi * (width * 0.5) * (height * 0.5)
    raise LightUnitError(f"reason=light_shape value={shape}")


def classify_point_emitter(power_w: float, radius_m: float, position: list[float]) -> dict:
    """Isotropic W/sr conversion, only for a zero-radius point with a real pose."""
    radius = float(radius_m)
    origin = bool(np.linalg.norm(np.asarray(position, dtype=np.float64)) <= 1e-4)
    record = {
        "type": "point",
        "included": False,
        "power_w": float(power_w),
        "radius_m": radius,
        "position": [float(value) for value in position],
        "polarization": "unpolarized",
        "color_space": "linear",
    }
    if origin:
        record["reason"] = "light_transform_at_origin"
        record["conversion"] = "not_converted"
        return record
    if not np.isfinite(radius) or radius < 0.0:
        record["reason"] = "point_radius"
        record["conversion"] = "not_converted"
        return record
    if radius > 1e-8:
        area = 4.0 * math.pi * radius * radius
        record["type"] = "sphere"
        record["included"] = True
        record["area_m2"] = area
        record["radiance"] = float(power_w) / (area * HEMISPHERE_LAMBERT)
        record["radiance_unit"] = "W/sr/m^2"
        record["conversion"] = "soft_sphere_lambertian_power_over_area_pi"
        record["reason"] = ""
        record["note"] = "Blender shadow_soft_size is nonzero. The emitter is a Lambertian sphere, not an isotropic point. intensity = power/(4 pi) is not used."
        return record
    record["included"] = True
    record["intensity"] = point_intensity_from_watts(power_w)
    record["intensity_unit"] = "W/sr"
    record["conversion"] = "isotropic_power_over_4pi"
    record["reason"] = ""
    return record


def classify_area_emitter(
    power_w: float,
    *,
    shape: str,
    size: float,
    size_y: float,
    scale: tuple[float, float, float],
    two_sided: bool,
    spread_rad: float,
    position: list[float],
) -> dict:
    """One-sided Lambertian radiance, or an explicit two-sided / unsupported record.

    One-sided: ``radiance = power / (pi * area)``.
    Two-sided: the same power is split across both faces,
    ``radiance = power / (2 * pi * area)``. That is a different conversion.
    A spread other than pi radians is not a Lambertian hemisphere.
    """
    origin = bool(np.linalg.norm(np.asarray(position, dtype=np.float64)) <= 1e-4)
    area = emitter_area_m2(shape, size, size_y, scale)
    record = {
        "type": "area",
        "included": False,
        "power_w": float(power_w),
        "shape": str(shape).upper(),
        "area_m2": float(area),
        "size_m": [float(size), float(size_y)],
        "scale": [float(value) for value in scale],
        "two_sided": bool(two_sided),
        "spread_rad": float(spread_rad),
        "position": [float(value) for value in position],
        "emitting_side": "both" if two_sided else "local_negative_z",
        "polarization": "unpolarized",
        "color_space": "linear",
        "radiance_unit": "W/sr/m^2",
    }
    if origin:
        record["reason"] = "light_transform_at_origin"
        record["conversion"] = "not_converted"
        return record
    if area <= 0.0:
        record["reason"] = "area"
        record["conversion"] = "not_converted"
        return record
    if abs(float(spread_rad) - math.pi) > 1e-3:
        record["reason"] = "non_lambertian_spread"
        record["conversion"] = "spread_not_hemispheric"
        record["note"] = "Blender spread is not pi radians, so the Lambertian hemisphere formula is not applied."
        return record
    if two_sided:
        record["included"] = True
        record["radiance"] = float(power_w) / (2.0 * area * HEMISPHERE_LAMBERT)
        record["conversion"] = "two_sided_lambertian_power_over_two_area_pi"
        record["difference_from_one_sided"] = "Power is split across two faces. One-sided radiance would be twice this value."
        record["reason"] = ""
        return record
    record["included"] = True
    record["radiance"] = area_radiance_from_watts(power_w, area)
    record["conversion"] = "lambertian_power_over_area_pi"
    record["reason"] = ""
    return record


def polarization_role(family: str) -> dict:
    """Say whether the Mitsuba family keeps polarization or destroys it.

    Diffuse reflection in this pin is depolarizing. Specular dielectric,
    plastic, and conductor families can produce polarization. Calling a
    depolarizing family polarization-accurate is a report error.
    """
    depolarizing = {"diffuse", "thindielectric", "roughdielectric", "roughplastic"}
    aware = {"pplastic", "dielectric", "conductor", "roughconductor", "measured_polarized"}
    if family in depolarizing:
        return {"family": family, "polarization": "depolarizing", "can_polarize": False}
    if family in aware:
        return {"family": family, "polarization": "polarization_aware", "can_polarize": True}
    return {"family": family, "polarization": "unknown", "can_polarize": False}

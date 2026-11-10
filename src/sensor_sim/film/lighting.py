"""Area-light units shared by Mitsuba and Blender.

``intensity`` in the YAML is Mitsuba's area-emitter radiance. Read as a
physical quantity it is watts per steradian per square meter. Blender's
area-light ``energy`` is radiant flux in watts. Cycles (Blender 4.5),
with Normalize on and exposure 0, turns that flux into radiance by

    L = energy / (pi * width * height)

so the matching power is ``pi * width * height * radiance``.
"""

from __future__ import annotations

import math

import numpy as np

# Multiplier used only for the PNG and the Blender viewport. The EXR stays
# linear. A smooth IOR-1.5 surface reflects about 4% at normal incidence, so
# the filmed ball is much darker than the walls. Exposure 4 lifts that body
# into the mid-grays. The shoulder keeps the walls from clipping to white.
DISPLAY_EXPOSURE = 4.0
DISPLAY_GAMMA = 2.2


def area_power_watts(size_x: float, size_y: float, radiance: float) -> float:
    """Blender area-light power that emits ``radiance`` toward the scene."""
    width = float(size_x)
    height = float(size_y)
    emitted = float(radiance)
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"area light size must be positive, got {(width, height)}")
    if emitted < 0.0:
        raise ValueError(f"radiance must be non-negative, got {emitted}")
    return math.pi * width * height * emitted


def display_srgb(linear: np.ndarray, exposure: float = DISPLAY_EXPOSURE) -> np.ndarray:
    """Display map for the film PNGs. This is not applied to the EXR.

    mapped = (exposure * linear) / (1 + exposure * linear), then
    display = clip(mapped, 0, 1) ** (1/2.2).
    """
    scaled = np.clip(np.asarray(linear, dtype=np.float64), 0.0, None) * float(exposure)
    mapped = scaled / (1.0 + scaled)
    return np.clip(mapped, 0.0, 1.0) ** (1.0 / DISPLAY_GAMMA)

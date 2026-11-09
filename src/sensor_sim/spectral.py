"""Eight-band spectral film and the radiance-versus-wavelength profile.

Each band is a ``regular`` spectrum equal to 1 from its lower edge to its
upper edge. ``specfilm`` does not accept the ``uniform`` spectrum plugin,
so this flat regular spectrum is the box response for that window.

Band names sort alphabetically into wavelength order. A region's value in
the profile is the mean of that linear band over pixels with

    row in [y0 * height, y1 * height)
    col in [x0 * width, x1 * width)

``frame`` uses the whole image. Means stay linear. They are not tonemapped.
"""

from __future__ import annotations

from dataclasses import dataclass

# 400 to 700 nm in eight equal windows of 37.5 nm.
BAND_START_NM = 400.0
BAND_WIDTH_NM = 37.5
BAND_COUNT = 8


@dataclass(frozen=True)
class SpectralBand:
    name: str
    wavelength_min: float
    wavelength_max: float

    @property
    def center_nm(self) -> float:
        return 0.5 * (self.wavelength_min + self.wavelength_max)


def spectral_bands() -> tuple[SpectralBand, ...]:
    bands = []
    for index in range(BAND_COUNT):
        lo = BAND_START_NM + index * BAND_WIDTH_NM
        hi = lo + BAND_WIDTH_NM
        bands.append(SpectralBand(f"band_{index + 1:02d}_{int(lo)}", lo, hi))
    return tuple(bands)


def specfilm_dict(resolution: int) -> dict:
    """Film that writes one linear channel per band."""
    film: dict = {
        "type": "specfilm",
        "width": int(resolution),
        "height": int(resolution),
        "component_format": "float32",
        "rfilter": {"type": "gaussian"},
    }
    for band in spectral_bands():
        film[band.name] = {
            "type": "regular",
            "wavelength_min": band.wavelength_min,
            "wavelength_max": band.wavelength_max,
            "values": "1, 1",
        }
    return film


# Fractions of the Cornell box frame, measured on a 64x64 scalar_rgb render.
# The light is the bright rectangle in the upper center. The red wall is the
# left strip. The green wall is the right strip. The box face is the lit
# small box below the light.
CORNELL_REGIONS: dict[str, tuple[float, float, float, float] | None] = {
    "frame": None,
    "light": (0.10, 0.28, 0.35, 0.65),
    "red_wall": (0.30, 0.70, 0.00, 0.10),
    "green_wall": (0.30, 0.70, 0.90, 1.00),
    "box_face": (0.50, 0.75, 0.48, 0.70),
}

# The ball sits in the lower middle of the same camera.
BALL_REGIONS: dict[str, tuple[float, float, float, float] | None] = {
    "frame": None,
    "ball": (0.42, 0.88, 0.22, 0.78),
}


def region_slice(
    height: int,
    width: int,
    bounds: tuple[float, float, float, float] | None,
) -> tuple[slice, slice]:
    if bounds is None:
        return slice(0, height), slice(0, width)
    y0, y1, x0, x1 = bounds
    return (
        slice(int(y0 * height), max(int(y1 * height), int(y0 * height) + 1)),
        slice(int(x0 * width), max(int(x1 * width), int(x0 * width) + 1)),
    )

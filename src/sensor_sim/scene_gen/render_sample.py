"""Stokes and AOV renders of one exported XML file."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from sensor_sim.film.images import write_exr
from sensor_sim.film.lighting import display_srgb
from sensor_sim.scene_gen.convention import S0_MIN


def choose_variant(preferred: str = "cuda_ad_spectral_polarized") -> str:
    import mitsuba as mi

    if preferred in mi.variants():
        return preferred
    fallback = "llvm_ad_spectral_polarized"
    if fallback not in mi.variants():
        raise RuntimeError(f"neither {preferred} nor {fallback} is installed")
    return fallback


def _names(scene, image: np.ndarray) -> list[str]:
    aovs = list(scene.integrator().aov_names())
    count = int(image.shape[2])
    if count == len(aovs):
        return aovs
    if count == 3 + len(aovs):
        return ["R", "G", "B"] + aovs
    raise RuntimeError(f"render has {count} channels and AOVs {aovs}")


def split_image(scene, image) -> tuple[dict[str, np.ndarray], list[str]]:
    array = np.array(image)
    names = _names(scene, array)
    channels = {names[index]: array[:, :, index] for index in range(array.shape[2])}
    return channels, names


def stack_stokes(channels: dict[str, np.ndarray], component: str) -> np.ndarray:
    keys = [f"{component}.{band}" for band in ("R", "G", "B")]
    if all(key in channels for key in keys):
        return np.stack([channels[key] for key in keys], axis=-1)
    if component in channels:
        return np.asarray(channels[component], dtype=np.float64)
    raise RuntimeError(f"Stokes component {component} is not in {sorted(channels)}")


def _load(xml: Path, variant: str):
    import mitsuba as mi

    mi.set_variant(variant)
    return mi.load_file(str(xml))


def render_stokes(xml: Path, variant: str, spp: int, seed: int):
    import mitsuba as mi

    scene = _load(xml, variant)
    image = mi.render(scene, spp=int(spp), seed=int(seed))
    channels, names = split_image(scene, image)
    bitmap = mi.Bitmap(np.array(image), channel_names=names)
    return scene, channels, names, bitmap


GEOMETRY_AOVS = "nn:sh_normal,geo:geo_normal,idx:shape_index,pos:position,depth:depth"
DIAGNOSTIC_AOVS = "nn:sh_normal,geo:geo_normal,idx:shape_index,pi:prim_index,pos:position,depth:depth"


def render_aovs(xml: Path, variant: str, aovs: str = GEOMETRY_AOVS):
    """Same file and sensor, with the integrator replaced by the geometry AOVs."""
    import mitsuba as mi

    tree = ET.parse(xml)
    root = tree.getroot()
    integrator = next(child for child in list(root) if child.tag.rsplit("}", 1)[-1] == "integrator")
    root.remove(integrator)
    aov = ET.Element("integrator", {"type": "aov"})
    ET.SubElement(aov, "string", {"name": "aovs", "value": aovs})
    root.insert(0, aov)
    text = ET.tostring(root, encoding="unicode")
    mi.set_variant(variant)
    previous = Path.cwd()
    try:
        os.chdir(xml.parent)
        scene = mi.load_string(text)
        image = mi.render(scene, spp=1, seed=0)
    finally:
        os.chdir(previous)
    channels, names = split_image(scene, image)
    return scene, channels, names


def write_preview(path: Path, s0: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = display_srgb(np.asarray(s0, dtype=np.float64))
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(np.clip(shown, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=80)
    plt.close(figure)


def polarization_mask(foreground: np.ndarray, s0: np.ndarray) -> np.ndarray:
    return np.asarray(foreground, dtype=bool) & (np.asarray(s0, dtype=np.float64) > S0_MIN)


def save_maps(directory: Path, arrays: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        image = np.asarray(values, dtype=np.float32)
        if image.ndim == 2:
            write_exr(directory / f"{name}.exr", image, [name])
        else:
            write_exr(directory / f"{name}.exr", image, [f"{name}.{axis}" for axis in ("X", "Y", "Z")])

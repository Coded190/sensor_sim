"""Remeasure BSDF polarization on the published Z-up sphere."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.generate import material_profile
from sensor_sim.scene_gen.materials import METALS, REFERENCE_RESPONSE, fixed_bsdf
from sensor_sim.scene_gen.render_sample import render_stokes, stack_stokes
from sensor_sim.scene_gen.xml_patch import bsdf_element
from sensor_sim.stokes import polarization_maps


def _replace_bsdf(source: Path, dest: Path, ident: str, bsdf: dict) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    current = None
    parent = None
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == "bsdf" and node.get("id") == ident:
            current = node
            break
    if current is None:
        raise RuntimeError(f"{source} has no bsdf {ident}")
    for node in root.iter():
        if current in list(node):
            parent = node
            break
    index = list(parent).index(current)
    parent.remove(current)
    parent.insert(index, bsdf_element(bsdf, ident))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(ET.tostring(root, encoding="unicode") + "\n")


def _masked_stats(s0, s1, s2, s3, mask: np.ndarray) -> dict:
    maps = polarization_maps(s0, s1, s2, s3)
    use = np.asarray(mask, dtype=bool) & maps["valid"]
    if int(use.sum()) == 0:
        return {"n": 0, "max_dolp": 0.0, "mean_dolp": 0.0, "mean_dop": 0.0, "fraction_above_0_2": 0.0, "n_stokes_inequality": 0}
    magnitude = np.sqrt(maps["s1"] ** 2 + maps["s2"] ** 2 + maps["s3"] ** 2)
    broken = int(np.count_nonzero(use & (maps["s0"] + 1e-4 < magnitude)))
    dolp = maps["dolp"][use]
    return {
        "n": int(use.sum()),
        "max_dolp": float(dolp.max()),
        "mean_dolp": float(dolp.mean()),
        "median_dolp": float(np.median(dolp)),
        "p90_dolp": float(np.percentile(dolp, 90)),
        "p95_dolp": float(np.percentile(dolp, 95)),
        "mean_dop": float(maps["dop"][use].mean()),
        "fraction_above_0_2": float(np.mean(dolp > 0.2)),
        "fraction_denominator": "object pixels with S0 > 1e-4",
        "n_stokes_inequality": broken,
    }


def measure_bsdf(scene_xml: Path, bsdf: dict, mask: np.ndarray, variant: str, spp: int, work: Path) -> dict:
    token = f"{bsdf['type']}_{bsdf.get('material', 'x')}_{bsdf.get('int_ior', 'x')}_{bsdf.get('alpha', 'x')}"
    dest = Path(work) / f"{token}.xml"
    dest = scene_xml.parent / dest.name
    _replace_bsdf(scene_xml, dest, "mat-obj_sphere", bsdf)
    _scene, channels, _names, _bitmap = render_stokes(dest, variant, spp, 0)
    stats = _masked_stats(
        stack_stokes(channels, "S0"),
        stack_stokes(channels, "S1"),
        stack_stokes(channels, "S2"),
        stack_stokes(channels, "S3"),
        mask,
    )
    expected = REFERENCE_RESPONSE.get(bsdf["type"], "untested")
    profile = material_profile(stats["max_dolp"], stats["mean_dolp"], stats["fraction_above_0_2"], stats["mean_dop"], expected)
    profile.update(stats)
    profile["bsdf"] = bsdf
    return profile


def presets() -> list[tuple[str, dict]]:
    rows = [(name, fixed_bsdf(name)) for name in ("diffuse", "dielectric", "roughdielectric", "pplastic", "plastic")]
    for metal in METALS:
        rows.append((f"conductor_{metal}", {"type": "conductor", "material": metal}))
        rows.append(
            (
                f"roughconductor_{metal}",
                {"type": "roughconductor", "material": metal, "distribution": "beckmann", "alpha": 0.1},
            )
        )
    return rows

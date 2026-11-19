"""Asset meshes in the Z-up film room.

Procedural families come from ``mesh_for_family`` without the polyethylene
shell. ``stand_dragon`` is not modified. The local dragon sits on ``z = 0``
with its XY center at the origin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sensor_sim.film import ROOT
from sensor_sim.film.geometry import read_ascii_ply
from sensor_sim.film.shape_mesh import mesh_for_family, sample_shape_params
from sensor_sim.scene_gen.meshcheck import require_clean

FAMILIES = (
    "sphere",
    "ellipsoid",
    "rounded_box",
    "capsule",
    "torus",
    "cylinder",
    "cone",
    "bumpy",
)
DRAGON_ID = "dragon"
ASSET_IDS = FAMILIES + (DRAGON_ID,)
DRAGON_PLY = ROOT / "assets" / "dragon" / "dragon_vrip_res4.ply"
RADIUS_RANGE = (0.03, 0.08)
DRAGON_HEIGHT_RANGE = (0.08, 0.14)


def dragon_local(height: float, path: Path | None = None) -> dict[str, np.ndarray]:
    """Y-up scan to Z-up, scaled to ``height``, centered in XY, bottom at ``z = 0``."""
    mesh = read_ascii_ply(Path(path) if path is not None else DRAGON_PLY)
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    spun = np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], axis=1)
    extent = float(spun[:, 2].max() - spun[:, 2].min())
    if extent <= 0.0:
        raise RuntimeError("dragon mesh has no height after the Y-up rotation")
    spun *= float(height) / extent
    spun[:, 0] -= 0.5 * float(spun[:, 0].min() + spun[:, 0].max())
    spun[:, 1] -= 0.5 * float(spun[:, 1].min() + spun[:, 1].max())
    spun[:, 2] -= float(spun[:, 2].min())
    return {"vertices": spun, "faces": np.asarray(mesh["faces"], dtype=np.int64)}


def build_asset(asset_id: str, rng, radius: float | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Local-frame mesh and the parameters that produced it."""
    if asset_id == DRAGON_ID:
        low, high = DRAGON_HEIGHT_RANGE
        height = float(rng.uniform(low, high))
        mesh = dragon_local(height)
        require_clean(mesh["vertices"], mesh["faces"], closed=False, name=DRAGON_ID)
        return mesh, {"height": height, "family": DRAGON_ID}
    if asset_id not in FAMILIES:
        raise ValueError(f"unknown asset {asset_id!r}")
    spec = sample_shape_params(asset_id, rng)
    low, high = RADIUS_RANGE
    size = float(rng.uniform(low, high) if radius is None else radius)
    mesh = mesh_for_family(asset_id, spec, size)
    require_clean(mesh["vertices"], mesh["faces"], closed=True, name=asset_id)
    spec = dict(spec)
    spec["radius"] = size
    spec["family"] = asset_id
    return mesh, spec


def horizontal_radius(vertices: np.ndarray) -> float:
    """Largest XY distance from the mesh center."""
    points = np.asarray(vertices, dtype=np.float64)
    center = points.mean(axis=0)
    offset = points[:, :2] - center[:2]
    return float(np.max(np.linalg.norm(offset, axis=1)))

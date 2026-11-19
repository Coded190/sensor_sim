"""The normal integrator recovers a plane and a constant-normal tilted plane."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.film_shape_sheet import integrate_log_depth


def _view(height: int = 64, width: int = 64, fov_deg: float = 50.0) -> np.ndarray:
    scale = 0.5 / np.tan(np.deg2rad(fov_deg) / 2.0)
    columns = (np.arange(width) + 0.5) / width
    rows = (np.arange(height) + 0.5) / height
    x_over_z = (columns - 0.5) / scale
    y_over_z = (rows - 0.5) / scale
    y_grid, x_grid = np.meshgrid(y_over_z, x_over_z, indexing="ij")
    directions = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def _disk(shape: tuple[int, int]) -> np.ndarray:
    rows = np.arange(shape[0]) - (shape[0] - 1) / 2.0
    cols = np.arange(shape[1]) - (shape[1] - 1) / 2.0
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    return xx**2 + yy**2 <= (0.32 * shape[0]) ** 2


def test_frontal_plane_integrates_flat():
    view = _view()
    mask = _disk(view.shape[:2])
    normal = np.zeros_like(view)
    normal[..., 2] = -1.0
    log_depth, used = integrate_log_depth(normal, view, mask)
    assert int(used.sum()) > 100
    assert np.nanmax(log_depth[used]) - np.nanmin(log_depth[used]) < 1e-6


def test_tilted_plane_matches_the_closed_form():
    view = _view()
    mask = _disk(view.shape[:2])
    normal = np.zeros_like(view)
    normal[..., 0] = -0.25
    normal[..., 1] = 0.15
    normal[..., 2] = -1.0
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    ray_z = view[..., 2]
    ray = np.stack([view[..., 0] / ray_z, view[..., 1] / ray_z, np.ones(view.shape[:2])], axis=-1)
    closed = 1.0 / np.sum(normal * ray, axis=-1)
    log_depth, used = integrate_log_depth(normal, view, mask)
    raw = np.exp(log_depth)
    scale = np.median(closed[used] / raw[used])
    ratio = np.abs(scale * raw[used] - closed[used]) / np.abs(closed[used])
    assert float(np.median(ratio)) < 1e-4

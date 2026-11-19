"""Rebuild the held-out film surfaces from camera-frame normals and draw them.

The Stokes network predicts a normal, not a depth. The reported run saved those
normals as 8-bit PNGs. This script decodes ``normal_stokes.png``, integrates
that field and the ground-truth normals into optical-axis depth, and draws each
test shape beside the render and the angular-error map.

Depth is the camera-frame Z of ``P = Z * (x, y, 1)``, where ``(x, y, 1)`` is
the pinhole ray stored in ``measurement["view"]`` before it was normalized.
A normal pointing back at the camera satisfies

    d(log Z) / d(column) = -n_x * dx/d(column) / (n · (x, y, 1))
    d(log Z) / d(row)    = -n_y * dy/d(row)    / (n · (x, y, 1))

The least-squares solution fixes one additive constant per connected component
of the film mask, which is a multiplicative scale on Z. The scale used in the
figure and in the depth error is the median of ``Z_mitsuba / Z_integrated``
on the film pixels that both the ray test and the integrator kept.

``Z_mitsuba`` is the camera-frame Z of the first hit on ``film.obj`` along the
pixel-center ray. Relative depth error is ``|s * Z_hat - Z_mitsuba| / Z_mitsuba``
on those pixels. The reported number is the median of that ratio.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr

from sensor_sim.film.lighting import display_srgb
from sensor_sim.film.sfp import OUTPUT, SEED, _loss_mask
from sensor_sim.film.sfp_math import angular_error_deg, angular_error_summary
from sensor_sim.film.shape_mesh import build_shape, shape_catalog

logger = logging.getLogger("sensor_sim.film_shape_sheet")

ROOT = OUTPUT / "02_heldout_shapes"
DENOM_MIN = 0.2
SHEET_PATH = ROOT / "test_shape_contact.png"
METRICS_PATH = ROOT / "test_shape_depth.json"


def _load_record(path: Path) -> dict:
    meta_path = path / "metadata.json"
    meta = json.loads(meta_path.read_text())
    measurement = {key: np.asarray(value) for key, value in np.load(path / "measurement.npz").items()}
    meta["measurement"] = measurement
    meta["out_dir"] = path
    return meta


def _gradient_field(normal: np.ndarray, view: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel derivatives of log optical-axis depth, and the denominator."""
    view = np.asarray(view, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    ray_z = np.maximum(view[..., 2], 1e-8)
    x_over_z = view[..., 0] / ray_z
    y_over_z = view[..., 1] / ray_z
    dx = float(np.median(np.diff(x_over_z, axis=1)))
    dy = float(np.median(np.diff(y_over_z, axis=0)))
    if abs(dx) < 1e-12 or abs(dy) < 1e-12:
        raise RuntimeError(f"pinhole ray slopes are degenerate, dx={dx} dy={dy}")
    denominator = normal[..., 0] * x_over_z + normal[..., 1] * y_over_z + normal[..., 2]
    safe = np.where(np.abs(denominator) < 1e-8, np.inf, denominator)
    column_slope = -normal[..., 0] * dx / safe
    row_slope = -normal[..., 1] * dy / safe
    return column_slope, row_slope, denominator


def integrate_log_depth(normal: np.ndarray, view: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares log depth. The return is ``(log_z, used_mask)``.

    ``used_mask`` drops film pixels whose ray is nearly tangent to the normal,
    because ``n · (x, y, 1)`` is then too small to give a stable slope. Each
    connected component is pinned to log depth 0 at its most frontal pixel.
    """
    column_slope, row_slope, denominator = _gradient_field(normal, view)
    usable = np.asarray(mask, dtype=bool) & np.isfinite(denominator) & (np.abs(denominator) >= DENOM_MIN)
    usable &= np.isfinite(column_slope) & np.isfinite(row_slope)
    height, width = usable.shape
    labels, count = ndimage.label(usable)
    if count == 0:
        raise RuntimeError("the film mask has no pixel stable enough to integrate")
    index = -np.ones((height, width), dtype=np.int64)
    index[usable] = np.arange(int(usable.sum()), dtype=np.int64)
    equation_rows: list[np.ndarray] = []
    equation_cols: list[np.ndarray] = []
    equation_data: list[np.ndarray] = []
    rhs_parts: list[np.ndarray] = []
    equation = 0

    def append_pairs(left_id: np.ndarray, right_id: np.ndarray, slope: np.ndarray) -> None:
        nonlocal equation
        count_pairs = int(left_id.shape[0])
        if count_pairs == 0:
            return
        base = np.arange(equation, equation + count_pairs, dtype=np.int64)
        equation_rows.append(np.repeat(base, 2))
        equation_cols.append(np.stack([right_id, left_id], axis=1).reshape(-1))
        equation_data.append(np.tile(np.array([1.0, -1.0]), count_pairs))
        rhs_parts.append(slope.astype(np.float64))
        equation += count_pairs

    right = usable[:, :-1] & usable[:, 1:]
    right_rows, right_cols = np.nonzero(right)
    append_pairs(
        index[right_rows, right_cols],
        index[right_rows, right_cols + 1],
        0.5 * (column_slope[right_rows, right_cols] + column_slope[right_rows, right_cols + 1]),
    )
    down = usable[:-1, :] & usable[1:, :]
    down_rows, down_cols = np.nonzero(down)
    append_pairs(
        index[down_rows, down_cols],
        index[down_rows + 1, down_cols],
        0.5 * (row_slope[down_rows, down_cols] + row_slope[down_rows + 1, down_cols]),
    )
    frontal = np.abs(normal[..., 2])
    pin_ids = []
    for label in range(1, count + 1):
        component = labels == label
        flat = np.flatnonzero(component)
        pin = int(flat[np.argmax(frontal[component])])
        pin_row, pin_col = np.unravel_index(pin, labels.shape)
        pin_ids.append(int(index[pin_row, pin_col]))
    pin_ids_array = np.asarray(pin_ids, dtype=np.int64)
    base = np.arange(equation, equation + pin_ids_array.shape[0], dtype=np.int64)
    equation_rows.append(base)
    equation_cols.append(pin_ids_array)
    equation_data.append(np.full(pin_ids_array.shape[0], 100.0))
    rhs_parts.append(np.zeros(pin_ids_array.shape[0], dtype=np.float64))
    equation += int(pin_ids_array.shape[0])
    matrix = sparse.csr_matrix(
        (
            np.concatenate(equation_data),
            (np.concatenate(equation_rows), np.concatenate(equation_cols)),
        ),
        shape=(equation, int(usable.sum())),
    )
    rhs = np.concatenate(rhs_parts)
    solved = lsqr(matrix, np.asarray(rhs, dtype=np.float64), atol=1e-8, btol=1e-8, iter_lim=4000)
    if int(solved[1]) not in (0, 1, 2, 4, 5):
        raise RuntimeError(f"log-depth solve failed with lsqr code {solved[1]}")
    log_depth = np.full((height, width), np.nan, dtype=np.float64)
    log_depth[usable] = solved[0]
    return log_depth, usable


def _camera_depth(spec: dict, mesh_dir: Path, view: np.ndarray) -> np.ndarray:
    """Camera-frame Z of the first film-mesh hit along each pixel-center ray."""
    import mitsuba as mi

    camera = build_shape(spec)["camera"]
    mi.set_variant("cuda_ad_rgb")
    scene = mi.load_dict(
        {
            "type": "scene",
            "sensor": {
                "type": "perspective",
                "fov": float(camera["fov_deg"]),
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=list(camera["position"]),
                    target=list(camera["target"]),
                    up=list(camera["up"]),
                ),
                "film": {
                    "type": "hdrfilm",
                    "width": int(view.shape[1]),
                    "height": int(view.shape[0]),
                    "pixel_format": "rgb",
                    "rfilter": {"type": "box"},
                },
            },
            "film": {"type": "obj", "filename": str(mesh_dir / "film.obj")},
        }
    )
    sensor = scene.sensors()[0]
    height, width = view.shape[:2]
    columns = (np.arange(width, dtype=np.float64) + 0.5) / width
    rows = (np.arange(height, dtype=np.float64) + 0.5) / height
    grid_y, grid_x = np.meshgrid(rows, columns, indexing="ij")
    position = mi.Point2f(grid_x.reshape(-1), grid_y.reshape(-1))
    rays, _weight = sensor.sample_ray(0.0, 0.5, position, mi.Point2f(0.5, 0.5))
    hit = scene.ray_intersect(rays)
    distance = np.asarray(hit.t, dtype=np.float64).reshape(height, width)
    valid = np.asarray(hit.is_valid(), dtype=bool).reshape(height, width)
    optical = distance * view[..., 2]
    optical = np.where(valid, optical, np.nan)
    del scene, sensor, hit, rays
    return optical


def _align(log_depth: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Scale integrated depth onto the Mitsuba depth. Returns depth, scale, and the shared mask."""
    shared = mask & np.isfinite(log_depth) & np.isfinite(reference) & (reference > 0.0)
    if int(shared.sum()) < 100:
        raise RuntimeError(f"only {int(shared.sum())} pixels can be compared with the mesh depth")
    raw = np.exp(log_depth)
    scale = float(np.median(reference[shared] / raw[shared]))
    return scale * raw, scale, shared


def relative_depth_error(depth: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict:
    """Median and mean of ``|depth - reference| / reference`` on ``mask``."""
    ratio = np.abs(depth[mask] - reference[mask]) / reference[mask]
    return {
        "pixels": int(mask.sum()),
        "median_relative_depth_error": float(np.median(ratio)),
        "mean_relative_depth_error": float(np.mean(ratio)),
    }


def decode_normal_png(path: Path) -> np.ndarray:
    """Invert ``_save_normals``. The PNG stores ``normal * 0.5 + 0.5`` in 8 bits.

    A round trip through this encoding on the ground-truth ellipsoid normals is
    about 0.33° mean angular error, which is the quantization of the file.
    """
    image = plt.imread(path)
    decoded = np.asarray(image[..., :3], dtype=np.float64) * 2.0 - 1.0
    length = np.linalg.norm(decoded, axis=-1, keepdims=True)
    unit = np.divide(decoded, np.maximum(length, 1e-8))
    present = np.asarray(image[..., :3]).max(axis=-1) > 0.02
    unit[~present] = 0.0
    return unit


def relief_mm(depth: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Depth minus the median mesh depth, in millimeters. Invalid pixels are NaN."""
    baseline = float(np.median(reference[mask]))
    relief = (depth - baseline) * 1000.0
    return np.where(mask & np.isfinite(depth), relief, np.nan)


def _depth_stats(depth: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict:
    """Absolute and relief-relative depth error on ``mask``.

    ``median_abs_mm`` is the median of ``|Z_aligned - Z_mesh|`` in millimeters.
    ``mesh_relief_mm`` is the 95th minus the 5th percentile of ``Z_mesh`` on the
    same pixels, so it is the front-to-back span of the visible surface and not
    the distance from the camera. ``median_abs_over_relief`` divides those two.
    """
    ratio = relative_depth_error(depth, reference, mask)
    absolute_mm = np.abs(depth[mask] - reference[mask]) * 1000.0
    mesh_mm = reference[mask] * 1000.0
    relief = float(np.percentile(mesh_mm, 95) - np.percentile(mesh_mm, 5))
    median_abs = float(np.median(absolute_mm))
    ratio["median_abs_mm"] = median_abs
    ratio["mesh_relief_mm"] = relief
    ratio["median_abs_over_relief"] = float(median_abs / relief) if relief > 0.0 else float("nan")
    return ratio


def _draw_cut(axis, mesh_mm: np.ndarray, stokes_mm: np.ndarray, column: int) -> None:
    """Vertical cut. Image row 0, the top of the frame, is at the top of the axes."""
    rows = np.arange(mesh_mm.shape[0])
    axis.plot(mesh_mm[:, column], rows, color="black", lw=1.3, label="mesh")
    axis.plot(stokes_mm[:, column], rows, color="#d62728", lw=1.1, label="Stokes")
    finite_rows = np.nonzero(np.isfinite(mesh_mm[:, column]) | np.isfinite(stokes_mm[:, column]))[0]
    if finite_rows.size:
        pad = 8
        top = max(0, int(finite_rows.min()) - pad)
        bottom = min(mesh_mm.shape[0] - 1, int(finite_rows.max()) + pad)
        axis.set_ylim(bottom, top)
    axis.tick_params(labelsize=7)
    axis.set_xlabel("Z − median (mm)", fontsize=8)
    axis.grid(True, lw=0.3, alpha=0.5)


def build_sheet(test: list[dict]) -> dict:
    figure, axes = plt.subplots(
        len(test),
        5,
        figsize=(16.5, 2.35 * len(test)),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.15, 1.15, 1.15, 1.05]},
    )
    depth_cmap = plt.cm.viridis.copy()
    depth_cmap.set_bad("white")
    rows = []
    error_image = None
    headers = ("render", "Stokes angular error", "Stokes depth", "mesh depth", "vertical cut")
    for row, record in enumerate(test):
        measurement = record["measurement"]
        predicted = decode_normal_png(Path(record["out_dir"]) / "normal_stokes.png")
        mask = _loss_mask(measurement, set(record["object_indices"]))
        summary = angular_error_summary(predicted, measurement["normal"], mask)
        reference = _camera_depth(record["spec"], Path(record["out_dir"]) / "meshes", measurement["view"])
        gt_log, gt_used = integrate_log_depth(measurement["normal"], measurement["view"], mask)
        pred_log, pred_used = integrate_log_depth(predicted, measurement["view"], mask)
        gt_depth, gt_scale, gt_shared = _align(gt_log, reference, gt_used)
        pred_depth, pred_scale, pred_shared = _align(pred_log, reference, pred_used)
        gt_error = _depth_stats(gt_depth, reference, gt_shared)
        pred_error = _depth_stats(pred_depth, reference, pred_shared)
        compared = gt_shared & pred_shared
        baseline_mask = compared if int(compared.sum()) > 100 else gt_shared
        gt_mm = relief_mm(reference, reference, baseline_mask)
        stokes_mm = relief_mm(pred_depth, reference, baseline_mask)
        finite = np.isfinite(gt_mm) | np.isfinite(stokes_mm)
        low, high = np.nanpercentile(np.concatenate([gt_mm[finite], stokes_mm[finite]]), [2, 98])
        pad = 0.08 * max(high - low, 1.0)
        low, high = float(low - pad), float(high + pad)
        column = int(np.argmax(mask.sum(axis=0)))
        logger.info(
            "shape sheet stage=depth view=%s mae_deg=%.4f pixels=%d gt_median_mm=%.3f stokes_median_mm=%.3f "
            "stokes_over_relief=%.4f relief_mm=%.2f gt_median_rel=%.6f stokes_median_rel=%.6f",
            record["view"],
            summary["mae_deg"],
            summary["pixels"],
            gt_error["median_abs_mm"],
            pred_error["median_abs_mm"],
            pred_error["median_abs_over_relief"],
            pred_error["mesh_relief_mm"],
            gt_error["median_relative_depth_error"],
            pred_error["median_relative_depth_error"],
        )
        rgb = np.clip(display_srgb(measurement["rgb"]), 0.0, 1.0)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{record['view']}\n{record['family']}", fontsize=9)
        degrees = np.where(mask, angular_error_deg(predicted, measurement["normal"]), np.nan)
        error_image = axes[row, 1].imshow(degrees, vmin=0.0, vmax=90.0, cmap="magma")
        axes[row, 1].set_title(f"MAE {summary['mae_deg']:.2f}°", fontsize=9)
        axes[row, 2].imshow(stokes_mm, cmap=depth_cmap, vmin=low, vmax=high)
        axes[row, 2].axvline(column, color="white", lw=0.7, alpha=0.85)
        axes[row, 2].set_title(f"med |dZ| {pred_error['median_abs_mm']:.2f} mm", fontsize=9)
        axes[row, 3].imshow(gt_mm, cmap=depth_cmap, vmin=low, vmax=high)
        axes[row, 3].axvline(column, color="white", lw=0.7, alpha=0.85)
        axes[row, 3].set_title(f"{low:.1f} to {high:.1f} mm", fontsize=9)
        _draw_cut(axes[row, 4], gt_mm, stokes_mm, column)
        if row == 0:
            axes[row, 4].legend(fontsize=7, loc="best", frameon=False)
        for column_id in range(4):
            axes[row, column_id].axis("off")
        if row == 0:
            for column_id, header in enumerate(headers):
                axes[row, column_id].set_title(header + "\n" + axes[row, column_id].get_title(), fontsize=9)
        rows.append(
            {
                "view": record["view"],
                "family": record["family"],
                "mae_deg": summary["mae_deg"],
                "median_deg": summary["median_deg"],
                "pixels": summary["pixels"],
                "normal_source": "normal_stokes.png decoded from 8-bit, then renormalized",
                "gt_normal_integration": gt_error,
                "stokes_integration": pred_error,
                "gt_scale": gt_scale,
                "stokes_scale": pred_scale,
            }
        )
    if error_image is not None:
        figure.colorbar(error_image, ax=axes[:, 1], fraction=0.04, pad=0.01, label="angular error (deg)")
    figure.savefig(SHEET_PATH, dpi=120)
    plt.close(figure)
    payload = {
        "sheet": str(SHEET_PATH),
        "normal_source": (
            "Decoded normal_stokes.png from the reported run. "
            "The 8-bit round trip is about 0.33 degrees on ground-truth normals."
        ),
        "depth": (
            "Camera-frame Z in P = Z * (x, y, 1). Integration recovers log Z up to one "
            "additive constant per connected film component. The scale is the median of "
            "Z_mesh / exp(log Z). The sheet colors Z minus the median Z_mesh, in millimeters. "
            "Purple is closer to the camera than that median. The white line is the vertical cut."
        ),
        "median_abs_mm": "median of |Z_aligned - Z_mesh| in millimeters",
        "mesh_relief_mm": "95th percentile of Z_mesh minus the 5th, in millimeters, on the compared pixels",
        "median_abs_over_relief": "median_abs_mm / mesh_relief_mm",
        "relative_depth_error": "median of |Z_aligned - Z_mesh| / Z_mesh. Z_mesh includes the distance from the camera, so this fraction is smaller than median_abs_over_relief.",
        "shapes": rows,
        "unweighted_mean": {
            "mae_deg": float(np.mean([item["mae_deg"] for item in rows])),
            "gt_median_abs_mm": float(np.mean([item["gt_normal_integration"]["median_abs_mm"] for item in rows])),
            "stokes_median_abs_mm": float(np.mean([item["stokes_integration"]["median_abs_mm"] for item in rows])),
            "stokes_median_abs_over_relief": float(np.mean([item["stokes_integration"]["median_abs_over_relief"] for item in rows])),
            "gt_median_relative_depth_error": float(np.mean([item["gt_normal_integration"]["median_relative_depth_error"] for item in rows])),
            "stokes_median_relative_depth_error": float(np.mean([item["stokes_integration"]["median_relative_depth_error"] for item in rows])),
        },
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("shape sheet stage=wrote path=%s", SHEET_PATH)
    return payload


def _attach_spec(record: dict, catalog: dict[str, dict]) -> dict:
    spec = catalog[record["view"]]
    record["spec"] = spec
    record["family"] = spec["family"]
    return record


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    catalog = {spec["folder"]: spec for spec in shape_catalog(SEED)}
    test = [_attach_spec(_load_record(path), catalog) for path in sorted((ROOT / "test").iterdir()) if path.is_dir()]
    build_sheet(test)


if __name__ == "__main__":
    main()

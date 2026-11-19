"""Measured partial occlusion. Nearby boxes are not enough."""

from __future__ import annotations

import numpy as np

from sensor_sim.film.shape_mesh import project_ndc

TRAINING_MIN_PIXELS = 80


def rasterize(vertices: np.ndarray, faces: np.ndarray, camera: dict, width: int, height: int) -> np.ndarray:
    """Pixel mask of the projected triangles. This is the unoccluded silhouette."""
    ndc = project_ndc(vertices, camera["position"], camera["target"], float(camera["fov_deg"]))
    pixels = np.stack([ndc[:, 0] * width, ndc[:, 1] * height], axis=1)
    mask = np.zeros((height, width), dtype=bool)
    for face in np.asarray(faces, dtype=np.int64):
        triangle = pixels[face]
        min_x = max(int(np.floor(triangle[:, 0].min())), 0)
        max_x = min(int(np.ceil(triangle[:, 0].max())), width - 1)
        min_y = max(int(np.floor(triangle[:, 1].min())), 0)
        max_y = min(int(np.ceil(triangle[:, 1].max())), height - 1)
        if min_x > max_x or min_y > max_y:
            continue
        xs = np.arange(min_x, max_x + 1) + 0.5
        ys = np.arange(min_y, max_y + 1) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        point = np.stack([grid_x, grid_y], axis=-1)
        v0 = triangle[1] - triangle[0]
        v1 = triangle[2] - triangle[0]
        v2 = point - triangle[0]
        denom = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(float(denom)) < 1e-12:
            continue
        bary_u = (v2[..., 0] * v1[1] - v1[0] * v2[..., 1]) / denom
        bary_v = (v0[0] * v2[..., 1] - v2[..., 0] * v0[1]) / denom
        inside = (bary_u >= 0.0) & (bary_v >= 0.0) & (bary_u + bary_v <= 1.0)
        mask[min_y : max_y + 1, min_x : max_x + 1] |= inside
    return mask


def partial_occlusion_report(
    object_ids: np.ndarray,
    objects: list[dict],
    camera: dict,
    width: int,
    height: int,
    normal_valid: np.ndarray | None = None,
) -> dict:
    """Accept measured rear occlusion of 10–50 percent and two supervised objects.

    The isolated count is the projected silhouette. The joint count is the
    painted object-id image. ``visible_pixels`` in this report is that image
    count, not the legacy ownership field.
    """
    from sensor_sim.scene_gen.dataset import discrete_object_ids

    labels, fractional = discrete_object_ids(object_ids)
    rows = []
    silhouettes = []
    for item in objects:
        if item.get("kind") == "plane" or item.get("asset_id") == "plane":
            continue
        label = int(item["object_index"])
        assigned = (labels == label) & ~fractional
        visible = int(np.count_nonzero(assigned))
        silhouette = rasterize(item["vertices"], item["faces"], camera, width, height)
        unoccluded = int(silhouette.sum())
        fraction = None if unoccluded == 0 else (unoccluded - visible) / unoccluded
        supervised = None
        if normal_valid is not None:
            supervised = int(np.count_nonzero(assigned & np.asarray(normal_valid, dtype=bool)))
        rows.append(
            {
                "name": item["name"],
                "object_index": label,
                "visible_pixels": visible,
                "object_id_pixel_count": visible,
                "unoccluded_pixels": unoccluded,
                "occlusion_fraction": fraction,
                "visibility_ratio": None if unoccluded == 0 else visible / unoccluded,
                "normal_supervised_pixel_count": supervised,
            }
        )
        silhouettes.append(silhouette)
    overlap = 0
    if len(silhouettes) >= 2:
        overlap = int(np.count_nonzero(silhouettes[0] & silhouettes[1]))
    rear = None
    if rows:
        # The rear object is the one with the larger measured visibility reduction.
        rear = max(rows, key=lambda row: -1.0 if row["occlusion_fraction"] is None else float(row["occlusion_fraction"]))
    both_eligible = len(rows) >= 2 and all(row["object_id_pixel_count"] >= TRAINING_MIN_PIXELS for row in rows[:2])
    fraction = None if rear is None else rear["occlusion_fraction"]
    in_band = fraction is not None and 0.10 <= float(fraction) <= 0.50
    supervised_ok = normal_valid is None or (
        len(rows) >= 2 and all(int(row["normal_supervised_pixel_count"] or 0) >= 1 for row in rows[:2])
    )
    passed = bool(overlap > 0 and in_band and both_eligible and supervised_ok)
    return {
        "pass": passed,
        "overlap_pixels": overlap,
        "objects": rows,
        "rear_object": None if rear is None else rear["name"],
        "occlusion_fraction": fraction,
        "both_training_eligible": both_eligible,
        "rear_visibility_reduced": in_band,
        "fraction_limits": [0.10, 0.50],
        "rule": "occlusion_fraction = (isolated_rear - jointly_visible_rear) / isolated_rear in [0.10, 0.50]; both objects >= 80 object-id pixels and at least one supervised normal",
    }

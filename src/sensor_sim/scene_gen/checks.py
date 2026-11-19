"""Camera, BSDF, Stokes, and sphere checks. Raw Stokes values are not clamped."""

from __future__ import annotations

import numpy as np

from sensor_sim.film.polar import analyzer_images
from sensor_sim.film.sfp_math import camera_matrix, view_directions, world_normals_to_camera
from sensor_sim.scene_gen.convention import DOP_EXCESS, S0_MIN, SHAPE_INDEX_TOL
from sensor_sim.stokes import polarization_maps, summarize_polarization


def stokes_report(s0, s1, s2, s3, tolerance: float = 1e-4) -> dict:
    """DoP and DoLP summaries plus the count of pixels that break the Stokes inequality."""
    maps = polarization_maps(s0, s1, s2, s3)
    summary = summarize_polarization(maps)
    images = analyzer_images(maps["s0"], maps["s1"], maps["s2"])
    valid = maps["valid"]
    magnitude = np.sqrt(maps["s1"] ** 2 + maps["s2"] ** 2 + maps["s3"] ** 2)
    broken = valid & (maps["s0"] + float(tolerance) < magnitude)
    summary = dict(summary)
    summary["n_stokes_inequality"] = int(np.count_nonzero(broken))
    summary["n_dolp_above_excess"] = int(np.count_nonzero(valid & (maps["dop"] > DOP_EXCESS)))
    summary["maps"] = maps
    summary["analyzers"] = images
    return summary


def integer_index(values: np.ndarray, index: int, tol: float = SHAPE_INDEX_TOL) -> np.ndarray:
    """True where the AOV is an integer match, not a blend of two shapes."""
    array = np.asarray(values, dtype=np.float64)
    nearest = np.rint(array)
    whole = np.abs(array - nearest) < float(tol)
    return whole & (nearest == int(index))


def compare_camera(sensor, camera: dict, rays: dict, width: int, height: int) -> dict:
    """Compare the loaded sensor with the config and the Blender rays."""
    import mitsuba as mi

    origin = np.asarray(camera["position"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    up = np.asarray(camera["up"], dtype=np.float64)
    expected = mi.ScalarTransform4f.look_at(origin=origin.tolist(), target=target.tolist(), up=up.tolist())
    got = camera_matrix(sensor)
    wanted = np.array(expected.matrix, dtype=np.float64).reshape(4, 4)
    matrix_error = float(np.max(np.abs(got - wanted)))
    params = mi.traverse(sensor)
    def _scalar(key_suffix: str) -> float:
        key = next(key for key in params.keys() if key.endswith(key_suffix))
        return float(np.array(params[key]).reshape(-1)[0])

    fov = _scalar("x_fov")
    near = _scalar("near_clip")
    shift_x = _scalar("principal_point_offset_x")
    shift_y = _scalar("principal_point_offset_y")
    direction_errors = []
    origin_errors = []
    for ray in rays["rays"]:
        col, row = int(ray["pixel"][0]), int(ray["pixel"][1])
        sample = mi.Point2f((col + 0.5) / float(width), (row + 0.5) / float(height))
        sampled, _weight = sensor.sample_ray(mi.Float(0.0), mi.Float(0.5), sample, mi.Point2f(0.5, 0.5))
        mitsuba_origin = np.array(sampled.o, dtype=np.float64).reshape(-1)[:3]
        mitsuba_direction = np.array(sampled.d, dtype=np.float64).reshape(-1)[:3]
        mitsuba_direction /= np.linalg.norm(mitsuba_direction)
        blender_direction = np.asarray(ray["direction"], dtype=np.float64)
        blender_direction /= np.linalg.norm(blender_direction)
        direction_errors.append(float(np.linalg.norm(mitsuba_direction - blender_direction)))
        # The ray starts on the near plane, which is perpendicular to camera +Z.
        cosine = float(np.dot(mitsuba_direction, got[:3, 2]))
        pinhole = mitsuba_origin - (near / cosine) * mitsuba_direction
        origin_errors.append(float(np.linalg.norm(pinhole - origin)))
        origin_errors.append(float(np.linalg.norm(np.asarray(ray["origin"], dtype=np.float64) - origin)))
    forward = target - origin
    forward /= np.linalg.norm(forward)
    camera_forward = got[:3, 2]
    camera_forward /= np.linalg.norm(camera_forward)
    report = {
        "matrix_error": matrix_error,
        "fov_deg": fov,
        "fov_error": abs(fov - float(camera["fov_deg"])),
        "near_clip": near,
        "principal_point_offset": [shift_x, shift_y],
        "forward_error": float(np.linalg.norm(camera_forward - forward)),
        "direction_error": float(max(direction_errors)),
        "origin_error": float(max(origin_errors)),
        "blender_fov_x_deg": float(rays["fov_x_deg"]),
        "blender_fov_y_deg": float(rays["fov_y_deg"]),
        "sensor_width_mm": float(rays["sensor_width_mm"]),
        "sensor_height_mm": float(rays["sensor_height_mm"]),
        "shift_xy": [float(rays["shift_x"]), float(rays["shift_y"])],
        "pixel_aspect": float(rays["pixel_aspect"]),
        "row_0_is_top": bool(rays["row_0_is_top"]),
        "resolution": [int(width), int(height)],
    }
    if matrix_error > 1e-5 or report["fov_error"] > 1e-4:
        raise RuntimeError(f"camera projection does not match the config: {report}")
    if report["direction_error"] > 1e-4 or report["origin_error"] > 1e-4:
        raise RuntimeError(f"camera rays do not match: {report}")
    if abs(shift_x) > 1e-6 or abs(shift_y) > 1e-6:
        raise RuntimeError(f"principal point is shifted: {shift_x}, {shift_y}")
    return report


def bsdf_eta(scene, shape_index: int) -> float:
    """Runtime ``eta`` of one shape. Dielectric stores ``int_ior / ext_ior`` here."""
    import mitsuba as mi

    shape = scene.shapes()[int(shape_index)]
    params = mi.traverse(shape.bsdf())
    for key in params.keys():
        if key.endswith("eta") or key.endswith("eta.value"):
            return float(np.array(params[key]).reshape(-1)[0])
    raise RuntimeError(f"shape {shape_index} BSDF has no eta in {list(params.keys())}")


def sphere_geometry(scene, shape_index: int, center: list[float], radius: float) -> dict:
    box = scene.shapes()[int(shape_index)].bbox()
    low = np.array(box.min, dtype=np.float64)
    high = np.array(box.max, dtype=np.float64)
    found = 0.5 * (low + high)
    wanted = np.asarray(center, dtype=np.float64)
    error = float(np.max(np.abs(found - wanted)))
    extent = high - low
    radius_error = float(max(abs(extent[0] / 2.0 - radius), abs(extent[1] / 2.0 - radius)))
    if error > 1e-4 or radius_error > 1e-3:
        raise RuntimeError(
            f"sphere bbox center {found.tolist()} radius extents {extent.tolist()} "
            f"do not match center {wanted.tolist()} radius {radius}"
        )
    return {"center": found.tolist(), "center_error": error, "radius_error": radius_error}


def analytic_sphere(
    geo_world: np.ndarray,
    depth: np.ndarray,
    interior: np.ndarray,
    *,
    center: np.ndarray,
    radius: float,
    origin: np.ndarray,
    sensor,
    near_clip: float,
    width: int,
    height: int,
    position: np.ndarray | None = None,
    angle_deg: float = 1.0,
    max_angle_deg: float = 8.0,
    depth_tol_m: float = 0.01,
) -> dict:
    """Compare interior normals with the radius through the mesh hit.

    The hit is ``origin + (depth + near_clip / cos) * ray``. ``cos`` is the
    dot of the ray with camera +Z, because the ray starts on the near plane.
    The median of that angle is the 1 degree gate. The max stays under 8 degrees.
    The continuous sphere is reported separately and is not the gate.
    """
    directions = view_directions(sensor, width, height)
    rotation = camera_matrix(sensor)[:3, :3]
    world = directions @ rotation.T
    world /= np.linalg.norm(world, axis=-1, keepdims=True)
    forward = rotation[:, 2]
    forward = forward / np.linalg.norm(forward)
    cosine = np.clip(np.sum(world * forward, axis=-1), 1e-4, None)
    ray_distance = np.asarray(depth, dtype=np.float64) + float(near_clip) / cosine
    if position is None:
        rendered_hit = origin.reshape(1, 1, 3) + ray_distance[..., None] * world
    else:
        rendered_hit = np.asarray(position, dtype=np.float64)
    expected = rendered_hit - center.reshape(1, 1, 3)
    expected /= np.maximum(np.linalg.norm(expected, axis=-1, keepdims=True), 1e-12)
    offset = origin.reshape(1, 1, 3) - center.reshape(1, 1, 3)
    half_b = np.sum(world * offset, axis=-1)
    discriminant = half_b ** 2 - (np.sum(offset * offset, axis=-1) - float(radius) ** 2)
    root = -half_b - np.sqrt(np.clip(discriminant, 0.0, None))
    actual = np.asarray(geo_world, dtype=np.float64)
    lengths = np.linalg.norm(actual, axis=-1)
    unit = actual / np.maximum(lengths[..., None], 1e-12)
    facing_cosine = np.sum(unit * -world, axis=-1)
    interior_face = interior & (facing_cosine > 0.5) & np.isfinite(ray_distance)
    if position is not None:
        radial_distance = np.linalg.norm(rendered_hit - center.reshape(1, 1, 3), axis=-1)
        interior_face &= np.abs(radial_distance - float(radius)) < 0.004
    cosine_angle = np.clip(np.sum(unit * expected, axis=-1), -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine_angle))
    analytic_hit = origin.reshape(1, 1, 3) + root[..., None] * world
    analytic = analytic_hit - center.reshape(1, 1, 3)
    analytic /= np.maximum(np.linalg.norm(analytic, axis=-1, keepdims=True), 1e-12)
    analytic_angles = np.degrees(np.arccos(np.clip(np.sum(unit * analytic, axis=-1), -1.0, 1.0)))
    depth_error = np.abs(ray_distance - root)
    positive = interior_face
    if int(np.count_nonzero(positive)) < 10:
        raise RuntimeError(
            f"the sphere covers fewer than 10 interior pixels "
            f"(mask count {int(np.count_nonzero(interior))})"
        )
    angle_max = float(np.max(angles[positive]))
    angle_median = float(np.median(angles[positive]))
    angle_p95 = float(np.percentile(angles[positive], 95))
    report = {
        "n_interior": int(np.count_nonzero(positive)),
        "angle_median_deg": angle_median,
        "angle_p95_deg": angle_p95,
        "angle_max_deg": angle_max,
        "analytic_sphere_median_deg": float(np.median(analytic_angles[positive])),
        "reference": "radius through the mesh hit reconstructed from depth and the near plane",
    }
    chord = float(np.median(depth_error[positive]))
    report["mesh_chord_depth_median_m"] = chord
    if angle_median > float(angle_deg) or angle_p95 > float(max_angle_deg) or chord > float(depth_tol_m):
        raise RuntimeError(f"sphere mesh-hit normals are outside the gate: {report}")
    camera_normals = world_normals_to_camera(actual, camera_matrix(sensor))
    facing = camera_normals[positive, 2]
    if float(np.max(facing)) > 1e-4:
        raise RuntimeError("a camera-space geometric normal does not point toward the camera")
    report["camera_z_max"] = float(np.max(facing))
    return report

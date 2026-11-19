"""Floor placement with a bounding-box gap and a fixed camera frame."""

from __future__ import annotations

import numpy as np

from sensor_sim.film.shape_mesh import place_mesh, project_ndc
from sensor_sim.scene_gen.assets import horizontal_radius
from sensor_sim.scene_gen.narrow import narrow_phase

FLOOR_Z = 0.001
DEFAULT_INSET = {"x": (-0.18, 0.18), "y": (-0.05, 0.18)}
DEFAULT_GAP = 0.005
FRAME_MARGIN = 0.08


def _box(vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    low = points.min(axis=0)
    high = points.max(axis=0)
    return np.stack([low, high], axis=0)


def _separated(first: np.ndarray, second: np.ndarray, gap: float) -> bool:
    return bool(
        first[1, 0] + gap <= second[0, 0]
        or second[1, 0] + gap <= first[0, 0]
        or first[1, 1] + gap <= second[0, 1]
        or second[1, 1] + gap <= first[0, 1]
    )


def _inside(box: np.ndarray, inset: dict) -> bool:
    x0, x1 = inset["x"]
    y0, y1 = inset["y"]
    return bool(box[0, 0] >= x0 and box[1, 0] <= x1 and box[0, 1] >= y0 and box[1, 1] <= y1)


def _in_frame(vertices: np.ndarray, camera: dict) -> bool:
    """The projected mesh must sit in the frame, not only its center."""
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape[0] > 400:
        points = points[:: max(1, points.shape[0] // 400)]
    try:
        ndc = project_ndc(points, camera["position"], camera["target"], float(camera["fov_deg"]))
    except RuntimeError:
        return False
    center = ndc.mean(axis=0)
    inside = (
        FRAME_MARGIN <= float(center[0]) <= 1.0 - FRAME_MARGIN
        and FRAME_MARGIN <= float(center[1]) <= 1.0 - FRAME_MARGIN
    )
    on_film = bool(np.any((ndc[:, 0] > 0.0) & (ndc[:, 0] < 1.0) & (ndc[:, 1] > 0.0) & (ndc[:, 1] < 1.0)))
    return inside and on_film


def _floor_area(inset: dict) -> float:
    return (float(inset["x"][1]) - float(inset["x"][0])) * (float(inset["y"][1]) - float(inset["y"][0]))


def _footprint(radius: float, gap: float) -> float:
    side = 2.0 * float(radius) + float(gap)
    return side * side


def _projected_occlusion_in_band(vertices: np.ndarray, faces: np.ndarray, previous: dict, camera: dict) -> bool:
    """Reject a pose whose projected overlap is outside 10–50 percent.

    This is the placement proposal. The rendered object-id count remains the
    acceptance measurement.
    """
    from sensor_sim.scene_gen.occlusion import rasterize

    current = rasterize(vertices, faces, camera, 128, 128)
    other = rasterize(np.asarray(previous["vertices"], dtype=np.float64), previous["faces"], camera, 128, 128)
    isolated_current = int(current.sum())
    isolated_other = int(other.sum())
    overlap = int(np.count_nonzero(current & other))
    if overlap <= 0 or isolated_current < 80 or isolated_other < 80:
        return False
    if isolated_current - overlap < 80 or isolated_other - overlap < 80:
        return False
    fraction_current = overlap / isolated_current
    fraction_other = overlap / isolated_other
    return 0.10 <= max(fraction_current, fraction_other) <= 0.50


def _silhouette_overlap(first: np.ndarray, second: np.ndarray, camera: dict) -> bool:
    from sensor_sim.film.shape_mesh import project_ndc

    def box(points: np.ndarray) -> tuple[float, float, float, float] | None:
        ndc = project_ndc(points[:: max(1, len(points) // 80)], camera["position"], camera["target"], float(camera["fov_deg"]))
        inside = ndc[(ndc[:, 0] > 0.0) & (ndc[:, 0] < 1.0) & (ndc[:, 1] > 0.0) & (ndc[:, 1] < 1.0)]
        if inside.shape[0] < 2:
            return None
        return float(inside[:, 0].min()), float(inside[:, 0].max()), float(inside[:, 1].min()), float(inside[:, 1].max())

    left, right = box(first), box(second)
    if left is None or right is None:
        return False
    return left[0] < right[1] and right[0] < left[1] and left[2] < right[3] and right[2] < left[3]


def place_objects(
    meshes: list[dict[str, np.ndarray]],
    rng,
    camera: dict,
    *,
    inset: dict | None = None,
    gap: float = DEFAULT_GAP,
    max_attempts: int = 40,
    seed: int = 0,
    require_image_overlap: bool = False,
) -> list[dict]:
    """Yaw about Z and rest each mesh on ``z = 0.001``. Overlap raises."""
    inset = DEFAULT_INSET if inset is None else inset
    radii = [horizontal_radius(np.asarray(mesh["vertices"], dtype=np.float64)) for mesh in meshes]
    footprints = [_footprint(radius, gap) for radius in radii]
    if sum(footprints) > _floor_area(inset):
        raise RuntimeError(
            f"scene_gen seed={seed} status=cannot_fit footprints={footprints} floor_area={_floor_area(inset)}"
        )
    order = sorted(range(len(meshes)), key=lambda index: radii[index], reverse=True)
    placed_by_index: dict[int, dict] = {}
    boxes = []
    placed_meshes: list[dict] = []
    for index in order:
        mesh = meshes[index]
        local = np.asarray(mesh["vertices"], dtype=np.float64)
        radius = radii[index]
        x0, x1 = float(inset["x"][0]) + radius, float(inset["x"][1]) - radius
        y0, y1 = float(inset["y"][0]) + radius, float(inset["y"][1]) - radius
        if x0 >= x1 or y0 >= y1:
            raise RuntimeError(f"scene_gen seed={seed} object={index} status=too_large_for_inset radius_xy={radius}")
        found = None
        directed = []
        if require_image_overlap and placed_meshes:
            previous_center = np.asarray(placed_meshes[-1]["vertices"], dtype=np.float64).mean(axis=0)
            for distance in (0.03, 0.045, 0.06, 0.08, 0.1):
                for shift in (-0.04, -0.02, 0.0, 0.02, 0.04):
                    directed.append((float(previous_center[0] + shift), float(previous_center[1] + distance)))
        for attempt_index in range(int(max_attempts)):
            yaw = float(rng.uniform(0.0, 2.0 * np.pi))
            if attempt_index < len(directed):
                center = directed[attempt_index]
            else:
                center = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
            vertices = place_mesh(local, yaw, center)
            box = _box(vertices)
            if not _inside(box, inset):
                continue
            if not require_image_overlap and any(not _separated(box, previous, gap) for previous in boxes):
                continue
            if not _in_frame(vertices, camera):
                continue
            if float(np.min(vertices[:, 2])) < FLOOR_Z - 1e-4:
                continue
            candidate = {"vertices": vertices, "faces": mesh["faces"]}
            narrow = [narrow_phase(candidate, previous, gap) for previous in placed_meshes]
            if any(not item["separated"] for item in narrow):
                continue
            if require_image_overlap and placed_meshes:
                if not any(_projected_occlusion_in_band(vertices, mesh["faces"], previous, camera) for previous in placed_meshes):
                    continue
            found = {
                "vertices": vertices,
                "yaw": yaw,
                "xy": [center[0], center[1]],
                "box": box,
                "narrow_phase": narrow,
            }
            break
        if found is None:
            raise RuntimeError(
                f"scene_gen seed={seed} object={index} status=placement_failed attempts={max_attempts} "
                f"footprints={footprints} floor_area={_floor_area(inset)}"
            )
        placed_by_index[index] = found
        boxes.append(found["box"])
        placed_meshes.append({"vertices": found["vertices"], "faces": mesh["faces"]})
    placed = [placed_by_index[index] for index in range(len(meshes))]
    for item in placed:
        bottom = float(np.min(item["vertices"][:, 2]))
        if abs(bottom - FLOOR_Z) > 1e-8:
            raise RuntimeError(f"placed mesh bottom is {bottom}, expected {FLOOR_Z}")
    return placed

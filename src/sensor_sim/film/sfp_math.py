"""Camera-frame normals, Fresnel inversion, and angular error.

AoLP is ``0.5 * arctan2(S2, S1)`` in the camera Stokes frame, and it is
treated as 0 where DoLP is below 0.05. Zenith is the angle between the
surface normal and the direction from the surface back toward the camera.
The diffuse branch sets the normal's azimuth equal to AoLP. The specular
branch adds π/2. Both keep the normal pointing toward the camera.
"""

from __future__ import annotations

import numpy as np

from sensor_sim.stokes import AOLP_DOLP_MIN, S0_MIN

IOR = 1.5
ZENITH_SAMPLES = 1801


def camera_matrix(sensor) -> np.ndarray:
    """4×4 camera-to-world matrix. Columns of the rotation are the camera axes."""
    matrix = np.array(sensor.world_transform().matrix, dtype=np.float64).reshape(4, 4)
    return matrix


def view_directions(sensor, width: int, height: int) -> np.ndarray:
    """Unit ray direction in camera space for each pixel center. +Z points into the scene."""
    projection = np.array(sensor.projection_transform().matrix, dtype=np.float64).reshape(4, 4)
    columns = (np.arange(int(width), dtype=np.float64) + 0.5) / float(width)
    rows = (np.arange(int(height), dtype=np.float64) + 0.5) / float(height)
    x_over_z = (columns - projection[0, 2]) / projection[0, 0]
    y_over_z = (rows - projection[1, 2]) / projection[1, 1]
    y_grid, x_grid = np.meshgrid(y_over_z, x_over_z, indexing="ij")
    directions = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def world_normals_to_camera(world_normals: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    """Rotate world normals into the camera and make them point toward the camera.

    A normal that faces the camera has a negative camera-space Z, because the
    camera's +Z axis points into the scene.
    """
    rotation = np.asarray(camera_to_world, dtype=np.float64)[:3, :3]
    normals = np.asarray(world_normals, dtype=np.float64) @ rotation
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = length[..., 0] > 1e-8
    unit = np.divide(normals, np.maximum(length, 1e-8))
    facing_away = unit[..., 2] > 0.0
    unit[facing_away] *= -1.0
    unit[~valid] = 0.0
    return unit


def aolp_pair(aolp: np.ndarray, dolp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``cos(2 AoLP)`` and ``sin(2 AoLP)``. AoLP is 0 where DoLP < 0.05."""
    angle = np.asarray(aolp, dtype=np.float64)
    masked = np.asarray(dolp, dtype=np.float64) < AOLP_DOLP_MIN
    angle = np.where(masked, 0.0, angle)
    doubled = 2.0 * angle
    return np.cos(doubled), np.sin(doubled)


def aolp_from_pair(cosine: np.ndarray, sine: np.ndarray) -> np.ndarray:
    """Invert :func:`aolp_pair`. The result is in ``[-π/2, π/2]``."""
    return 0.5 * np.arctan2(np.asarray(sine, dtype=np.float64), np.asarray(cosine, dtype=np.float64))


def _transmitted_cosine(ior: float, theta: np.ndarray) -> np.ndarray:
    sin_t2 = np.sin(theta) ** 2 / float(ior) ** 2
    return np.sqrt(np.clip(1.0 - sin_t2, 0.0, None))


def diffuse_dolp(theta: np.ndarray, ior: float = IOR) -> np.ndarray:
    """DoLP of unpolarized light reflected by a diffuse dielectric.

    ``(n - 1/n)^2 sin^2(θ) / (2 + 2n^2 - (n + 1/n)^2 sin^2(θ) + 4 cos(θ) sqrt(n^2 - sin^2(θ)))``.
    θ is the zenith angle in radians. At θ = 0 the value is 0.
    """
    theta = np.asarray(theta, dtype=np.float64)
    eta = float(ior)
    sine = np.sin(theta)
    cosine = np.cos(theta)
    contrast = eta - 1.0 / eta
    numerator = (contrast ** 2) * sine ** 2
    inside = np.clip(eta ** 2 - sine ** 2, 0.0, None)
    denominator = 2.0 + 2.0 * eta ** 2 - (contrast ** 2) * sine ** 2 + 4.0 * cosine * np.sqrt(inside)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)


def specular_dolp(theta: np.ndarray, ior: float = IOR) -> np.ndarray:
    """DoLP of unpolarized light reflected by a smooth dielectric, ``|Rs - Rp| / (Rs + Rp)``."""
    theta = np.asarray(theta, dtype=np.float64)
    eta = float(ior)
    cos_i = np.cos(theta)
    cos_t = _transmitted_cosine(eta, theta)
    rs = ((cos_i - eta * cos_t) / (cos_i + eta * cos_t)) ** 2
    rp = ((eta * cos_i - cos_t) / (eta * cos_i + cos_t)) ** 2
    total = rs + rp
    return np.divide(np.abs(rs - rp), total, out=np.zeros_like(total), where=total > 0.0)


def zenith_from_dolp(dolp: np.ndarray, branch: str, ior: float = IOR) -> np.ndarray:
    """Smallest zenith whose Fresnel DoLP matches ``dolp``.

    The specular curve rises to 1 near Brewster's angle and falls again, so
    one DoLP can come from two angles. The smaller angle is the one used.
    That choice does not look at the ground-truth normal.
    """
    if branch not in ("diffuse", "specular"):
        raise ValueError(f"branch must be diffuse or specular, got {branch}")
    formula = diffuse_dolp if branch == "diffuse" else specular_dolp
    samples = np.linspace(0.0, 0.5 * np.pi, ZENITH_SAMPLES)
    curve = formula(samples, ior)
    observed = np.clip(np.asarray(dolp, dtype=np.float64), 0.0, 1.0)
    flat = observed.reshape(-1)
    chosen = np.empty(flat.shape[0], dtype=np.float64)
    chunk = 8192
    for start in range(0, flat.shape[0], chunk):
        block = flat[start : start + chunk]
        nearest = np.abs(curve.reshape(1, -1) - block.reshape(-1, 1)).argmin(axis=1)
        chosen[start : start + chunk] = samples[nearest]
    return chosen.reshape(observed.shape)


def _azimuth_basis(view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axes in the plane perpendicular to ``view``. The first follows camera +X."""
    camera_x = np.zeros_like(view)
    camera_x[..., 0] = 1.0
    tangent = camera_x - view * np.sum(view * camera_x, axis=-1, keepdims=True)
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    degenerate = length[..., 0] < 1e-6
    camera_y = np.zeros_like(view)
    camera_y[..., 1] = 1.0
    fallback = camera_y - view * np.sum(view * camera_y, axis=-1, keepdims=True)
    tangent = np.where(degenerate[..., None], fallback, tangent)
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    first = tangent / np.maximum(length, 1e-8)
    second = np.cross(view, first)
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    return first, second


def normal_from_zenith_azimuth(view: np.ndarray, zenith: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """Camera-space normal. ``view`` points into the scene. The normal points back at the camera."""
    view = np.asarray(view, dtype=np.float64)
    first, second = _azimuth_basis(view)
    zenith = np.asarray(zenith, dtype=np.float64)[..., None]
    azimuth = np.asarray(azimuth, dtype=np.float64)[..., None]
    transverse = np.cos(azimuth) * first + np.sin(azimuth) * second
    normal = np.cos(zenith) * (-view) + np.sin(zenith) * transverse
    return normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)


def fresnel_normals(
    dolp: np.ndarray,
    aolp: np.ndarray,
    view: np.ndarray,
    branch: str,
    *,
    ior: float = IOR,
    azimuth_shift: float = 0.0,
) -> np.ndarray:
    """One Fresnel normal per pixel. ``azimuth_shift`` is 0 or π, not chosen from ground truth."""
    zenith = zenith_from_dolp(dolp, branch, ior)
    azimuth = np.where(np.asarray(dolp, dtype=np.float64) < AOLP_DOLP_MIN, 0.0, np.asarray(aolp, dtype=np.float64))
    if branch == "specular":
        azimuth = azimuth + 0.5 * np.pi
    azimuth = azimuth + float(azimuth_shift)
    return normal_from_zenith_azimuth(view, zenith, azimuth)


def angular_error_deg(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Angle in degrees between two normals. Identical unit normals score 0."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    dot = np.clip(np.sum(prediction * truth, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def mean_angular_error_deg(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Mean angular error on ``mask``. An empty mask scores NaN."""
    chosen = np.asarray(mask, dtype=bool)
    if int(chosen.sum()) == 0:
        return float("nan")
    return float(np.mean(angular_error_deg(prediction, truth)[chosen]))


# Same cutoffs as SfPUEL's ``normal_criterion.py``. The comparison is strict.
ANGLE_CUTOFFS_DEG = (11.25, 22.5, 30.0)
# fp32 weight, gradient, and the two Adam moments.
PARAMETER_STATE_BYTES = 16
# Activations for a 256^2 crop in a batch of 4. This is the space check, not an 8 GiB gate.
ACTIVATION_ALLOWANCE_BYTES = 1024**3


def angle_fractions(degrees: np.ndarray) -> dict:
    """Share of angles strictly below 11.25°, 22.5°, and 30°. An empty input is NaN."""
    degrees = np.asarray(degrees, dtype=np.float64).reshape(-1)
    if degrees.size == 0:
        return {
            "fraction_below_11_25": float("nan"),
            "fraction_below_22_5": float("nan"),
            "fraction_below_30": float("nan"),
        }
    return {
        "fraction_below_11_25": float(np.mean(degrees < ANGLE_CUTOFFS_DEG[0])),
        "fraction_below_22_5": float(np.mean(degrees < ANGLE_CUTOFFS_DEG[1])),
        "fraction_below_30": float(np.mean(degrees < ANGLE_CUTOFFS_DEG[2])),
    }


def angular_error_summary(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    """Mean, median, and the fraction of angles strictly below 11.25°, 22.5°, and 30°.

    The mean is ``mean_angular_error_deg``. The median is the midpoint of the
    same angles. Each fraction is the count of angles below the cutoff divided
    by the number of masked pixels. An empty mask scores NaN and zero pixels.
    """
    chosen = np.asarray(mask, dtype=bool)
    pixels = int(chosen.sum())
    if pixels == 0:
        return {
            "mae_deg": float("nan"),
            "median_deg": float("nan"),
            **angle_fractions(np.array([])),
            "pixels": 0,
        }
    degrees = angular_error_deg(prediction, truth)[chosen]
    return {
        "mae_deg": float(np.mean(degrees)),
        "median_deg": float(np.median(degrees)),
        **angle_fractions(degrees),
        "pixels": pixels,
    }


def mean_abs_s3_over_s0(s3: np.ndarray, s0: np.ndarray, mask: np.ndarray) -> float:
    """Mean of ``|S3| / S0`` on masked pixels with S0 above the Stokes floor."""
    chosen = np.asarray(mask, dtype=bool) & (np.asarray(s0, dtype=np.float64) > S0_MIN)
    if int(chosen.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(s3, dtype=np.float64)[chosen]) / np.asarray(s0, dtype=np.float64)[chosen]))


def batch_floor_bytes(n_parameters: int, activation_bytes: int = ACTIVATION_ALLOWANCE_BYTES) -> int:
    """Bytes one Adam step needs: parameter state plus the activation allowance.

    ``n_parameters`` is the trained scalar count. Each scalar keeps an fp32
    weight, a gradient, and two Adam moments, which is 16 bytes. The activation
    allowance is a fixed 1 GiB for a 256^2 batch of 4. A card with more free
    memory than this floor is used.
    """
    if int(n_parameters) < 1:
        raise ValueError(f"n_parameters must be positive, got {n_parameters}")
    return int(n_parameters) * PARAMETER_STATE_BYTES + int(activation_bytes)


def release_render_caches() -> str:
    """Drop Mitsuba's cached buffers. This does not import Torch.

    Importing Torch while Mitsuba is still rendering can reset the CUDA context.
    The Torch cache is emptied later, in ``release_gpu_caches``, after the frames.
    """
    import gc

    gc.collect()
    notes = ["gc=collected"]
    try:
        import drjit as dr

        flush = getattr(dr, "flush_malloc_cache", None)
        if flush is not None:
            flush()
        sync = getattr(dr, "sync_thread", None)
        if sync is not None:
            sync()
        notes.append("drjit_cache=flushed")
    except Exception as exc:
        notes.append(f"drjit_cache=skipped error={type(exc).__name__}")
    return " ".join(notes)


def release_gpu_caches() -> str:
    """Flush Dr.Jit, then the Torch cache when Torch is already imported.

    Call this after the renders, when the free-memory read is about to run.
    Nothing else on the machine is killed.
    """
    notes = [release_render_caches()]
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            notes.append("torch_cache=emptied")
        else:
            notes.append("torch_cache=cuda_unavailable")
    except Exception as exc:
        notes.append(f"torch_cache=skipped error={type(exc).__name__}")
    return " ".join(notes)


def choose_device(cuda_available: bool, free_bytes: int | None, minimum_bytes: int) -> tuple[str, str]:
    """GPU when CUDA works, unless the flush left fewer bytes than the batch floor.

    A missing free-memory reading does not select the CPU: CUDA is still used,
    and the caller logs that the reading failed. Nothing is killed to free the GPU.
    """
    floor = int(minimum_bytes)
    if not cuda_available:
        return "cpu", "torch.cuda.is_available() is false"
    if free_bytes is None:
        return "cuda", "free GPU memory could not be read; CUDA is available"
    if int(free_bytes) < floor:
        return (
            "cpu",
            f"after flush, free GPU memory {int(free_bytes)} bytes is below the batch floor {floor} bytes",
        )
    return "cuda", f"after flush, free GPU memory {int(free_bytes)} bytes covers the batch floor {floor} bytes"


def select_torch_device(n_parameters: int) -> tuple[str, str, dict]:
    """Flush, read free memory, and choose cuda or cpu for ``n_parameters``."""
    import torch

    release_note = release_gpu_caches()
    floor = batch_floor_bytes(n_parameters)
    available = bool(torch.cuda.is_available())
    free = None
    total = None
    if available:
        try:
            free_raw, total_raw = torch.cuda.mem_get_info()
            free, total = int(free_raw), int(total_raw)
        except Exception:
            free, total = None, None
    device, reason = choose_device(available, free, floor)
    info = {
        "selected": device,
        "reason": reason,
        "torch_version": torch.__version__,
        "cuda_available": available,
        "free_bytes": free,
        "total_bytes": total,
        "batch_floor_bytes": floor,
        "n_parameters": int(n_parameters),
        "release": release_note,
    }
    return device, reason, info


def object_pixel_mask(shape_index: np.ndarray, object_indices: set[int], s0: np.ndarray) -> np.ndarray:
    """Object pixels whose S0 clears the Stokes floor. Walls are not included."""
    index = np.rint(np.asarray(shape_index)).astype(np.int64)
    selected = np.isin(index, list(object_indices))
    return selected & (np.asarray(s0, dtype=np.float64) > S0_MIN)

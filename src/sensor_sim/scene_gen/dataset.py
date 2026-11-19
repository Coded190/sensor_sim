"""Atomic dataset samples and the loader that reads them."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from sensor_sim.film.images import read_exr
from sensor_sim.scene_gen.convention import S0_MIN
from sensor_sim.scene_gen.render_sample import stack_stokes

GROUND_TRUTH = (
    "normal_geo_world",
    "normal_geo_camera",
    "normal_shading_world",
    "normal_shading_camera",
    "depth",
    "position",
    "object_ids",
    "material_ids",
    "valid_mask",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def channels_to_array(channels: dict[str, np.ndarray]) -> np.ndarray:
    """One EXR split dict as an array. A single multi-channel image is kept."""
    if len(channels) == 1:
        return np.asarray(next(iter(channels.values())), dtype=np.float64)

    def rank(key: str) -> int:
        for suffix, order in (("X", 0), ("Y", 1), ("Z", 2), ("R", 0), ("G", 1), ("B", 2)):
            if key == suffix or key.endswith("." + suffix):
                return order
        return 10

    ordered = [channels[key] for key in sorted(channels, key=rank)]
    return np.stack([np.asarray(value, dtype=np.float64) for value in ordered], axis=-1)


def _finite(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all())


def validate_sample(directory: Path) -> None:
    """Reject an incomplete or non-finite sample before it is registered."""
    directory = Path(directory)
    required = [
        directory / "scene_config.json",
        directory / "scene_exported.xml",
        directory / "scene.xml",
        directory / "polarization.exr",
        directory / "metadata.json",
    ]
    required.extend(directory / "ground_truth" / f"{name}.exr" for name in GROUND_TRUTH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"sample is missing {missing}")
    stokes = read_exr(directory / "polarization.exr")
    components = [stack_stokes(stokes, name) for name in ("S0", "S1", "S2", "S3")]
    shapes = {component.shape[:2] for component in components}
    if len(shapes) != 1:
        raise RuntimeError(f"Stokes shapes disagree: {shapes}")
    height, width = components[0].shape[:2]
    for name in GROUND_TRUTH:
        image = read_exr(directory / "ground_truth" / f"{name}.exr")
        array = np.concatenate([values.reshape(values.shape[0], values.shape[1], -1) for values in image.values()], axis=-1)
        if array.shape[0] != height or array.shape[1] != width:
            raise RuntimeError(f"{name} is {array.shape[:2]}, Stokes is {(height, width)}")
        if not _finite(array):
            raise RuntimeError(f"{name} has a NaN or infinite value")
    for component in components:
        if not _finite(component):
            raise RuntimeError("a Stokes component has a NaN or infinite value")
    valid = channels_to_array(read_exr(directory / "ground_truth" / "valid_mask.exr"))
    depth = channels_to_array(read_exr(directory / "ground_truth" / "depth.exr"))
    geo_array = channels_to_array(read_exr(directory / "ground_truth" / "normal_geo_world.exr"))
    if geo_array.ndim == 2:
        raise RuntimeError("geometric normals have one channel")
    mask = np.asarray(valid).reshape(height, width) > 0.5
    if int(mask.sum()) == 0:
        raise RuntimeError("valid mask is empty")
    lengths = np.linalg.norm(geo_array, axis=-1)
    if float(np.max(np.abs(lengths[mask] - 1.0))) > 1e-2:
        raise RuntimeError("a foreground geometric normal is not unit length")
    if float(np.min(np.asarray(depth).reshape(height, width)[mask])) <= 0.0:
        raise RuntimeError("foreground depth is not positive")


def publish(staging: Path, destination: Path) -> None:
    """Move a validated staging directory into the dataset."""
    from sensor_sim.scene_gen.provenance import publication_issues

    staging = Path(staging)
    destination = Path(destination)
    issues = publication_issues(staging)
    if issues:
        raise RuntimeError("publication audit failed: " + "; ".join(issues))
    validate_sample(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"{destination} already exists")
    os.rename(staging, destination)


def record_failure(root: Path, scene_id: str, message: str, **extra) -> None:
    """Write a failure that can be rerun. ``error`` stays the human message."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene_id": scene_id,
        "status": "failed",
        "error": message,
        "exception_type": extra.pop("exception_type", "RuntimeError"),
        "exception_message": message,
    }
    payload.update(extra)
    (root / f"{scene_id}.json").write_text(json.dumps(payload) + "\n")


def _normal(directory: Path, name: str) -> np.ndarray:
    return channels_to_array(read_exr(directory / "ground_truth" / f"{name}.exr"))


def load_sample(directory: Path, *, dataset_v1: bool = False) -> dict:
    """Stokes tensors in order S0, S1, S2, S3 and the aligned ground truth.

    ``normal_supervision_mask`` is a separate output. The five training views
    are unchanged. Schema 2 metadata is not read as a schema 3 object-id count.
    """
    directory = Path(directory)
    stokes = read_exr(directory / "polarization.exr")
    components = [np.asarray(stack_stokes(stokes, name), dtype=np.float64) for name in ("S0", "S1", "S2", "S3")]
    if components[0].ndim == 3:
        averaged = [component.mean(axis=2) for component in components]
    else:
        averaged = components
    height, width = averaged[0].shape
    valid = channels_to_array(read_exr(directory / "ground_truth" / "valid_mask.exr"))
    valid = np.asarray(valid, dtype=np.float64).reshape(height, width) > 0.5
    raw_object_ids = np.asarray(
        channels_to_array(read_exr(directory / "ground_truth" / "object_ids.exr")),
        dtype=np.float64,
    ).reshape(height, width)
    labels, fractional = discrete_object_ids(raw_object_ids)
    object_ids = np.where(fractional, 0, labels).astype(np.int32)
    material_ids = channels_to_array(read_exr(directory / "ground_truth" / "material_ids.exr"))
    material_ids = np.rint(np.asarray(material_ids, dtype=np.float64).reshape(height, width)).astype(np.int32)
    s0 = averaged[0]
    measurement = valid & (s0 > S0_MIN)
    root = stokes.get("<root>")
    if root is None:
        rgb = np.zeros((height, width, 3), dtype=np.float64)
    else:
        rgb = np.asarray(root, dtype=np.float64)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    sample = {
        "rgb": rgb[:, :, :3],
        "s0": averaged[0],
        "s1": averaged[1],
        "s2": averaged[2],
        "s3": averaged[3],
        "normal_geo_world": _normal(directory, "normal_geo_world"),
        "normal_geo_camera": _normal(directory, "normal_geo_camera"),
        "normal_shading_world": _normal(directory, "normal_shading_world"),
        "normal_shading_camera": _normal(directory, "normal_shading_camera"),
        "depth": np.asarray(channels_to_array(read_exr(directory / "ground_truth" / "depth.exr")), dtype=np.float64).reshape(height, width),
        "object_ids": object_ids,
        "material_ids": material_ids,
        "valid_mask": valid,
        "geometry_mask": valid,
        "object_mask": object_ids > 0,
        "material_mask": material_ids > 0,
        "instance_mask": object_ids > 0,
        "measurement_mask": measurement,
        "polarization_mask": measurement,
        "invalid_fractional_pixel_count": int(fractional.sum()),
    }
    metadata = _read_metadata(directory)
    if dataset_v1:
        schema = 0 if metadata is None else int(metadata.get("schema_version") or 0)
        if schema != 3:
            raise RuntimeError(f"Dataset v1 load requires schema 3, found {schema}")
        issues = object_id_publication_issues(raw_object_ids, metadata)
        if issues:
            raise RuntimeError("Dataset v1 object ids failed: " + "; ".join(issues))
    sample["normal_supervision_mask"] = normal_supervision_mask(
        sample,
        metadata if metadata is not None and int(metadata.get("schema_version") or 0) >= 3 else None,
    )
    return sample


def _read_metadata(directory: Path) -> dict | None:
    path = Path(directory) / "metadata.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


ID_TOLERANCE = 1e-3
UNIT_NORMAL_TOLERANCE = 1e-2
VISIBILITY_POLICY_VERSION = 1
TRAINING_MIN_OBJECT_PIXELS = 80


def discrete_object_ids(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer ids and the fractional mask.

    A pixel is valid when it is within ``1e-3`` of an integer. Background is 0.
    Fractional pixels are not assigned to an object.
    """
    raw = np.asarray(values, dtype=np.float64)
    raw = raw.reshape(raw.shape[0], raw.shape[1]) if raw.ndim > 2 else raw
    fractional = np.abs(raw - np.rint(raw)) >= ID_TOLERANCE
    labels = np.where(fractional, -1, np.rint(raw)).astype(np.int32)
    return labels, fractional


def raw_object_counts(object_ids: np.ndarray) -> dict[int, int]:
    """Count object-id pixels from the image. Background id 0 is omitted.

    Fractional ids are excluded. This is ``object_id_pixel_count``, not the
    legacy position-ownership ``visible_pixels`` field.
    """
    labels, fractional = discrete_object_ids(object_ids)
    return {
        int(value): int(np.count_nonzero((labels == int(value)) & ~fractional))
        for value in np.unique(labels)
        if int(value) > 0
    }


def normal_valid_mask(normals: np.ndarray) -> np.ndarray:
    """Finite unit shading normals. The stored normals are not modified."""
    array = np.asarray(normals, dtype=np.float64)
    lengths = np.linalg.norm(array, axis=-1)
    return np.isfinite(array).all(axis=-1) & (np.abs(lengths - 1.0) < UNIT_NORMAL_TOLERANCE)


def normal_supervision_mask(sample: dict, metadata: dict | None = None) -> np.ndarray:
    """Pixels whose shading normal may supervise training.

    The mask is false for background, fully hidden objects, and objects with
    1–79 object-id pixels. It is true only for an object with at least 80
    object-id pixels where the shading normal is a finite unit vector. Schema 3
    metadata must agree with the image. Schema 2 ``visible_pixels`` is not
    treated as that count.
    """
    labels, fractional = discrete_object_ids(sample["object_ids"])
    counts = raw_object_counts(sample["object_ids"])
    if metadata is not None and int(metadata.get("schema_version") or 0) >= 3:
        for item in metadata.get("shape_map") or []:
            if item.get("asset_id") == "plane" or str(item.get("object_id")) == "ground":
                continue
            label = int(item["object_index"])
            recorded = item.get("object_id_pixel_count")
            if recorded is None:
                raise RuntimeError(f"object {label} is missing object_id_pixel_count")
            actual = counts.get(label, 0)
            if int(recorded) != actual:
                raise RuntimeError(
                    f"object {label} metadata object_id_pixel_count {recorded} != image count {actual}"
                )
    valid = normal_valid_mask(sample["normal_shading_camera"])
    mask = np.zeros(labels.shape, dtype=bool)
    for label, count in counts.items():
        if count >= TRAINING_MIN_OBJECT_PIXELS:
            mask |= (labels == label) & ~fractional & valid
    return mask


def object_id_publication_issues(raw_ids: np.ndarray, metadata: dict) -> list[str]:
    """Schema 3 image checks. Legacy ``visible_pixels`` is not consulted."""
    issues = []
    if int(metadata.get("schema_version") or 0) != 3:
        issues.append(f"schema_version {metadata.get('schema_version')} is not 3")
    if int(metadata.get("visibility_policy_version") or 0) != VISIBILITY_POLICY_VERSION:
        issues.append("visibility policy version is not 1")
    labels, fractional = discrete_object_ids(raw_ids)
    invalid = int(fractional.sum())
    if invalid != int(metadata.get("invalid_fractional_pixel_count") or 0):
        issues.append(f"invalid fractional pixel count {invalid} disagrees with metadata")
    if invalid:
        issues.append(f"object-id image has {invalid} fractional pixels")
    known = {int(item["object_index"]) for item in metadata.get("shape_map") or []}
    unknown = (~fractional) & (labels > 0) & ~np.isin(labels, np.array(sorted(known) or [-1], dtype=np.int32))
    if int(unknown.sum()):
        issues.append(f"object-id image has {int(unknown.sum())} unknown non-background pixels")
    counts = raw_object_counts(raw_ids)
    for item in metadata.get("shape_map") or []:
        label = int(item["object_index"])
        recorded = item.get("object_id_pixel_count")
        actual = counts.get(label, 0)
        if recorded is None or int(recorded) != actual:
            issues.append(f"object {label} object_id_pixel_count {recorded} != image count {actual}")
    return issues


def visibility_category(raw_pixels: int) -> str:
    if int(raw_pixels) <= 0:
        return "fully_occluded"
    if int(raw_pixels) < TRAINING_MIN_OBJECT_PIXELS:
        return "low_visibility"
    return "training_eligible"


def training_views(sample: dict) -> dict[str, np.ndarray]:
    """Float32 model inputs. Raw Stokes are not clamped and are not overwritten."""
    s0 = np.asarray(sample["s0"], dtype=np.float32)
    s1 = np.asarray(sample["s1"], dtype=np.float32)
    s2 = np.asarray(sample["s2"], dtype=np.float32)
    s3 = np.asarray(sample["s3"], dtype=np.float32)
    rgb = np.asarray(sample["rgb"], dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    linear = np.sqrt(s1**2 + s2**2)
    dolp = np.zeros_like(s0)
    np.divide(linear, s0, out=dolp, where=s0 > S0_MIN)
    aolp = (0.5 * np.arctan2(s2, s1)).astype(np.float32)
    aolp = np.where((s0 > S0_MIN) & (dolp >= 0.05), aolp, 0.0).astype(np.float32)
    stokes = np.stack([s0, s1, s2, s3], axis=-1)
    return {
        "s0": s0[:, :, None],
        "s0_s2": np.stack([s0, s1, s2], axis=-1),
        "stokes": stokes,
        "dolp_aolp": np.stack([dolp.astype(np.float32), aolp], axis=-1),
        "rgb_stokes": np.concatenate([rgb[:, :, :3], stokes], axis=-1),
    }


def sample_integrity(sample: dict) -> dict:
    """Geometry validity is not the same thing as a polarized measurement."""
    height, width = sample["s0"].shape[:2]
    issues = []
    for name in ("s0", "object_ids", "material_ids", "depth", "geometry_mask"):
        if sample[name].shape[:2] != (height, width):
            issues.append(f"{name} shape {sample[name].shape}")
    for name in ("normal_geo_camera", "normal_shading_camera", "normal_geo_world", "normal_shading_world"):
        if sample[name].shape[:2] != (height, width) or sample[name].shape[-1] != 3:
            issues.append(f"{name} shape {sample[name].shape}")
    mask = np.asarray(sample["geometry_mask"], dtype=bool)
    for name in ("normal_geo_world", "normal_shading_world", "normal_geo_camera", "normal_shading_camera"):
        if int(mask.sum()) == 0:
            continue
        lengths = np.linalg.norm(sample[name][mask], axis=-1)
        if not np.isfinite(sample[name][mask]).all() or float(np.max(np.abs(lengths - 1.0))) > 1e-2:
            issues.append(f"{name} is not a finite unit normal on the geometry mask")
    if int(mask.sum()) and float(np.min(sample["depth"][mask])) <= 0.0:
        issues.append("depth is not positive on the geometry mask")
    finite_normal = np.isfinite(sample["normal_geo_world"]).all(axis=-1)
    if int(np.count_nonzero(~finite_normal & (sample["object_ids"] > 0))):
        issues.append("a non-finite normal pixel has an object id")
    low = mask & (sample["s0"] > S0_MIN) & (np.sqrt(sample["s1"] ** 2 + sample["s2"] ** 2) / np.maximum(sample["s0"], 1e-8) < 0.05)
    return {
        "ok": not issues,
        "issues": issues,
        "n_geometry": int(mask.sum()),
        "n_measurement": int(np.asarray(sample["measurement_mask"]).sum()),
        "n_low_dolp_still_geometry": int(low.sum()),
        "representation": "broadband RGB-converted Stokes, not wavelength-resolved",
    }


def exr_schema(path: Path) -> dict:
    """File channel names, and the groups the Mitsuba reader actually returns.

    A Stokes film is often described as 16 channels, RGBA plus RGB for S0-S3.
    This writer stores 15 float32 channels: a leading RGB and RGB for S0-S3.
    ``Bitmap.split`` groups those into five images and does not return the
    leading RGB under the S0-S3 names. The loader averages S0-S3 and does not
    read the leading RGB.
    """
    import mitsuba as mi

    bitmap = mi.Bitmap(str(path))
    file_channels = [field.name for field in bitmap.struct_()]
    parts = []
    for name, image in bitmap.split():
        parts.append({"name": str(name), "channels": [field.name for field in image.struct_()]})
    return {
        "path": str(path),
        "pixel_format": str(bitmap.pixel_format()),
        "component_format": str(bitmap.component_format()),
        "height": int(bitmap.height()),
        "width": int(bitmap.width()),
        "file_channel_count": len(file_channels),
        "file_channels": file_channels,
        "reader_parts": parts,
        "reader_channel_count": len(parts),
        "not_in_file": "alpha",
        "loader_uses": ["S0", "S1", "S2", "S3"],
        "loader_ignores": "<root> RGB",
    }

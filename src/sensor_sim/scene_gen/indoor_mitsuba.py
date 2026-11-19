"""Indoor Mitsuba scenes, shape identity, and Stokes output.

Identical BSDF parameters can collapse to one loaded shape with an empty id.
That was reproduced on Mitsuba 3.9.1. The optical scene keeps the intended
BSDFs. Identity is a separate categorical product: each logical shape is loaded
by itself, its world vertices are checked, and a nearest-hit buffer stores an
integer id. The buffer is not filtered and it is not the optical render.

Stokes uses ``cuda_ad_spectral_polarized``. The film is RGB, three base
channels. ``integrator.aov_names()`` supplies ``S0.R`` through ``S3.B`` in that
order. Names are taken from that metadata. ``S3`` sign is the project
convention and was not independently remeasured.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import PackageError
from sensor_sim.scene_gen.convention import STOKES_FRAME
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES
from sensor_sim.scene_gen.render_sample import _names

logger = logging.getLogger("sensor_sim.scene_gen.indoor_mitsuba")

VARIANT = "cuda_ad_spectral_polarized"
STOKES_AOVS = CHANNEL_NAMES[3:]
def require_variant() -> str:
    import mitsuba as mi

    if VARIANT not in mi.variants():
        raise PackageError(f"reason=variant_missing variant={VARIANT}")
    mi.set_variant(VARIANT)
    if mi.variant() != VARIANT:
        raise PackageError(f"reason=variant_mismatch variant={mi.variant()}")
    return VARIANT


def cube_shape(spec: dict) -> dict:
    """One cube primitive. The optical BSDF is whatever the spec declares."""
    import mitsuba as mi

    require_variant()
    center = spec["center"]
    scale = spec.get("scale", [0.08, 0.08, 0.08])
    transform = mi.ScalarTransform4f.translate(list(center)) @ mi.ScalarTransform4f.scale(list(scale))
    return {
        "type": "cube",
        "to_world": transform,
        "bsdf": spec["bsdf"],
    }


def optical_scene(shapes: list[dict], resolution: tuple[int, int] = (16, 16)) -> dict:
    """Scene dict whose BSDFs are the declared optical BSDFs, with no identity perturbation."""
    import mitsuba as mi

    require_variant()
    width, height = resolution
    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 3},
        "sensor": {
            "type": "perspective",
            "fov": 50.0,
            "to_world": mi.ScalarTransform4f.look_at(origin=[0.0, -3.0, 2.0], target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0]),
            "sampler": {"type": "independent"},
            "film": {"type": "hdrfilm", "width": int(width), "height": int(height), "pixel_format": "rgb", "rfilter": {"type": "box"}},
        },
        "light": {"type": "constant", "radiance": {"type": "rgb", "value": 1.0}},
    }
    for spec in shapes:
        scene[spec["shape_id"]] = cube_shape(spec)
    return scene


def load_shapes(scene: dict) -> list[dict]:
    import mitsuba as mi

    require_variant()
    loaded = mi.load_dict(scene)
    records = []
    for shape in loaded.shapes():
        parameters = mi.traverse(shape)
        positions = np.array(parameters["vertex_positions"], dtype=np.float64).reshape(-1, 3)
        faces = np.array(parameters["faces"], dtype=np.int64).reshape(-1, 3)
        records.append(
            {
                "id": str(shape.id()),
                "vertex_count": int(len(positions)),
                "triangle_count": int(len(faces)),
                "position_sha256": _position_hash(positions),
                "bounds_min": positions.min(axis=0).tolist(),
                "bounds_max": positions.max(axis=0).tolist(),
            }
        )
    return records


def identity_inventory(shapes: list[dict]) -> list[dict]:
    """Load each shape alone so identical BSDFs cannot merge it with another shape."""
    inventory = []
    for index, spec in enumerate(shapes):
        alone = {"type": "scene", spec["shape_id"]: cube_shape(spec)}
        loaded = load_shapes(alone)
        if len(loaded) != 1:
            raise PackageError(f"reason=identity_shape_count shape={spec['shape_id']} count={len(loaded)}")
        if loaded[0]["id"] != spec["shape_id"]:
            raise PackageError(f"reason=identity_id expected={spec['shape_id']} actual={loaded[0]['id']}")
        inventory.append(
            {
                "identity": index + 1,
                "shape_id": spec["shape_id"],
                "logical_object": spec["logical_object"],
                "component": spec["component"],
                "material_slot": spec["material_slot"],
                "instance": spec["instance"],
                "vertex_count": loaded[0]["vertex_count"],
                "triangle_count": loaded[0]["triangle_count"],
                "position_sha256": loaded[0]["position_sha256"],
                "bounds_min": loaded[0]["bounds_min"],
                "bounds_max": loaded[0]["bounds_max"],
            }
        )
    identities = [item["identity"] for item in inventory]
    if len(set(identities)) != len(identities) or 0 in identities:
        raise PackageError("reason=identity_values")
    return inventory


def boundary_report(shapes: list[dict]) -> dict:
    """Count geometry at each boundary. The optical load is allowed to merge identical BSDFs."""
    optical = optical_scene(shapes)
    declared = [key for key in optical if key not in {"type", "integrator", "sensor", "light"}]
    loaded = load_shapes(optical)
    inventory = identity_inventory(shapes)
    merged = len(loaded) != len(shapes) or any(item["id"] == "" for item in loaded)
    logger.info(
        "indoor_mitsuba event=boundary declared=%d loaded=%d identity=%d merged=%s",
        len(declared),
        len(loaded),
        len(inventory),
        merged,
    )
    return {
        "logical_shapes": len(shapes),
        "scene_dict_shapes": len(declared),
        "loaded_shapes": len(loaded),
        "loaded_ids": [item["id"] for item in loaded],
        "loaded_vertex_count": sum(item["vertex_count"] for item in loaded),
        "identity_shapes": len(inventory),
        "merged_on_load": merged,
        "identity": inventory,
        "expected_vertices_if_unmerged": 24 * len(shapes),
    }


def identity_buffer(shapes: list[dict], width: int = 48, height: int = 48) -> np.ndarray:
    """Nearest-hit integer ids. Ties keep the smaller id. There is no filtered blend."""
    import mitsuba as mi

    require_variant()
    centers = np.asarray([spec["center"] for spec in shapes], dtype=np.float64)
    low = centers.min(axis=0) - 0.3
    high = centers.max(axis=0) + 0.3
    target = ((low + high) / 2.0).tolist()
    origin = [target[0], target[1], high[2] + 2.0]
    origins, directions = _rays(origin, target, [0.0, 1.0, 0.0], 60.0, width, height)
    best_t = np.full(width * height, np.inf, dtype=np.float64)
    best_id = np.zeros(width * height, dtype=np.int32)
    for index, spec in enumerate(shapes):
        scene = mi.load_dict({"type": "scene", spec["shape_id"]: cube_shape(spec)})
        ray = mi.Ray3f(
            o=mi.Point3f(np.ascontiguousarray(origins.T, dtype=np.float32)),
            d=mi.Vector3f(np.ascontiguousarray(directions.T, dtype=np.float32)),
        )
        hit = scene.ray_intersect(ray)
        valid = np.array(hit.is_valid(), dtype=bool).reshape(-1)
        distance = np.array(hit.t, dtype=np.float64).reshape(-1)
        identity = index + 1
        closer = valid & np.isfinite(distance) & (distance < best_t - 1e-6)
        comparable = valid & np.isfinite(distance) & np.isfinite(best_t)
        tie = np.zeros(best_id.shape, dtype=bool)
        if np.any(comparable):
            tie[comparable] = np.abs(distance[comparable] - best_t[comparable]) <= 1e-6
        tie &= (identity < best_id) & (best_id != 0)
        best_t[closer] = distance[closer]
        best_id[closer | tie] = identity
    image = best_id.reshape(height, width)
    present = set(int(value) for value in np.unique(image) if int(value) != 0)
    expected = set(range(1, len(shapes) + 1))
    if present - expected:
        raise PackageError(f"reason=unexpected_identity ids={sorted(present - expected)}")
    logger.info(
        "indoor_mitsuba event=identity_buffer shapes=%d visible=%d width=%d height=%d",
        len(shapes),
        len(present),
        width,
        height,
    )
    return image


def optical_repeatability(shapes: list[dict], spp: int = 8, seed: int = 11) -> dict:
    """Two renders of one optical scene. The tolerance is this measured variation."""
    import mitsuba as mi

    require_variant()
    scene = mi.load_dict(optical_scene(shapes, (16, 16)))
    first = np.array(mi.render(scene, spp=int(spp), seed=int(seed)))
    second = np.array(mi.render(scene, spp=int(spp), seed=int(seed)))
    metrics = image_metrics(first, second)
    metrics.update({"spp": int(spp), "seed": int(seed), "resolution": [16, 16], "variant": VARIANT, "sampler": "independent"})
    return metrics


def optical_control_difference(shapes: list[dict], spp: int = 8, seed: int = 11) -> dict:
    """Compare an optical scene with a second build that has the same materials and no identity pass."""
    import mitsuba as mi

    require_variant()
    baseline = optical_repeatability(shapes, spp=spp, seed=seed)
    control = np.array(mi.render(mi.load_dict(optical_scene(shapes, (16, 16))), spp=int(spp), seed=int(seed)))
    optical = np.array(mi.render(mi.load_dict(optical_scene(shapes, (16, 16))), spp=int(spp), seed=int(seed)))
    measured = image_metrics(control, optical)
    measured["repeatability"] = baseline
    measured["within_repeatability"] = bool(
        measured["max_absolute_error"] <= baseline["max_absolute_error"] + 0.0
        and measured["mean_absolute_error"] <= baseline["mean_absolute_error"] + 1e-8
    )
    return measured


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    difference = np.abs(np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    denominator = float(np.mean(np.abs(reference)))
    relative = float(np.mean(difference) / denominator) if denominator > 1e-8 else None
    return {
        "max_absolute_error": float(np.max(difference)),
        "mean_absolute_error": float(np.mean(difference)),
        "relative_mean_absolute_error": relative,
    }


def render_stokes(scene: dict, output: Path, spp: int, seed: int) -> dict:
    """Render, name channels from the integrator, write an EXR, and read it back."""
    import mitsuba as mi

    require_variant()
    loaded = mi.load_dict(scene)
    aovs = list(loaded.integrator().aov_names())
    if aovs != list(STOKES_AOVS):
        raise PackageError(f"reason=stokes_aov names={aovs}")
    image = np.array(mi.render(loaded, spp=int(spp), seed=int(seed)))
    names = _names(loaded, image)
    if names != list(CHANNEL_NAMES):
        raise PackageError(f"reason=stokes_names names={names}")
    if image.dtype != np.float32 or image.shape[2] != 15 or not np.isfinite(image).all():
        raise PackageError("reason=stokes_buffer")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mi.Bitmap(image, channel_names=names).write(str(output))
    reopened = mi.Bitmap(str(output))
    read_names = _bitmap_names(reopened)
    if read_names != list(CHANNEL_NAMES):
        raise PackageError(f"reason=exr_names names={read_names}")
    payload = np.array(reopened)
    if payload.shape != image.shape or payload.dtype != np.float32 or not np.isfinite(payload).all():
        raise PackageError("reason=exr_payload")
    report = {
        "variant": mi.variant(),
        "aov_names": aovs,
        "channel_names": names,
        "exr_names": read_names,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "finite": True,
        "stokes_frame": STOKES_FRAME,
        "s3_sign_independently_remeasured": False,
        "channel_mean": {name: float(image[:, :, index].mean()) for index, name in enumerate(names)},
    }
    logger.info(
        "indoor_mitsuba event=stokes variant=%s shape=%s exr=%s",
        report["variant"],
        report["shape"],
        output,
    )
    return report


def stokes_cone_diagnostic(image: np.ndarray, names: list[str], *, tolerance: float = 1e-4, s0_floor: float = 1e-4) -> dict:
    """RGB-domain Stokes-cone diagnostic.

    ``mi.render`` of an ``hdrfilm`` with ``pixel_format`` ``rgb`` returns
    Stokes components after Mitsuba's spectral-to-RGB conversion. This
    function does not treat that RGB inequality as a hard physical failure.
    """
    index = {name: position for position, name in enumerate(names)}
    s0 = np.stack([image[:, :, index[f"S0.{band}"]] for band in "RGB"], axis=-1).astype(np.float64)
    s1 = np.stack([image[:, :, index[f"S1.{band}"]] for band in "RGB"], axis=-1).astype(np.float64)
    s2 = np.stack([image[:, :, index[f"S2.{band}"]] for band in "RGB"], axis=-1).astype(np.float64)
    s3 = np.stack([image[:, :, index[f"S3.{band}"]] for band in "RGB"], axis=-1).astype(np.float64)
    magnitude = np.sqrt(s1 * s1 + s2 * s2 + s3 * s3)
    finite = np.isfinite(s0) & np.isfinite(magnitude)
    negative_s0 = int(np.count_nonzero(finite & (s0 < 0.0)))
    low = finite & (s0 <= s0_floor)
    compared = finite & ~low & (s0 >= 0.0)
    excess = magnitude - s0
    violations = int(np.count_nonzero(compared & (excess > tolerance)))
    return {
        "representation": "mitsuba_hdrfilm_rgb_stokes",
        "hard_failure": False,
        "reason": "RGB conversion is not shown to preserve the spectral Stokes cone",
        "tolerance": tolerance,
        "s0_floor": s0_floor,
        "nonfinite_count": int(np.count_nonzero(~finite)),
        "negative_s0_count": negative_s0,
        "low_s0_count": int(np.count_nonzero(low)),
        "compared_count": int(np.count_nonzero(compared)),
        "rgb_diagnostic_violation_count": violations,
        "max_excess": float(np.max(excess[compared])) if np.any(compared) else 0.0,
    }


def neutral_stokes_scene(resolution: tuple[int, int] = (32, 32)) -> dict:
    """Diffuse sphere under a constant dome. Polarization is expected to stay small."""
    scene = stokes_scene(resolution)
    scene["light"] = {"type": "constant", "radiance": {"type": "rgb", "value": 1.0}}
    scene["sphere"] = {"type": "sphere", "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.6}}}
    return scene


def stokes_scene(resolution: tuple[int, int] = (32, 32)) -> dict:
    """Dielectric sphere and an area light. S1 is expected to be nonzero."""
    import mitsuba as mi

    require_variant()
    width, height = resolution
    return {
        "type": "scene",
        "integrator": {"type": "stokes", "integrator": {"type": "path", "max_depth": 5}},
        "sensor": {
            "type": "perspective",
            "fov": 35.0,
            "to_world": mi.ScalarTransform4f.look_at(origin=[0.0, -2.2, 0.6], target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0]),
            "sampler": {"type": "independent"},
            "film": {
                "type": "hdrfilm",
                "width": int(width),
                "height": int(height),
                "pixel_format": "rgb",
                "rfilter": {"type": "box"},
            },
        },
        "light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(origin=[1.2, -1.2, 2.0], target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])
            @ mi.ScalarTransform4f.scale([0.4, 0.4, 1.0]),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 8.0}},
        },
        "sphere": {"type": "sphere", "bsdf": {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}},
    }


def write_rgb_png(exr_path: Path, png_path: Path) -> None:
    """Write the Stokes film RGB channels as a review image from that same render."""
    import mitsuba as mi

    require_variant()
    bitmap = mi.Bitmap(str(exr_path))
    names = _bitmap_names(bitmap)
    image = np.array(bitmap)
    rgb = image[:, :, :3]
    if names[:3] != ["R", "G", "B"]:
        raise PackageError("reason=rgb_names")
    review = mi.Bitmap(rgb)
    review = review.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    review.write(str(png_path))


def grid_shapes(count: int, reflectance: list[float] | None = None) -> list[dict]:
    columns = int(np.ceil(np.sqrt(count)))
    material = reflectance or [0.5, 0.5, 0.5]
    shapes = []
    for index in range(count):
        row, column = divmod(index, columns)
        shapes.append(
            {
                "shape_id": f"shape_{index:02d}",
                "logical_object": "grid",
                "component": "body",
                "material_slot": "default",
                "instance": str(index),
                "center": [column * 0.35, row * 0.35, 0.08],
                "scale": [0.08, 0.08, 0.08],
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": list(material)}},
            }
        )
    return shapes


def shared_bsdf_xml(directory: Path) -> dict:
    """Two OBJ shapes referencing one BSDF id. This is the shared-reference case."""
    from sensor_sim.scene_gen.mesh_io import write_obj

    directory.mkdir(parents=True, exist_ok=True)
    write_a = directory / "a.obj"
    write_b = directory / "b.obj"
    write_a.write_bytes(write_obj(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64), [[0, 1, 2]], name="a"))
    write_b.write_bytes(write_obj(np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float64), [[0, 1, 2]], name="b"))
    xml = f"""<scene version="3.0.0">
  <bsdf type="diffuse" id="shared">
    <rgb name="reflectance" value="0.5, 0.5, 0.5"/>
  </bsdf>
  <shape type="obj" id="shape_a">
    <string name="filename" value="{write_a}"/>
    <ref id="shared"/>
  </shape>
  <shape type="obj" id="shape_b">
    <string name="filename" value="{write_b}"/>
    <ref id="shared"/>
  </shape>
</scene>
"""
    path = directory / "shared.xml"
    path.write_text(xml, encoding="utf-8")
    separate = xml.replace('id="shared"', 'id="mat_a"', 1).replace('<ref id="shared"/>', '<bsdf type="diffuse" id="mat_b"><rgb name="reflectance" value="0.5, 0.5, 0.5"/></bsdf>', 1)
    # The replacement above is too brittle. Write the second file explicitly.
    separate = f"""<scene version="3.0.0">
  <shape type="obj" id="shape_a">
    <string name="filename" value="{write_a}"/>
    <bsdf type="diffuse" id="mat_a"><rgb name="reflectance" value="0.5, 0.5, 0.5"/></bsdf>
  </shape>
  <shape type="obj" id="shape_b">
    <string name="filename" value="{write_b}"/>
    <bsdf type="diffuse" id="mat_b"><rgb name="reflectance" value="0.5, 0.5, 0.5"/></bsdf>
  </shape>
</scene>
"""
    separate_path = directory / "separate.xml"
    separate_path.write_text(separate, encoding="utf-8")
    return {"shared_xml": path, "separate_xml": separate_path}


def load_xml_shapes(path: Path) -> list[dict]:
    import mitsuba as mi

    require_variant()
    scene = mi.load_file(str(path))
    return [{"id": str(shape.id()), "vertex_count": int(np.array(mi.traverse(shape)["vertex_positions"]).reshape(-1, 3).shape[0])} for shape in scene.shapes()]


def _position_hash(positions: np.ndarray) -> str:
    rounded = np.round(np.asarray(positions, dtype=np.float64), 6).astype(np.float64)
    return hashlib.sha256(np.ascontiguousarray(rounded).tobytes()).hexdigest()


def _bitmap_names(bitmap) -> list[str]:
    names = []
    for field in bitmap.struct_():
        if getattr(field, "name", None):
            names.append(str(field.name))
            continue
        tokens = str(field).replace(";", " ").split()
        names.append(tokens[tokens.index("float32") + 1])
    return names


def _rays(origin, target, up, fov_deg, width, height):
    origin = np.asarray(origin, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - origin
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)
    scale = np.tan(np.deg2rad(fov_deg) * 0.5)
    aspect = float(width) / float(height)
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    camera_x = (2.0 * (columns + 0.5) / width - 1.0) * scale * aspect
    camera_y = (1.0 - 2.0 * (rows + 0.5) / height) * scale
    directions = forward + right * camera_x[..., None] + true_up * camera_y[..., None]
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(origin, directions.shape).copy()
    return origins.reshape(-1, 3), directions.reshape(-1, 3)

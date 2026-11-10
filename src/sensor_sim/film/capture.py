"""Render one Blender-exported scene to RGB, Stokes, analyzers, and a film mask."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

from sensor_sim.film.blender_build import build_blend, find_blender
from sensor_sim.film.images import read_exr, write_exr, write_gray_png
from sensor_sim.film.mitsuba_scene import scene_dict, without_film
from sensor_sim.film.polar import stokes_bundle
from sensor_sim.film.lighting import DISPLAY_EXPOSURE, display_srgb
from sensor_sim.render import _split_bitmap
from sensor_sim.stokes import DOP_EXCESS, summarize_polarization

logger = logging.getLogger("sensor_sim.film")

RECONSTRUCTION_ATOL = 1e-5
ANALYZER_ATOL = 1e-4
DOLP_MIN = 0.05


def blend_parameters(
    *,
    radius: float,
    scale: tuple[float, float, float],
    thickness: float,
    coverage: float,
    axis: list[float],
    camera: dict,
    light: dict,
    floor_gap: float,
    margin: float,
    blend: Path,
    out_dir: Path,
    reflectance: float,
    room_reflectance: float,
    ior: float,
) -> dict:
    out_dir = Path(out_dir)
    return {
        "radius": float(radius),
        "scale": [float(v) for v in scale],
        "thickness": float(thickness),
        "coverage": float(coverage),
        "axis": [float(v) for v in axis],
        "floor_gap": float(floor_gap),
        "margin": float(margin),
        "camera": camera,
        "light": light,
        "reflectance": float(reflectance),
        "room_reflectance": float(room_reflectance),
        "ior": float(ior),
        "blend": str(blend),
        "out_dir": str(out_dir),
        "poses": str(out_dir / "poses.json"),
    }


def project_point(sensor, point: list[float], width: int, height: int) -> tuple[int, int, list[float]]:
    """Pixel of a world point. Mitsuba's projection already divides by depth."""
    transformed = sensor.world_transform().inverse() @ mi.Point3f(float(point[0]), float(point[1]), float(point[2]))
    ndc = np.array(sensor.projection_transform() @ transformed).reshape(-1)
    column = int(np.floor(float(ndc[0]) * width))
    row = int(np.floor(float(ndc[1]) * height))
    return row, column, [float(v) for v in ndc[:3]]


def _render(scene: dict, spp: int, seed: int):
    loaded = mi.load_dict(scene)
    started = time.perf_counter()
    image = mi.render(loaded, spp=int(spp), seed=int(seed))
    array = np.array(image)
    elapsed = time.perf_counter() - started
    if not np.isfinite(array).all():
        n_bad = int(np.size(array) - np.isfinite(array).sum())
        raise RuntimeError(f"render produced {n_bad} non-finite values")
    return loaded, image, array, elapsed


def _object_mask(index_image: np.ndarray, loaded, poses: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    height, width = index_image.shape
    sensor = loaded.sensors()[0]
    ids = [shape.id() for shape in loaded.shapes()]
    points = poses["points"]
    film_row, film_col, film_ndc = project_point(sensor, points["film"], width, height)
    if not (0 <= film_row < height and 0 <= film_col < width):
        raise RuntimeError(f"film point projects outside the image row={film_row} col={film_col} ndc={film_ndc}")
    shell_index = int(round(float(index_image[film_row, film_col])))
    if shell_index == 0 or "film-shell" not in ids:
        raise RuntimeError(f"film shape index is missing at the cap point ids={ids} index={shell_index}")
    mask = (np.rint(index_image) == shell_index).astype(np.uint8)
    ball_row, ball_col, ball_ndc = project_point(sensor, points["uncovered"], width, height)
    floor_row, floor_col, floor_ndc = project_point(sensor, points["floor"], width, height)
    report = {
        "shape_ids": ids,
        "film_index": shell_index,
        "film_pixel": [film_row, film_col],
        "film_ndc": film_ndc,
        "uncovered_pixel": [ball_row, ball_col],
        "uncovered_ndc": ball_ndc,
        "floor_pixel": [floor_row, floor_col],
        "floor_ndc": floor_ndc,
        "n_film_pixels": int(mask.sum()),
    }
    if mask[film_row, film_col] != 1:
        raise RuntimeError(f"cap point is not labeled film: {report}")
    if points["uncovered_visible"]:
        if not _has_zero_near(mask, ball_row, ball_col, radius=max(2, width // 32)):
            raise RuntimeError(f"no unlabeled pixel near the uncovered ball point: {report}")
    if _in_frame(floor_row, floor_col, height, width) and mask[floor_row, floor_col] != 0:
        raise RuntimeError(f"background point is labeled film: {report}")
    if int(mask.sum()) == 0 or int(mask.sum()) == mask.size:
        raise RuntimeError(f"mask is all {int(mask.flat[0])}: {report}")
    nonzero = np.unique(np.rint(index_image)[mask == 0])
    report["unlabeled_indices"] = [int(value) for value in nonzero if value != 0]
    if not report["unlabeled_indices"]:
        raise RuntimeError(f"no uncovered surface index outside the film: {report}")
    return mask, index_image, report


def _in_frame(row: int, col: int, height: int, width: int) -> bool:
    return 0 <= row < height and 0 <= col < width


def _has_zero_near(mask: np.ndarray, row: int, col: int, radius: int) -> bool:
    """True when a 0 label lies within ``radius`` pixels of the projected point."""
    if not _in_frame(row, col, mask.shape[0], mask.shape[1]):
        return False
    r0 = max(0, row - radius)
    r1 = min(mask.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(mask.shape[1], col + radius + 1)
    return bool(np.any(mask[r0:r1, c0:c1] == 0))


def render_mask(scene_kwargs: dict, poses: dict, spp_seed: int) -> tuple[np.ndarray, dict, float]:
    scene = scene_dict(**scene_kwargs, polarized=False, spp=1, rfilter="box")
    scene["integrator"] = {"type": "aov", "aovs": "idx:shape_index"}
    loaded, _image, array, elapsed = _render(scene, spp=1, seed=spp_seed)
    index_image = array[:, :, 0] if array.ndim == 3 else array
    mask, _index, report = _object_mask(index_image, loaded, poses)
    report["elapsed_s"] = elapsed
    return mask, report, elapsed


def _split_rendered(loaded, image):
    """Name channels for an RGB film or a monochrome film."""
    array = np.array(image)
    count = 1 if array.ndim == 2 else int(array.shape[2])
    aovs = list(loaded.integrator().aov_names())
    if count == 3 + len(aovs):
        names = ["R", "G", "B"] + aovs
    elif count == 1 + len(aovs):
        names = ["Y"] + aovs
    else:
        raise RuntimeError(f"render has {count} channels and {len(aovs)} AOVs")
    packed = array if array.ndim == 3 else array[:, :, None]
    bitmap = mi.Bitmap(packed, channel_names=names)
    return bitmap, dict(bitmap.split()), names


def _stokes_from_image(loaded, image) -> dict:
    _bitmap, channels, names = _split_rendered(loaded, image)
    missing = [key for key in ("S0", "S1", "S2", "S3") if key not in channels]
    if missing:
        raise RuntimeError(f"Stokes channels {missing} missing from {names}")
    bundle = stokes_bundle(channels["S0"], channels["S1"], channels["S2"], channels["S3"])
    bundle["bitmap"] = _bitmap
    bundle["channel_names"] = names
    bundle["rgb"] = np.array(channels["<root>"], dtype=np.float64)
    return bundle


def _save_polarization(out_dir: Path, bundle: dict) -> None:
    polar = out_dir / "polarization"
    polar.mkdir(parents=True, exist_ok=True)
    bundle["bitmap"].write(str(polar / "stokes.exr"))
    for name in ("s0", "s1", "s2", "s3", "dolp", "aolp"):
        image = np.array(bundle[name], dtype=np.float32)
        if name == "aolp":
            image = np.where(bundle["aolp_masked"], 0.0, image)
        filename = f"{name.upper()}.exr" if name.startswith("s") else f"{name}.exr"
        write_exr(polar / filename, image, [name])
    write_exr(polar / "analyzer_000.exr", bundle["analyzer_000"], ["I0"])
    write_exr(polar / "analyzer_045.exr", bundle["analyzer_045"], ["I45"])
    write_exr(polar / "analyzer_090.exr", bundle["analyzer_090"], ["I90"])
    write_exr(polar / "analyzer_135.exr", bundle["analyzer_135"], ["I135"])


def _overlay(path: Path, rgb: np.ndarray, mask: np.ndarray) -> None:
    display = display_srgb(rgb)
    if display.ndim == 2:
        display = np.repeat(display[:, :, None], 3, axis=2)
    edge = np.zeros(mask.shape, dtype=bool)
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    painted = display.copy()
    painted[edge] = (1.0, 0.15, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.clip(display, 0.0, 1.0))
    axes[0].set_title(f"RGB  display = (({DISPLAY_EXPOSURE:g}*linear)/(1+{DISPLAY_EXPOSURE:g}*linear)) ** (1/2.2)")
    axes[1].imshow(np.clip(painted, 0.0, 1.0))
    axes[1].set_title("film mask edge")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def check_bundle(bundle: dict, mask: np.ndarray, control_dolp: np.ndarray | None) -> dict:
    """Reconstruction, nonnegativity, DoP, and the unfilmed DoLP comparison."""
    error = bundle["reconstruction_error"]["max"]
    if error > RECONSTRUCTION_ATOL:
        raise RuntimeError(f"analyzer reconstruction error {error} exceeds {RECONSTRUCTION_ATOL}")
    floor = -ANALYZER_ATOL
    analyzer_min = {}
    for name in ("analyzer_000", "analyzer_045", "analyzer_090", "analyzer_135"):
        analyzer_min[name] = float(np.min(bundle[name]))
        if analyzer_min[name] < floor:
            raise RuntimeError(f"{name} min {analyzer_min[name]} is below {floor}")
    summary = summarize_polarization(bundle)
    if summary["frac_dop_gt_1_05"] > 0.0:
        raise RuntimeError(f"frac DoP>{DOP_EXCESS} is {summary['frac_dop_gt_1_05']}")
    report = {"reconstruction_max_abs": error, "analyzer_min": analyzer_min, "summary": summary}
    if control_dolp is not None:
        film_pixels = mask.astype(bool)
        film_max = float(bundle["dolp"][film_pixels].max()) if film_pixels.any() else 0.0
        control_max = float(control_dolp[film_pixels].max()) if film_pixels.any() else 0.0
        report["film_max_dolp"] = film_max
        report["control_max_dolp"] = control_max
        matches_control = film_max < DOLP_MIN or film_max <= control_max * 1.05
        report["dolp_matches_unfilmed_control"] = matches_control
        if matches_control:
            raise RuntimeError(
                f"DoLP on the cap matches the unfilmed control film_max={film_max:.6f} "
                f"control_max={control_max:.6f}"
            )
    return report


def capture_sample(
    *,
    variant: str,
    scene_kwargs: dict,
    poses: dict,
    out_dir: Path,
    seed: int,
    spp: int,
    resolution: tuple[int, int],
    control: bool,
    mono_variant: str | None,
    metadata: dict,
) -> dict:
    """Write one sample directory. ``scene_kwargs`` omits polarized, spp, and rfilter."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mi.set_variant(variant)
    logger.info(
        "film sample=%s stage=rgb variant=%s res=%s spp=%d seed=%d",
        out_dir.name,
        variant,
        resolution,
        spp,
        seed,
    )
    rgb_scene = scene_dict(**scene_kwargs, polarized=False, spp=spp, rfilter="gaussian")
    _loaded, rgb_image, _array, rgb_elapsed = _render(rgb_scene, spp, seed)
    rgb_bitmap, rgb_channels, _names = _split_bitmap(_loaded, rgb_image)
    rgb = np.array(rgb_channels["<root>"], dtype=np.float64)
    rgb_bitmap.write(str(out_dir / "rgb.exr"))

    logger.info("film sample=%s stage=stokes variant=%s", out_dir.name, variant)
    stokes_scene = scene_dict(**scene_kwargs, polarized=True, spp=spp, rfilter="gaussian")
    stokes_loaded, stokes_image, _array, stokes_elapsed = _render(stokes_scene, spp, seed)
    bundle = _stokes_from_image(stokes_loaded, stokes_image)
    _save_polarization(out_dir, bundle)

    logger.info("film sample=%s stage=mask variant=%s", out_dir.name, variant)
    mask, mask_report, mask_elapsed = render_mask(scene_kwargs, poses, seed)
    truth = out_dir / "ground_truth"
    truth.mkdir(parents=True, exist_ok=True)
    write_gray_png(truth / "film_mask.png", mask)
    write_exr(truth / "film_coverage.exr", mask.astype(np.float32), ["coverage"])
    _overlay(out_dir / "overlay.png", rgb, mask)

    control_dolp = None
    control_elapsed = None
    if control:
        logger.info("film sample=%s stage=control variant=%s", out_dir.name, variant)
        control_scene = without_film(scene_dict(**scene_kwargs, polarized=True, spp=spp, rfilter="gaussian"))
        control_loaded, control_image, _array, control_elapsed = _render(control_scene, spp, seed)
        control_bundle = _stokes_from_image(control_loaded, control_image)
        control_dolp = control_bundle["dolp"]
    checks = check_bundle(bundle, mask, control_dolp)

    mono_report = None
    if mono_variant:
        logger.info("film sample=%s stage=mono variant=%s", out_dir.name, mono_variant)
        mi.set_variant(mono_variant)
        mono_scene = scene_dict(**scene_kwargs, polarized=True, spp=spp, rfilter="gaussian")
        mono_loaded, mono_image, _array, mono_elapsed = _render(mono_scene, spp, seed)
        mono_bundle = _stokes_from_image(mono_loaded, mono_image)
        both = bundle["valid"] & mono_bundle["valid"]
        if int(both.sum()) == 0:
            diff = None
        else:
            diff = float(np.mean(np.abs(bundle["dolp"][both] - mono_bundle["dolp"][both])))
        mono_report = {
            "variant": mono_variant,
            "elapsed_s": mono_elapsed,
            "mean_abs_dolp_diff": diff,
            "max_dolp": float(mono_bundle["dolp"].max()),
        }
        mi.set_variant(variant)

    mean_s0 = float(bundle["s0"][mask.astype(bool)].mean()) if mask.any() else 0.0
    record = {
        **metadata,
        "variant": variant,
        "resolution": list(resolution),
        "spp": int(spp),
        "seed": int(seed),
        "wavelength": (
            "llvm_ad_spectral_polarized samples the visible range and stores three "
            "spectral copies. Those copies are averaged. IOR is constant, so this is "
            "not one sampled wavelength."
        ),
        "channel_definitions": {
            "rgb": "linear path-integrator RGB",
            "stokes": "S0 S1 S2 S3 after averaging the three spectral copies, camera frame",
            "analyzers": "I0=(S0+S1)/2 I45=(S0+S2)/2 I90=(S0-S1)/2 I135=(S0-S2)/2",
            "s3": "saved, not measured by the four linear analyzers, not a model input",
            "dolp": "sqrt(S1^2+S2^2)/S0 where S0>1e-4 else 0",
            "aolp": "0.5*arctan2(S2,S1); 0 where DoLP<0.05 or S0<=1e-4",
            "mask": "1 where the first-hit shape_index is film-shell; uint8 values 0 and 1",
            "display": (
                f"PNG only: mapped=(exposure*linear)/(1+exposure*linear), "
                f"display=clip(mapped,0,1)**(1/2.2), exposure={DISPLAY_EXPOSURE:g}. EXR stays linear."
            ),
        },
        "elapsed_s": {
            "rgb": rgb_elapsed,
            "stokes": stokes_elapsed,
            "mask": mask_elapsed,
            "control": control_elapsed,
        },
        "mean_s0_film": mean_s0,
        "mask": mask_report,
        "checks": {key: value for key, value in checks.items() if key != "summary"},
        "polarization_summary": checks["summary"],
        "mono": mono_report,
        "poses": {
            "camera": poses["camera"],
            "light": poses["light"],
            "room_low": poses["room_low"],
            "room_high": poses["room_high"],
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    logger.info(
        "film sample=%s status=ok elapsed_stokes_s=%.3f max_dolp=%.6f mean_s0_film=%.6g n_film_pixels=%d",
        out_dir.name,
        stokes_elapsed,
        checks["summary"]["max_dolp"],
        mean_s0,
        mask_report["n_film_pixels"],
    )
    return record


def probe_mean_s0(variant: str, scene_kwargs: dict, poses: dict, spp: int, seed: int) -> float:
    """Mean S0 on film pixels at the probe resolution already stored in scene_kwargs."""
    mi.set_variant(variant)
    mask, _report, _elapsed = render_mask(scene_kwargs, poses, seed)
    scene = scene_dict(**scene_kwargs, polarized=True, spp=spp, rfilter="gaussian")
    loaded, image, _array, elapsed = _render(scene, spp, seed)
    bundle = _stokes_from_image(loaded, image)
    mean_s0 = float(bundle["s0"][mask.astype(bool)].mean()) if mask.any() else 0.0
    logger.info(
        "film stage=probe variant=%s spp=%d elapsed_s=%.3f mean_s0_film=%.6g n_film_pixels=%d",
        variant,
        spp,
        elapsed,
        mean_s0,
        int(mask.sum()),
    )
    return mean_s0


def ensure_scene(params: dict) -> dict:
    blender = find_blender()
    if blender is None:
        raise FileNotFoundError("blender was not found")
    return build_blend(params, blender)


def _single_channel(channels: dict[str, np.ndarray]) -> np.ndarray:
    image = next(iter(channels.values()))
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array


def load_measurement(sample_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Linear RGB plus four analyzers, and the uint8 mask. Variant must be set."""
    from sensor_sim.film.images import read_gray_png

    sample_dir = Path(sample_dir)
    rgb = np.asarray(read_exr(sample_dir / "rgb.exr")["<root>"], dtype=np.float64)
    if rgb.ndim == 2:
        rgb = rgb[:, :, None]
    polar = sample_dir / "polarization"
    analyzers = [
        _single_channel(read_exr(polar / name))[:, :, None]
        for name in ("analyzer_000.exr", "analyzer_045.exr", "analyzer_090.exr", "analyzer_135.exr")
    ]
    mask = read_gray_png(sample_dir / "ground_truth" / "film_mask.png")
    return np.concatenate([rgb, *analyzers], axis=2), mask

"""Phase 1 renders: CUDA Cornell box, then the ball and film.

Stokes is written first. The 8-band cube is a second render, and the
spectral profile is the mean of each cube band. ``specfilm`` is not
combined with the Stokes integrator.

``uv run phase1`` writes ``outputs/phase1/``. A CUDA failure is logged
and raised. The Phase 0 CPU images are not copied in its place.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

from sensor_sim.render import (
    _rgb_metrics,
    _split_bitmap,
    _write_polarized,
    _write_rgb_only,
    configure_logging,
    tonemap_srgb,
)
from sensor_sim.scenes import (
    ball_film_scene,
    cornell_box_scene,
    cornell_box_stokes_scene,
    set_film,
)
from sensor_sim.spectral import (
    BALL_REGIONS,
    CORNELL_REGIONS,
    region_slice,
    specfilm_dict,
    spectral_bands,
)
from sensor_sim.stokes import DEFINITIONS

logger = logging.getLogger("phase1")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase1"

STOKES_VARIANT = "cuda_ad_spectral_polarized"
CUBE_VARIANT = "cuda_ad_spectral"
RGB_VARIANT = "cuda_ad_rgb"
RESOLUTION = 256
STOKES_SPP = 64
CUBE_SPP = 256
RGB_SPP = 64


def _load(scene: dict, resolution: int, spp: int, *, cube: bool):
    if cube:
        scene["sensor"]["film"] = specfilm_dict(resolution)
        scene["sensor"]["sampler"]["sample_count"] = int(spp)
    else:
        set_film(scene, resolution, spp)
    return mi.load_dict(scene)


def _render(job: str, variant: str, scene_fn, resolution: int, spp: int, seed: int, *, cube: bool):
    mi.set_variant(variant)
    scene = scene_fn()
    loaded = _load(scene, resolution, spp, cube=cube)
    started = time.perf_counter()
    try:
        image = mi.render(loaded, spp=spp, seed=seed)
    except Exception as exc:
        logger.error(
            "phase1 job=%s variant=%s status=failed error=%s",
            job,
            variant,
            exc,
        )
        raise
    elapsed_s = time.perf_counter() - started
    array = np.array(image)
    if not np.isfinite(array).all():
        n_bad = int(np.size(array) - np.isfinite(array).sum())
        logger.error(
            "phase1 job=%s variant=%s status=failed nonfinite_values=%d",
            job,
            variant,
            n_bad,
        )
        raise RuntimeError(f"{job} produced {n_bad} non-finite pixels")
    logger.info(
        "phase1 job=%s variant=%s res=%d spp=%d seed=%d elapsed_s=%.3f shape=%s",
        job,
        variant,
        resolution,
        spp,
        seed,
        elapsed_s,
        tuple(array.shape),
    )
    return loaded, image, array, elapsed_s


def _band_means(cube: np.ndarray, regions: dict) -> dict[str, list[float]]:
    height, width, count = cube.shape
    if count != len(spectral_bands()):
        raise ValueError(f"expected {len(spectral_bands())} bands, got {count}")
    means = {}
    for name, bounds in regions.items():
        rows, cols = region_slice(height, width, bounds)
        region = cube[rows, cols, :]
        means[name] = [float(np.mean(region[:, :, band])) for band in range(count)]
    return means


def _write_cube(out_dir: Path, cube: np.ndarray) -> None:
    names = [band.name for band in spectral_bands()]
    bitmap = mi.Bitmap(cube, channel_names=names)
    bitmap.write(str(out_dir / "render.exr"))
    # One shared display scale so a brighter red band stays brighter.
    finite = cube[np.isfinite(cube)]
    vmax = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, band, values in zip(axes.ravel(), spectral_bands(), range(cube.shape[2])):
        image = tonemap_srgb(np.clip(cube[:, :, values] / vmax, 0.0, 1.0))
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"{band.wavelength_min:.0f}–{band.wavelength_max:.0f} nm")
        axis.axis("off")
    fig.suptitle("specfilm bands, display = clip(band / 99th percentile, 0, 1) ** (1/2.2)")
    fig.tight_layout()
    fig.savefig(out_dir / "bands.png", dpi=120)
    plt.close(fig)


def write_spectral_profile(
    path: Path,
    means: dict[str, list[float]],
    regions: dict,
) -> None:
    """Plot mean linear radiance of each band. ``means`` is region to 8 values."""
    centers = [band.center_nm for band in spectral_bands()]
    colors = {
        "frame": "C0",
        "light": "darkorange",
        "red_wall": "crimson",
        "green_wall": "green",
        "box_face": "purple",
        "ball": "sienna",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in means.items():
        ax.plot(centers, values, marker="o", label=name, color=colors.get(name))
    ax.set_xlabel("Band center (nm)")
    ax.set_ylabel("Mean linear radiance")
    ax.set_title("Spectral profile: mean of each specfilm band")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    payload = {
        "definition": (
            "Mean of the linear specfilm band over the named region. "
            "Regions are fractions (y0, y1, x0, x1) of the image. "
            "frame is the full image. Each band is a regular spectrum "
            "equal to 1 between wavelength_min and wavelength_max."
        ),
        "bands": [
            {
                "name": band.name,
                "wavelength_min": band.wavelength_min,
                "wavelength_max": band.wavelength_max,
                "center_nm": band.center_nm,
            }
            for band in spectral_bands()
        ],
        "regions": {
            name: None if bounds is None else {"y0": bounds[0], "y1": bounds[1], "x0": bounds[2], "x1": bounds[3]}
            for name, bounds in regions.items()
        },
        "mean_radiance": means,
    }
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_metrics(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics, indent=2) + "\n")


def render_stokes(job: str, scene_fn, out_dir: Path, resolution: int, spp: int, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded, image, _array, elapsed_s = _render(
        job, STOKES_VARIANT, scene_fn, resolution, spp, seed, cube=False
    )
    bitmap, channels, channel_names = _split_bitmap(loaded, image)
    if "S0" not in channels:
        raise RuntimeError(f"{job} did not produce Stokes channels: {channel_names}")
    summary = _write_polarized(out_dir, bitmap, channels)
    metrics = {
        "job": job,
        "variant": STOKES_VARIANT,
        "resolution": resolution,
        "spp": spp,
        "seed": seed,
        "elapsed_s": elapsed_s,
        "channel_names": channel_names,
        "definitions": DEFINITIONS,
        **summary,
    }
    _write_metrics(out_dir / "metrics.json", metrics)
    logger.info(
        "phase1 job=%s variant=%s mean_dolp=%.6f max_dolp=%.6f mean_abs_s3_over_s0=%.6g",
        job,
        STOKES_VARIANT,
        summary["mean_dolp"],
        summary["max_dolp"],
        summary["mean_abs_s3_over_s0"],
    )
    return metrics


def render_cube(
    job: str,
    scene_fn,
    out_dir: Path,
    profile_path: Path,
    regions: dict,
    resolution: int,
    spp: int,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _loaded, _image, cube, elapsed_s = _render(
        job, CUBE_VARIANT, scene_fn, resolution, spp, seed, cube=True
    )
    if cube.ndim != 3 or cube.shape[2] != 8:
        raise RuntimeError(f"{job} cube shape {cube.shape}, expected HxWx8")
    _write_cube(out_dir, cube)
    means = _band_means(cube, regions)
    write_spectral_profile(profile_path, means, regions)
    metrics = {
        "job": job,
        "variant": CUBE_VARIANT,
        "resolution": resolution,
        "spp": spp,
        "seed": seed,
        "elapsed_s": elapsed_s,
        "channel_names": [band.name for band in spectral_bands()],
        "srf": "regular spectrum equal to 1 on each wavelength window",
        "mean_radiance": means,
    }
    _write_metrics(out_dir / "metrics.json", metrics)
    logger.info(
        "phase1 job=%s variant=%s frame_band_means=%s",
        job,
        CUBE_VARIANT,
        " ".join(f"{value:.4g}" for value in means["frame"]),
    )
    return metrics


def render_rgb(
    job: str,
    scene_fn,
    out_dir: Path,
    resolution: int,
    spp: int,
    seed: int,
    variant: str = RGB_VARIANT,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded, image, _array, elapsed_s = _render(
        job, variant, scene_fn, resolution, spp, seed, cube=False
    )
    bitmap, channels, channel_names = _split_bitmap(loaded, image)
    rgb = np.array(channels["<root>"], copy=True)
    _write_rgb_only(out_dir, bitmap, rgb, title=job)
    stats = _rgb_metrics(rgb)
    metrics = {
        "job": job,
        "variant": variant,
        "resolution": resolution,
        "spp": spp,
        "seed": seed,
        "elapsed_s": elapsed_s,
        "channel_names": channel_names,
        **stats,
    }
    _write_metrics(out_dir / "metrics.json", metrics)
    logger.info(
        "phase1 job=%s variant=%s mean_rgb=%.6f max_rgb=%.6f",
        job,
        variant,
        stats["mean_rgb"],
        stats["max_rgb"],
    )
    return metrics


def render_phase1(output_root: Path = DEFAULT_OUTPUT_ROOT, seed: int = 0, resolution: int = RESOLUTION) -> dict:
    """Run the Phase 1 figure set. Stokes jobs run before cube jobs."""
    output_root = Path(output_root)
    cornell_dir = output_root / "cornell_cuda"
    ball_dir = output_root / "ball_film"
    results = {
        "cornell_stokes": render_stokes(
            "cornell_stokes",
            cornell_box_stokes_scene,
            cornell_dir / "stokes",
            resolution,
            STOKES_SPP,
            seed,
        ),
        "cornell_cube": render_cube(
            "cornell_cube",
            cornell_box_scene,
            cornell_dir / "cube",
            cornell_dir / "spectral_profile.png",
            CORNELL_REGIONS,
            resolution,
            CUBE_SPP,
            seed,
        ),
        "ball_rgb": render_rgb(
            "ball_rgb",
            lambda: ball_film_scene(polarized=False),
            ball_dir / "rgb",
            resolution,
            RGB_SPP,
            seed,
            # cuda_ad_rgb cannot evaluate the ball's regular spectrum.
            # This hdrfilm render is the spectral variant integrated to RGB.
            variant=CUBE_VARIANT,
        ),
        "ball_stokes": render_stokes(
            "ball_stokes",
            lambda: ball_film_scene(polarized=True),
            ball_dir / "stokes",
            resolution,
            STOKES_SPP,
            seed,
        ),
        "ball_cube": render_cube(
            "ball_cube",
            lambda: ball_film_scene(polarized=False),
            ball_dir / "cube",
            ball_dir / "spectral_profile.png",
            BALL_REGIONS,
            resolution,
            CUBE_SPP,
            seed,
        ),
    }
    return results


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Render Phase 1 CUDA Cornell box and ball film.")
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    logger.info(
        "phase1 mitsuba=%s drjit=%s",
        getattr(mi, "__version__", "unknown"),
        __import__("drjit").__version__,
    )
    render_phase1(output_root=args.output_root, seed=args.seed, resolution=args.resolution)


if __name__ == "__main__":
    main()

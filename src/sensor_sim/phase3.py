"""Phase 3 proof dataset: 32 randomized frames of the ball.

Each frame stores a Stokes EXR (its RGB channels are the image), an
8-band cube, a first-hit film mask, and the pose, light, and material.
``uv run phase3`` writes ``outputs/phase3/``. It is not part of
``uv run phase0``. A CUDA Stokes or cube failure is logged and raised.
The mask tries ``cuda_ad_spectral`` and falls back to ``llvm_ad_spectral``
only when that AOV render fails.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

from sensor_sim.phase1 import _render
from sensor_sim.render import _save_image, _split_bitmap, _write_polarized, configure_logging, tonemap_srgb
from sensor_sim.scenes import (
    BALL_FILM_THICKNESS,
    BALL_INNER_RADIUS,
    BALL_OUTER_RADIUS,
    FLOOR_GAP,
    FLOOR_Y,
    LIGHT_RADIANCE,
    PHASE2_ALPHA,
    PHASE2_EXT_IOR,
    PHASE2_INT_IOR,
    ball_film_scene,
    ball_rest_center,
    set_film,
)
from sensor_sim.spectral import spectral_bands
from sensor_sim.stokes import polarization_maps, summarize_polarization

logger = logging.getLogger("phase3")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase3"

STOKES_VARIANT = "cuda_ad_spectral_polarized"
CUBE_VARIANT = "cuda_ad_spectral"
MASK_VARIANT = "cuda_ad_spectral"
MASK_FALLBACK_VARIANT = "llvm_ad_spectral"
RESOLUTION = 256
STOKES_SPP = 64
CUBE_SPP = 256
FRAME_COUNT = 32
FILM_COUNT = 24
PPLASTIC_COUNT = 8
XZ_LIMIT = 0.35
LIGHT_SHIFT_LIMIT = 0.25
RADIANCE_SCALE_MIN = 0.5
RADIANCE_SCALE_MAX = 1.5

MASK_DEFINITION = (
    "1 where the first-hit shape_index equals the outer dielectric sphere "
    "named film-shell. The AOV uses a box filter and 1 sample so indices "
    "are not blended. The mask is 0 everywhere when film-shell is absent."
)


def sample_frames(seed: int = 0, count: int = FRAME_COUNT) -> list[dict]:
    """Deterministic pose, light, and material records.

    The first ``FILM_COUNT`` frames are the dielectric shell. The rest are
    ``pplastic``. Pose and light are drawn from ``numpy`` Generator(seed).
    Material is not a coin flip.
    """
    if count != FRAME_COUNT:
        raise ValueError(f"this proof set is {FRAME_COUNT} frames, got {count}")
    rng = np.random.default_rng(seed)
    frames = []
    for index in range(count):
        material = "dielectric" if index < FILM_COUNT else "pplastic"
        x = float(rng.uniform(-XZ_LIMIT, XZ_LIMIT))
        z = float(rng.uniform(-XZ_LIMIT, XZ_LIMIT))
        shift_x = float(rng.uniform(-LIGHT_SHIFT_LIMIT, LIGHT_SHIFT_LIMIT))
        shift_z = float(rng.uniform(-LIGHT_SHIFT_LIMIT, LIGHT_SHIFT_LIMIT))
        scale = float(rng.uniform(RADIANCE_SCALE_MIN, RADIANCE_SCALE_MAX))
        frames.append(
            {
                "index": index,
                "seed": int(seed),
                "material": material,
                "center": ball_rest_center(x, z),
                "light_shift": [shift_x, shift_z],
                "radiance_scale": scale,
                "radiance": float(LIGHT_RADIANCE * scale),
                "int_ior": PHASE2_INT_IOR,
                "ext_ior": PHASE2_EXT_IOR,
                "film_thickness": round(BALL_FILM_THICKNESS, 10) if material == "dielectric" else 0.0,
                "alpha": None if material == "dielectric" else PHASE2_ALPHA,
                "outer_radius": BALL_OUTER_RADIUS,
                "inner_radius": BALL_INNER_RADIUS if material == "dielectric" else None,
                "floor_y": FLOOR_Y,
                "floor_gap": FLOOR_GAP,
            }
        )
    return frames


def _scene(record: dict, *, polarized: bool) -> dict:
    return ball_film_scene(
        polarized=polarized,
        center=record["center"],
        light_shift=(record["light_shift"][0], record["light_shift"][1]),
        radiance_scale=record["radiance_scale"],
        material=record["material"],
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _project_center(sensor, center: list[float], width: int, height: int) -> tuple[int, int]:
    """Pixel of the ball center. Mitsuba's projection already divides by depth."""
    point = sensor.world_transform().inverse() @ mi.Point3f(center[0], center[1], center[2])
    ndc = np.array(sensor.projection_transform() @ point).reshape(-1)
    col = int(np.floor(float(ndc[0]) * width))
    row = int(np.floor(float(ndc[1]) * height))
    if not (0 <= col < width and 0 <= row < height):
        raise RuntimeError(f"ball center projects outside the film at row={row} col={col}")
    return row, col


def _shape_index_image(variant: str, record: dict, resolution: int) -> tuple[np.ndarray, object]:
    mi.set_variant(variant)
    scene = _scene(record, polarized=False)
    scene["integrator"] = {"type": "aov", "aovs": "idx:shape_index"}
    set_film(scene, resolution, 1)
    scene["sensor"]["film"]["rfilter"] = {"type": "box"}
    loaded = mi.load_dict(scene)
    image = np.array(mi.render(loaded, spp=1, seed=record["seed"]))
    if image.ndim == 3:
        image = image[:, :, 0]
    return image, loaded


def render_mask(record: dict, resolution: int) -> tuple[np.ndarray, str]:
    """Binary film mask. Falls back to LLVM if the CUDA AOV render fails."""
    variant = MASK_VARIANT
    try:
        index_image, loaded = _shape_index_image(variant, record, resolution)
    except Exception as exc:
        logger.error(
            "phase3 frame=%03d mask variant=%s status=fallback error=%s",
            record["index"],
            variant,
            exc,
        )
        variant = MASK_FALLBACK_VARIANT
        index_image, loaded = _shape_index_image(variant, record, resolution)
        logger.info(
            "phase3 frame=%03d mask variant=%s status=fallback_ok",
            record["index"],
            variant,
        )
    if not np.isfinite(index_image).all():
        raise RuntimeError(f"frame {record['index']} mask has non-finite shape indices")
    ids = [shape.id() for shape in loaded.shapes()]
    mask = np.zeros(index_image.shape, dtype=np.uint8)
    if "film-shell" in ids:
        row, col = _project_center(loaded.sensors()[0], record["center"], index_image.shape[1], index_image.shape[0])
        shell_index = int(round(float(index_image[row, col])))
        if shell_index == 0:
            raise RuntimeError(
                f"frame {record['index']} ball center pixel is a miss, so the shell index is unknown"
            )
        mask[np.rint(index_image) == shell_index] = 1
    logger.info(
        "phase3 frame=%03d mask variant=%s n_film_pixels=%d",
        record["index"],
        variant,
        int(mask.sum()),
    )
    return mask, variant


def _save_mask(path: Path, mask: np.ndarray) -> None:
    plt.imsave(path, mask, cmap="gray", vmin=0, vmax=1)


def render_frame(
    record: dict,
    frame_dir: Path,
    resolution: int = RESOLUTION,
    stokes_spp: int = STOKES_SPP,
    cube_spp: int = CUBE_SPP,
    showcase: bool = False,
) -> tuple[dict, np.ndarray]:
    """Render Stokes, the cube, and the mask for one frame. Returns the record and display RGB."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    label = f"frame={record['index']:03d} material={record['material']}"
    loaded, image, _array, elapsed_s = _render(
        f"phase3_stokes_{record['index']:03d}",
        STOKES_VARIANT,
        lambda: _scene(record, polarized=True),
        resolution,
        stokes_spp,
        record["seed"],
        cube=False,
    )
    bitmap, channels, channel_names = _split_bitmap(loaded, image)
    if "S0" not in channels:
        raise RuntimeError(f"{label} did not produce Stokes channels: {channel_names}")
    stokes_dir = frame_dir / "stokes"
    stokes_dir.mkdir(parents=True, exist_ok=True)
    bitmap.write(str(stokes_dir / "render.exr"))
    rgb = np.array(channels["<root>"], copy=True)
    maps = polarization_maps(
        np.array(channels["S0"]),
        np.array(channels["S1"]),
        np.array(channels["S2"]),
        np.array(channels["S3"]),
    )
    summary = summarize_polarization(maps)
    _save_image(
        frame_dir / "rgb.png",
        tonemap_srgb(rgb),
        title="RGB  display = clip(linear, 0, 1) ** (1/2.2)",
    )
    _save_image(
        frame_dir / "dolp.png",
        maps["dolp"],
        title="DoLP = sqrt(S1^2 + S2^2) / S0",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    if showcase:
        _write_polarized(stokes_dir, bitmap, channels)
    logger.info(
        "phase3 %s variant=%s status=ok max_dolp=%.6f mean_dolp=%.6f elapsed_s=%.3f",
        label,
        STOKES_VARIANT,
        summary["max_dolp"],
        summary["mean_dolp"],
        elapsed_s,
    )

    _loaded, _image, cube, cube_elapsed_s = _render(
        f"phase3_cube_{record['index']:03d}",
        CUBE_VARIANT,
        lambda: _scene(record, polarized=False),
        resolution,
        cube_spp,
        record["seed"],
        cube=True,
    )
    if cube.ndim != 3 or cube.shape[2] != len(spectral_bands()):
        raise RuntimeError(f"{label} cube shape {cube.shape}")
    cube_dir = frame_dir / "cube"
    cube_dir.mkdir(parents=True, exist_ok=True)
    mi.Bitmap(cube, channel_names=[band.name for band in spectral_bands()]).write(str(cube_dir / "render.exr"))
    logger.info(
        "phase3 %s cube variant=%s status=ok elapsed_s=%.3f",
        label,
        CUBE_VARIANT,
        cube_elapsed_s,
    )

    mask, mask_variant = render_mask(record, resolution)
    _save_mask(frame_dir / "mask.png", mask)

    written = {
        **record,
        "resolution": int(resolution),
        "stokes_spp": int(stokes_spp),
        "cube_spp": int(cube_spp),
        "render_seed": int(record["seed"]),
        "stokes_variant": STOKES_VARIANT,
        "cube_variant": CUBE_VARIANT,
        "mask_variant": mask_variant,
        "mask_definition": MASK_DEFINITION,
        "channel_names": channel_names,
        "cube_channels": [band.name for band in spectral_bands()],
        "max_dolp": summary["max_dolp"],
        "mean_dolp": summary["mean_dolp"],
        "mean_abs_s3_over_s0": summary["mean_abs_s3_over_s0"],
        "n_film_pixels": int(mask.sum()),
        "stokes_elapsed_s": elapsed_s,
        "cube_elapsed_s": cube_elapsed_s,
    }
    _write_json(frame_dir / "params.json", written)
    return written, tonemap_srgb(rgb)


def _write_index(path: Path, images: list[np.ndarray]) -> None:
    columns = 8
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.6, rows * 1.6))
    for axis, image in zip(np.ravel(axes), images):
        axis.imshow(image)
        axis.axis("off")
    for axis in np.ravel(axes)[len(images) :]:
        axis.axis("off")
    fig.suptitle("Phase 3 frames, display RGB")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_phase3(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    seed: int = 0,
    resolution: int = RESOLUTION,
    stokes_spp: int = STOKES_SPP,
    cube_spp: int = CUBE_SPP,
) -> list[dict]:
    """Render the 32-frame proof set."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = sample_frames(seed)
    written = []
    previews = []
    first_pplastic = next(record["index"] for record in records if record["material"] == "pplastic")
    for record in records:
        frame_dir = output_root / "frames" / f"{record['index']:03d}"
        result, preview = render_frame(
            record,
            frame_dir,
            resolution=resolution,
            stokes_spp=stokes_spp,
            cube_spp=cube_spp,
            showcase=record["index"] in (0, first_pplastic),
        )
        written.append(result)
        previews.append(preview)
    _write_index(output_root / "index.png", previews)
    manifest = {
        "seed": int(seed),
        "frame_count": len(written),
        "film_count": sum(row["material"] == "dielectric" for row in written),
        "pplastic_count": sum(row["material"] == "pplastic" for row in written),
        "resolution": int(resolution),
        "stokes_spp": int(stokes_spp),
        "cube_spp": int(cube_spp),
        "stokes_variant": STOKES_VARIANT,
        "cube_variant": CUBE_VARIANT,
        "mask_definition": MASK_DEFINITION,
        "bands": [
            {
                "name": band.name,
                "wavelength_min": band.wavelength_min,
                "wavelength_max": band.wavelength_max,
                "center_nm": band.center_nm,
            }
            for band in spectral_bands()
        ],
        "frames": written,
    }
    _write_json(output_root / "manifest.json", manifest)
    logger.info(
        "phase3 frames=%d film=%d pplastic=%d",
        manifest["frame_count"],
        manifest["film_count"],
        manifest["pplastic_count"],
    )
    return written


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Render the Phase 3 32-frame proof dataset.")
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--stokes-spp", type=int, default=STOKES_SPP)
    parser.add_argument("--cube-spp", type=int, default=CUBE_SPP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    logger.info(
        "phase3 mitsuba=%s drjit=%s",
        getattr(mi, "__version__", "unknown"),
        __import__("drjit").__version__,
    )
    render_phase3(
        output_root=args.output_root,
        seed=args.seed,
        resolution=args.resolution,
        stokes_spp=args.stokes_spp,
        cube_spp=args.cube_spp,
    )


if __name__ == "__main__":
    main()

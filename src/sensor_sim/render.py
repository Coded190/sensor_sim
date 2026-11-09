"""Render Phase 0 scenes and write Stokes figures.

Outputs for one job live in ``outputs/phase0/<scene>_<variant>/``:

- ``render.exr`` is the linear OpenEXR image. Polarized scenes store
  RGB plus ``S0`` through ``S3``. This is the measurement file.
- ``rgb.png``, ``s0.png`` through ``s3.png``, ``dolp.png``, ``aolp.png``
  are display images. RGB and ``S0`` use ``clip(linear, 0, 1) ** (1/2.2)``.
  Signed Stokes components use a diverging scale at the 99th percentile
  of the absolute value.
- ``contact_sheet.png`` places those panels on one figure.
- ``metrics.json`` records the run and the definitions of each scalar.

Run ``uv run phase0`` to render the saved-figure set. Tests and debug
renders should pass ``variant=scalar_spectral_polarized``.
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
from matplotlib.colors import TwoSlopeNorm

from sensor_sim.scenes import PHASE2_MATERIALS, SCENE_NAMES, load_scene
from sensor_sim.stokes import DEFINITIONS, polarization_maps, summarize_polarization

logger = logging.getLogger("phase0")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase0"

FIGURE_JOBS = (
    ("cornell_box", "scalar_rgb", 256, 64),
    ("cbox_pol", "llvm_ad_spectral_polarized", 256, 64),
    ("dielectric_sphere", "llvm_ad_spectral_polarized", 256, 64),
    ("conductor_spheres", "llvm_ad_spectral_polarized", 256, 64),
)

DISPLAY_GAMMA = 2.2
SIGNED_PERCENTILE = 99.0


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def tonemap_srgb(linear: np.ndarray) -> np.ndarray:
    """Display transform for radiance. The EXR keeps the linear values."""
    return np.clip(np.clip(linear, 0.0, None), 0.0, 1.0) ** (1.0 / DISPLAY_GAMMA)


def _symmetric_limit(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(finite, SIGNED_PERCENTILE))
    return max(limit, 1e-6)


def _save_image(path: Path, image: np.ndarray, title: str, **imshow_kwargs) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    artist = ax.imshow(image, **imshow_kwargs)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if image.ndim == 2 or np.ma.isMaskedArray(image):
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _split_bitmap(scene, image):
    aov_names = list(scene.integrator().aov_names())
    channel_names = ["R", "G", "B"] + aov_names
    bitmap = mi.Bitmap(image, channel_names=channel_names)
    channels = dict(bitmap.split())
    return bitmap, channels, channel_names


def _rgb_metrics(rgb: np.ndarray) -> dict[str, float]:
    finite = rgb[np.isfinite(rgb)]
    return {
        "mean_rgb": float(finite.mean()) if finite.size else 0.0,
        "max_rgb": float(finite.max()) if finite.size else 0.0,
    }


def _write_rgb_only(out_dir: Path, bitmap, rgb: np.ndarray, title: str = "RGB") -> None:
    bitmap.write(str(out_dir / "render.exr"))
    _save_image(
        out_dir / "rgb.png",
        tonemap_srgb(rgb),
        title="RGB  display = clip(linear, 0, 1) ** (1/2.2)",
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(tonemap_srgb(rgb))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "contact_sheet.png", dpi=120)
    plt.close(fig)


def _write_polarized(out_dir: Path, bitmap, channels: dict) -> dict:
    rgb = np.array(channels["<root>"], copy=True)
    s0 = np.array(channels["S0"], copy=True)
    s1 = np.array(channels["S1"], copy=True)
    s2 = np.array(channels["S2"], copy=True)
    s3 = np.array(channels["S3"], copy=True)
    maps = polarization_maps(s0, s1, s2, s3)
    summary = summarize_polarization(maps)
    summary["mean_abs_s0_minus_rgb"] = float(np.mean(np.abs(s0 - rgb)))

    bitmap.write(str(out_dir / "render.exr"))
    s0_display = tonemap_srgb(s0.mean(axis=2))
    rgb_display = tonemap_srgb(rgb)
    _save_image(out_dir / "rgb.png", rgb_display, title="RGB  display = clip(linear, 0, 1) ** (1/2.2)")
    _save_image(
        out_dir / "s0.png",
        s0_display,
        title="S0 intensity  display = clip(mean RGB, 0, 1) ** (1/2.2)",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    signed_panels = []
    for name, values, label in (
        ("s1", maps["s1"], "S1 horizontal vs vertical"),
        ("s2", maps["s2"], "S2 +45 deg vs -45 deg"),
        ("s3", maps["s3"], "S3 right vs left circular"),
    ):
        limit = _symmetric_limit(values)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        title = f"{label}\n99th |value| = {limit:.4g}"
        _save_image(out_dir / f"{name}.png", values, title=title, cmap="coolwarm", norm=norm)
        signed_panels.append((values, title, norm))

    _save_image(
        out_dir / "dolp.png",
        maps["dolp"],
        title="DoLP = sqrt(S1^2 + S2^2) / S0",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    aolp_deg = np.ma.array(np.degrees(maps["aolp"]), mask=maps["aolp_masked"])
    _save_image(
        out_dir / "aolp.png",
        aolp_deg,
        title="AoLP deg = 0.5*arctan2(S2, S1); masked where DoLP < 0.05",
        cmap="hsv",
        vmin=-90.0,
        vmax=90.0,
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    panels = [
        (axes[0, 0], rgb_display, "RGB", None),
        (axes[0, 1], s0_display, "S0", dict(cmap="gray", vmin=0.0, vmax=1.0)),
        (axes[0, 2], signed_panels[0][0], "S1", dict(cmap="coolwarm", norm=signed_panels[0][2])),
        (axes[0, 3], signed_panels[1][0], "S2", dict(cmap="coolwarm", norm=signed_panels[1][2])),
        (axes[1, 0], signed_panels[2][0], "S3", dict(cmap="coolwarm", norm=signed_panels[2][2])),
        (axes[1, 1], maps["dolp"], "DoLP", dict(cmap="viridis", vmin=0.0, vmax=1.0)),
        (axes[1, 2], aolp_deg, "AoLP (deg)", dict(cmap="hsv", vmin=-90.0, vmax=90.0)),
    ]
    for ax, image, title, kwargs in panels:
        if kwargs is None:
            ax.imshow(image)
        else:
            ax.imshow(image, **kwargs)
        ax.set_title(title)
        ax.axis("off")
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.0,
        0.5,
        "\n".join(
            [
                "DoP = sqrt(S1^2+S2^2+S3^2)/S0",
                "DoLP = sqrt(S1^2+S2^2)/S0",
                "AoLP = 0.5*arctan2(S2,S1)",
                f"mean DoLP = {summary['mean_dolp']:.4f}",
                f"max DoLP = {summary['max_dolp']:.4f}",
                f"max DoP = {summary['max_dop']:.4f}",
                f"frac DoP>1.05 = {summary['frac_dop_gt_1_05']:.4f}",
                f"mean |S3| = {summary['mean_abs_s3']:.4g}",
                f"mean |S3|/S0 = {summary['mean_abs_s3_over_s0']:.4g}",
            ]
        ),
        family="monospace",
        fontsize=10,
        va="center",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "contact_sheet.png", dpi=120)
    plt.close(fig)
    return summary


def render_job(
    scene_name: str,
    variant: str,
    resolution: int,
    spp: int,
    seed: int = 0,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    """Render one scene and write EXR, PNGs, and metrics.json."""
    if scene_name not in SCENE_NAMES and scene_name not in PHASE2_MATERIALS:
        raise ValueError(f"unknown scene {scene_name!r}")
    out_dir = Path(output_root) / f"{scene_name}_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mi.set_variant(variant)
    scene = load_scene(scene_name, resolution, spp)
    started = time.perf_counter()
    image = mi.render(scene, spp=spp, seed=seed)
    elapsed_s = time.perf_counter() - started
    image_np = np.array(image)
    if not np.isfinite(image_np).all():
        n_bad = int(np.size(image_np) - np.isfinite(image_np).sum())
        logger.error(
            "phase0 scene=%s variant=%s nonfinite_values=%d",
            scene_name,
            variant,
            n_bad,
        )
        raise RuntimeError(f"{scene_name} produced {n_bad} non-finite pixels")

    bitmap, channels, channel_names = _split_bitmap(scene, image)
    polarized = "S0" in channels
    metrics: dict = {
        "scene": scene_name,
        "variant": variant,
        "resolution": int(resolution),
        "spp": int(spp),
        "seed": int(seed),
        "elapsed_s": elapsed_s,
        "channel_names": channel_names,
        "definitions": DEFINITIONS,
    }
    if polarized:
        summary = _write_polarized(out_dir, bitmap, channels)
        metrics.update(summary)
        logger.info(
            "phase0 scene=%s variant=%s res=%d spp=%d seed=%d elapsed_s=%.3f "
            "mean_dolp=%.6f max_dolp=%.6f max_dop=%.6f frac_dop_gt_1_05=%.6f mean_abs_s3=%.6g "
            "mean_abs_s3_over_s0=%.6g mean_abs_s0_minus_rgb=%.6g",
            scene_name,
            variant,
            resolution,
            spp,
            seed,
            elapsed_s,
            summary["mean_dolp"],
            summary["max_dolp"],
            summary["max_dop"],
            summary["frac_dop_gt_1_05"],
            summary["mean_abs_s3"],
            summary["mean_abs_s3_over_s0"],
            summary["mean_abs_s0_minus_rgb"],
        )
    else:
        rgb = np.array(channels["<root>"], copy=True)
        _write_rgb_only(out_dir, bitmap, rgb)
        rgb_stats = _rgb_metrics(rgb)
        metrics.update(rgb_stats)
        logger.info(
            "phase0 scene=%s variant=%s res=%d spp=%d seed=%d elapsed_s=%.3f mean_rgb=%.6f max_rgb=%.6f",
            scene_name,
            variant,
            resolution,
            spp,
            seed,
            elapsed_s,
            rgb_stats["mean_rgb"],
            rgb_stats["max_rgb"],
        )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def render_figure_jobs(output_root: Path = DEFAULT_OUTPUT_ROOT, seed: int = 0) -> list[dict]:
    results = []
    for scene_name, variant, resolution, spp in FIGURE_JOBS:
        results.append(
            render_job(
                scene_name,
                variant,
                resolution,
                spp,
                seed=seed,
                output_root=output_root,
            )
        )
    return results


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Render Phase 0 Mitsuba scenes.")
    parser.add_argument("--scene", choices=SCENE_NAMES, help="Render one scene. Default: all figure jobs.")
    parser.add_argument("--variant", help="Mitsuba variant. Default depends on the scene.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    logger.info(
        "phase0 mitsuba=%s drjit=%s",
        getattr(mi, "__version__", "unknown"),
        __import__("drjit").__version__,
    )
    if args.scene is None:
        render_figure_jobs(output_root=args.output_root, seed=args.seed)
        return

    variant = args.variant
    if variant is None:
        variant = "scalar_rgb" if args.scene == "cornell_box" else "llvm_ad_spectral_polarized"
    render_job(
        args.scene,
        variant,
        args.resolution,
        args.spp,
        seed=args.seed,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()

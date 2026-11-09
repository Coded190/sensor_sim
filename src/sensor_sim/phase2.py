"""Compare six sphere BSDFs and recommend the film material.

``uv run phase2`` writes ``outputs/phase2/<material>_llvm_ad_spectral_polarized/``
with the same EXR, PNGs, contact sheet, and ``metrics.json`` as Phase 0.
``comparison.json`` and ``comparison.png`` summarize ``max_dolp`` and
``mean_abs_s3_over_s0``. These jobs are not part of ``uv run phase0``.

A BSDF keeps polarization when ``max_dolp > 0.2`` and ``max_dolp`` is at
least five times the diffuse control. ``max_dolp`` is the maximum of
``sqrt(S1^2 + S2^2) / S0`` over pixels with ``S0 > 1e-4``.
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

from sensor_sim.render import configure_logging, render_job
from sensor_sim.scenes import PHASE2_MATERIALS, material_bsdf

logger = logging.getLogger("phase2")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase2"
VARIANT = "llvm_ad_spectral_polarized"
RESOLUTION = 256
SPP = 64
KEEPS_MAX_DOLP = 0.2
KEEPS_RATIO = 5.0

_UNSUPPORTED_MARKERS = (
    "not implemented",
    "not supported",
    "unsupported",
    "not compatible",
    "eval_pol",
)


def keeps_polarization(max_dolp: float, diffuse_max_dolp: float) -> bool:
    """True when this BSDF is polarized well above the diffuse control."""
    return max_dolp > KEEPS_MAX_DOLP and max_dolp >= KEEPS_RATIO * diffuse_max_dolp


def _unsupported(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _UNSUPPORTED_MARKERS)


def render_material(
    material: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    variant: str = VARIANT,
    resolution: int = RESOLUTION,
    spp: int = SPP,
    seed: int = 0,
) -> dict:
    """Render one sphere. An unsupported polarized BSDF is recorded and does not raise."""
    output_root = Path(output_root)
    out_dir = output_root / f"{material}_{variant}"
    bsdf = material_bsdf(material)
    try:
        metrics = render_job(
            material,
            variant,
            resolution,
            spp,
            seed=seed,
            output_root=output_root,
        )
    except Exception as exc:
        if not _unsupported(exc):
            logger.error(
                "phase2 material=%s variant=%s status=failed error=%s",
                material,
                variant,
                exc,
            )
            raise
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "scene": material,
            "material": material,
            "variant": variant,
            "resolution": int(resolution),
            "spp": int(spp),
            "seed": int(seed),
            "status": "unsupported",
            "error": f"{type(exc).__name__}: {exc}",
            "bsdf": bsdf,
            "alpha": bsdf.get("alpha"),
            "max_dolp": None,
            "mean_abs_s3_over_s0": None,
        }
        (out_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        logger.error(
            "phase2 material=%s variant=%s status=unsupported error=%s",
            material,
            variant,
            exc,
        )
        return record

    metrics["material"] = material
    metrics["status"] = "ok"
    metrics["bsdf"] = bsdf
    metrics["alpha"] = bsdf.get("alpha")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    logger.info(
        "phase2 material=%s variant=%s status=ok max_dolp=%.6f mean_abs_s3_over_s0=%.6g alpha=%s",
        material,
        variant,
        metrics["max_dolp"],
        metrics["mean_abs_s3_over_s0"],
        metrics["alpha"],
    )
    return metrics


def recommend(results: list[dict]) -> dict:
    """Apply the decision rule. A contradicted bullet leaves the choice open."""
    by_name = {row["material"]: row for row in results}
    diffuse = by_name["diffuse"]
    flags: dict[str, bool] = {}
    if diffuse.get("status") != "ok" or diffuse.get("max_dolp") is None:
        flags = {name: False for name in PHASE2_MATERIALS}
        contradictions = ["diffuse control did not render"]
        diffuse_max = None
    else:
        diffuse_max = float(diffuse["max_dolp"])
        for name in PHASE2_MATERIALS:
            row = by_name[name]
            if row.get("status") != "ok" or row.get("max_dolp") is None:
                flags[name] = False
            else:
                flags[name] = keeps_polarization(float(row["max_dolp"]), diffuse_max)
        contradictions = []
        if not (flags["dielectric"] and not flags["thindielectric"]):
            contradictions.append(
                "dielectric should keep polarization and thindielectric should not"
            )
        if not (flags["pplastic"] and not flags["plastic"]):
            contradictions.append("pplastic should keep polarization and plastic should not")
        if flags["roughdielectric"]:
            contradictions.append(
                "roughdielectric kept polarization, so roughness would be a workaround"
            )

    if contradictions:
        choice = None
        opaque_alternative = None
        text = (
            "The renders contradict the decision rule. The choice is open. "
            + " ".join(contradictions)
            + "."
        )
    else:
        choice = "dielectric"
        opaque_alternative = "pplastic"
        text = (
            "Use the closed dielectric shell for the transparent film. "
            "pplastic is the alternative if the object is later treated as opaque coated plastic. "
            "thindielectric, roughdielectric, plastic, and diffuse do not keep polarization."
        )
    return {
        "keeps_polarization": flags,
        "diffuse_max_dolp": diffuse_max,
        "rule": (
            f"keeps polarization when max_dolp > {KEEPS_MAX_DOLP} and "
            f"max_dolp >= {KEEPS_RATIO:g} * diffuse max_dolp"
        ),
        "choice": choice,
        "opaque_alternative": opaque_alternative,
        "contradictions": contradictions,
        "text": text,
    }


def _bar_value(row: dict, key: str) -> float:
    value = row.get(key)
    if row.get("status") != "ok" or value is None:
        return 0.0
    return float(value)


def write_comparison(results: list[dict], output_root: Path, recommendation: dict) -> dict:
    """Write comparison.json and a two-panel bar chart."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in results:
        rows.append(
            {
                "material": row["material"],
                "status": row["status"],
                "variant": row["variant"],
                "resolution": row["resolution"],
                "spp": row["spp"],
                "seed": row["seed"],
                "alpha": row.get("alpha"),
                "bsdf": row.get("bsdf"),
                "max_dolp": row.get("max_dolp"),
                "mean_abs_s3_over_s0": row.get("mean_abs_s3_over_s0"),
                "mean_dolp": row.get("mean_dolp"),
                "keeps_polarization": recommendation["keeps_polarization"][row["material"]],
                "error": row.get("error"),
            }
        )
    payload = {
        "definition_max_dolp": "max of sqrt(S1^2 + S2^2) / S0 over pixels with S0 > 1e-4",
        "definition_mean_abs_s3_over_s0": "mean of |S3| / S0 over pixels with S0 > 1e-4",
        "recommendation": recommendation,
        "materials": rows,
    }
    (output_root / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")

    labels = []
    for row in rows:
        label = row["material"]
        if row["status"] != "ok":
            label = f"{label}\n({row['status']})"
        labels.append(label)
    max_dolp = [_bar_value(row, "max_dolp") for row in rows]
    s3 = [_bar_value(row, "mean_abs_s3_over_s0") for row in rows]
    positions = list(range(len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(positions, max_dolp, color="teal")
    axes[0].axhline(KEEPS_MAX_DOLP, color="black", linestyle="--", linewidth=1, label="0.2 threshold")
    axes[0].set_ylabel("max DoLP")
    axes[0].set_title("max DoLP = max sqrt(S1^2+S2^2)/S0")
    axes[0].legend(fontsize=8)
    axes[1].bar(positions, s3, color="slategray")
    axes[1].set_ylabel("mean |S3| / S0")
    axes[1].set_title("mean |S3| / S0")
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_root / "comparison.png", dpi=120)
    plt.close(fig)
    logger.info(
        "phase2 comparison choice=%s opaque_alternative=%s",
        recommendation["choice"],
        recommendation["opaque_alternative"],
    )
    return payload


def render_phase2(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    seed: int = 0,
    resolution: int = RESOLUTION,
    spp: int = SPP,
    variant: str = VARIANT,
) -> dict:
    """Render the six materials, then write the comparison."""
    output_root = Path(output_root)
    results = []
    for material in PHASE2_MATERIALS:
        results.append(
            render_material(
                material,
                output_root=output_root,
                variant=variant,
                resolution=resolution,
                spp=spp,
                seed=seed,
            )
        )
    recommendation = recommend(results)
    comparison = write_comparison(results, output_root, recommendation)
    logger.info("phase2 recommendation=%s", recommendation["text"])
    return comparison


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Compare Phase 2 sphere BSDFs.")
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--spp", type=int, default=SPP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default=VARIANT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    logger.info(
        "phase2 mitsuba=%s drjit=%s variant=%s",
        getattr(mi, "__version__", "unknown"),
        __import__("drjit").__version__,
        args.variant,
    )
    render_phase2(
        output_root=args.output_root,
        seed=args.seed,
        resolution=args.resolution,
        spp=args.spp,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()

"""Train the held-out Stokes U-Net and save the weights the policy will freeze.

The held-out trainer used to drop the network after writing PNG normals. This
module loads those measurements, runs the same fit, and writes ``stokes.pt``
plus the mean angular error on the ten test shapes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from sensor_sim.film.heldout import _fit_model
from sensor_sim.film.sfp import OUTPUT, SEED, _channels, _loss_mask
from sensor_sim.film.sfp_math import angular_error_summary
from sensor_sim.film.shape_mesh import shape_catalog
from sensor_sim.film.unet import FilmUNet

logger = logging.getLogger("sensor_sim.policy")

ROOT = OUTPUT / "02_heldout_shapes"
CHECKPOINT = OUTPUT / "policy" / "stokes.pt"
REPORT = OUTPUT / "policy" / "stokes_mae.json"


def load_records(split: str) -> list[dict]:
    """Metadata plus ``measurement.npz`` for one split. Catalog fields are attached."""
    catalog = {spec["folder"]: spec for spec in shape_catalog(SEED)}
    records = []
    for path in sorted((ROOT / split).iterdir()):
        if not path.is_dir() or not (path / "measurement.npz").is_file():
            continue
        record = json.loads((path / "metadata.json").read_text())
        record["measurement"] = {key: np.asarray(value) for key, value in np.load(path / "measurement.npz").items()}
        record["spec"] = catalog[record["view"]]
        record["out_dir"] = path
        records.append(record)
    if not records:
        raise RuntimeError(f"no measurements in {ROOT / split}")
    return records


def save_stokes_checkpoint(device: str = "cuda") -> dict:
    """Fit Stokes on the 50 training shapes and score the 10 held-out shapes."""
    train = load_records("train")
    test = load_records("test")
    if len(train) != 50 or len(test) != 10:
        raise RuntimeError(f"expected 50 train and 10 test, got {len(train)} and {len(test)}")
    logger.info("policy stage=stokes_train device=%s train=%d test=%d", device, len(train), len(test))
    predictions, info = _fit_model(train, test, "stokes", device, crops_per_image=1, checkpoint_path=CHECKPOINT)
    rows = []
    for record, predicted in zip(test, predictions, strict=True):
        mask = _loss_mask(record["measurement"], set(record["object_indices"]))
        summary = angular_error_summary(predicted, record["measurement"]["normal"], mask)
        rows.append(
            {
                "view": record["view"],
                "family": record["family"],
                "mae_deg": summary["mae_deg"],
                "median_deg": summary["median_deg"],
                "pixels": summary["pixels"],
            }
        )
        logger.info(
            "policy stage=stokes_mae view=%s mae_deg=%.4f median_deg=%.4f pixels=%d",
            record["view"],
            summary["mae_deg"],
            summary["median_deg"],
            summary["pixels"],
        )
    report = {
        "checkpoint": str(CHECKPOINT),
        "definition": (
            "Mean angular error is the mean, over film pixels with S0 > 1e-4 and a unit "
            "normal, of arccos(clip(dot(prediction, truth), -1, 1)) in degrees. "
            "The headline is the unweighted mean of the ten test shapes."
        ),
        "shapes": rows,
        "unweighted_mean_mae_deg": float(np.mean([row["mae_deg"] for row in rows])),
        "train_info": {key: value for key, value in info.items() if key != "standardization"},
        "standardization_in_checkpoint": True,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    logger.info(
        "policy stage=stokes_done mean_mae_deg=%.4f path=%s",
        report["unweighted_mean_mae_deg"],
        CHECKPOINT,
    )
    return report


def load_stokes(device: str, path: Path | None = None):
    """Return the frozen network, the training mean, and the training std."""
    payload = torch.load(Path(path or CHECKPOINT), map_location=device, weights_only=False)
    model = FilmUNet(int(payload["in_channels"]), out_channels=3).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, np.asarray(payload["mean"], dtype=np.float64), np.asarray(payload["std"], dtype=np.float64)


def predict_stokes(model, measurement: dict, mean: np.ndarray, std: np.ndarray, device: str) -> np.ndarray:
    """Unit camera-frame normals. Pixels outside the film mask are left for the caller to zero."""
    image = _channels(measurement, "stokes")
    standardized = (image - mean) / std
    tensor = torch.tensor(np.transpose(standardized, (2, 0, 1))[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = model(tensor)[0]
        prediction = prediction / prediction.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return np.transpose(prediction.detach().cpu().numpy(), (1, 2, 0))

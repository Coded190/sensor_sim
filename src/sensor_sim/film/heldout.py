"""Held-out filmed shapes. ``uv run film-sfp`` enters here.

Experiment 01 retrains the Cornell film and leaves the copper mesh out of
the loss, the error images, and the summary. Experiment 02 trains on 50
generated shapes and scores 10 held-out shapes. Neither SfPUEL archive is
downloaded. The device is chosen after the Dr.Jit and Torch caches are flushed.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sensor_sim.film.capture import _render, _stokes_from_image
from sensor_sim.film.config import scene_config
from sensor_sim.film.lighting import display_srgb
from sensor_sim.film.sfp import (
    BATCH,
    CROP,
    EPOCHS,
    LEARNING_RATE,
    MAX_DEPTH,
    MODELS,
    OUTPUT,
    RESOLUTION,
    SEED,
    SPP,
    VARIANT,
    _channels,
    _index_at,
    _loss_mask,
    _save_error,
    _save_normals,
    _save_rgb,
    _split_aov,
    _write_prediction_images,
)
from sensor_sim.film.geometry import write_obj
from sensor_sim.film.sfp_math import (
    angular_error_deg,
    angular_error_summary,
    aolp_pair,
    camera_matrix,
    fresnel_normals,
    mean_abs_s3_over_s0,
    release_render_caches,
    select_torch_device,
    view_directions,
    world_normals_to_camera,
)
from sensor_sim.film.sfp_scene import (
    CORNELL_CAMERA_B,
    STANDARD_LIGHT_RADIANCE,
    filmed_object_dict,
    write_cornell_film,
    write_gray_room,
)
from sensor_sim.film.shape_mesh import (
    MIN_FILM_FRACTION,
    TEST_COUNTS,
    TRAIN_COUNTS,
    build_shape,
    shape_catalog,
)
from sensor_sim.render import configure_logging
from sensor_sim.stokes import S0_MIN

logger = logging.getLogger("sensor_sim.sfp")

RENDER_PNGS = ("rgb.png", "normal_gt.png", "dolp.png", "aolp.png", "s3.png")
EXPERIMENT_01 = "01_cornell_film"
EXPERIMENT_02 = "02_heldout_shapes"
SFPUEL_NOTE = (
    "Neither SfPUEL archive was downloaded. Their training zip is about 20,000 "
    "Mitsuba 2 environment-map renders of bare metals and dielectrics, with no "
    "polyethylene film. Their test set is four polarizer PNGs scored on an object "
    "mask, stored as relative 8-bit or 16-bit intensities, with no linear Stokes "
    "vector and no S3. The angle columns below use their strict cutoffs so the "
    "numbers can be read beside their table. The pixels and the lighting are ours."
)


def score_definitions() -> dict:
    """How every number in the two summary files is computed."""
    return {
        "mae_deg": (
            "Mean, over film pixels with S0 > 1e-4 and a unit normal, of "
            "arccos(clip(dot(prediction, truth), -1, 1)) in degrees. The truth is "
            "the Mitsuba shading normal nn, rotated into the camera and pointed back "
            "at the camera."
        ),
        "median_deg": "Median of those same per-pixel angles.",
        "fraction_below_11_25": "Fraction of those angles strictly below 11.25 degrees.",
        "fraction_below_22_5": "Fraction of those angles strictly below 22.5 degrees.",
        "fraction_below_30": "Fraction of those angles strictly below 30 degrees.",
        "test_mean": (
            "Unweighted mean of the ten per-shape scores. A shape with more film "
            "pixels does not count more than a shape with fewer."
        ),
        "mean_abs_s3_over_s0": (
            "Mean of |S3| / S0 on film pixels with S0 > 1e-4. S3 is written to "
            "measurement.npz and s3.png. It is not a network input."
        ),
        "lamp": (
            "Unpolarized area light on the ceiling. No polarizer and no quarter-wave "
            "plate are in the scene."
        ),
        "fresnel_specular": (
            "Baseline. Azimuth equals AoLP + pi/2, on the film only. The smaller "
            "zenith whose Fresnel DoLP matches the pixel is kept. Not applied to copper."
        ),
        "fresnel_oracle": (
            "Same zenith, but each pixel keeps azimuth or azimuth + pi, whichever is "
            "closer to the truth. This is not the baseline."
        ),
        "loss": (
            "Mean of 1 - dot(unit prediction, truth) on film pixels with S0 > 1e-4 "
            "and a unit normal. Walls, the inner diffuse solid, and the copper mesh "
            "are omitted."
        ),
        "standardization": (
            "Per-channel mean and NumPy standard deviation (divide by N) on the "
            "training film pixels. A channel whose std is below 1e-8 is left unscaled. "
            "The same numbers are applied to every evaluated frame."
        ),
    }


def assert_no_dragon_key(payload) -> None:
    """The copper mesh is in the Cornell picture and absent from the summary keys."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if "dragon" in str(key).lower():
                raise RuntimeError(f"summary key {key} names the copper mesh")
            assert_no_dragon_key(value)
    elif isinstance(payload, list):
        for item in payload:
            assert_no_dragon_key(item)


def pack_measurement(bundle: dict, normals: np.ndarray, shape_index: np.ndarray, view: np.ndarray) -> dict:
    """Arrays written to measurement.npz. S3 is included and is not a model input."""
    cosine, sine = aolp_pair(bundle["aolp"], bundle["dolp"])
    rgb = np.asarray(bundle["rgb"], dtype=np.float64)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    payload = {
        "rgb": rgb,
        "s0": np.asarray(bundle["s0"], dtype=np.float64),
        "s1": np.asarray(bundle["s1"], dtype=np.float64),
        "s2": np.asarray(bundle["s2"], dtype=np.float64),
        "s3": np.asarray(bundle["s3"], dtype=np.float64),
        "dolp": np.asarray(bundle["dolp"], dtype=np.float64),
        "aolp": np.asarray(bundle["aolp"], dtype=np.float64),
        "aolp_cos": cosine,
        "aolp_sin": sine,
        "i0": np.asarray(bundle["analyzer_000"], dtype=np.float64),
        "i45": np.asarray(bundle["analyzer_045"], dtype=np.float64),
        "i90": np.asarray(bundle["analyzer_090"], dtype=np.float64),
        "i135": np.asarray(bundle["analyzer_135"], dtype=np.float64),
        "normal": np.asarray(normals, dtype=np.float64),
        "shape_index": np.asarray(shape_index, dtype=np.float64),
        "view": np.asarray(view, dtype=np.float64),
    }
    if "s3" not in payload:
        raise RuntimeError("measurement is missing s3")
    return payload


def _save_scalar(path: Path, values: np.ndarray, mask: np.ndarray, title: str, cmap: str, vmin: float, vmax: float) -> None:
    shown = np.where(mask, np.asarray(values, dtype=np.float64), np.nan)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(shown, vmin=vmin, vmax=vmax, cmap=cmap)
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _write_render_pngs(out_dir: Path, measurement: dict, film_mask: np.ndarray, s3_ratio: float) -> None:
    s0 = measurement["s0"]
    bright = film_mask & (s0 > S0_MIN)
    _save_rgb(out_dir / "rgb.png", measurement["rgb"])
    unit = np.linalg.norm(measurement["normal"], axis=-1) > 0.5
    _save_normals(out_dir / "normal_gt.png", measurement["normal"], film_mask & unit)
    _save_scalar(out_dir / "dolp.png", measurement["dolp"], bright, "DoLP on the film", "viridis", 0.0, 1.0)
    aolp_mask = bright & (measurement["dolp"] >= 0.05)
    _save_scalar(
        out_dir / "aolp.png",
        measurement["aolp"],
        aolp_mask,
        "AoLP (rad) where DoLP >= 0.05",
        "twilight",
        -0.5 * np.pi,
        0.5 * np.pi,
    )
    if int(bright.sum()) == 0:
        limit = 1e-6
    else:
        limit = float(np.percentile(np.abs(measurement["s3"][bright]), 99))
        limit = max(limit, 1e-6)
    _save_scalar(
        out_dir / "s3.png",
        measurement["s3"],
        bright,
        f"S3  mean |S3|/S0 {s3_ratio:.4f}",
        "coolwarm",
        -limit,
        limit,
    )
    missing = [name for name in RENDER_PNGS if not (out_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"{out_dir} is missing {missing}")


def _widest_normal_parameter_count() -> int:
    """Stokes U-Net width. It has more input channels than the analyzer and RGB nets."""
    from sensor_sim.film.unet import FilmUNet, parameter_count

    model = FilmUNet(9, out_channels=3)
    count = parameter_count(model)
    del model
    return count


def select_device() -> tuple[str, str, dict]:
    """Flush Dr.Jit and Torch, then use CUDA unless the batch floor is not free."""
    device, reason, info = select_torch_device(_widest_normal_parameter_count())
    logger.info(
        "sfp stage=release_gpu %s",
        info["release"],
    )
    logger.info(
        "sfp stage=device cuda_free_bytes=%s cuda_total_bytes=%s batch_floor_bytes=%d n_parameters=%d",
        info["free_bytes"],
        info["total_bytes"],
        info["batch_floor_bytes"],
        info["n_parameters"],
    )
    logger.info(
        "sfp stage=device selected=%s reason=%s torch=%s",
        device,
        reason,
        info["torch_version"],
    )
    return device, reason, info


def _set_cuda_variant() -> str:
    """Select the polarized CUDA variant. There is no CPU fallback."""
    import mitsuba as mi

    mi.set_variant(VARIANT)
    active = mi.variant()
    if active != VARIANT:
        raise RuntimeError(f"render variant {active} is not {VARIANT}")
    return active


def _require_cuda_variant() -> None:
    """Log the variant once at startup. Later frames use the same setting."""
    logger.info("sfp stage=render variant=%s", _set_cuda_variant())


def _film_aov(build, probe_point: list[float]) -> dict:
    """One-sample shape index and shading normals. This runs before the Stokes render."""
    _set_cuda_variant()
    aov_scene = build(polarized=False, spp=1, rfilter="box")
    aov_scene["integrator"] = {"type": "aov", "aovs": "nn:sh_normal,idx:shape_index"}
    aov_loaded, _aov_image, aov_array, _aov_elapsed = _render(aov_scene, spp=1, seed=SEED)
    world_normals, shape_index = _split_aov(aov_loaded, aov_array)
    aov_sensor = aov_loaded.sensors()[0]
    normals = world_normals_to_camera(world_normals, camera_matrix(aov_sensor))
    film_index, row, column = _index_at(aov_sensor, shape_index, probe_point)
    del aov_loaded, aov_array, aov_sensor
    index = np.rint(shape_index).astype(np.int64)
    film_pixels = int((index == film_index).sum())
    fraction = film_pixels / float(shape_index.size)
    release_render_caches()
    return {
        "normals": normals,
        "shape_index": shape_index,
        "film_index": film_index,
        "row": row,
        "column": column,
        "film_pixels": film_pixels,
        "fraction": fraction,
    }


def render_film_view(
    *,
    experiment: str,
    view_name: str,
    out_dir: Path,
    build,
    probe_point: list[float],
    require_fraction: bool,
) -> dict:
    """Stokes plus shading normals. The film-pixel gate runs before the 64-sample render."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("sfp experiment=%s view=%s stage=normals", experiment, view_name)
    aov = _film_aov(build, probe_point)
    film_index = aov["film_index"]
    row = aov["row"]
    column = aov["column"]
    film_pixels = aov["film_pixels"]
    fraction = aov["fraction"]
    shape_index = aov["shape_index"]
    normals = aov["normals"]
    if film_index == 0:
        raise RuntimeError(f"{experiment} {view_name} film probe missed every surface")
    if require_fraction and fraction < MIN_FILM_FRACTION:
        raise RuntimeError(
            f"{experiment} {view_name} film covers {fraction:.4f} of the frame, below {MIN_FILM_FRACTION}"
        )
    if film_pixels < 100:
        raise RuntimeError(f"{experiment} {view_name} has only {film_pixels} film pixels")
    logger.info(
        "sfp experiment=%s view=%s stage=stokes variant=%s spp=%d seed=%d depth=%d",
        experiment,
        view_name,
        VARIANT,
        SPP,
        SEED,
        MAX_DEPTH,
    )
    loaded, image, _array, elapsed = _render(build(polarized=True, spp=SPP, rfilter="gaussian"), SPP, SEED)
    bundle = _stokes_from_image(loaded, image)
    height, width = bundle["s0"].shape
    view = view_directions(loaded.sensors()[0], width, height)
    del loaded, image
    release_render_caches()
    measurement = pack_measurement(bundle, normals, shape_index, view)
    film_mask = np.rint(shape_index).astype(np.int64) == film_index
    s3_ratio = mean_abs_s3_over_s0(measurement["s3"], measurement["s0"], film_mask)
    logger.info(
        "sfp experiment=%s view=%s stage=render status=ok film_index=%d film_pixels=%d "
        "film_fraction=%.4f mean_abs_s3_over_s0=%.6f elapsed_stokes_s=%.3f",
        experiment,
        view_name,
        film_index,
        film_pixels,
        fraction,
        s3_ratio,
        elapsed,
    )
    np.savez_compressed(out_dir / "measurement.npz", **measurement)
    _write_render_pngs(out_dir, measurement, film_mask, s3_ratio)
    record = {
        "experiment": experiment,
        "view": view_name,
        "out_dir": str(out_dir),
        "film_only": True,
        "variant": VARIANT,
        "resolution": list(RESOLUTION),
        "spp": SPP,
        "seed": SEED,
        "max_depth": MAX_DEPTH,
        "elapsed_stokes_s": elapsed,
        "shapes": {"film-shell": {"index": film_index, "pixel": [row, column], "point": list(probe_point)}},
        "object_indices": [film_index],
        "film_pixels": film_pixels,
        "film_pixel_fraction": fraction,
        "mean_abs_s3_over_s0": s3_ratio,
        "s0_min": S0_MIN,
        "lamp": "unpolarized ceiling rectangle",
        "measurement": measurement,
    }
    _write_json(out_dir / "metadata.json", {key: value for key, value in record.items() if key != "measurement"})
    return record


def _jsonable(value):
    """JSON objects. Non-finite floats become null so the summary stays valid JSON."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_json(path: Path, payload: dict) -> None:
    assert_no_dragon_key(payload)
    Path(path).write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def _crop_containing(mask: np.ndarray, generator, crop: int) -> tuple[int, int]:
    """A crop that contains one film pixel. BatchNorm cannot train on an empty film."""
    import torch

    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise RuntimeError("crop requested on a frame with no film pixels")
    pick = int(torch.randint(0, int(rows.size), (1,), generator=generator))
    row_offset = int(torch.randint(0, crop, (1,), generator=generator))
    column_offset = int(torch.randint(0, crop, (1,), generator=generator))
    height, width = mask.shape
    top = int(np.clip(int(rows[pick]) - row_offset, 0, height - crop))
    left = int(np.clip(int(columns[pick]) - column_offset, 0, width - crop))
    return top, left


def _unweighted_mean(summaries: list[dict]) -> dict:
    """Mean of per-shape scores. Each shape counts once."""
    keys = ("mae_deg", "median_deg", "fraction_below_11_25", "fraction_below_22_5", "fraction_below_30")
    return {key: float(np.mean([item[key] for item in summaries])) for key in keys}


def score_film_fresnel(experiment: str, record: dict) -> dict:
    """Specular Fresnel on the film, plus the oracle that may add pi. The oracle is labeled."""
    measurement = record["measurement"]
    index = np.rint(measurement["shape_index"]).astype(np.int64)
    region = (measurement["s0"] > S0_MIN) & (np.linalg.norm(measurement["normal"], axis=-1) > 0.5)
    region = region & (index == record["shapes"]["film-shell"]["index"])
    baseline = fresnel_normals(measurement["dolp"], measurement["aolp"], measurement["view"], "specular")
    shifted = fresnel_normals(
        measurement["dolp"], measurement["aolp"], measurement["view"], "specular", azimuth_shift=np.pi
    )
    baseline_error = angular_error_deg(baseline, measurement["normal"])
    shifted_error = angular_error_deg(shifted, measurement["normal"])
    chosen = np.where((shifted_error < baseline_error)[..., None], shifted, baseline)
    specular = angular_error_summary(baseline, measurement["normal"], region)
    oracle = angular_error_summary(chosen, measurement["normal"], region)
    out_dir = Path(record["out_dir"])
    _save_normals(out_dir / "normal_fresnel.png", baseline, region)
    _save_error(
        out_dir / "error_fresnel.png",
        baseline_error,
        region,
        f"{experiment} {record['view']} fresnel specular  MAE {specular['mae_deg']:.2f} deg",
    )
    logger.info(
        "sfp experiment=%s view=%s fresnel=specular mae_deg=%.4f median_deg=%.4f oracle_mae_deg=%.4f pixels=%d",
        experiment,
        record["view"],
        specular["mae_deg"],
        specular["median_deg"],
        oracle["mae_deg"],
        specular["pixels"],
    )
    return {"fresnel_specular": specular, "fresnel_oracle": oracle}


def _fit_model(
    train_records: list[dict],
    eval_records: list[dict],
    model_name: str,
    device: str,
    crops_per_image: int,
    checkpoint_path: Path | None = None,
):
    """One U-Net. Standardization uses training film pixels only."""
    import torch

    from sensor_sim.film.unet import FilmUNet, parameter_count

    expected = {"stokes": 9, "analyzer": 7, "rgb": 6}
    train_raw = [_channels(record["measurement"], model_name) for record in train_records]
    eval_raw = [_channels(record["measurement"], model_name) for record in eval_records]
    train_masks = [_loss_mask(record["measurement"], set(record["object_indices"])) for record in train_records]
    samples = np.concatenate([image[mask] for image, mask in zip(train_raw, train_masks)], axis=0)
    if samples.shape[0] < 100:
        raise RuntimeError(f"{model_name} has only {samples.shape[0]} training film pixels")
    mean = samples.mean(axis=0)
    scale = samples.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)

    def apply(image: np.ndarray) -> np.ndarray:
        return (image - mean) / scale

    train_std = [apply(image) for image in train_raw]
    eval_std = [apply(image) for image in eval_raw]
    in_channels = int(train_std[0].shape[-1])
    if in_channels != expected[model_name]:
        raise RuntimeError(f"{model_name} expected {expected[model_name]} channels, got {in_channels}")
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = FilmUNet(in_channels, out_channels=3).to(device)
    param_device = next(model.parameters()).device
    logger.info(
        "sfp stage=train model=%s param_device=%s requested=%s",
        model_name,
        param_device,
        device,
    )
    if param_device.type != device:
        raise RuntimeError(f"{model_name} parameters are on {param_device}, requested {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.0)
    sources = [torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32) for image in train_std]
    truths = [torch.tensor(record["measurement"]["normal"], dtype=torch.float32) for record in train_records]
    generator = torch.Generator()
    generator.manual_seed(SEED)
    model.train()
    last_loss = float("nan")
    for epoch in range(EPOCHS):
        crops = []
        for index, mask in enumerate(train_masks):
            for _copy in range(int(crops_per_image)):
                top, left = _crop_containing(mask, generator, CROP)
                crops.append((index, top, left))
        order = torch.randperm(len(crops), generator=generator).tolist()
        losses = []
        film_counts = []
        for start in range(0, len(order), BATCH):
            chosen = [crops[item] for item in order[start : start + BATCH]]
            if len(chosen) < 2:
                logger.info(
                    "sfp stage=train model=%s epoch=%d skip_crops=%d reason=batch_norm_needs_two",
                    model_name,
                    epoch,
                    len(chosen),
                )
                continue
            batch_in = []
            batch_truth = []
            batch_mask = []
            for index, top, left in chosen:
                crop_mask = train_masks[index][top : top + CROP, left : left + CROP]
                batch_in.append(sources[index][:, top : top + CROP, left : left + CROP])
                batch_truth.append(truths[index][top : top + CROP, left : left + CROP])
                batch_mask.append(torch.tensor(crop_mask))
                film_counts.append(int(crop_mask.sum()))
            prediction = model(torch.stack(batch_in).to(device))
            prediction = prediction / prediction.norm(dim=1, keepdim=True).clamp_min(1e-8)
            target = torch.stack(batch_truth).permute(0, 3, 1, 2).to(device)
            chosen_mask = torch.stack(batch_mask).to(device)
            if int(chosen_mask.sum()) == 0:
                raise RuntimeError(f"{model_name} epoch {epoch} produced a crop with no film pixels")
            cosine = (prediction * target).sum(dim=1)
            loss = (1.0 - cosine)[chosen_mask].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else float("nan")
        logger.info(
            "sfp stage=train model=%s epoch=%d loss=%.6f device=%s batches=%d crops=%d final_batch=%d mean_film_pixels_per_crop=%.1f",
            model_name,
            epoch,
            last_loss,
            device,
            len(losses),
            len(crops),
            len(crops) % BATCH or BATCH,
            float(np.mean(film_counts)) if film_counts else float("nan"),
        )
    if not np.isfinite(last_loss):
        raise RuntimeError(f"{model_name} produced no training loss")
    model.eval()
    predictions = []
    with torch.no_grad():
        for image in eval_std:
            tensor = torch.tensor(np.transpose(image, (2, 0, 1))[None], dtype=torch.float32, device=device)
            prediction = model(tensor)[0]
            prediction = prediction / prediction.norm(dim=0, keepdim=True).clamp_min(1e-8)
            predictions.append(np.transpose(prediction.detach().cpu().numpy(), (1, 2, 0)))
    info = {
        "parameters": parameter_count(model),
        "in_channels": in_channels,
        "standardization": {"mean": mean.tolist(), "std": scale.tolist()},
        "final_epoch_loss": last_loss,
        "epochs": EPOCHS,
        "crops_per_image": int(crops_per_image),
        "batch": BATCH,
        "crop": CROP,
        "optimizer": "adam",
        "lr": LEARNING_RATE,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
        "seed": SEED,
        "device": device,
        "param_device": str(param_device),
    }
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mean": mean,
                "std": scale,
                "in_channels": in_channels,
                "model_name": model_name,
                "info": info,
            },
            path,
        )
        logger.info("sfp stage=checkpoint path=%s", path)
    del model, sources, truths
    if device == "cuda":
        torch.cuda.empty_cache()
    return predictions, info


def train_and_score(
    experiment: str,
    train_records: list[dict],
    eval_records: list[dict],
    device: str,
    crops_per_image: int,
    crop_note: str,
) -> tuple[dict, list[dict]]:
    logger.info(
        "sfp experiment=%s stage=train device=%s crops_per_image=%d train_frames=%d eval_frames=%d note=%s",
        experiment,
        device,
        crops_per_image,
        len(train_records),
        len(eval_records),
        crop_note,
    )
    per_model = {}
    for model_name in MODELS:
        predictions, info = _fit_model(train_records, eval_records, model_name, device, crops_per_image)
        scored = []
        for record, predicted in zip(eval_records, predictions, strict=True):
            mask = _loss_mask(record["measurement"], set(record["object_indices"]))
            summary = angular_error_summary(predicted, record["measurement"]["normal"], mask)
            _write_prediction_images(
                experiment,
                record,
                model_name,
                predicted,
                {"all_objects_mae_deg": summary["mae_deg"]},
            )
            if model_name == "stokes":
                record["stokes_degrees"] = angular_error_deg(predicted, record["measurement"]["normal"])
                record["stokes_mask"] = mask
            scored.append(summary)
            logger.info(
                "sfp experiment=%s view=%s model=%s mae_deg=%.4f median_deg=%.4f "
                "frac_below_11_25=%.4f frac_below_22_5=%.4f frac_below_30=%.4f pixels=%d",
                experiment,
                record["view"],
                model_name,
                summary["mae_deg"],
                summary["median_deg"],
                summary["fraction_below_11_25"],
                summary["fraction_below_22_5"],
                summary["fraction_below_30"],
                summary["pixels"],
            )
        per_model[model_name] = {"info": info, "views": scored}
    fresnel = [score_film_fresnel(experiment, record) for record in eval_records]
    return per_model, fresnel


def _contact_sheet(path: Path, records: list[dict]) -> None:
    """Ten test RGBs beside the Stokes error. Blank pixels are outside the film score."""
    figure, axes = plt.subplots(len(records), 2, figsize=(8, 2.1 * len(records)), squeeze=False)
    last = None
    for row, record in enumerate(records):
        rgb = np.clip(display_srgb(record["measurement"]["rgb"]), 0.0, 1.0)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{record['view']} {record['family']}")
        axes[row, 0].axis("off")
        shown = np.where(record["stokes_mask"], record["stokes_degrees"], np.nan)
        last = axes[row, 1].imshow(shown, vmin=0.0, vmax=90.0, cmap="magma")
        mae = float(np.nanmean(shown))
        axes[row, 1].set_title(f"Stokes MAE {mae:.2f} deg")
        axes[row, 1].axis("off")
    figure.colorbar(last, ax=axes[:, 1], fraction=0.03, label="angular error (deg)")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    logger.info("sfp experiment=%s stage=contact path=%s rows=%d", EXPERIMENT_02, path, len(records))


def _view_metrics(record: dict, per_model: dict, fresnel: dict, index: int) -> dict:
    return {
        "view": record["view"],
        "film_pixels": record["film_pixels"],
        "film_pixel_fraction": record["film_pixel_fraction"],
        "mean_abs_s3_over_s0": record["mean_abs_s3_over_s0"],
        "stokes": per_model["stokes"]["views"][index],
        "analyzer": per_model["analyzer"]["views"][index],
        "rgb": per_model["rgb"]["views"][index],
        "fresnel_specular": fresnel["fresnel_specular"],
        "fresnel_oracle": fresnel["fresnel_oracle"],
    }


def run_cornell() -> dict:
    root = OUTPUT / EXPERIMENT_01
    mesh_dir = root / "meshes"
    logger.info("sfp experiment=%s stage=meshes dir=%s", EXPERIMENT_01, mesh_dir)
    poses = write_cornell_film(mesh_dir)
    scene = scene_config()
    kwargs = {
        "resolution": RESOLUTION,
        "max_depth": MAX_DEPTH,
        "reflectance": float(scene["object"]["reflectance"]),
        "room_reflectance": float(scene["room"]["reflectance"]),
        "ior": float(scene["film"]["ior"]),
        "roughness": float(scene["film"]["roughness"]),
    }
    from sensor_sim.film.sfp_scene import cornell_stress_dict

    records = []
    for view_name, position in (("view_a", None), ("view_b", list(CORNELL_CAMERA_B))):
        view_poses = json.loads(json.dumps(poses))
        if position is not None:
            view_poses["camera"]["position"] = position

        def build(*, polarized: bool, spp: int, rfilter: str, poses=view_poses) -> dict:
            return cornell_stress_dict(
                poses, mesh_dir, polarized=polarized, spp=spp, rfilter=rfilter, **kwargs
            )

        records.append(
            render_film_view(
                experiment=EXPERIMENT_01,
                view_name=view_name,
                out_dir=root / view_name,
                build=build,
                probe_point=list(poses["points"]["film"]),
                require_fraction=False,
            )
        )
    device, _reason, info = select_device()
    per_model, fresnel = train_and_score(
        EXPERIMENT_01,
        records[:1],
        records,
        device,
        crops_per_image=4,
        crop_note="four_crops_of_view_a_so_batch_norm_batch_is_4",
    )
    summary = {
        "experiment": EXPERIMENT_01,
        "lamp": "unpolarized ceiling rectangle",
        "lamp_radiance": float(poses["light"]["intensity"]),
        "lamp_note": (
            "The Cornell rectangle is the lamp in configs/film/scene.yaml. "
            "Experiment 02 uses the gray-room rectangle at radiance "
            f"{STANDARD_LIGHT_RADIANCE}. Neither lamp is polarized."
        ),
        "variant": VARIANT,
        "resolution": list(RESOLUTION),
        "spp": SPP,
        "max_depth": MAX_DEPTH,
        "device": info,
        "crop_note": (
            "View A is the only training frame. Each epoch draws 4 crops of that frame "
            "so the batch stays 4. One crop would make BatchNorm see a batch of 1."
        ),
        "loss_pixels": "outer film only",
        "definitions": score_definitions(),
        "views": {
            record["view"]: _view_metrics(record, per_model, fresnel[index], index)
            for index, record in enumerate(records)
        },
        "models": {name: per_model[name]["info"] for name in MODELS},
    }
    _write_json(root / "summary.json", summary)
    logger.info("sfp experiment=%s stage=done summary=%s device=%s", EXPERIMENT_01, root / "summary.json", device)
    return summary


def _shape_jobs() -> list[dict]:
    """Write the gray room and every filmed mesh. No Stokes render yet."""
    catalog = shape_catalog(SEED)
    root = OUTPUT / EXPERIMENT_02
    room_dir = root / "room"
    logger.info("sfp experiment=%s stage=room dir=%s", EXPERIMENT_02, room_dir)
    write_gray_room(room_dir)
    jobs = []
    for spec in catalog:
        out_dir = root / spec["split"] / spec["folder"]
        built = build_shape(spec)
        mesh_dir = out_dir / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        write_obj(mesh_dir / "solid.obj", built["solid"]["vertices"], built["solid"]["faces"], smooth=True)
        write_obj(mesh_dir / "film.obj", built["film"]["vertices"], built["film"]["faces"], smooth=True)
        camera = built["camera"]
        logger.info(
            "sfp experiment=%s stage=mesh split=%s folder=%s family=%s seed=%d radius_m=%.4f "
            "sagitta_m=%.3e hull_fraction=%.4f",
            EXPERIMENT_02,
            spec["split"],
            spec["folder"],
            spec["family"],
            int(spec["instance_seed"]),
            float(spec["radius"]),
            float(built["sagitta_m"]),
            float(built["frame_fraction"]),
        )

        def build(*, polarized: bool, spp: int, rfilter: str, camera=camera, mesh_dir=mesh_dir) -> dict:
            return filmed_object_dict(
                room_dir,
                mesh_dir,
                camera,
                resolution=RESOLUTION,
                spp=spp,
                max_depth=MAX_DEPTH,
                polarized=polarized,
                rfilter=rfilter,
            )

        jobs.append(
            {
                "spec": spec,
                "built": built,
                "out_dir": out_dir,
                "build": build,
                "probe": [float(value) for value in built["probe"]],
                "folder": spec["folder"],
                "split": spec["split"],
                "family": spec["family"],
            }
        )
    return jobs


def preflight_shape_coverage(jobs: list[dict] | None = None) -> list[dict]:
    """1-sample film-pixel fraction for every shape, before any 64-sample Stokes frame."""
    jobs = _shape_jobs() if jobs is None else jobs
    for job in jobs:
        aov = _film_aov(job["build"], job["probe"])
        fraction = float(aov["fraction"])
        logger.info(
            "sfp stage=coverage folder=%s split=%s family=%s film_pixels=%d film_fraction=%.4f",
            job["folder"],
            job["split"],
            job["family"],
            int(aov["film_pixels"]),
            fraction,
        )
        if aov["film_index"] == 0:
            raise RuntimeError(f"{job['folder']} film probe missed every surface")
        if fraction < MIN_FILM_FRACTION:
            raise RuntimeError(
                f"{job['folder']} film covers {fraction:.4f} of the frame, below {MIN_FILM_FRACTION}"
            )
        job["coverage_fraction"] = fraction
    return jobs


def run_shapes(jobs: list[dict] | None = None) -> dict:
    root = OUTPUT / EXPERIMENT_02
    if jobs is None:
        jobs = preflight_shape_coverage()
    records = []
    for job in jobs:
        spec = job["spec"]
        built = job["built"]
        out_dir = job["out_dir"]
        record = render_film_view(
            experiment=EXPERIMENT_02,
            view_name=spec["folder"],
            out_dir=out_dir,
            build=job["build"],
            probe_point=job["probe"],
            require_fraction=True,
        )
        record["split"] = spec["split"]
        record["family"] = spec["family"]
        record["instance_seed"] = int(spec["instance_seed"])
        record["radius_m"] = float(spec["radius"])
        record["sagitta_m"] = float(built["sagitta_m"])
        record["hull_fraction"] = float(built["frame_fraction"])
        record["yaw_rad"] = float(spec["yaw"])
        record["floor_xy"] = [float(spec["x"]), float(spec["y"])]
        _write_json(out_dir / "metadata.json", {key: value for key, value in record.items() if key != "measurement"})
        records.append(record)
    train = [item for item in records if item["split"] == "train"]
    test = [item for item in records if item["split"] == "test"]
    if len(train) != 50 or len(test) != 10:
        raise RuntimeError(f"expected 50 train and 10 test renders, got {len(train)} and {len(test)}")
    if any(item["family"] == "sphere" for item in test):
        raise RuntimeError("the test split contains a sphere")
    if {item["instance_seed"] for item in train} & {item["instance_seed"] for item in test}:
        raise RuntimeError("a test seed also appears in train")
    device, _reason, info = select_device()
    per_model, fresnel = train_and_score(
        EXPERIMENT_02,
        train,
        records,
        device,
        crops_per_image=1,
        crop_note="one_crop_per_training_shape_batched_by_4_final_batch_2",
    )
    _contact_sheet(root / "test_contact.png", test)
    test_rows = []
    for record in test:
        index = records.index(record)
        row = _view_metrics(record, per_model, fresnel[index], index)
        row["family"] = record["family"]
        row["instance_seed"] = record["instance_seed"]
        row["sagitta_m"] = record["sagitta_m"]
        row["hull_fraction"] = record["hull_fraction"]
        test_rows.append(row)
    summary = {
        "experiment": EXPERIMENT_02,
        "lamp": "unpolarized ceiling rectangle",
        "lamp_radiance": STANDARD_LIGHT_RADIANCE,
        "lamp_tracks_object": False,
        "variant": VARIANT,
        "resolution": list(RESOLUTION),
        "spp": SPP,
        "max_depth": MAX_DEPTH,
        "device": info,
        "train_shapes": 50,
        "test_shapes": 10,
        "families_train": TRAIN_COUNTS,
        "families_test": TEST_COUNTS,
        "crop_note": (
            "Each epoch draws one 256 crop from every training shape and groups them "
            "in batches of 4. The last batch has the remaining 2 shapes. Experiment 01 "
            "draws 4 crops of its only frame instead, because a batch of 1 breaks BatchNorm."
        ),
        "loss_pixels": "outer film only",
        "sfpuel": SFPUEL_NOTE,
        "definitions": score_definitions(),
        "test": test_rows,
        "test_mean": {
            "stokes": _unweighted_mean([row["stokes"] for row in test_rows]),
            "analyzer": _unweighted_mean([row["analyzer"] for row in test_rows]),
            "rgb": _unweighted_mean([row["rgb"] for row in test_rows]),
            "fresnel_specular": _unweighted_mean([row["fresnel_specular"] for row in test_rows]),
            "fresnel_oracle": _unweighted_mean([row["fresnel_oracle"] for row in test_rows]),
        },
        "models": {name: per_model[name]["info"] for name in MODELS},
    }
    _write_json(root / "summary.json", summary)
    logger.info("sfp experiment=%s stage=done summary=%s device=%s", EXPERIMENT_02, root / "summary.json", device)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run experiment 01, experiment 02, or both. Returns None so a script exit stays 0."""
    configure_logging()
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1 or (args and args[0] not in {"01", "02", "both"}):
        raise SystemExit("usage: film-sfp [01|02|both]")
    choice = args[0] if args else "both"
    _require_cuda_variant()
    logger.info(
        "sfp stage=start choice=%s variant=%s resolution=%s spp=%d lamp=unpolarized_ceiling_rectangle",
        choice,
        VARIANT,
        list(RESOLUTION),
        SPP,
    )
    jobs = preflight_shape_coverage() if choice in {"02", "both"} else None
    if choice in {"01", "both"}:
        run_cornell()
    if choice in {"02", "both"}:
        run_shapes(jobs)
    logger.info("sfp stage=done choice=%s", choice)


if __name__ == "__main__":
    main()

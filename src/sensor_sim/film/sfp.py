"""Render polarized views and fit camera-frame normals on the outer film.

``uv run film-sfp`` runs experiment 01 (Cornell, film loss only) and
experiment 02 (50 training shapes and 10 held-out shapes). Renders use
``cuda_ad_spectral_polarized``. Training uses the GPU when Torch can see
one with at least 8 GiB free, and the CPU otherwise. The earlier gray-ball
run is ``run_previous_sfp`` and is not what the entry point calls.

Mean angular error is the mean, over pixels in one region with S0 > 1e-4,
of ``arccos(clip(dot(prediction, truth), -1, 1))`` in degrees.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

from sensor_sim.film import ROOT
from sensor_sim.film.capture import _render, _stokes_from_image, project_point
from sensor_sim.film.config import rendering_config, scene_config
from sensor_sim.film.geometry import DRAGON_CENTER_XY, _basis, _direction, theta_max
from sensor_sim.film.lighting import display_srgb
from sensor_sim.film.sfp_math import (
    angular_error_deg,
    aolp_pair,
    camera_matrix,
    fresnel_normals,
    mean_angular_error_deg,
    object_pixel_mask,
    view_directions,
    world_normals_to_camera,
)
from sensor_sim.film.sfp_scene import (
    CORNELL_CAMERA_B,
    DRAGON_DOLP_GATE,
    STANDARD_CAMERA_A,
    STANDARD_CAMERA_B,
    ball_center,
    copper_bsdf,
    cornell_stress_dict,
    pplastic_bsdf,
    standard_probe_points,
    standard_scene_dict,
    write_gray_room,
)
from sensor_sim.render import configure_logging
from sensor_sim.stokes import S0_MIN

logger = logging.getLogger("sensor_sim.sfp")

VARIANT = "cuda_ad_spectral_polarized"
OUTPUT = ROOT / "outputs" / "film" / "sfp"
RESOLUTION = (512, 512)
SPP = 64
SEED = 0
MAX_DEPTH = 8
EPOCHS = 20
STEPS_PER_EPOCH = 4
BATCH = 4
CROP = 256
LEARNING_RATE = 1e-3
MODELS = ("stokes", "analyzer", "rgb")


def _split_aov(loaded, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = list(loaded.integrator().aov_names())
    packed = np.array(array)
    if packed.ndim == 2:
        packed = packed[:, :, None]
    offset = 0
    if packed.shape[-1] == len(names) + 3:
        offset = 3
    elif packed.shape[-1] != len(names):
        raise RuntimeError(f"AOV image has {packed.shape[-1]} channels for names {names}")
    lookup = {name: packed[:, :, offset + index] for index, name in enumerate(names)}
    missing = [name for name in ("nn.X", "nn.Y", "nn.Z", "idx.I") if name not in lookup]
    if missing:
        raise RuntimeError(f"AOV channels {missing} missing from {names}")
    normal = np.stack([lookup["nn.X"], lookup["nn.Y"], lookup["nn.Z"]], axis=-1)
    return normal.astype(np.float64), np.asarray(lookup["idx.I"], dtype=np.float64)


def _index_at(sensor, index_image: np.ndarray, point: list[float]) -> tuple[int, int, int]:
    height, width = index_image.shape
    row, column, _ndc = project_point(sensor, point, width, height)
    if not (0 <= row < height and 0 <= column < width):
        raise RuntimeError(f"point {point} projects outside the image at row={row} col={column}")
    return int(round(float(index_image[row, column]))), row, column


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    plt.imsave(path, np.clip(display_srgb(rgb), 0.0, 1.0))


def _save_normals(path: Path, normals: np.ndarray, mask: np.ndarray) -> None:
    image = np.clip(normals * 0.5 + 0.5, 0.0, 1.0)
    image[~mask] = 0.0
    plt.imsave(path, image)


def _save_error(path: Path, degrees: np.ndarray, mask: np.ndarray, title: str) -> None:
    shown = np.where(mask, degrees, np.nan)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(shown, vmin=0.0, vmax=90.0, cmap="magma")
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046, label="angular error (deg)")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def render_standard(view_name: str, camera: dict) -> dict:
    mesh_dir = OUTPUT / "standard" / "meshes"
    write_gray_room(mesh_dir)
    out_dir = OUTPUT / "standard" / view_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def build(*, polarized: bool, spp: int, rfilter: str) -> dict:
        return standard_scene_dict(
            mesh_dir,
            camera,
            resolution=RESOLUTION,
            spp=spp,
            max_depth=MAX_DEPTH,
            polarized=polarized,
            rfilter=rfilter,
        )

    return _render_view("standard", view_name, out_dir, build, standard_probe_points(), ("ball",))


def render_cornell(view_name: str, camera_position: list[float] | None) -> dict:
    poses = json.loads((ROOT / "assets" / "mitsuba" / "film_ball" / "poses.json").read_text())
    if camera_position is not None:
        poses = copy.deepcopy(poses)
        poses["camera"] = dict(poses["camera"])
        poses["camera"]["position"] = list(camera_position)
    scene = scene_config()
    rendering = rendering_config()
    mesh_dir = ROOT / "assets" / "mitsuba" / "film_ball"
    out_dir = OUTPUT / "cornell" / view_name
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "resolution": RESOLUTION,
        "max_depth": int(rendering["max_depth"]),
        "reflectance": float(scene["object"]["reflectance"]),
        "room_reflectance": float(scene["room"]["reflectance"]),
        "ior": float(scene["film"]["ior"]),
        "roughness": float(scene["film"]["roughness"]),
    }

    def build(*, polarized: bool, spp: int, rfilter: str) -> dict:
        return cornell_stress_dict(
            poses,
            mesh_dir,
            polarized=polarized,
            spp=spp,
            rfilter=rfilter,
            **kwargs,
        )

    dragon = [float(DRAGON_CENTER_XY[0]), float(DRAGON_CENTER_XY[1]), 0.06]
    # The cap reaches about 84°. A point 0.12 rad past that rim is the visible
    # diffuse ball. The stored uncovered point still falls inside the film's pixels.
    center = np.array(poses["ball_center"], dtype=np.float64)
    axis = np.array(scene["film"]["axis"], dtype=np.float64)
    direction, tangent, bitangent = _basis(axis)
    offset = _direction(direction, tangent, bitangent, theta_max(float(scene["film"]["coverage"])) + 0.12, 0.5)
    radius = float(scene["object"]["radius"]) - float(scene["film"]["thickness"])
    ball_point = (center + offset * radius).tolist()
    points = {
        "film-shell": list(poses["points"]["film"]),
        "ball": ball_point,
        "dragon": dragon,
    }
    return _render_view("cornell", view_name, out_dir, build, points, ("film-shell", "ball", "dragon"))


def _render_view(scene_name: str, view_name: str, out_dir: Path, build, points: dict, objects: tuple[str, ...]) -> dict:
    mi.set_variant(VARIANT)
    logger.info("sfp scene=%s view=%s stage=stokes variant=%s spp=%d", scene_name, view_name, VARIANT, SPP)
    stokes_scene = build(polarized=True, spp=SPP, rfilter="gaussian")
    loaded, image, _array, elapsed = _render(stokes_scene, SPP, SEED)
    bundle = _stokes_from_image(loaded, image)
    sensor = loaded.sensors()[0]
    height, width = bundle["s0"].shape
    view = view_directions(sensor, width, height)

    logger.info("sfp scene=%s view=%s stage=normals", scene_name, view_name)
    aov_scene = build(polarized=False, spp=1, rfilter="box")
    aov_scene["integrator"] = {"type": "aov", "aovs": "nn:sh_normal,idx:shape_index"}
    aov_loaded, _aov_image, aov_array, _aov_elapsed = _render(aov_scene, spp=1, seed=SEED)
    world_normals, shape_index = _split_aov(aov_loaded, aov_array)
    normals = world_normals_to_camera(world_normals, camera_matrix(aov_loaded.sensors()[0]))

    aov_sensor = aov_loaded.sensors()[0]
    names = {}
    for name, point in points.items():
        index, row, column = _index_at(aov_sensor, shape_index, point)
        names[name] = {"index": index, "pixel": [row, column], "point": list(point)}
        logger.info(
            "sfp scene=%s view=%s shape=%s index=%d pixel=%d,%d",
            scene_name,
            view_name,
            name,
            index,
            row,
            column,
        )
    object_indices = {names[name]["index"] for name in objects}
    s0 = np.asarray(bundle["s0"], dtype=np.float64)
    cosine, sine = aolp_pair(bundle["aolp"], bundle["dolp"])
    rgb = np.asarray(bundle["rgb"], dtype=np.float64)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    measurement = {
        "rgb": rgb,
        "s0": s0,
        "s1": np.asarray(bundle["s1"], dtype=np.float64),
        "s2": np.asarray(bundle["s2"], dtype=np.float64),
        "dolp": np.asarray(bundle["dolp"], dtype=np.float64),
        "aolp": np.asarray(bundle["aolp"], dtype=np.float64),
        "aolp_cos": cosine,
        "aolp_sin": sine,
        "i0": np.asarray(bundle["analyzer_000"], dtype=np.float64),
        "i45": np.asarray(bundle["analyzer_045"], dtype=np.float64),
        "i90": np.asarray(bundle["analyzer_090"], dtype=np.float64),
        "i135": np.asarray(bundle["analyzer_135"], dtype=np.float64),
        "normal": normals,
        "shape_index": shape_index,
        "view": view,
    }
    np.savez_compressed(out_dir / "measurement.npz", **measurement)
    _save_rgb(out_dir / "rgb.png", rgb)
    valid_normal = np.linalg.norm(normals, axis=-1) > 0.5
    _save_normals(out_dir / "normal_gt.png", normals, valid_normal)
    if "ball" not in names:
        raise RuntimeError(f"{scene_name} has no ball probe point")
    for name, record in names.items():
        if record["index"] == 0:
            raise RuntimeError(f"{scene_name} {view_name} {name} probe missed every surface")
    ball_pixels = int((np.rint(shape_index).astype(np.int64) == names["ball"]["index"]).sum())
    fraction = ball_pixels / float(shape_index.size)
    region_dolp = {}
    for name, record in names.items():
        region = np.rint(shape_index).astype(np.int64) == record["index"]
        usable = region & (s0 > S0_MIN)
        region_dolp[name] = float(measurement["dolp"][usable].max()) if int(usable.sum()) else 0.0
        logger.info(
            "sfp scene=%s view=%s shape=%s pixels=%d max_dolp=%.6f mean_s0=%.6g",
            scene_name,
            view_name,
            name,
            int(region.sum()),
            region_dolp[name],
            float(s0[region].mean()) if int(region.sum()) else 0.0,
        )
    meta = {
        "scene": scene_name,
        "view": view_name,
        "variant": VARIANT,
        "resolution": list(RESOLUTION),
        "spp": SPP,
        "seed": SEED,
        "elapsed_stokes_s": elapsed,
        "shapes": names,
        "object_indices": sorted(object_indices),
        "ball_pixel_fraction": fraction,
        "max_dolp": region_dolp,
        "s0_min": S0_MIN,
        "pplastic": pplastic_bsdf(),
        "copper": copper_bsdf(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info(
        "sfp scene=%s view=%s status=ok ball_fraction=%.4f elapsed_stokes_s=%.3f",
        scene_name,
        view_name,
        fraction,
        elapsed,
    )
    meta["measurement"] = measurement
    return meta


def _channels(measurement: dict, model_name: str) -> np.ndarray:
    view = measurement["view"]
    if model_name == "stokes":
        stacked = [
            measurement["s0"],
            measurement["s1"],
            measurement["s2"],
            measurement["dolp"],
            measurement["aolp_cos"],
            measurement["aolp_sin"],
        ]
    elif model_name == "analyzer":
        stacked = [measurement["i0"], measurement["i45"], measurement["i90"], measurement["i135"]]
    elif model_name == "rgb":
        return np.concatenate([measurement["rgb"], view], axis=-1)
    else:
        raise ValueError(f"unknown model {model_name}")
    return np.concatenate([np.stack(stacked, axis=-1), view], axis=-1)


def _standardize(train: np.ndarray, other: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-channel mean and NumPy std (divide by N) on the training object pixels."""
    samples = train[mask]
    mean = samples.mean(axis=0)
    scale = samples.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (train - mean) / scale, (other - mean) / scale, {"mean": mean.tolist(), "std": scale.tolist()}


def _loss_mask(measurement: dict, object_indices: set[int]) -> np.ndarray:
    finite = np.linalg.norm(measurement["normal"], axis=-1) > 0.5
    return object_pixel_mask(measurement["shape_index"], object_indices, measurement["s0"]) & finite


def train_models(scene_name: str, train_view: dict, test_view: dict) -> dict:
    """Fit three U-Nets on view A. Importing the network hides the GPU from Torch."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    from sensor_sim.film.unet import FilmUNet, parameter_count

    torch.manual_seed(SEED)
    object_indices = set(train_view["object_indices"])
    train_mask = _loss_mask(train_view["measurement"], object_indices)
    test_mask = _loss_mask(test_view["measurement"], object_indices)
    if int(train_mask.sum()) < 100:
        raise RuntimeError(f"{scene_name} view A has only {int(train_mask.sum())} training pixels")
    results = {}
    for model_name in MODELS:
        torch.manual_seed(SEED)
        raw_train = _channels(train_view["measurement"], model_name)
        raw_test = _channels(test_view["measurement"], model_name)
        train_input, test_input, stats = _standardize(raw_train, raw_test, train_mask)
        model = FilmUNet(train_input.shape[-1], out_channels=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.0)
        truth = torch.tensor(train_view["measurement"]["normal"], dtype=torch.float32)
        source = torch.tensor(np.transpose(train_input, (2, 0, 1)), dtype=torch.float32)
        height, width = train_mask.shape
        generator = torch.Generator()
        generator.manual_seed(SEED)
        model.train()
        last_loss = None
        for epoch in range(EPOCHS):
            losses = []
            for _step in range(STEPS_PER_EPOCH):
                batch_in = []
                batch_truth = []
                batch_mask = []
                for _crop in range(BATCH):
                    top = int(torch.randint(0, height - CROP + 1, (1,), generator=generator))
                    left = int(torch.randint(0, width - CROP + 1, (1,), generator=generator))
                    crop_mask = train_mask[top : top + CROP, left : left + CROP]
                    if int(crop_mask.sum()) < 16:
                        continue
                    batch_in.append(source[:, top : top + CROP, left : left + CROP])
                    batch_truth.append(truth[top : top + CROP, left : left + CROP])
                    batch_mask.append(torch.tensor(crop_mask))
                if not batch_in:
                    continue
                prediction = model(torch.stack(batch_in))
                prediction = prediction / prediction.norm(dim=1, keepdim=True).clamp_min(1e-8)
                target = torch.stack(batch_truth).permute(0, 3, 1, 2)
                chosen = torch.stack(batch_mask)
                cosine = (prediction * target).sum(dim=1)
                loss = (1.0 - cosine)[chosen].mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            last_loss = float(np.mean(losses)) if losses else float("nan")
            logger.info(
                "sfp train scene=%s model=%s epoch=%d loss=%.6f pixels=%d",
                scene_name,
                model_name,
                epoch,
                last_loss,
                int(train_mask.sum()),
            )
        scores = {}
        model.eval()
        with torch.no_grad():
            for view in (train_view, test_view):
                raw = train_input if view["view"] == train_view["view"] else test_input
                tensor = torch.tensor(np.transpose(raw, (2, 0, 1))[None], dtype=torch.float32)
                prediction = model(tensor)[0]
                prediction = prediction / prediction.norm(dim=0, keepdim=True).clamp_min(1e-8)
                predicted = np.transpose(prediction.numpy(), (1, 2, 0))
                # Each view has its own shape-index numbers. View A's indices do not name view B's surfaces.
                mask = _loss_mask(view["measurement"], set(view["object_indices"]))
                region_scores = _region_scores(view, predicted, mask)
                scores[view["view"]] = region_scores
                _write_prediction_images(scene_name, view, model_name, predicted, region_scores)
        results[model_name] = {
            "parameters": parameter_count(model),
            "in_channels": int(train_input.shape[-1]),
            "standardization": stats,
            "final_crop_loss": last_loss,
            "views": scores,
            "optimizer": "adam",
            "lr": LEARNING_RATE,
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "batch": BATCH,
            "crop": CROP,
            "seed": SEED,
        }
        del model
    return results


def _region_scores(view: dict, predicted: np.ndarray, loss_mask: np.ndarray) -> dict:
    truth = view["measurement"]["normal"]
    index = np.rint(view["measurement"]["shape_index"]).astype(np.int64)
    s0 = view["measurement"]["s0"]
    usable = (s0 > S0_MIN) & (np.linalg.norm(truth, axis=-1) > 0.5)
    scores = {"all_objects_mae_deg": mean_angular_error_deg(predicted, truth, loss_mask)}
    named = {name for name in view["shapes"]}
    for name in sorted(named):
        region = usable & (index == view["shapes"][name]["index"])
        scores[name] = {
            "mae_deg": mean_angular_error_deg(predicted, truth, region),
            "pixels": int(region.sum()),
        }
    other = usable.copy()
    for name in named:
        other &= index != view["shapes"][name]["index"]
    scores["other"] = {"mae_deg": mean_angular_error_deg(predicted, truth, other), "pixels": int(other.sum())}
    return scores


def _view_dir(scene_name: str, view: dict) -> Path:
    recorded = view.get("out_dir")
    if recorded is not None:
        return Path(recorded)
    return OUTPUT / scene_name / view["view"]


def _write_prediction_images(scene_name: str, view: dict, model_name: str, predicted: np.ndarray, scores: dict) -> None:
    out_dir = _view_dir(scene_name, view)
    if view.get("film_only"):
        mask = _loss_mask(view["measurement"], set(view["object_indices"]))
    else:
        mask = np.linalg.norm(predicted, axis=-1) > 0.5
    _save_normals(out_dir / f"normal_{model_name}.png", predicted, mask)
    degrees = angular_error_deg(predicted, view["measurement"]["normal"])
    title = (
        f"{scene_name} {view['view']} {model_name}  "
        f"object MAE {scores['all_objects_mae_deg']:.2f} deg"
    )
    loss_mask = _loss_mask(view["measurement"], set(view["object_indices"]))
    _save_error(out_dir / f"error_{model_name}.png", degrees, loss_mask, title)


def score_fresnel(scene_name: str, view: dict) -> dict:
    """Fixed branch, no ground-truth disambiguation. The oracle tries azimuth + π."""
    measurement = view["measurement"]
    if scene_name == "standard":
        regions = {"ball": "diffuse"}
    else:
        regions = {"film-shell": "specular"}
    scores = {}
    index = np.rint(measurement["shape_index"]).astype(np.int64)
    usable = (measurement["s0"] > S0_MIN) & (np.linalg.norm(measurement["normal"], axis=-1) > 0.5)
    combined = np.zeros(measurement["s0"].shape, dtype=bool)
    for name, branch in regions.items():
        region = usable & (index == view["shapes"][name]["index"])
        combined |= region
        baseline = fresnel_normals(measurement["dolp"], measurement["aolp"], measurement["view"], branch)
        oracle = fresnel_normals(
            measurement["dolp"],
            measurement["aolp"],
            measurement["view"],
            branch,
            azimuth_shift=np.pi,
        )
        baseline_error = angular_error_deg(baseline, measurement["normal"])
        oracle_error = angular_error_deg(oracle, measurement["normal"])
        use_oracle = oracle_error < baseline_error
        chosen = np.where(use_oracle[..., None], oracle, baseline)
        scores[name] = {
            "branch": branch,
            "mae_deg": mean_angular_error_deg(baseline, measurement["normal"], region),
            "oracle_mae_deg": mean_angular_error_deg(chosen, measurement["normal"], region),
            "pixels": int(region.sum()),
            "zenith_rule": "smallest zenith whose Fresnel DoLP matches the pixel",
            "oracle": "picks azimuth or azimuth+pi, whichever is closer to the ground truth",
        }
        logger.info(
            "sfp fresnel scene=%s view=%s shape=%s branch=%s mae_deg=%.4f oracle_mae_deg=%.4f pixels=%d",
            scene_name,
            view["view"],
            name,
            branch,
            scores[name]["mae_deg"],
            scores[name]["oracle_mae_deg"],
            scores[name]["pixels"],
        )
    if combined.any():
        # Draw the fixed-branch normal on the regions this formula applies to.
        painted = np.zeros_like(measurement["normal"])
        for name, branch in regions.items():
            region = usable & (index == view["shapes"][name]["index"])
            painted[region] = fresnel_normals(
                measurement["dolp"], measurement["aolp"], measurement["view"], branch
            )[region]
        out_dir = _view_dir(scene_name, view)
        _save_normals(out_dir / "normal_fresnel.png", painted, combined)
        degrees = angular_error_deg(painted, measurement["normal"])
        mean = mean_angular_error_deg(painted, measurement["normal"], combined)
        _save_error(
            out_dir / "error_fresnel.png",
            degrees,
            combined,
            f"{scene_name} {view['view']} fresnel  MAE {mean:.2f} deg",
        )
    return scores


def _drop_measurement(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "measurement"}


def main() -> None:
    """Render and train experiment 01, experiment 02, or both.

    The previous gray-ball and Cornell entry point is ``run_previous_sfp``.
    ``uv run film-sfp`` now prepares the held-out film experiments. Numeric
    tables are written by that run, under ``01_cornell_film`` and
    ``02_heldout_shapes``, and the old ``summary.json`` is left in place.
    """
    from sensor_sim.film.heldout import main as run_heldout

    run_heldout()


def run_previous_sfp() -> None:
    """The two-view gray ball and the Cornell stress scene that includes the dragon in the loss."""
    configure_logging()
    logger.info(
        "sfp variant=%s note=previous_experiment",
        VARIANT,
    )
    standard_a = render_standard("view_a", STANDARD_CAMERA_A)
    standard_b = render_standard("view_b", STANDARD_CAMERA_B)
    if standard_a["ball_pixel_fraction"] < 0.15:
        raise RuntimeError(
            f"standard ball covers {standard_a['ball_pixel_fraction']:.3f} of view A, below 0.15"
        )
    ball_dolp = standard_a["max_dolp"]["ball"]
    if ball_dolp <= DRAGON_DOLP_GATE:
        raise RuntimeError(f"pplastic ball max DoLP {ball_dolp:.6f} is at or below {DRAGON_DOLP_GATE}")
    cornell_a = render_cornell("view_a", None)
    cornell_b = render_cornell("view_b", CORNELL_CAMERA_B)
    cornell_ids = {name: cornell_a["shapes"][name]["index"] for name in ("film-shell", "ball", "dragon")}
    if len(set(cornell_ids.values())) != 3:
        raise RuntimeError(f"Cornell probes did not land on three surfaces: {cornell_ids}")
    dragon_dolp = cornell_a["max_dolp"]["dragon"]
    cornell_gate = dragon_dolp > DRAGON_DOLP_GATE
    logger.info(
        "sfp cornell dragon_max_dolp=%.6f gate=%.3f status=%s",
        dragon_dolp,
        DRAGON_DOLP_GATE,
        "pass" if cornell_gate else "dolp_below_gate",
    )
    summary = {
        "definitions": {
            "mae_deg": (
                "Mean of arccos(clip(dot(prediction, truth), -1, 1)) in degrees. "
                "The mean uses pixels in the named region with S0 > 1e-4 and a unit normal."
            ),
            "ball_pixel_fraction": "Pixels whose shape index is the ball, divided by width times height.",
            "dolp": "sqrt(S1^2 + S2^2) / S0 where S0 > 1e-4, else 0.",
            "aolp": "0.5 * arctan2(S2, S1). Stored as 0 where DoLP < 0.05.",
            "loss": "Mean of 1 - dot(prediction, truth) on object pixels with S0 > 1e-4. Walls are excluded.",
            "fresnel_diffuse": "Azimuth equals AoLP. Used on the pplastic ball.",
            "fresnel_specular": "Azimuth equals AoLP + pi/2. Used on the dielectric film. Not used on copper.",
            "oracle_mae_deg": "Same zenith, but azimuth or azimuth+pi is chosen by the ground-truth normal.",
            "variant": VARIANT,
        },
        "standard": {
            "views": {"view_a": _drop_measurement(standard_a), "view_b": _drop_measurement(standard_b)},
        },
        "cornell": {
            "dragon_max_dolp": dragon_dolp,
            "dragon_dolp_gate": DRAGON_DOLP_GATE,
            "trained": cornell_gate,
            "views": {"view_a": _drop_measurement(cornell_a), "view_b": _drop_measurement(cornell_b)},
        },
    }
    summary["standard"]["fresnel"] = {
        "view_a": score_fresnel("standard", standard_a),
        "view_b": score_fresnel("standard", standard_b),
    }
    summary["cornell"]["fresnel"] = {
        "view_a": score_fresnel("cornell", cornell_a),
        "view_b": score_fresnel("cornell", cornell_b),
    }
    summary["standard"]["models"] = train_models("standard", standard_a, standard_b)
    if cornell_gate:
        summary["cornell"]["models"] = train_models("cornell", cornell_a, cornell_b)
    else:
        summary["cornell"]["models"] = None
        logger.error("sfp cornell status=skipped reason=dragon_dolp_below_gate")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("sfp status=ok summary=%s", OUTPUT / "summary.json")


if __name__ == "__main__":
    main()

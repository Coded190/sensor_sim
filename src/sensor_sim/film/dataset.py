"""Small grouped dataset. One scene factor changes at a time."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mitsuba as mi
import numpy as np

from sensor_sim.film import ROOT
from sensor_sim.film.capture import blend_parameters, capture_sample, ensure_scene, load_measurement
from sensor_sim.film.config import dataset_config, rendering_config, scene_config
from sensor_sim.film.images import read_gray_png
from sensor_sim.render import configure_logging

logger = logging.getLogger("sensor_sim.film")

DATASET_ROOT = ROOT / "outputs" / "film" / "dataset"
ASSET_ROOT = ROOT / "assets"


def _light(spec: dict, base_intensity: float) -> dict:
    light = dict(spec)
    scale = float(light.pop("intensity_scale", 1.0))
    intensity = light.get("intensity")
    light["intensity"] = (float(base_intensity) if intensity is None else float(intensity)) * scale
    return light


def iter_views(dataset: dict, base_intensity: float) -> list[dict]:
    """Flat sample list. Views of one scene id share that scene's split."""
    samples = []
    index = 1
    for scene in dataset["scenes"]:
        for view in scene["views"]:
            camera = dataset["cameras"][view["camera"]]
            light = _light(dataset["lights"][view["light"]], base_intensity)
            sample_id = f"sample_{index:06d}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "scene_id": scene["id"],
                    "split": scene["split"],
                    "geometry": scene["geometry"],
                    "scale": list(scene["scale"]),
                    "coverage": float(scene["coverage"]),
                    "radius": float(scene["radius"]),
                    "thickness": float(scene["thickness"]),
                    "axis": list(scene["axis"]),
                    "camera_name": view["camera"],
                    "light_name": view["light"],
                    "camera": camera,
                    "light": light,
                }
            )
            index += 1
    return samples


def assert_split_groups(samples: list[dict]) -> None:
    """A scene id stays in one split. Held-out coverages and the ellipsoid are test-only."""
    splits: dict[str, set[str]] = {}
    for sample in samples:
        splits.setdefault(sample["scene_id"], set()).add(sample["split"])
        if sample["geometry"] == "ellipsoid" and sample["split"] != "test":
            raise RuntimeError(f"{sample['scene_id']} ellipsoid is in {sample['split']}")
    leaked = {key: value for key, value in splits.items() if len(value) > 1}
    if leaked:
        raise RuntimeError(f"scene ids cross splits: {leaked}")
    sphere_coverages: dict[str, set[float]] = {}
    for sample in samples:
        if sample["geometry"] == "sphere":
            sphere_coverages.setdefault(sample["split"], set()).add(float(sample["coverage"]))
    overlap = sphere_coverages.get("test", set()) & (
        sphere_coverages.get("train", set()) | sphere_coverages.get("validation", set())
    )
    if overlap:
        raise RuntimeError(f"sphere coverages {overlap} appear in the test split and in train or validation")


def validate_dataset(root: Path) -> dict:
    """File, shape, finite-value, mask, and split checks from the manifest."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    samples = manifest["samples"]
    assert_split_groups(samples)
    errors = []
    seen = set()
    for sample in samples:
        sample_dir = root / sample["sample_id"]
        key = (sample["scene_id"], sample["camera_name"], sample["light_name"], sample["coverage"], sample["radius"])
        if key in seen:
            errors.append(f"duplicate {key}")
        seen.add(key)
        required = [
            sample_dir / "rgb.exr",
            sample_dir / "polarization" / "stokes.exr",
            sample_dir / "polarization" / "S0.exr",
            sample_dir / "polarization" / "S1.exr",
            sample_dir / "polarization" / "S2.exr",
            sample_dir / "polarization" / "S3.exr",
            sample_dir / "polarization" / "analyzer_000.exr",
            sample_dir / "polarization" / "analyzer_045.exr",
            sample_dir / "polarization" / "analyzer_090.exr",
            sample_dir / "polarization" / "analyzer_135.exr",
            sample_dir / "polarization" / "dolp.exr",
            sample_dir / "polarization" / "aolp.exr",
            sample_dir / "ground_truth" / "film_mask.png",
            sample_dir / "ground_truth" / "film_coverage.exr",
            sample_dir / "metadata.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            errors.append(f"{sample['sample_id']} missing {missing}")
            continue
        meta = json.loads((sample_dir / "metadata.json").read_text())
        if meta["sample_id"] != sample["sample_id"] or meta["scene_id"] != sample["scene_id"]:
            errors.append(f"{sample['sample_id']} metadata id mismatch")
        mi.set_variant("scalar_rgb")
        image, mask = load_measurement(sample_dir)
        if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
            errors.append(f"{sample['sample_id']} shape {image.shape} vs mask {mask.shape}")
        if image.shape[2] != 7:
            errors.append(f"{sample['sample_id']} has {image.shape[2]} channels")
        if not np.isfinite(image).all():
            errors.append(f"{sample['sample_id']} has non-finite measurements")
        values = set(np.unique(mask).tolist())
        if not values <= {0, 1} or values != {0, 1}:
            errors.append(f"{sample['sample_id']} mask values {values}")
        loaded = read_gray_png(sample_dir / "ground_truth" / "film_mask.png")
        if loaded.dtype != np.uint8:
            errors.append(f"{sample['sample_id']} mask dtype {loaded.dtype}")
    if errors:
        for error in errors:
            logger.error("film dataset status=invalid error=%s", error)
        raise RuntimeError(f"dataset validation failed with {len(errors)} errors")
    counts = {split: sum(sample["split"] == split for sample in samples) for split in ("train", "validation", "test")}
    logger.info("film dataset status=ok n=%d counts=%s", len(samples), counts)
    return {"n": len(samples), "counts": counts}


def render_dataset() -> dict:
    configure_logging()
    rendering = rendering_config()
    if str(rendering["variant"]).startswith("cuda"):
        raise RuntimeError(f"refusing GPU variant {rendering['variant']}")
    scene = scene_config()
    dataset = dataset_config()
    samples = iter_views(dataset, float(scene["lighting"]["intensity"]))
    assert_split_groups(samples)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    rendered = []
    for sample in samples:
        mesh_dir = ASSET_ROOT / "mitsuba" / "dataset" / sample["sample_id"]
        blend = ASSET_ROOT / "blender" / "dataset" / f"{sample['sample_id']}.blend"
        params = blend_parameters(
            radius=sample["radius"],
            scale=tuple(sample["scale"]),
            thickness=sample["thickness"],
            coverage=sample["coverage"],
            axis=sample["axis"],
            camera={**sample["camera"], "resolution": scene["camera"]["resolution"]},
            light=sample["light"],
            floor_gap=float(scene["room"]["floor_gap"]),
            margin=float(scene["room"]["margin"]),
            blend=blend,
            out_dir=mesh_dir,
            reflectance=float(scene["object"]["reflectance"]),
            room_reflectance=float(scene["room"]["reflectance"]),
            ior=float(scene["film"]["ior"]),
        )
        poses = ensure_scene(params)
        resolution = tuple(int(v) for v in scene["camera"]["resolution"])
        metadata = {
            "sample_id": sample["sample_id"],
            "scene_id": sample["scene_id"],
            "split": sample["split"],
            "scene_seed": int(dataset["seed"]),
            "object_geometry": sample["geometry"],
            "object_scale": sample["scale"],
            "object_material": "diffuse",
            "object_reflectance": float(scene["object"]["reflectance"]),
            "substrate_radius": sample["radius"] - sample["thickness"],
            "substrate_radius": sample["radius"] - sample["thickness"],
            "film_material": scene["film"]["material"],
            "film_thickness": sample["thickness"],
            "film_coverage": sample["coverage"],
            "film_ior": float(scene["film"]["ior"]),
            "film_roughness": float(scene["film"]["roughness"]),
            "film_axis": sample["axis"],
            "camera_name": sample["camera_name"],
            "light_name": sample["light_name"],
        }
        kwargs = {
            "poses": poses,
            "mesh_dir": mesh_dir,
            "resolution": resolution,
            "max_depth": int(rendering["max_depth"]),
            "reflectance": float(scene["object"]["reflectance"]),
            "room_reflectance": float(scene["room"]["reflectance"]),
            "ior": float(scene["film"]["ior"]),
            "roughness": float(scene["film"]["roughness"]),
        }
        record = capture_sample(
            variant=rendering["variant"],
            scene_kwargs=kwargs,
            poses=poses,
            out_dir=DATASET_ROOT / sample["sample_id"],
            seed=int(rendering["seed"]),
            spp=int(rendering["spp"]),
            resolution=resolution,
            control=False,
            mono_variant=None,
            metadata=metadata,
        )
        rendered.append({**sample, "mean_s0_film": record["mean_s0_film"], "n_film_pixels": record["mask"]["n_film_pixels"]})
    manifest = {"seed": dataset["seed"], "variant": rendering["variant"], "samples": rendered}
    (DATASET_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return validate_dataset(DATASET_ROOT)


def main() -> None:
    render_dataset()


if __name__ == "__main__":
    main()

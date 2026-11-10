"""Build the Blender scene and render the one validation sample."""

from __future__ import annotations

import logging
from pathlib import Path

from sensor_sim.film import ROOT
from sensor_sim.film.capture import blend_parameters, capture_sample, ensure_scene
from sensor_sim.film.config import rendering_config, scene_config, write_yaml
from sensor_sim.render import configure_logging

logger = logging.getLogger("sensor_sim.film")

SCENE_PATH = ROOT / "configs" / "film" / "scene.yaml"
MESH_DIR = ROOT / "assets" / "mitsuba" / "film_ball"
BLEND = ROOT / "assets" / "blender" / "film_ball.blend"
OUTPUT = ROOT / "outputs" / "film" / "sample_000001"


def scene_kwargs(poses: dict, scene: dict, rendering: dict, resolution: tuple[int, int]) -> dict:
    return {
        "poses": poses,
        "mesh_dir": MESH_DIR,
        "resolution": resolution,
        "max_depth": int(rendering["max_depth"]),
        "reflectance": float(scene["object"]["reflectance"]),
        "room_reflectance": float(scene["room"]["reflectance"]),
        "ior": float(scene["film"]["ior"]),
        "roughness": float(scene["film"]["roughness"]),
    }


def base_parameters(scene: dict) -> dict:
    camera = scene["camera"]
    light = scene["lighting"]
    return blend_parameters(
        radius=float(scene["object"]["radius"]),
        scale=(1.0, 1.0, 1.0),
        thickness=float(scene["film"]["thickness"]),
        coverage=float(scene["film"]["coverage"]),
        axis=list(scene["film"]["axis"]),
        camera=camera,
        light=light,
        floor_gap=float(scene["room"]["floor_gap"]),
        margin=float(scene["room"]["margin"]),
        blend=BLEND,
        out_dir=MESH_DIR,
        reflectance=float(scene["object"]["reflectance"]),
        room_reflectance=float(scene["room"]["reflectance"]),
        ior=float(scene["film"]["ior"]),
    )


def render_validation_sample() -> dict:
    configure_logging()
    rendering = rendering_config()
    if str(rendering["variant"]).startswith("cuda"):
        raise RuntimeError(f"refusing GPU variant {rendering['variant']}")
    floor = float(rendering["s0_floor"])
    factor = float(rendering["intensity_factor"])
    tries = int(rendering["intensity_tries"])
    record = None
    for attempt in range(tries):
        scene = scene_config(SCENE_PATH)
        poses = ensure_scene(base_parameters(scene))
        resolution = tuple(int(v) for v in scene["camera"]["resolution"])
        metadata = {
            "sample_id": "sample_000001",
            "scene_id": "film_ball",
            "scene_seed": int(rendering["seed"]),
            "object_geometry": "sphere",
            "object_material": "diffuse",
            "object_reflectance": float(scene["object"]["reflectance"]),
            "substrate_radius": float(scene["object"]["radius"]) - float(scene["film"]["thickness"]),
            "substrate_radius": float(scene["object"]["radius"]) - float(scene["film"]["thickness"]),
            "film_material": scene["film"]["material"],
            "film_thickness": float(scene["film"]["thickness"]),
            "film_coverage": float(scene["film"]["coverage"]),
            "film_ior": float(scene["film"]["ior"]),
            "film_roughness": float(scene["film"]["roughness"]),
            "film_axis": list(scene["film"]["axis"]),
            "intensity_attempt": attempt,
        }
        record = capture_sample(
            variant=rendering["variant"],
            scene_kwargs=scene_kwargs(poses, scene, rendering, resolution),
            poses=poses,
            out_dir=OUTPUT,
            seed=int(rendering["seed"]),
            spp=int(rendering["spp"]),
            resolution=resolution,
            control=True,
            mono_variant=rendering["mono_variant"],
            metadata=metadata,
        )
        if record["mean_s0_film"] >= floor:
            logger.info(
                "film stage=intensity status=ok attempt=%d intensity=%.6g mean_s0_film=%.6g",
                attempt,
                scene["lighting"]["intensity"],
                record["mean_s0_film"],
            )
            return record
        scene["lighting"]["intensity"] = float(scene["lighting"]["intensity"]) * factor
        write_yaml(SCENE_PATH, scene)
        logger.info(
            "film stage=intensity status=raise attempt=%d intensity=%.6g mean_s0_film=%.6g floor=%.6g",
            attempt,
            scene["lighting"]["intensity"],
            record["mean_s0_film"],
            floor,
        )
    raise RuntimeError(f"mean film S0 stayed below {floor} after {tries} full-resolution tries")


def main() -> None:
    render_validation_sample()


if __name__ == "__main__":
    main()

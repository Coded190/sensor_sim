"""Separate capsule transforms. Inverse-transpose normals stay untested."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.film.shape_mesh import place_mesh, project_ndc
from sensor_sim.scene_gen.assets import build_asset
from sensor_sim.scene_gen.generate import PROTOTYPE_CAMERA, generate_sample
from sensor_sim.scene_gen.qualify_v1 import _config, _object

OUT = Path("dataset/qualification/convergence_v2/transforms")
DIELECTRIC = {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}
DEPTH_TOLERANCE_M = 0.0004


def _capsule(name: str, vertices: np.ndarray, faces: np.ndarray) -> dict:
    return _object(name, 1, "capsule", vertices, faces, DIELECTRIC)


def _cases() -> dict[str, np.ndarray]:
    mesh, _spec = build_asset("capsule", np.random.default_rng(4), radius=0.03)
    base = place_mesh(np.asarray(mesh["vertices"], dtype=np.float64), 0.0, (0.0, 0.04))
    faces = mesh["faces"]
    shifted = {
        "translate_x": base + np.array([0.03, 0.0, 0.0]),
        "translate_y": base + np.array([0.0, 0.025, 0.0]),
        "translate_z": base + np.array([0.0, 0.0, 0.02]),
        "yaw_z": place_mesh(np.asarray(mesh["vertices"], dtype=np.float64), 0.7, (0.0, 0.04)),
        "uniform_scale": place_mesh(np.asarray(mesh["vertices"], dtype=np.float64) * 1.2, 0.0, (0.0, 0.04)),
        "combined": place_mesh(np.asarray(mesh["vertices"], dtype=np.float64) * 1.1, 0.4, (0.02, 0.05)) + np.array([0.0, 0.0, 0.01]),
    }
    return {name: (vertices, faces) for name, vertices in shifted.items()}


def _loaded_vertices(sample: Path, name: str) -> np.ndarray:
    import mitsuba as mi

    mi.set_variant("cuda_ad_spectral_polarized")
    scene = mi.load_file(str(sample / "scene.xml"))
    shape = next(item for item in scene.shapes() if str(item.id()) == f"mesh-{name}")
    count = int(shape.vertex_count())
    return np.stack([np.array(shape.vertex_position(index)).reshape(-1)[:3] for index in range(count)])


def _vertex_error(source: np.ndarray, loaded: np.ndarray) -> dict:
    if source.shape == loaded.shape:
        error = np.linalg.norm(source - loaded, axis=1)
        paired = True
    else:
        step = max(1, source.shape[0] // 400)
        error = np.array([np.linalg.norm(loaded - point, axis=1).min() for point in source[::step]])
        paired = False
    return {
        "paired": paired,
        "n": int(error.size),
        "median_m": float(np.median(error)),
        "p95_m": float(np.percentile(error, 95)),
        "max_m": float(np.max(error)),
    }


def _angles(reference: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> dict:
    left = reference[mask]
    right = actual[mask]
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(np.sum(left * right, axis=1), -1.0, 1.0)
    degrees = np.degrees(np.arccos(cosine))
    return {"n": int(mask.sum()), "median_deg": float(np.median(degrees)), "p95_deg": float(np.percentile(degrees, 95)), "max_deg": float(np.max(degrees))}


def _depth(sample: Path) -> dict:
    import mitsuba as mi
    from sensor_sim.film.sfp_math import camera_matrix
    from sensor_sim.scene_gen.dataset import load_sample

    mi.set_variant("cuda_ad_spectral_polarized")
    scene = mi.load_file(str(sample / "scene.xml"))
    loaded = load_sample(sample)
    matrix = camera_matrix(scene.sensors()[0])
    forward = matrix[:3, 2]
    forward = forward / np.linalg.norm(forward)
    origin = np.array(PROTOTYPE_CAMERA["position"], dtype=np.float64)
    position = loaded["normal_geo_world"]  # replaced below
    from sensor_sim.film.images import read_exr
    from sensor_sim.scene_gen.dataset import channels_to_array

    position = channels_to_array(read_exr(sample / "ground_truth" / "position.exr"))
    position = np.asarray(position, dtype=np.float64)
    offset = position - origin
    distance = np.linalg.norm(offset, axis=-1)
    ray = offset / np.maximum(distance[..., None], 1e-12)
    expected = float(PROTOTYPE_CAMERA["near_clip"]) / np.maximum(np.sum(ray * forward, axis=-1), 1e-8)
    residual = (distance - loaded["depth"]) - expected
    mask = loaded["valid_mask"] & (loaded["object_ids"] > 0)
    values = np.abs(residual[mask])
    return {
        "formula": "abs((|position-camera| - depth) - near_clip/dot(ray, camera_forward))",
        "tolerance_m": DEPTH_TOLERANCE_M,
        "n": int(mask.sum()),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(np.max(values)),
        "pass": bool(float(np.max(values)) <= DEPTH_TOLERANCE_M),
    }


def _normals(sample: Path, source_vertices: np.ndarray, faces: np.ndarray) -> dict:
    from sensor_sim.film.sfp_math import world_normals_to_camera, camera_matrix
    from sensor_sim.scene_gen.dataset import load_sample
    import mitsuba as mi

    mi.set_variant("cuda_ad_spectral_polarized")
    scene = mi.load_file(str(sample / "scene.xml"))
    loaded = load_sample(sample)
    converted = world_normals_to_camera(loaded["normal_shading_world"], camera_matrix(scene.sensors()[0]))
    mask = loaded["valid_mask"] & (loaded["object_ids"] > 0)
    shading = _angles(converted, loaded["normal_shading_camera"], mask)
    triangles = source_vertices[np.asarray(faces, dtype=np.int64)]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    geo = loaded["normal_geo_world"][mask]
    # Nearest face normal to each geometric sample. This is the mesh criterion.
    step = max(1, geo.shape[0] // 2000)
    sampled = geo[::step]
    cosine = np.clip(np.abs(sampled @ face_normals.T), 0.0, 1.0)
    degrees = np.degrees(np.arccos(cosine.max(axis=1)))
    return {
        "shading_camera_matches_world_normals_to_camera": shading,
        "geometric_vs_nearest_face": {
            "n": int(degrees.size),
            "median_deg": float(np.median(degrees)),
            "p95_deg": float(np.percentile(degrees, 95)),
            "max_deg": float(np.max(degrees)),
            "note": "nearest face normal, not the continuous analytic capsule",
        },
        "inverse_transpose_nonuniform": "untested because nonuniform scale stays rejected",
    }


def main() -> None:
    rows = {}
    built = _cases()
    for name, (vertices, faces) in built.items():
        config = _config(f"transform_{name}", [_capsule("capsule", vertices, faces)], resolution=(128, 128), spp=2)
        config["run_id"] = "convergence-v2-transforms"
        config["config_hash"] = f"convergence-v2-transforms-{name}"
        sample = generate_sample(config, OUT, overwrite=True)
        loaded = _loaded_vertices(sample, "capsule")
        source_ndc = project_ndc(vertices[:: max(1, len(vertices) // 20)], PROTOTYPE_CAMERA["position"], PROTOTYPE_CAMERA["target"], PROTOTYPE_CAMERA["fov_deg"])
        loaded_ndc = project_ndc(loaded[:: max(1, len(loaded) // 20)], PROTOTYPE_CAMERA["position"], PROTOTYPE_CAMERA["target"], PROTOTYPE_CAMERA["fov_deg"])
        rows[name] = {
            "status": "blender_export_integration_tested",
            "vertices": _vertex_error(vertices, loaded),
            "landmark_vertex_max_m": None
        if vertices.shape != loaded.shape
        else float(np.max(np.linalg.norm(vertices[:: max(1, len(vertices) // 8)] - loaded[:: max(1, len(loaded) // 8)], axis=1))),
        "landmark_ndc_max_abs": float(np.max(np.abs(source_ndc - loaded_ndc))) if source_ndc.shape == loaded_ndc.shape else None,
            "depth": _depth(sample),
            "normals": _normals(sample, vertices, faces),
            "sample": str(sample),
        }
        print("transform", name, rows[name]["vertices"]["max_m"], rows[name]["depth"]["pass"], flush=True)
    destination = Path("dataset/qualification/convergence_v2/reports/transforms.json")
    destination.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

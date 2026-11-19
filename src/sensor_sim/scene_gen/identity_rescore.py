"""Rescore exported identity scenes with independent rays. Not a unit-test closure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.dataset import channels_to_array
from sensor_sim.scene_gen.pipeline_closure import OUT

IDENTITY = OUT / "identity"


def _interior_boundary(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sensor_sim.scene_gen.metric_audit import interior_and_boundary

    return interior_and_boundary(ids)


def _scores(painted: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict:
    chosen_paint = painted[mask]
    chosen_ref = reference[mask]
    labels = sorted(set(int(value) for value in np.unique(chosen_paint) if int(value) > 0) | set(int(value) for value in np.unique(chosen_ref) if int(value) > 0))
    per_object = {}
    for label in labels:
        predicted = chosen_paint == label
        truth = chosen_ref == label
        tp = int(np.count_nonzero(predicted & truth))
        per_object[str(label)] = {
            "precision": None if int(predicted.sum()) == 0 else tp / int(predicted.sum()),
            "recall": None if int(truth.sum()) == 0 else tp / int(truth.sum()),
            "n_painted": int(predicted.sum()),
            "n_reference": int(truth.sum()),
            "true_positive": tp,
        }
    confusion = {}
    for pred in labels:
        row = {}
        for truth in labels:
            row[str(truth)] = int(np.count_nonzero((chosen_paint == pred) & (chosen_ref == truth)))
        confusion[str(pred)] = row
    agree = int(np.count_nonzero(chosen_paint == chosen_ref)) if chosen_paint.size else 0
    return {
        "n": int(mask.sum()),
        "agreement": None if mask.sum() == 0 else agree / int(mask.sum()),
        "per_object": per_object,
        "confusion": confusion,
    }


def score_sample(sample: Path) -> dict:
    import mitsuba as mi
    from sensor_sim.film.images import read_exr
    from sensor_sim.film.sfp_math import camera_matrix, view_directions
    from sensor_sim.scene_gen.reference_rays import reference_ray_ids

    mi.set_variant("cuda_ad_spectral_polarized")
    scene = mi.load_file(str(sample / "scene.xml"))
    meta = json.loads((sample / "metadata.json").read_text())
    raw_ids = channels_to_array(read_exr(sample / "ground_truth" / "object_ids.exr"))
    raw_ids = np.asarray(raw_ids, dtype=np.float64).reshape(raw_ids.shape[0], raw_ids.shape[1])
    fractional = int(np.count_nonzero(np.abs(raw_ids - np.rint(raw_ids)) >= 1e-3))
    painted = np.rint(raw_ids).astype(np.int32)
    materials = channels_to_array(read_exr(sample / "ground_truth" / "material_ids.exr"))
    materials = np.rint(np.asarray(materials, dtype=np.float64).reshape(painted.shape)).astype(np.int32)
    sensor = scene.sensors()[0]
    height, width = painted.shape
    directions = view_directions(sensor, width, height)
    world = directions.reshape(-1, 3) @ camera_matrix(sensor)[:3, :3].T
    origin = np.repeat(np.array(meta["camera"]["position"], dtype=np.float64).reshape(1, 3), world.shape[0], axis=0)
    meshes = []
    expected_material = {}
    for item in meta["objects"]:
        expected_material[int(item["object_index"])] = int(item["material_id"])
        if item.get("asset_id") == "plane":
            continue
        shape = next(candidate for candidate in scene.shapes() if str(candidate.id()) == f"mesh-{item['name']}")
        count = int(shape.vertex_count())
        vertices = np.stack([np.array(shape.vertex_position(index)).reshape(-1)[:3] for index in range(count)])
        faces = np.array(shape.faces_buffer()).reshape(-1, 3)
        meshes.append({"vertices": vertices, "faces": faces, "object_index": int(item["object_index"]), "name": item["name"]})
    rays = reference_ray_ids(origin, world, meshes).reshape(painted.shape)
    interior, boundary = _interior_boundary(painted)
    material_expected = np.zeros(painted.shape, dtype=np.int32)
    for label, material in expected_material.items():
        material_expected[rays == label] = material
    material_mask = rays > 0
    material_agree = int(np.count_nonzero((materials == material_expected) & material_mask))
    hidden = []
    for item in meta["objects"]:
        if item.get("asset_id") == "plane":
            continue
        visible = int(np.count_nonzero(painted == int(item["object_index"])))
        if visible == 0:
            hidden.append({"name": item["name"], "visible_object_id_pixels": 0, "kept_in_metadata": True})
    return {
        "scene_id": sample.name,
        "resolution": [int(width), int(height)],
        "status": "blender_export_integration_tested",
        "reference": "reference_ray_ids on exported Mitsuba meshes",
        "fractional_ids": fractional,
        "n_pixels": int(painted.size),
        "interior": _scores(painted, rays, interior),
        "boundary": _scores(painted, rays, boundary),
        "material_agreement": None if int(material_mask.sum()) == 0 else material_agree / int(material_mask.sum()),
        "n_material": int(material_mask.sum()),
        "hidden": hidden,
        "objects_in_metadata": [item["name"] for item in meta["objects"] if item.get("asset_id") != "plane"],
    }


def main() -> None:
    rows = []
    for sample in sorted(path for path in IDENTITY.iterdir() if (path / "metadata.json").is_file()):
        row = score_sample(sample)
        rows.append(row)
        print(
            "identity_rescore",
            row["scene_id"],
            "interior",
            row["interior"]["agreement"],
            "n",
            row["interior"]["n"],
            "boundary",
            row["boundary"]["agreement"],
            "hidden",
            row["hidden"],
            flush=True,
        )
    destination = Path("dataset/qualification/convergence_v2/reports/identity.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

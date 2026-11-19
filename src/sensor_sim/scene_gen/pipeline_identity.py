"""Eight Blender identity scenes and one capsule transform. Not Dataset v1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.film.shape_mesh import place_mesh
from sensor_sim.scene_gen.assets import build_asset
from sensor_sim.scene_gen.dataset import load_sample
from sensor_sim.scene_gen.generate import generate_sample
from sensor_sim.scene_gen.materials import fixed_bsdf
from sensor_sim.scene_gen.pipeline_closure import OUT, _report, _save
from sensor_sim.scene_gen.qualify_v1 import _config, _object
from sensor_sim.scene_gen.reference_rays import reference_ray_ids

DIELECTRIC = {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}
PLASTIC = fixed_bsdf("pplastic")
COPPER = {"type": "conductor", "material": "Cu"}


def _mesh(family: str, seed: int, radius: float, name: str, xy, bsdf: dict, yaw: float = 0.0, scale: float = 1.0) -> dict:
    built, _spec = build_asset(family, np.random.default_rng(seed), radius=radius)
    vertices = np.asarray(built["vertices"], dtype=np.float64) * float(scale)
    vertices = place_mesh(vertices, yaw, xy)
    item = _object(name, 1, family, vertices, built["faces"], bsdf)
    return item


def _interior(ids: np.ndarray, value: int) -> int:
    count = 0
    rows, cols = np.where(ids == value)
    height, width = ids.shape
    for row, col in zip(rows, cols):
        if row == 0 or col == 0 or row == height - 1 or col == width - 1:
            continue
        if np.all(ids[row - 1 : row + 2, col - 1 : col + 2] == value):
            count += 1
    return count


def _score(sample: Path) -> dict:
    import mitsuba as mi
    from sensor_sim.film.sfp_math import camera_matrix, view_directions

    mi.set_variant("cuda_ad_spectral_polarized")
    scene = mi.load_file(str(sample / "scene.xml"))
    loaded = load_sample(sample)
    meta = json.loads((sample / "metadata.json").read_text())
    ids = loaded["object_ids"]
    sensor = scene.sensors()[0]
    directions = view_directions(sensor, ids.shape[1], ids.shape[0])
    world = directions @ camera_matrix(sensor)[:3, :3].T
    origin = np.array(meta["camera"]["position"], dtype=np.float64)
    meshes = []
    for shape in scene.shapes():
        ident = str(shape.id())
        if not ident.startswith("mesh-") or ident == "mesh-ground":
            continue
        name = ident.removeprefix("mesh-")
        item = next(obj for obj in meta["objects"] if obj["name"] == name)
        count = int(shape.vertex_count())
        vertices = np.stack([np.array(shape.vertex_position(i)).reshape(-1)[:3] for i in range(count)])
        faces = np.array(shape.faces_buffer()).reshape(-1, 3)
        meshes.append({"vertices": vertices, "faces": faces, "object_index": int(item["object_index"]), "name": name})
    present = sorted(int(value) for value in np.unique(ids) if int(value) > 0)
    per_object = {}
    for value in present:
        interior = _interior(ids, value)
        rows, cols = np.where(ids == value)
        take = np.linspace(0, len(rows) - 1, num=min(24, len(rows)), dtype=int) if len(rows) else []
        if len(take) == 0:
            per_object[str(value)] = {"interior_pixels": interior, "visible_pixels": 0, "agree": None, "n_scored": 0}
            continue
        hit = reference_ray_ids(np.repeat(origin.reshape(1, 3), len(take), 0), world[rows[take], cols[take]], meshes)
        per_object[str(value)] = {
            "interior_pixels": interior,
            "visible_pixels": int(len(rows)),
            "n_scored": int(len(take)),
            "agree": int(np.count_nonzero(hit == value)),
            "reference": "independent ray-triangle on exported meshes; not the shape-index voter",
        }
    hidden = [item["name"] for item in meta["objects"] if item.get("asset_id") != "plane" and int(np.count_nonzero(ids == item["object_index"])) == 0]
    return {"present": present, "objects": per_object, "hidden_names": hidden, "xml_ids": [str(shape.id()) for shape in scene.shapes()]}


def _emit(name: str, objects: list[dict], share: str | None = None) -> dict:
    if share:
        objects[1]["share_datablock_with"] = share
    config = _config(f"closure_{name}", objects, resolution=(256, 256), spp=2)
    config["run_id"] = "pipeline-closure-identity"
    existing = OUT / "identity" / config["scene_id"]
    sample = existing if (existing / "metadata.json").is_file() else generate_sample(config, OUT / "identity", overwrite=True)
    scored = _score(sample)
    scored["scene_id"] = config["scene_id"]
    scored["status"] = "blender_export_tested"
    print(name, {key: value.get("interior_pixels") for key, value in scored["objects"].items()}, scored["hidden_names"], flush=True)
    return scored


def main() -> None:
    report = _report()
    shared_a = _mesh("sphere", 1, 0.04, "copy_a", (0.0, 0.04), DIELECTRIC)
    shared_b = _mesh("sphere", 1, 0.04, "copy_b", (0.0, 0.04), DIELECTRIC)
    shared_a["object_location"] = [-0.08, 0.0, 0.0]
    shared_b["object_location"] = [0.08, 0.0, 0.0]
    report["identity"] = {
        "shared_datablock": _emit("shared_datablock", [shared_a, shared_b], share="copy_a"),
        "different_transforms": _emit("different_transforms", [
            _mesh("capsule", 2, 0.035, "left", (-0.08, 0.05), DIELECTRIC, yaw=0.4),
            _mesh("capsule", 3, 0.035, "right", (0.08, 0.05), DIELECTRIC, yaw=1.2),
        ]),
        "same_geometry_different_bsdf": _emit("same_geometry_different_bsdf", [
            _mesh("sphere", 4, 0.04, "glass", (-0.07, 0.04), DIELECTRIC),
            _mesh("sphere", 4, 0.04, "metal", (0.07, 0.04), COPPER),
        ]),
        "different_geometry_same_bsdf": _emit("different_geometry_same_bsdf", [
            _mesh("sphere", 5, 0.04, "round", (-0.07, 0.04), PLASTIC),
            _mesh("capsule", 6, 0.035, "long", (0.08, 0.04), PLASTIC),
        ]),
        "several_copies": _emit("several_copies", [
            _mesh("sphere", 7, 0.03, "a", (-0.09, 0.02), DIELECTRIC),
            _mesh("sphere", 7, 0.03, "b", (0.0, 0.08), DIELECTRIC),
            _mesh("sphere", 7, 0.03, "c", (0.09, 0.02), DIELECTRIC),
        ]),
        "partial_occlusion": _emit("partial_occlusion", [
            _mesh("sphere", 8, 0.05, "front", (0.0, 0.0), DIELECTRIC),
            _mesh("ellipsoid", 9, 0.05, "back", (0.04, 0.05), COPPER),
        ]),
        "identical_vertices_two_datablocks": _emit("identical_vertices", [
            _mesh("sphere", 10, 0.04, "one", (-0.08, 0.04), DIELECTRIC),
            _mesh("sphere", 10, 0.04, "two", (0.08, 0.04), PLASTIC),
        ]),
    }
    front = _mesh("sphere", 11, 0.07, "front", (0.0, 0.0), DIELECTRIC)
    back = _mesh("sphere", 12, 0.015, "back", (0.0, 0.12), PLASTIC)
    back["vertices"] = back["vertices"].copy()
    back["vertices"][:, 2] += float(front["vertices"][:, 2].mean() - back["vertices"][:, 2].mean())
    report["identity"]["hidden"] = _emit("hidden", [front, back])
    _save(report)
    print("identity written", flush=True)


if __name__ == "__main__":
    main()

"""Export accepted placements and measure the loaded triangles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.assets import build_asset
from sensor_sim.scene_gen.generate import PROTOTYPE_CAMERA, generate_sample
from sensor_sim.scene_gen.narrow import floor_status, mesh_clearance, narrow_phase
from sensor_sim.scene_gen.placement import FLOOR_Z, place_objects
from sensor_sim.scene_gen.qualify_v1 import _config, _object

OUT = Path("dataset/qualification/dataset_v1_readiness/placement_integration")
DIELECTRIC = {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}
FAMILIES = ("sphere", "capsule", "ellipsoid", "cylinder")


def _accepted(count: int) -> dict:
    rng = np.random.default_rng(4100 + count)
    meshes = []
    families = []
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        radius = 0.03 + 0.004 * index
        mesh, _spec = build_asset(family, rng, radius=radius)
        meshes.append(mesh)
        families.append(family)
    placed = place_objects(meshes, rng, PROTOTYPE_CAMERA, seed=4100 + count, max_attempts=80)
    objects = []
    for index, pose in enumerate(placed):
        objects.append(_object(f"obj_{index:02d}", index + 1, families[index], pose["vertices"], meshes[index]["faces"], DIELECTRIC))
    config = _config(f"placement_count_{count}", objects, resolution=(128, 128), spp=2)
    config["run_id"] = "dataset-v1-placement"
    config["dataset_id"] = "placement_qualification"
    config["config_hash"] = f"placement-count-{count}"
    sample = generate_sample(config, OUT, overwrite=True)
    import mitsuba as mi

    mi.set_variant("cuda_ad_spectral_polarized")
    scene = mi.load_file(str(sample / "scene.xml"))
    loaded = {}
    for shape in scene.shapes():
        ident = str(shape.id())
        if not ident.startswith("mesh-") or ident == "mesh-ground":
            continue
        count_vertices = int(shape.vertex_count())
        vertices = np.stack([np.array(shape.vertex_position(i)).reshape(-1)[:3] for i in range(count_vertices)])
        faces = np.array(shape.faces_buffer()).reshape(-1, 3)
        loaded[ident] = {"vertices": vertices, "faces": faces, "floor": floor_status(vertices, FLOOR_Z)}
    names = sorted(loaded)
    pairs = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            report = mesh_clearance(loaded[left_name], loaded[right_name])
            pairs.append({"pair": [left_name, right_name], "clearance_m": report["clearance_m"], "surface_intersection": report["surface_intersection"]})
    floors = {name: item["floor"] for name, item in loaded.items()}
    return {
        "count": count,
        "families": families,
        "sample": str(sample),
        "pairs": pairs,
        "floors": floors,
        "pass": all(pair["clearance_m"] + 1e-6 >= 0.005 and not pair["surface_intersection"] for pair in pairs)
        and all(not item["penetrating"] for item in floors.values()),
    }


def _rejected() -> dict:
    rng = np.random.default_rng(7)
    small, _spec = build_asset("sphere", rng, radius=0.03)
    other, _spec = build_asset("sphere", np.random.default_rng(8), radius=0.03)
    close = {"vertices": small["vertices"] + np.array([0.064, 0.0, 0.0]), "faces": small["faces"]}
    near = narrow_phase({"vertices": small["vertices"], "faces": small["faces"]}, close, 0.005)
    sunk = small["vertices"].copy()
    sunk[:, 2] -= float(sunk[:, 2].min())
    floor = floor_status(sunk, FLOOR_Z)
    return {
        "sub_5_mm_separated": near["separated"],
        "sub_5_mm_clearance_m": near["clearance_m"],
        "floor_penetration_rejected": floor["penetrating"],
        "floor_min_z_m": floor["min_z_m"],
    }


def main() -> None:
    rows = {"rejected": _rejected(), "accepted": []}
    for count in (1, 2, 3, 4):
        try:
            row = _accepted(count)
        except RuntimeError as exc:
            row = {"count": count, "pass": False, "error": str(exc)}
        rows["accepted"].append(row)
        print("placement", count, row.get("pass"), row.get("error", ""), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Identity, the remaining convergence scenes, RGB, and the frozen spec."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.film.shape_mesh import place_mesh
from sensor_sim.scene_gen.assets import build_asset
from sensor_sim.scene_gen.final_follow import OUT, _load, _save, convergence, passes
from sensor_sim.scene_gen.generate import generate_sample
from sensor_sim.scene_gen.materials import fixed_bsdf
from sensor_sim.scene_gen.qualify_v1 import _config, _object

ROOT = Path(__file__).resolve().parents[3]


def _sphere(name: str, xy, radius: float, bsdf: dict, rng) -> dict:
    mesh, _spec = build_asset("sphere", rng, radius=radius)
    return _object(name, 1, "sphere", place_mesh(mesh["vertices"], 0.0, xy), mesh["faces"], bsdf)


def identity_scenes() -> dict:
    from sensor_sim.scene_gen.dataset import load_sample
    from sensor_sim.scene_gen.qualify_v1 import _ray_check

    dielectric = {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}
    plastic = fixed_bsdf("pplastic")
    rng = np.random.default_rng(0)
    cases = {
        "separated": [
            _sphere("left", (-0.08, 0.02), 0.04, dielectric, rng),
            _sphere("right", (0.08, 0.08), 0.04, plastic, rng),
        ],
        "partial": [
            _sphere("front", (-0.01, 0.0), 0.045, dielectric, rng),
            _sphere("back", (0.05, 0.06), 0.05, plastic, rng),
        ],
        "hidden": [
            _sphere("front", (0.0, 0.0), 0.07, dielectric, rng),
            _sphere("back", (0.0, 0.12), 0.015, plastic, rng),
        ],
    }
    # Lift the hidden sphere into the front sphere's shadow.
    back = cases["hidden"][1]["vertices"]
    front = cases["hidden"][0]["vertices"]
    back = back.copy()
    back[:, 2] += float(front[:, 2].mean() - back[:, 2].mean())
    cases["hidden"][1]["vertices"] = back
    rows = {}
    for name, objects in cases.items():
        config = _config(f"id256_{name}", objects, resolution=(256, 256), spp=2)
        existing = OUT / "identity256" / config["scene_id"]
        sample = existing if (existing / "metadata.json").is_file() else generate_sample(config, OUT / "identity256", overwrite=True)
        loaded = load_sample(sample)
        present = sorted(int(value) for value in np.unique(loaded["object_ids"]) if int(value) > 0)
        interior = {}
        ids = loaded["object_ids"]
        for value in present:
            count = 0
            ys, xs = np.where(ids == value)
            for row, col in zip(ys, xs):
                if row == 0 or col == 0 or row == ids.shape[0] - 1 or col == ids.shape[1] - 1:
                    continue
                block = ids[row - 1 : row + 2, col - 1 : col + 2]
                if np.all(block == value):
                    count += 1
            interior[str(value)] = count
        rays = _ray_check(sample, {1: objects[0]["name"], 2: objects[1]["name"]})
        rows[name] = {"present": present, "interior_pixels": interior, "rays": rays, "scene_id": config["scene_id"]}
        print("identity", name, interior, flush=True)
    return rows


def material_scenes() -> dict:
    from sensor_sim.scene_gen.xml_patch import bsdf_element
    import xml.etree.ElementTree as ET

    mesh, _spec = build_asset("sphere", np.random.default_rng(1), radius=0.05)
    base = _config(
        "conv_sphere",
        [_sphere("ball", (0.0, 0.04), 0.05, {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}, np.random.default_rng(1))],
        resolution=(128, 128),
        spp=16,
    )
    sample = generate_sample(base, OUT / "convergence", overwrite=True)
    presets = {
        "dielectric_1_33": {"type": "dielectric", "int_ior": 1.33, "ext_ior": 1.0},
        "dielectric_1_70": {"type": "dielectric", "int_ior": 1.70, "ext_ior": 1.0},
        "pplastic": fixed_bsdf("pplastic"),
        "copper": {"type": "conductor", "material": "Cu"},
    }
    rows = {"multi_object_rerun_269837089": json.loads((OUT / "evidence.json").read_text())["convergence_multi"]}
    for name, bsdf in presets.items():
        tree = ET.parse(sample / "scene.xml")
        root = tree.getroot()
        for node in list(root.iter()):
            if node.tag.rsplit("}", 1)[-1] == "bsdf" and node.get("id") == "mat-obj_ball":
                parent = next(parent for parent in root.iter() if node in list(parent))
                index = list(parent).index(node)
                parent.remove(node)
                parent.insert(index, bsdf_element(bsdf, "mat-obj_ball"))
                break
        dest = sample.parent / name / "scene.xml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Keep the mesh path valid by writing the xml beside a copy of the meshes via relative path.
        # The original xml uses meshes/ball.ply relative to the sphere folder. Write next to it instead.
        dest = sample / f"{name}.xml"
        dest.write_text(ET.tostring(root, encoding="unicode"))
        # convergence() loads scene.xml. Point it at a folder that contains this xml name.
        folder = OUT / "convergence" / name
        folder.mkdir(parents=True, exist_ok=True)
        text = dest.read_text().replace("meshes/ball.ply", "../conv_sphere/meshes/ball.ply")
        text = text.replace("meshes/ground.ply", "../conv_sphere/meshes/ground.ply")
        (folder / "scene.xml").write_text(text)
        # Copy ground truth mask from the base sphere. The object id image is the same geometry.
        import shutil
        shutil.copytree(sample / "ground_truth", folder / "ground_truth", dirs_exist_ok=True)
        shutil.copy(sample / "polarization.exr", folder / "polarization.exr")
        rows[name] = convergence(folder)
        print(name, {key: value.get("pass") for key, value in rows[name].items()}, flush=True)
    return rows


def reference_variability() -> dict:
    from sensor_sim.scene_gen.render_sample import render_stokes, stack_stokes
    from sensor_sim.scene_gen.dataset import load_sample
    import mitsuba as mi

    mi.set_variant("cuda_ad_spectral_polarized")
    sample = OUT / "rerun" / "rerun_269837089"
    loaded = load_sample(sample)
    first = render_stokes(sample / "scene.xml", "cuda_ad_spectral_polarized", 256, 0)
    second = render_stokes(sample / "scene.xml", "cuda_ad_spectral_polarized", 256, 1)
    stacks = []
    for rendered in (first, second):
        channels = rendered[1]
        stacks.append([stack_stokes(channels, name) for name in ("S0", "S1", "S2", "S3")])
    from sensor_sim.scene_gen.final_follow import stokes_table

    row = stokes_table(stacks[0], stacks[1], loaded["measurement_mask"])
    row["pass_within_caps"] = passes(row)
    return row


def main() -> None:
    report = _load()
    if "reference_variability_256" not in report:
        report["reference_variability_256"] = reference_variability()
        _save(report)
    print("variability", report["reference_variability_256"]["pass_within_caps"], flush=True)
    report["identity_256"] = identity_scenes()
    _save(report)
    try:
        report["convergence_scenes"] = material_scenes()
    except Exception as exc:
        report["convergence_scenes_error"] = str(exc)
        _save(report)
        raise
    chosen = 256
    report["chosen_spp"] = chosen
    report["chosen_spp_reason"] = (
        "16 and 64 spp miss the predeclared S0 p95 cap of 0.15 on the multi-object scene. "
        "The dataset spp is therefore 256 for every scene, not a mixture."
    )
    _save(report)
    print("rest written", flush=True)


if __name__ == "__main__":
    main()

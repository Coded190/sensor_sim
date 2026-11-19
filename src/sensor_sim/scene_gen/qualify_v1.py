"""Dataset v1 qualification renders. Run after the unit tests.

Writes ``dataset/qualification/report.json``. This is evidence for the fixed
camera and the measured materials. It is not a trained model and not Dataset v1.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from sensor_sim.film.sfp_math import camera_matrix, view_directions
from sensor_sim.film.shape_mesh import place_mesh
from sensor_sim.scene_gen.assets import FAMILIES, DRAGON_ID, build_asset
from sensor_sim.scene_gen.characterize import measure_bsdf
from sensor_sim.scene_gen.dataset import exr_schema, load_sample
from sensor_sim.scene_gen.generate import PROTOTYPE_CAMERA, PROTOTYPE_LIGHT, generate_sample
from sensor_sim.scene_gen.identity import ray_triangle_ids
from sensor_sim.scene_gen.materials import fixed_bsdf, primary_catalog
from sensor_sim.scene_gen.placement import place_objects
from sensor_sim.stokes import polarization_maps

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "dataset" / "qualification"


def _ground() -> dict:
    return {
        "name": "ground",
        "object_index": 99,
        "instance_id": "ground",
        "family": "plane",
        "asset_id": "plane",
        "kind": "plane",
        "size": 0.6,
        "location": [0.0, 0.0, 0.0],
        "color": [0.6, 0.6, 0.6],
        "blender_material": "diffuse",
        "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.6, 0.6, 0.6]}},
        "material_response": "does_not",
    }


def _object(name: str, index: int, family: str, vertices: np.ndarray, faces: np.ndarray, bsdf: dict) -> dict:
    return {
        "name": name,
        "object_index": index,
        "instance_id": f"{family}:{name}",
        "family": family,
        "asset_id": family,
        "kind": "obj",
        "shape": {"family": family},
        "bsdf": bsdf,
        "material_response": "keeps",
        "vertices": vertices,
        "faces": faces,
    }


def _config(scene_id: str, objects: list[dict], resolution=(64, 64), spp: int = 4) -> dict:
    numbered = []
    for index, item in enumerate(objects, start=1):
        item = dict(item)
        item["object_index"] = index
        numbered.append(item)
    ground = _ground()
    ground["object_index"] = len(numbered) + 1
    return {
        "mode": "identity",
        "scene_id": scene_id,
        "seed": 1,
        "geometry_seed": 1,
        "material_seed": 2,
        "render_seed": 3,
        "resolution": [int(resolution[0]), int(resolution[1])],
        "spp": int(spp),
        "max_depth": 8,
        "camera": dict(PROTOTYPE_CAMERA),
        "light": dict(PROTOTYPE_LIGHT),
        "objects": numbered + [ground],
    }


def _placed(family: str, xy: tuple[float, float], bsdf: dict, name: str, index: int, rng, lift: float = 0.0) -> dict:
    mesh, _spec = build_asset(family, rng, radius=0.04)
    vertices = place_mesh(mesh["vertices"], 0.0, xy)
    if lift:
        vertices = vertices.copy()
        vertices[:, 2] += float(lift)
    return _object(name, index, family, vertices, mesh["faces"], bsdf)


def _ray_check(sample: Path, expected: dict[int, str]) -> dict:
    """Compare a handful of labeled pixels with an independent ray-triangle test."""
    import mitsuba as mi

    from sensor_sim.scene_gen.render_sample import choose_variant

    try:
        mi.set_variant(choose_variant())
    except Exception:
        pass
    scene = mi.load_file(str(sample / "scene.xml"))
    loaded = load_sample(sample)
    sensor = scene.sensors()[0]
    height, width = loaded["object_ids"].shape
    directions = view_directions(sensor, width, height)
    rotation = camera_matrix(sensor)[:3, :3]
    world = directions @ rotation.T
    origin = np.array(PROTOTYPE_CAMERA["position"], dtype=np.float64)
    meshes = []
    for shape in scene.shapes():
        ident = shape.id()
        if not ident.startswith("mesh-") or ident == "mesh-ground":
            continue
        count = int(shape.vertex_count())
        vertices = np.stack([np.array(shape.vertex_position(i)).reshape(-1)[:3] for i in range(count)])
        faces = np.array(shape.faces_buffer()).reshape(-1, 3)
        name = ident.removeprefix("mesh-")
        index = next(item["object_index"] for item in json.loads((sample / "metadata.json").read_text())["objects"] if item["name"] == name)
        meshes.append({"vertices": vertices, "faces": faces, "object_index": index, "name": name})
    report = {}
    # A pixel on the silhouette can be a different surface than the ray through
    # its center. The reference uses pixels whose 8 neighbors share the id.
    interior = np.zeros_like(loaded["object_ids"], dtype=bool)
    for row in range(1, height - 1):
        for column in range(1, width - 1):
            block = loaded["object_ids"][row - 1 : row + 2, column - 1 : column + 2]
            interior[row, column] = bool(np.all(block == loaded["object_ids"][row, column]))
    for index, label in expected.items():
        rows, cols = np.where(interior & (loaded["object_ids"] == index))
        if len(rows) == 0:
            report[label] = {"pixels": 0, "agree": None}
            continue
        take = np.linspace(0, len(rows) - 1, num=min(12, len(rows)), dtype=int)
        origins = np.repeat(origin.reshape(1, 3), len(take), axis=0)
        dirs = world[rows[take], cols[take]]
        hit = ray_triangle_ids(origins, dirs, meshes)
        report[label] = {"pixels": int(len(rows)), "agree": int(np.count_nonzero(hit == index)), "checked": int(len(take))}
    return report


def identity_scenes() -> dict:
    dielectric = {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}
    plastic = fixed_bsdf("pplastic")
    copper = {"type": "conductor", "material": "Cu"}
    rng = np.random.default_rng(0)
    cases = {
        "separated": [
            _placed("sphere", (-0.08, 0.02), dielectric, "left", 1, rng),
            _placed("sphere", (0.08, 0.08), plastic, "right", 2, rng),
        ],
        "partial": [
            _placed("sphere", (0.0, 0.0), dielectric, "front", 1, rng),
            _placed("ellipsoid", (0.045, 0.09), plastic, "back", 2, rng),
        ],
        "gap": [
            _placed("sphere", (-0.06, 0.04), dielectric, "near", 1, rng),
            _placed("capsule", (0.02, 0.04), copper, "capsule", 2, rng),
        ],
        "duplicates": [
            _placed("sphere", (-0.07, 0.05), dielectric, "copy_a", 1, np.random.default_rng(1)),
            _placed("sphere", (0.07, 0.05), copper, "copy_b", 2, np.random.default_rng(1)),
        ],
        "hidden": [
            _placed("sphere", (0.0, 0.0), dielectric, "front", 1, rng),
            _placed("sphere", (0.0, 0.10), plastic, "back", 2, rng, lift=0.03),
        ],
    }
    # The front sphere in the hidden case is enlarged by using a second build.
    front = build_asset("sphere", np.random.default_rng(2), radius=0.06)
    cases["hidden"][0] = _object(
        "front",
        1,
        "sphere",
        place_mesh(front[0]["vertices"], 0.0, (0.0, -0.01)),
        front[0]["faces"],
        dielectric,
    )
    left = cases["gap"][0]["vertices"]
    right = cases["gap"][1]["vertices"]
    shift = float(left[:, 0].max() + 0.005 - right[:, 0].min())
    right = right.copy()
    right[:, 0] += shift
    cases["gap"][1]["vertices"] = right
    out = {}
    root = OUT / "identity"
    for name, objects in cases.items():
        config = _config(f"id_{name}", objects)
        try:
            sample = generate_sample(config, root)
        except Exception as exc:
            out[name] = {"error": str(exc)}
            print("identity", name, "FAILED", exc, flush=True)
            continue
        loaded = load_sample(sample)
        present = sorted(int(value) for value in np.unique(loaded["object_ids"]) if int(value) > 0)
        metadata = json.loads((sample / "metadata.json").read_text())
        names = [item["name"] for item in metadata["objects"]]
        expected = {1: objects[0]["name"], 2: objects[1]["name"]}
        out[name] = {
            "scene_id": config["scene_id"],
            "present_ids": present,
            "object_names": names,
            "rays": _ray_check(sample, expected),
            "git_dirty": metadata.get("git_dirty"),
            "render_seed": metadata.get("render_seed"),
        }
        print("identity", name, out[name]["present_ids"], out[name]["rays"], flush=True)
    return out


def asset_sweep() -> list[str]:
    done = []
    root = OUT / "assets"
    for index, family in enumerate(list(FAMILIES) + [DRAGON_ID]):
        rng = np.random.default_rng(10 + index)
        mesh, spec = build_asset(family, rng)
        vertices = place_mesh(mesh["vertices"], 0.2, (0.0, 0.06))
        bsdf = {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}
        config = _config(f"asset_{family}", [_object(f"obj_{family}", 1, family, vertices, mesh["faces"], bsdf)], spp=4)
        try:
            generate_sample(config, root)
        except Exception as exc:
            done.append(f"{family}:FAILED:{exc}")
            print("asset", family, "FAILED", exc, flush=True)
            continue
        done.append(family)
        print("asset", family, flush=True)
    return done


def material_distributions() -> dict:
    import mitsuba as mi

    try:
        mi.set_variant("cuda_ad_spectral_polarized")
    except Exception:
        pass
    sample = ROOT / "dataset" / "samples" / "scene_000000"
    work = OUT / "materials" / "sphere_base"
    if not (work / "scene.xml").is_file():
        shutil.copytree(sample, work)
    from sensor_sim.film.images import read_exr
    from sensor_sim.scene_gen.dataset import channels_to_array

    ids = np.squeeze(channels_to_array(read_exr(work / "ground_truth" / "object_ids.exr")))
    mask = np.rint(ids) == 1
    sample = work
    rows = {}
    for name, bsdf in primary_catalog():
        key = f"{name}_{bsdf.get('material', '')}_{bsdf.get('int_ior', '')}_{bsdf.get('alpha', '')}"
        stats = measure_bsdf(sample / "scene.xml", bsdf, mask, "cuda_ad_spectral_polarized", 16, OUT / "materials")
        stats["resolution"] = [256, 256]
        stats["spp"] = 16
        stats["note"] = "distribution on the published Z-up sphere, object id 1, S0 > 1e-4"
        rows[key] = {k: v for k, v in stats.items() if k != "maps"}
        print("material", key, "median", stats.get("median_dolp"), "frac", stats.get("fraction_above_0_2"), flush=True)
    rough = {
        "type": "roughdielectric",
        "int_ior": 1.5,
        "ext_ior": 1.0,
        "distribution": "beckmann",
        "alpha": 0.1,
    }
    stats = measure_bsdf(sample / "scene.xml", rough, mask, "cuda_ad_spectral_polarized", 16, OUT / "materials")
    stats["resolution"] = [256, 256]
    stats["spp"] = 16
    stats["note"] = "Mitsuba 3.9.1 polarized render on this sphere. Zero DoLP here is not a claim about every rough dielectric."
    rows["roughdielectric_1.5_0.1"] = {k: v for k, v in stats.items() if k != "maps"}
    return rows


def packing_probe() -> dict:
    camera = dict(PROTOTYPE_CAMERA)
    report = {}
    for count in (1, 2, 3, 4):
        ok = 0
        reasons = {}
        for seed in range(12):
            meshes = []
            rng = np.random.default_rng(1000 + count * 100 + seed)
            for _index in range(count):
                family = ("sphere", "capsule", "ellipsoid", "rounded_box")[_index % 4]
                meshes.append(build_asset(family, rng, radius=float(rng.uniform(0.03, 0.05)))[0])
            try:
                placed = place_objects(meshes, rng, camera, seed=seed, max_attempts=40)
            except RuntimeError as exc:
                text = str(exc)
                kind = "cannot_fit" if "cannot_fit" in text else "placement_failed" if "placement_failed" in text else "other"
                reasons[kind] = reasons.get(kind, 0) + 1
                continue
            if len(placed) != count:
                reasons["dropped"] = reasons.get("dropped", 0) + 1
                continue
            ok += 1
        report[str(count)] = {"successes": ok, "attempts": 12, "failures": reasons}
        print("packing", count, report[str(count)], flush=True)
    return report


def _aolp_error(reference: np.ndarray, other: np.ndarray, mask: np.ndarray) -> float:
    """Median absolute AoLP error in degrees. The period is 180 degrees."""
    delta = np.abs(reference - other)
    period = np.pi
    delta = np.minimum(delta, period - np.minimum(delta, period))
    # delta is already in [0, pi]; the smallest step on a pi-periodic circle is min(d, pi-d).
    delta = np.minimum(delta, np.pi - delta)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.degrees(np.median(delta[mask])))


def spp_compare(sample: Path) -> dict:
    import mitsuba as mi

    from sensor_sim.scene_gen.render_sample import render_stokes, stack_stokes

    mi.set_variant("cuda_ad_spectral_polarized")
    loaded = load_sample(sample)
    mask = loaded["measurement_mask"] & (loaded["object_ids"] == 1)
    stacks = {}
    for spp in (16, 64, 256):
        _scene, channels, _names, _bitmap = render_stokes(sample / "scene.xml", "cuda_ad_spectral_polarized", spp, 0)
        stacks[spp] = [stack_stokes(channels, name) for name in ("S0", "S1", "S2", "S3")]
        print("spp", spp, "rendered", flush=True)
    reference = stacks[256]
    ref_maps = polarization_maps(*[component.mean(axis=2) if component.ndim == 3 else component for component in reference])
    report = {}
    for spp, components in stacks.items():
        averaged = [component.mean(axis=2) if component.ndim == 3 else component for component in components]
        maps = polarization_maps(*averaged)
        use = mask & maps["valid"] & ref_maps["valid"] & (maps["dolp"] >= 0.05) & (ref_maps["dolp"] >= 0.05)
        report[str(spp)] = {
            "s0_mae": float(np.mean(np.abs(averaged[0] - reference[0].mean(axis=2))[mask])) if mask.any() else None,
            "dolp_mae": float(np.mean(np.abs(maps["dolp"] - ref_maps["dolp"])[mask])) if mask.any() else None,
            "aolp_median_deg": _aolp_error(ref_maps["aolp"], maps["aolp"], use),
            "aolp_period_deg": 180.0,
            "n_mask": int(mask.sum()),
            "n_aolp": int(use.sum()),
        }
    return report


def _save(report: dict) -> None:
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


def pilot_and_loader(report: dict) -> None:
    """Eight successful one-object scenes and eight successful tabletop scenes."""
    commands = [
        ("one_object", ROOT / "configs" / "scenes" / "one_object.yaml", OUT / "pilot" / "one"),
        ("tabletop", ROOT / "configs" / "scenes" / "tabletop.yaml", OUT / "pilot" / "table"),
    ]
    report["pilot"] = {}
    for name, config, directory in commands:
        samples = directory / "samples"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "sensor_sim.scene_gen.cli",
                "--config",
                str(config),
                "--num-scenes",
                "8",
                "--seed",
                "100",
                "--output",
                str(samples),
                "--split-mode",
                "scene",
                "--split-seed",
                "1",
            ],
            cwd=ROOT,
            check=False,
        )
        manifest = json.loads((directory / "run_manifest.json").read_text()) if (directory / "run_manifest.json").is_file() else {}
        report["pilot"][name] = {
            "returncode": int(completed.returncode),
            "completed": manifest.get("completed_ids", []),
            "failures": len(manifest.get("failures", [])),
            "status": manifest.get("status"),
        }
        _save(report)
        print("pilot", name, report["pilot"][name]["status"], len(report["pilot"][name]["completed"]), flush=True)
    one_ids = report["pilot"]["one_object"]["completed"]
    if one_ids:
        sample = OUT / "pilot" / "one" / "samples" / one_ids[0]
        import mitsuba as mi

        try:
            mi.set_variant("cuda_ad_spectral_polarized")
        except Exception:
            pass
        report["exr"] = exr_schema(sample / "polarization.exr")
        report["spp"] = spp_compare(sample)
        loaded = load_sample(sample)
        report["loader"] = {
            "shape": list(loaded["s0"].shape),
            "dtype": str(loaded["s0"].dtype),
            "normal_shape": list(loaded["normal_shading_camera"].shape),
            "s1_min": float(np.min(loaded["s1"])),
            "aligned": loaded["s0"].shape == loaded["object_ids"].shape == loaded["material_ids"].shape,
        }
    hidden = OUT / "identity" / "id_separated"
    if hidden.is_dir():
        moved = OUT / "moved_sample"
        if moved.exists():
            shutil.rmtree(moved)
        shutil.copytree(hidden, moved)
        import mitsuba as mi

        scene = mi.load_file(str(moved / "scene.xml"))
        report["moved_multi_object_shapes"] = [shape.id() for shape in scene.shapes()]
    _save(report)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    report["packing"] = packing_probe()
    _save(report)
    report["identity"] = identity_scenes()
    _save(report)
    report["assets"] = asset_sweep()
    _save(report)
    report["materials"] = material_distributions()
    _save(report)
    pilot_and_loader(report)
    print("wrote", OUT / "report.json", flush=True)


if __name__ == "__main__":
    main()

"""Scene construction, Stokes algebra, splits, and one Blender export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sensor_sim.film.geometry import DRAGON_CENTER_XY, stand_dragon, read_ascii_ply
from sensor_sim.scenes import material_bsdf
from sensor_sim.scene_gen.assets import DRAGON_PLY, FAMILIES, build_asset, dragon_local
from sensor_sim.scene_gen.checks import integer_index, stokes_report
from sensor_sim.scene_gen.dataset import load_sample
from sensor_sim.scene_gen.generate import child_seeds, generate_sample, prototype_config, random_config
from sensor_sim.scene_gen.materials import BSDF_NAMES, fixed_bsdf, sample_bsdf
from sensor_sim.scene_gen.placement import place_objects
from sensor_sim.scene_gen.splits import assign_splits
from sensor_sim.stokes import polarization_maps

CAMERA = {
    "position": [0.0, -0.55, 0.32],
    "target": [0.0, 0.0, 0.18],
    "up": [0.0, 0.0, 1.0],
    "fov_deg": 55.0,
}


def test_fixed_bsdfs_match_phase2():
    for name in ("diffuse", "dielectric", "roughdielectric", "pplastic", "plastic"):
        assert fixed_bsdf(name) == material_bsdf(name)
    assert "ext_ior" not in fixed_bsdf("plastic")
    with pytest.raises(ValueError):
        sample_bsdf("thindielectric", np.random.default_rng(0))


def test_sampled_parameters_stay_in_range():
    rng = np.random.default_rng(1)
    for name in BSDF_NAMES:
        bsdf = sample_bsdf(name, rng)
        assert bsdf["type"] == name
        if "int_ior" in bsdf:
            assert 1.33 <= bsdf["int_ior"] <= 1.70
            if name != "plastic":
                assert bsdf["ext_ior"] == 1.0
        if "alpha" in bsdf:
            assert 0.02 <= bsdf["alpha"] <= 0.5
        if name == "plastic":
            assert "ext_ior" not in bsdf


def test_geometry_and_material_seeds_are_independent():
    geometry_a, material_a = child_seeds(5)
    geometry_b, material_b = child_seeds(5)
    assert (geometry_a, material_a) == (geometry_b, material_b)
    other_geo, same_material = child_seeds(6)
    assert other_geo != geometry_a
    assert same_material != material_a


def test_same_geometry_seed_rebuilds_the_mesh():
    first, spec_a = build_asset("ellipsoid", np.random.default_rng(child_seeds(4)[0]))
    second, spec_b = build_asset("ellipsoid", np.random.default_rng(child_seeds(4)[0]))
    assert spec_a == spec_b
    assert np.allclose(first["vertices"], second["vertices"])
    material = sample_bsdf("dielectric", np.random.default_rng(child_seeds(4)[1]))
    other = sample_bsdf("dielectric", np.random.default_rng(child_seeds(9)[1]))
    assert material != other


def test_families_are_watertight_and_dragon_stays_local():
    rng = np.random.default_rng(0)
    for family in FAMILIES:
        mesh, _spec = build_asset(family, rng)
        assert mesh["faces"].size > 0
    local = dragon_local(0.1)
    vertices = local["vertices"]
    center = 0.5 * (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0))
    assert np.allclose(center, 0.0)
    assert abs(float(vertices[:, 2].min())) < 1e-8
    raw = read_ascii_ply(DRAGON_PLY)
    stood = stand_dragon(raw["vertices"])
    assert abs(float(stood[:, 2].min())) < 1e-8
    xy = 0.5 * (stood[:, :2].min(axis=0) + stood[:, :2].max(axis=0))
    assert np.allclose(xy, DRAGON_CENTER_XY)


def test_placement_rejects_overlap_and_rests_on_the_floor():
    rng = np.random.default_rng(2)
    meshes = [build_asset("sphere", rng, radius=0.04)[0] for _index in range(3)]
    placed = place_objects(meshes, np.random.default_rng(3), CAMERA, seed=3)
    assert len(placed) == 3
    for item in placed:
        assert abs(float(item["vertices"][:, 2].min()) - 0.001) < 1e-8
    boxes = [item["box"] for item in placed]
    for index, box in enumerate(boxes):
        for other in boxes[index + 1 :]:
            separated = (
                box[1, 0] + 0.005 <= other[0, 0]
                or other[1, 0] + 0.005 <= box[0, 0]
                or box[1, 1] + 0.005 <= other[0, 1]
                or other[1, 1] + 0.005 <= box[0, 1]
            )
            assert separated


def test_stokes_algebra_does_not_clamp():
    s0 = np.array([[1.0, 0.0], [0.2, 0.5]])
    s1 = np.array([[0.0, 0.0], [-0.1, 0.4]])
    s2 = np.array([[0.0, 0.0], [0.0, -0.2]])
    s3 = np.array([[0.0, 0.0], [0.0, 0.1]])
    report = stokes_report(s0, s1, s2, s3)
    maps = polarization_maps(s0, s1, s2, s3)
    assert np.allclose(report["maps"]["s1"], s1)
    assert report["maps"]["s1"][1, 0] < 0.0
    assert report["maps"]["s0"][0, 1] == 0.0
    assert report["maps"]["dolp"][0, 1] == 0.0
    assert np.allclose(report["analyzers"]["analyzer_000"][1, 1], 0.5 * (0.5 + 0.4))
    assert np.allclose(report["analyzers"]["analyzer_090"][1, 1], 0.5 * (0.5 - 0.4))
    assert maps["aolp"][1, 1] == pytest.approx(0.5 * np.arctan2(-0.2, 0.4))
    broken = np.array([[0.1, 0.0], [0.0, 0.0]])
    bad = stokes_report(broken, np.ones_like(broken), np.zeros_like(broken), np.zeros_like(broken))
    assert bad["n_stokes_inequality"] == 1
    assert np.allclose(bad["maps"]["s0"], broken)
    values = np.array([[0.0, 1.2, 2.0]])
    assert integer_index(values, 2).tolist() == [[False, False, True]]
    assert not integer_index(values, 1).any()


def test_reversed_face_is_rejected_and_families_pass():
    from sensor_sim.scene_gen.assets import build_asset, dragon_local
    from sensor_sim.scene_gen.meshcheck import mesh_issues, require_clean

    mesh, _spec = build_asset("sphere", np.random.default_rng(0), radius=0.05)
    require_clean(mesh["vertices"], mesh["faces"], closed=True, name="sphere")
    flipped = mesh["faces"].copy()
    flipped[0] = flipped[0, ::-1]
    assert "inverted_or_inconsistent_winding" in mesh_issues(mesh["vertices"], flipped, closed=True)
    degenerate = mesh["faces"].copy()
    vertices = mesh["vertices"].copy()
    vertices[degenerate[0, 1]] = vertices[degenerate[0, 0]]
    assert "degenerate" in mesh_issues(vertices, degenerate, closed=True)
    for family in ("ellipsoid", "capsule", "cylinder", "cone"):
        built, _spec = build_asset(family, np.random.default_rng(1), radius=0.05)
        require_clean(built["vertices"], built["faces"], closed=True, name=family)
    dragon = dragon_local(0.1)
    assert "not_closed" not in mesh_issues(dragon["vertices"], dragon["faces"], closed=False)


def test_success_count_ignores_failed_seeds(tmp_path: Path):
    from sensor_sim.scene_gen.run import config_fingerprint, resume_manifest, sample_is_valid, stream_seed

    spec = {"mode": "random", "seed": 3, "spp": 4}
    first = stream_seed(3, 0, 0, "geometry")
    assert first == stream_seed(3, 0, 0, "geometry")
    assert first != stream_seed(3, 0, 0, "material")
    assert first != stream_seed(3, 0, 1, "geometry")
    assert not sample_is_valid(tmp_path / "missing")
    manifest = resume_manifest(tmp_path / "run_manifest.json", spec, requested=2, base_seed=3, split_mode="scene", split_seed=1, sample_root=tmp_path)
    other = dict(spec)
    other["spp"] = 8
    try:
        resume_manifest(tmp_path / "nope.json", other, requested=2, base_seed=3, split_mode=None, split_seed=0, sample_root=tmp_path)
    except RuntimeError:
        pass
    else:
        manifest_path = tmp_path / "run_manifest.json"
        manifest_path.write_text(__import__("json").dumps(manifest))
        try:
            resume_manifest(manifest_path, other, requested=2, base_seed=3, split_mode=None, split_seed=0, sample_root=tmp_path)
            raise AssertionError("a changed config resumed")
        except RuntimeError:
            pass
    assert config_fingerprint(spec) != config_fingerprint(other)


def test_nearest_surface_ignores_a_hidden_object():
    from sensor_sim.scene_gen.generate import label_positions

    front = np.array([[0.0, 0.0, 0.05], [0.02, 0.0, 0.05]])
    back = np.array([[0.0, 0.0, 0.2]])
    position = np.zeros((2, 2, 3))
    position[0, 0] = [0.0, 0.0, 0.05]
    position[0, 1] = [0.01, 0.0, 0.05]
    owners = label_positions({"front": front, "back": back}, position, threshold=0.02)
    assert owners[0, 0] == "front"
    assert owners[0, 1] == "front"
    assert owners[1, 0] == ""


def test_holdout_keys_do_not_leak_into_train(tmp_path: Path):
    records = []
    for index, family in enumerate(["sphere", "sphere", "ellipsoid", "cone", "torus", "capsule"]):
        records.append(
            {
                "scene_id": f"scene_{index:06d}",
                "objects": [
                    {
                        "instance_id": f"{family}:{index}",
                        "family": family,
                        "bsdf": {"type": "dielectric", "int_ior": 1.4 if index % 2 == 0 else 1.6},
                    }
                ],
            }
        )
    for mode in ("scene", "asset_instance", "asset_family", "material", "material_range"):
        splits = assign_splits(records, mode, seed=1)
        assert set(splits["train"]) & set(splits["test"]) == set()
        assert sum(len(rows) for rows in splits.values()) == len(records)
    from sensor_sim.scene_gen.splits import write_splits

    first = write_splits(records, "asset_family", 1, tmp_path)
    second = write_splits(records, "asset_family", 1, tmp_path)
    assert first == second
    written = (tmp_path / "train.txt").read_text().split()
    assert set(written).isdisjoint(set((tmp_path / "test.txt").read_text().split()))


def test_two_spheres_get_different_material_ids(tmp_path: Path):
    import json
    from sensor_sim.scene_gen.dataset import channels_to_array, load_sample
    from sensor_sim.scene_gen.generate import generate_sample, random_config
    from sensor_sim.film.images import read_exr

    profiles = json.loads(Path("configs/scenes/material_profiles.json").read_text())
    spec = {
        "num_objects": [2, 2],
        "assets": ["sphere"],
        "materials": ["dielectric", "pplastic"],
        "profiles": profiles,
        "resolution": [48, 48],
        "spp": 2,
        "gap": 0.02,
    }
    config = random_config(spec, seed=11)
    config["objects"][0]["bsdf"] = {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}
    config["objects"][1]["bsdf"] = {
        "type": "pplastic",
        "int_ior": 1.5,
        "ext_ior": 1.0,
        "alpha": 0.1,
        "diffuse_reflectance": 0.5,
    }
    config["run_id"] = "test-two-spheres"
    sample = generate_sample(config, tmp_path / "dataset")
    loaded = load_sample(sample)
    materials = channels_to_array(read_exr(sample / "ground_truth" / "material_ids.exr")).reshape(loaded["object_ids"].shape)
    present = set(np.unique(loaded["object_ids"]))
    assert 1 in present and 2 in present
    on_first = set(np.unique(materials[np.rint(loaded["object_ids"]) == 1]))
    on_second = set(np.unique(materials[np.rint(loaded["object_ids"]) == 2]))
    assert on_first and on_second and on_first.isdisjoint(on_second)
    meta = json.loads((sample / "metadata.json").read_text())
    assert meta["schema_version"] == 3
    assert meta["visibility_policy_version"] == 1
    assert "normal_supervision_mask" in loaded
    assert loaded["normal_supervision_mask"].shape == loaded["object_ids"].shape
    assert loaded["normal_supervision_mask"].dtype == np.bool_
    for item in meta["shape_map"]:
        if item["asset_id"] == "plane":
            continue
        image_count = int(np.count_nonzero(loaded["object_ids"] == item["object_index"]))
        assert item["object_id_pixel_count"] == image_count
        assert "visible_pixels" in item
    assert {item["instance_id"] for item in meta["shape_map"] if item["asset_id"] == "sphere"} == {
        config["objects"][0]["instance_id"],
        config["objects"][1]["instance_id"],
    }


def test_prototype_sample_loads_after_it_is_moved(tmp_path: Path):
    config = prototype_config(seed=0, resolution=(48, 48), spp=4)
    config["run_id"] = "test-prototype"
    config["objects"][0]["segments"] = 192
    config["objects"][0]["rings"] = 96
    sample = generate_sample(config, tmp_path / "dataset")
    loaded = load_sample(sample)
    assert loaded["s0"].shape == (48, 48)
    assert loaded["s0"].dtype == np.float64
    assert loaded["normal_geo_camera"].shape == (48, 48, 3)
    assert loaded["object_ids"].dtype == np.int32
    moved = tmp_path / "elsewhere" / "scene_000000"
    shutil.copytree(sample, moved)
    import mitsuba as mi

    mi.set_variant("llvm_ad_spectral_polarized")
    scene = mi.load_file(str(moved / "scene.xml"))
    assert len(scene.shapes()) >= 2
    from sensor_sim.film.images import read_exr
    from sensor_sim.scene_gen.dataset import channels_to_array

    position = channels_to_array(read_exr(sample / "ground_truth" / "position.exr"))
    depth = channels_to_array(read_exr(sample / "ground_truth" / "depth.exr")).reshape(position.shape[:2])
    object_ids = channels_to_array(read_exr(sample / "ground_truth" / "object_ids.exr")).reshape(position.shape[:2])
    origin = np.array(config["camera"]["position"], dtype=np.float64)
    distance = np.linalg.norm(position - origin, axis=-1)
    on_sphere = np.rint(object_ids) == 1
    assert int(on_sphere.sum()) > 10
    assert float(np.median(np.abs(distance - (depth + 0.01))[on_sphere])) < 5e-4
    assert float(np.median(np.abs(distance - depth)[on_sphere])) > 5e-3
    metadata = json.loads((sample / "metadata.json").read_text())
    assert metadata["addon_commit"].startswith("8848ded")
    assert metadata["stokes_frame"]["aolp_formula"].startswith("0.5 * atan2")
    assert any(item["path"] == "bsdf.mat-obj_sphere.ext_ior" for item in metadata["xml_patches"])
    again = generate_sample(config, tmp_path / "dataset")
    assert again == sample
    assert loaded["material_ids"].shape == loaded["object_ids"].shape
    assert loaded["geometry_mask"].dtype == bool
    assert "git_dirty" in metadata
    assert "render_seed" in metadata


def test_one_seed_rebuilds_the_same_mesh():
    profiles = json.loads(Path("configs/scenes/material_profiles.json").read_text())
    spec = {
        "num_objects": [1, 1],
        "assets": ["sphere"],
        "materials": ["dielectric"],
        "profiles": profiles,
        "resolution": [32, 32],
        "spp": 1,
    }
    first = random_config(spec, seed=5)
    second = random_config(spec, seed=5)
    assert np.allclose(first["objects"][0]["vertices"], second["objects"][0]["vertices"])
    assert first["objects"][0]["instance_id"] == second["objects"][0]["instance_id"]
    assert first["geometry_seed"] == second["geometry_seed"]


def test_primary_sampler_draws_only_measured_presets():
    from sensor_sim.scene_gen.materials import primary_catalog, sample_primary

    catalog = primary_catalog()
    assert {name for name, _preset in catalog} == {"dielectric", "pplastic", "conductor", "roughconductor"}
    iors = sorted(preset["int_ior"] for name, preset in catalog if name == "dielectric")
    assert iors == [1.33, 1.4, 1.5, 1.6, 1.7]
    rough = [preset for name, preset in catalog if name == "roughconductor"]
    assert {preset["material"] for preset in rough} == {"Cu", "Au"}
    assert {preset["alpha"] for preset in rough} == {0.1}
    rng = np.random.default_rng(4)
    for _draw in range(30):
        name, preset = sample_primary(rng)
        assert (name, preset) in catalog


def test_oversized_footprints_fail_before_dropping_an_object():
    huge = build_asset("sphere", np.random.default_rng(0), radius=0.12)[0]
    try:
        place_objects([huge, huge, huge], np.random.default_rng(1), CAMERA, seed=9)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("three large spheres were placed")
    assert "cannot_fit" in message
    assert "dropped" not in message


def test_ray_reference_prefers_the_front_surface():
    from sensor_sim.scene_gen.identity import ray_triangle_ids

    front = {
        "vertices": np.array([[-1.0, -1.0, 0.2], [1.0, -1.0, 0.2], [0.0, 1.0, 0.2]]),
        "faces": np.array([[0, 1, 2]]),
        "object_index": 1,
    }
    back = {
        "vertices": np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
        "faces": np.array([[0, 1, 2]]),
        "object_index": 2,
    }
    ids = ray_triangle_ids(np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]]), [back, front])
    assert int(ids[0]) == 1
    miss = ray_triangle_ids(np.array([[5.0, 5.0, 1.0]]), np.array([[0.0, 0.0, -1.0]]), [front])
    assert int(miss[0]) == 0


def test_shared_asset_instance_stays_in_one_split():
    shared = {"instance_id": "sphere:same", "family": "sphere", "bsdf": {"type": "dielectric", "int_ior": 1.33}}
    records = [
        {"scene_id": "scene_a", "objects": [shared]},
        {"scene_id": "scene_b", "objects": [dict(shared)]},
    ]
    for index, family in enumerate(("ellipsoid", "capsule", "torus", "cylinder", "cone")):
        records.append(
            {
                "scene_id": f"scene_{index}",
                "objects": [{"instance_id": f"{family}:only", "family": family, "bsdf": {"type": "dielectric", "int_ior": 1.7}}],
            }
        )
    assigned = assign_splits(records, "asset_instance", seed=0)
    home = next(name for name, ids in assigned.items() if "scene_a" in ids)
    assert "scene_b" in assigned[home]

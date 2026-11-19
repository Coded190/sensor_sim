"""Run management, Stokes conventions, and packing gates. Renders live elsewhere."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sensor_sim.scene_gen.assets import build_asset
from sensor_sim.scene_gen.generate import _reject_nonuniform_scale
from sensor_sim.scene_gen.placement import place_objects
from sensor_sim.scene_gen.run import expand_schedule, new_manifest, resume_manifest, sample_id
from sensor_sim.stokes import polarization_maps

CAMERA = {
    "position": [0.0, -0.55, 0.32],
    "target": [0.0, 0.0, 0.18],
    "up": [0.0, 0.0, 1.0],
    "fov_deg": 55.0,
}


def test_unknown_preset_fails_and_a_pinned_preset_is_not_random():
    from sensor_sim.scene_gen.materials import primary_catalog, require_primary_preset, sample_primary

    with pytest.raises(RuntimeError, match="unknown primary preset"):
        require_primary_preset({"type": "dielectric", "int_ior": 1.2, "ext_ior": 1.0})
    pinned = {"type": "dielectric", "int_ior": 1.33, "ext_ior": 1.0}
    name, row = sample_primary(np.random.default_rng(0), families=["dielectric"], preset=pinned)
    assert name == "dielectric"
    assert row == pinned
    assert any(item.get("material") == "Au" for _name, item in primary_catalog())
    with pytest.raises(RuntimeError, match="not in"):
        sample_primary(np.random.default_rng(0), families=["dielectric"], preset={"type": "conductor", "material": "Cu"})


def test_production_count_override_does_not_reach_export(tmp_path: Path, monkeypatch):
    from sensor_sim.scene_gen.cli import main

    calls = {"n": 0}

    def explode(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("export started")

    monkeypatch.setattr("sensor_sim.scene_gen.generate.export_scene", explode)
    with pytest.raises(RuntimeError, match="before export"):
        main(["--config", "configs/scenes/dataset_v1.yaml", "--num-scenes", "20", "--spp", "16", "--output", str(tmp_path / "samples")])
    assert calls["n"] == 0


def test_publish_requires_a_run_id():
    from sensor_sim.scene_gen.generate import generate_sample

    with pytest.raises(RuntimeError, match="run_id"):
        generate_sample({"scene_id": "scene_x"}, Path("/tmp/unused"))


def test_config_view_preserves_seed_and_resolution():
    from sensor_sim.scene_gen.config_view import config_view, config_view_hash

    spec = {"assets": ["sphere", "cone"], "seed": 5, "resolution": [128, 128], "spp": 16, "materials": ["dielectric"], "gap": 0.005}
    view = config_view(spec)
    assert view["assets"]["families"] == ["sphere", "cone"]
    assert view["scene"]["seed"] == 5
    assert view["sensor"]["resolution"] == [128, 128]
    assert config_view_hash(spec) == config_view_hash(dict(spec))


def test_closed_mesh_inside_another_is_not_separated():
    from sensor_sim.scene_gen.narrow import narrow_phase

    def sphere(radius, center):
        mesh, _spec = build_asset("sphere", np.random.default_rng(0), radius=radius)
        vertices = mesh["vertices"].copy()
        vertices += np.asarray(center) - vertices.mean(axis=0)
        return {"vertices": vertices, "faces": mesh["faces"]}

    outer = sphere(0.08, (0.0, 0.0, 0.08))
    inner = sphere(0.02, (0.0, 0.0, 0.08))
    report = narrow_phase(inner, outer, 0.005)
    assert report["contained"]
    assert not report["separated"]


def test_tiny_and_bad_normals_are_rejected():
    from sensor_sim.scene_gen.meshcheck import mesh_issues

    vertices = np.array([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [0.0, 1e-6, 0.0]])
    faces = np.array([[0, 1, 2]])
    assert "tiny" in mesh_issues(vertices, faces, closed=False)
    normals = np.zeros((3, 3))
    assert "zero_normals" in mesh_issues(np.eye(3), faces, closed=False, normals=normals)


def test_schedule_retries_the_same_stratum_slot():
    strata = [{"family": "sphere", "material": "dielectric", "count": 1}, {"family": "cone", "material": "conductor", "count": 1}]
    slots = expand_schedule(strata, 2)
    assert slots[0]["family"] == "sphere"
    assert slots[1]["material"] == "conductor"
    with pytest.raises(RuntimeError):
        expand_schedule(strata, 3)


def test_two_runs_with_the_same_base_seed_do_not_share_ids():
    spec = {"mode": "random", "seed": 100}
    first = new_manifest(spec, requested=1, base_seed=100, split_mode=None, split_seed=0)
    second = new_manifest(spec, requested=1, base_seed=100, split_mode=None, split_seed=0)
    assert first["run_id"] != second["run_id"]
    assert sample_id(first["run_id"], 0) != sample_id(second["run_id"], 0)


def test_resume_rejects_a_different_config_and_a_corrupt_sample(tmp_path: Path):
    spec = {"spp": 4}
    manifest = new_manifest(spec, requested=1, base_seed=1, split_mode=None, split_seed=0)
    sample = tmp_path / "samples" / sample_id(manifest["run_id"], 0)
    sample.mkdir(parents=True)
    (sample / "metadata.json").write_text(json.dumps({"status": "complete", "config_hash": manifest["config_hash"]}))
    manifest["completed_ids"] = [sample.name]
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    resumed = resume_manifest(path, spec, requested=1, base_seed=1, split_mode=None, split_seed=0, sample_root=tmp_path / "samples")
    assert resumed["completed_ids"] == []
    other = dict(spec)
    other["spp"] = 8
    with pytest.raises(RuntimeError):
        resume_manifest(path, other, requested=1, base_seed=1, split_mode=None, split_seed=0, sample_root=tmp_path / "samples")


def test_injected_failures_publish_only_successes_and_stop_at_the_budget(tmp_path: Path, monkeypatch):
    from sensor_sim.scene_gen.cli import main

    config = tmp_path / "scene.yaml"
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"dielectric": {"response_class": "keeps"}}))
    config.write_text(
        "mode: random\nseed: 3\nnum_objects: [1, 1]\nassets: [sphere]\nmaterials: [dielectric]\n"
        f"profile_file: {profiles}\nresolution: [32, 32]\nspp: 1\n"
    )
    calls = {"n": 0}

    def fake_generate(built, output):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected failure")
        dest = Path(output) / built["scene_id"]
        if dest.exists():
            raise RuntimeError("duplicate publication")
        dest.mkdir(parents=True)
        (dest / "metadata.json").write_text(
            json.dumps({"status": "complete", "scene_id": built["scene_id"], "config_hash": built.get("config_hash"), "objects": []})
        )
        return dest

    monkeypatch.setattr("sensor_sim.scene_gen.cli.generate_sample", fake_generate)
    monkeypatch.setattr("sensor_sim.scene_gen.cli.sample_is_valid", lambda path, expected=None: True)
    out = tmp_path / "out" / "samples"
    main(["--config", str(config), "--num-scenes", "1", "--max-attempts", "3", "--output", str(out), "--split-mode", "scene"])
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["completed_ids"]) == 1
    assert len(list((tmp_path / "out" / "failures").glob("*.json"))) == 1
    failure = json.loads(next((tmp_path / "out" / "failures").glob("*.json")).read_text())
    for key in ("run_id", "sample_slot", "attempt_index", "scene_seed", "geometry_seed", "material_seed", "render_seed", "exception_type", "exception_message"):
        assert key in failure
    stray = out / "scene_stray"
    stray.mkdir()
    (stray / "metadata.json").write_text("{}\n")
    assert "scene_stray" not in manifest["completed_ids"]
    calls["n"] = 0

    def always_fail(built, output):
        raise RuntimeError("budget")

    monkeypatch.setattr("sensor_sim.scene_gen.cli.generate_sample", always_fail)
    with pytest.raises(SystemExit):
        main(["--config", str(config), "--num-scenes", "1", "--max-attempts", "1", "--output", str(tmp_path / "budget" / "samples"), "--split-mode", "scene"])
    assert not (tmp_path / "budget" / "splits").exists()


def test_nonuniform_scale_is_rejected_and_impossible_pack_is_cannot_fit():
    with pytest.raises(RuntimeError, match="nonuniform"):
        _reject_nonuniform_scale({"objects": [{"name": "a", "scale": [1.0, 2.0, 1.0]}]})
    huge = build_asset("sphere", np.random.default_rng(0), radius=0.12)[0]
    with pytest.raises(RuntimeError, match="cannot_fit"):
        place_objects([huge, huge, huge], np.random.default_rng(1), CAMERA, seed=4)


def test_analyzer_identities_and_signed_stokes_are_not_clamped():
    s0 = np.array([[1.0, 0.0], [0.2, 0.4]])
    s1 = np.array([[0.0, 0.0], [-0.1, 0.2]])
    s2 = np.array([[0.0, 0.0], [0.0, -0.1]])
    s3 = np.array([[0.0, 0.0], [0.05, -0.2]])
    maps = polarization_maps(s0, s1, s2, s3)
    assert maps["s1"][1, 0] == pytest.approx(-0.1)
    assert maps["s3"][1, 1] == pytest.approx(-0.2)
    assert maps["dolp"][0, 1] == 0.0
    theta = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    intensity = 0.5 * (0.4 + 0.2 * np.cos(2 * theta) + -0.1 * np.sin(2 * theta))
    named = np.array(
        [
            0.5 * (0.4 + 0.2),
            0.5 * (0.4 + -0.1),
            0.5 * (0.4 - 0.2),
            0.5 * (0.4 - -0.1),
        ]
    )
    assert intensity == pytest.approx(named)
    # A 180 degree analyzer rotation does not change the linear intensity.
    theta = 0.3
    forward = 0.5 * (0.4 + 0.2 * np.cos(2 * theta) + -0.1 * np.sin(2 * theta))
    rotated = 0.5 * (0.4 + 0.2 * np.cos(2 * (theta + np.pi)) + -0.1 * np.sin(2 * (theta + np.pi)))
    assert forward == pytest.approx(rotated)


def test_four_identical_dielectrics_keep_distinct_shape_ids(tmp_path: Path):
    import mitsuba as mi

    from sensor_sim.scene_gen.xml_patch import patch_xml

    shapes = []
    for index, radius in enumerate((0.2, 0.25, 0.3, 0.35)):
        shapes.append(
            f'<bsdf type="dielectric" id="mat-obj_obj_{index:02d}"><float name="int_ior" value="1.5"/><float name="ext_ior" value="1.0"/></bsdf>'
            f'<shape type="sphere" id="mesh-obj_{index:02d}"><float name="radius" value="{radius}"/><ref id="mat-obj_obj_{index:02d}" name="bsdf"/></shape>'
        )
    source = tmp_path / "exported.xml"
    source.write_text('<scene version="3.9.1"><integrator type="path"/>' + "".join(shapes) + "</scene>")
    patch_xml(source, tmp_path / "scene.xml", {}, 4)
    mi.set_variant("llvm_ad_spectral_polarized")
    scene = mi.load_file(str(tmp_path / "scene.xml"))
    ids = sorted(shape.id() for shape in scene.shapes())
    assert ids == [f"mesh-obj_{index:02d}" for index in range(4)]


def test_identical_bsdfs_are_not_merged_into_one_shape(tmp_path: Path):
    import mitsuba as mi

    from sensor_sim.scene_gen.xml_patch import patch_xml

    source = tmp_path / "exported.xml"
    source.write_text(
        """<scene version="3.9.1">
        <integrator type="path"/>
        <bsdf type="diffuse" id="mat-obj_a"><rgb name="reflectance" value="0.5 0.5 0.5"/></bsdf>
        <bsdf type="diffuse" id="mat-obj_b"><rgb name="reflectance" value="0.5 0.5 0.5"/></bsdf>
        <shape type="sphere" id="mesh-a" name="mesh-a"><float name="radius" value="0.2"/><ref id="mat-obj_a" name="bsdf"/></shape>
        <shape type="sphere" id="mesh-b" name="mesh-b"><float name="radius" value="0.3"/><ref id="mat-obj_b" name="bsdf"/></shape>
        </scene>
        """
    )
    patch_xml(source, tmp_path / "scene.xml", {}, 4)
    mi.set_variant("llvm_ad_spectral_polarized")
    scene = mi.load_file(str(tmp_path / "scene.xml"))
    ids = sorted(shape.id() for shape in scene.shapes())
    assert ids == ["mesh-a", "mesh-b"]


def test_shape_index_does_not_identify_a_mitsuba_instance():
    import mitsuba as mi

    mi.set_variant("llvm_ad_spectral_polarized")
    scene = mi.load_dict(
        {
            "type": "scene",
            "group": {
                "type": "shapegroup",
                "id": "group",
                "sphere": {"type": "sphere", "radius": 0.2, "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.2, 0.7, 0.2]}}},
            },
            "a": {"type": "instance", "id": "inst-a", "shape": {"type": "ref", "id": "group"}, "to_world": mi.ScalarTransform4f.translate([-0.6, 0, 0])},
            "b": {"type": "instance", "id": "inst-b", "shape": {"type": "ref", "id": "group"}, "to_world": mi.ScalarTransform4f.translate([0.6, 0, 0])},
        }
    )
    hits = []
    for origin in ([-0.6, -2, 0], [0.6, -2, 0]):
        si = scene.ray_intersect(mi.Ray3f(origin, [0, 1, 0]))
        hits.append(int(np.array(si.prim_index).reshape(-1)[0]))
        assert si.instance is not None
    assert hits[0] == hits[1]


def _triangle(vertices: np.ndarray) -> dict:
    return {"vertices": np.asarray(vertices, dtype=np.float64), "faces": np.array([[0, 1, 2]])}


def test_supervision_mask_excludes_low_visibility_and_keeps_eligible_objects():
    from sensor_sim.scene_gen.dataset import normal_supervision_mask, visibility_category

    ids = np.zeros((10, 20), dtype=np.int32)
    ids[:, :8] = 1
    ids[:, 8:10] = 2
    normals = np.zeros((10, 20, 3), dtype=np.float64)
    normals[..., 2] = 1.0
    sample = {"object_ids": ids, "normal_shading_camera": normals}
    mask = normal_supervision_mask(sample)
    assert mask[:, :8].all()
    assert not mask[:, 8:].any()
    assert visibility_category(0) == "fully_occluded"
    assert visibility_category(73) == "low_visibility"
    assert visibility_category(80) == "training_eligible"
    legacy = normal_supervision_mask(
        sample,
        {"schema_version": 2, "shape_map": [{"object_index": 1, "object_id": "object_00", "visible_pixels": 10, "asset_id": "sphere"}]},
    )
    assert np.array_equal(legacy, mask)
    with pytest.raises(RuntimeError, match="object_id_pixel_count"):
        normal_supervision_mask(
            sample,
            {
                "schema_version": 3,
                "shape_map": [{"object_index": 1, "object_id": "object_00", "object_id_pixel_count": 10, "asset_id": "sphere"}],
            },
        )


def test_supervision_loss_ignores_an_ineligible_object():
    """A 36-pixel object cannot contribute, and a 320-pixel object can."""
    ids = np.zeros((20, 20), dtype=np.int32)
    ids.reshape(-1)[:320] = 1
    ids.reshape(-1)[320:356] = 2
    normals = np.zeros(ids.shape + (3,), dtype=np.float64)
    normals[..., 2] = 1.0
    from sensor_sim.scene_gen.dataset import normal_supervision_mask

    mask = normal_supervision_mask({"object_ids": ids, "normal_shading_camera": normals})
    loss = np.ones(ids.shape, dtype=np.float64)
    weighted = loss * mask
    assert weighted[ids == 2].sum() == 0
    assert weighted[ids == 1].sum() == 320
    assert weighted[ids == 0].sum() == 0


def test_schedule_rejects_bad_slots_and_changes_its_hash(tmp_path: Path):
    from sensor_sim.scene_gen.production_schedule import recommended_slots, schedule_hash, validate_slots
    from sensor_sim.scene_gen.run import expand_schedule, new_manifest, resume_manifest, save_manifest

    slots = recommended_slots()
    margins = validate_slots(slots)
    assert margins["splits"] == {"train": 48, "validation": 8, "test": 8}
    expanded = expand_schedule(slots, 64)
    digest = schedule_hash(expanded)
    changed = json.loads(json.dumps(expanded))
    changed[0]["objects"][0]["radius"] = 0.011
    assert schedule_hash(changed) != digest
    broken = json.loads(json.dumps(slots))
    broken[0]["preset"] = {"type": "dielectric", "int_ior": 1.2, "ext_ior": 1.0}
    with pytest.raises(RuntimeError, match="unknown primary preset"):
        validate_slots(broken)
    gold = json.loads(json.dumps(slots))
    gold[0]["material"] = "roughconductor"
    gold[0]["preset"] = {"type": "roughconductor", "material": "Au", "distribution": "beckmann", "alpha": 0.1}
    with pytest.raises(RuntimeError, match="gold"):
        validate_slots(gold)
    counted = json.loads(json.dumps(slots))
    counted[0]["count_objects"] = 5
    counted[0]["objects"] = [{"family": "sphere", "radius": 0.02}] * 5
    with pytest.raises(RuntimeError, match="outside 1-4"):
        validate_slots(counted)
    duplicate = json.loads(json.dumps(slots))
    duplicate[1]["slot_id"] = duplicate[0]["slot_id"]
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_slots(duplicate)
    spec = {"seed": 17011, "strata": slots, "schedule_hash": digest}
    manifest = new_manifest(spec, requested=64, base_seed=17011, split_mode=None, split_seed=17011)
    path = tmp_path / "run_manifest.json"
    save_manifest(path, manifest)
    resumed = resume_manifest(path, spec, requested=64, base_seed=17011, split_mode=None, split_seed=17011, sample_root=tmp_path)
    assert resumed["config_hash"] == manifest["config_hash"]
    moved = json.loads(json.dumps(spec))
    moved["strata"][0]["objects"][0]["radius"] = 0.011
    moved["schedule_hash"] = schedule_hash(expand_schedule(moved["strata"], 64))
    with pytest.raises(RuntimeError, match="config hash changed"):
        resume_manifest(path, moved, requested=64, base_seed=17011, split_mode=None, split_seed=17011, sample_root=tmp_path)


def test_mesh_clearance_matches_a_known_parallel_gap():
    from sensor_sim.scene_gen.narrow import mesh_clearance

    first = {"vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), "faces": np.array([[0, 1, 2]])}
    second = {"vertices": np.array([[0.0, 0.0, 0.006], [1.0, 0.0, 0.006], [0.0, 1.0, 0.006]]), "faces": np.array([[0, 1, 2]])}
    independent = float(abs(first["vertices"][0, 2] - second["vertices"][0, 2]))
    assert mesh_clearance(first, second)["clearance_m"] == pytest.approx(independent)


def test_reference_rays_pick_the_nearest_mesh_and_ignore_a_miss():
    from sensor_sim.scene_gen.reference_rays import reference_ray_ids

    near = {
        "vertices": np.array([[-0.1, 0.0, 0.9], [0.1, 0.0, 0.9], [0.0, 0.1, 0.9]]),
        "faces": np.array([[0, 1, 2]]),
        "object_index": 2,
    }
    far = {
        "vertices": np.array([[-0.1, 0.0, 2.0], [0.1, 0.0, 2.0], [0.0, 0.1, 2.0]]),
        "faces": np.array([[0, 1, 2]]),
        "object_index": 4,
    }
    origins = np.array([[0.0, 0.02, 0.0], [0.5, 0.5, 0.0]])
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    assert reference_ray_ids(origins, directions, [far, near]).tolist() == [2, 0]


def test_triangle_queries_catch_penetration_clearance_and_floor_contact():
    from sensor_sim.scene_gen.narrow import floor_status, narrow_phase

    base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    piercing = np.array([[0.1, 0.1, -1.0], [0.1, 0.1, 1.0], [0.2, 0.2, 1.0]])
    hit = narrow_phase(_triangle(base), _triangle(piercing), 0.005)
    assert hit["surface_intersection"]
    assert not hit["separated"]
    legal = base.copy()
    legal[:, 2] = 0.005
    gap = narrow_phase(_triangle(base), _triangle(legal), 0.005)
    assert gap["clearance_m"] == pytest.approx(0.005)
    assert gap["separated"]
    close = base.copy()
    close[:, 2] = 0.004
    tight = narrow_phase(_triangle(base), _triangle(close), 0.005)
    assert tight["clearance_m"] == pytest.approx(0.004)
    assert not tight["separated"]
    plate = np.array([[0.4, 0.05, 0.008], [0.8, 0.05, 0.008], [0.6, 0.15, 0.008]])
    wall = {
        "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.4, 0.0], [0.0, 0.0, 0.03]]),
        "faces": np.array([[0, 1, 2], [0, 1, 3]]),
    }
    boxes = narrow_phase(wall, _triangle(plate), 0.005)
    assert boxes["aabb_overlap"]
    assert boxes["separated"]
    assert boxes["clearance_m"] == pytest.approx(0.008)
    contact = np.array([[0.0, 0.0, 0.001], [0.02, 0.0, 0.001], [0.0, 0.02, 0.001]])
    assert floor_status(contact)["permitted_contact"]
    sunk = contact.copy()
    sunk[:, 2] = 0.0
    assert floor_status(sunk)["penetrating"]


def test_failed_staging_never_enters_completed_ids(tmp_path: Path):
    from sensor_sim.scene_gen.dataset import publish
    from sensor_sim.scene_gen.provenance import admit_completed, publication_issues

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "metadata.json").write_text("{}\n")
    destination = tmp_path / "published" / "scene_a"
    with pytest.raises(RuntimeError, match="publication audit"):
        publish(staging, destination)
    assert not destination.exists()
    sample = tmp_path / "samples" / "scene_a"
    sample.mkdir(parents=True)
    (sample / "scene.xml").write_text("<scene><shape id='mesh-ball'/></scene>\n")
    (sample / "metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "run_id": "run",
                "config_hash": "abc",
                "scene_id": "scene_a",
                "logical_shape_ids": ["mesh-ball"],
                "artifact_sha256": {"scene.xml": "0" * 64},
                "objects": [{"name": "ball", "material_id": 1}],
                "shape_map": [{"object_id": "ball", "material_id": 2}],
            }
        )
    )
    issues = publication_issues(sample)
    assert any("hash mismatch" in item for item in issues)
    assert any("material id" in item for item in issues)
    manifest = {"completed_ids": []}
    assert admit_completed(manifest, tmp_path / "samples", "scene_a") is False
    assert manifest["completed_ids"] == []

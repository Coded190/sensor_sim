"""Material specification, independent seeds, and shell geometry checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sensor_sim.scene_gen.material_assign import replay_spec, sample_family
from sensor_sim.scene_gen.material_spec import MaterialSpecError, make_spec, preset, to_mitsuba
from sensor_sim.scene_gen.plastic_wrap import make_wrap_spec, mesh_hash, shell_report, store_result
from sensor_sim.scene_gen.structural import box_mesh
from tests.test_semantic_scenes import DATASET_V1, LOCKED

INDEX_SHA256 = "081bf07291a4cc3f8f2af1fe27b8f8beb0164153100244aefa25df1ba209c157"


def _box(minimum, maximum):
    vertices, faces = box_mesh(minimum, maximum)
    return {"vertices": vertices, "faces": faces}


def _shell():
    outer = _box([-0.51, -0.51, -0.51], [0.51, 0.51, 0.51])
    inner_vertices, inner_faces = box_mesh([-0.50, -0.50, -0.50], [0.50, 0.50, 0.50])
    inner_faces = [list(reversed(face)) for face in inner_faces]
    vertices = np.vstack([outer["vertices"], inner_vertices])
    faces = outer["faces"] + [[index + 8 for index in face] for face in inner_faces]
    target = _box([-0.40, -0.40, -0.40], [0.40, 0.40, 0.40])
    return {"vertices": vertices, "faces": faces}, target


def test_family_rejects_metallic_on_a_dielectric():
    with pytest.raises(MaterialSpecError, match="family_field"):
        make_spec("bad", "dielectric", {"int_ior": 1.5, "ext_ior": 1.0, "metallic": 0.2})


def test_mapper_does_not_substitute_a_family():
    spec = preset("principled_plastic")
    mapped = to_mitsuba(spec)
    assert mapped["type"] == "principled"
    with pytest.raises(MaterialSpecError):
        to_mitsuba(make_spec("nope", "diffuse", {"reflectance": [1.2, 0.0, 0.0]}))


def test_material_seed_changes_parameters_and_not_geometry():
    target = _box([-0.1, -0.1, 0.0], [0.1, 0.1, 0.2])
    before = mesh_hash(target["vertices"], np.asarray(target["faces"]))
    first = sample_family("pplastic", 21, "cup")
    second = sample_family("pplastic", 22, "cup")
    assert first["parameters"] != second["parameters"]
    assert mesh_hash(target["vertices"], np.asarray(target["faces"])) == before
    assert replay_spec(first)["parameters"] == first["parameters"]


def test_wrap_seed_does_not_change_the_target_spec():
    left = make_wrap_spec("shell", "box", "partial", offset_m=0.004, thickness_m=0.002, seed=31)
    right = make_wrap_spec("shell", "box", "partial", offset_m=0.004, thickness_m=0.002, seed=32)
    assert left["seed"] != right["seed"]
    assert left["target_id"] == right["target_id"]
    assert left["interfaces"]["outer"]["ext_ior"] == 1.0
    assert left["interfaces"]["inner"]["status"] == "not_validated_opaque_substrate"
    assert left["interfaces"]["sidewall"]["status"] == "assigned_partial_boundary"


def test_synthetic_shell_has_two_interfaces_and_keeps_clearance():
    shell, target = _shell()
    spec = make_wrap_spec("synthetic", "target", "full", offset_m=0.10, thickness_m=0.01, seed=1, substrate_family="diffuse")
    report = shell_report(shell, target, spec)
    assert report["passed_geometry"] is True
    assert report["counts"]["outer"] > 0 and report["counts"]["inner"] > 0
    assert report["minimum_clearance_m"] > 0.05
    assert report["optical_film_to_substrate"] == "not_validated"


def test_failed_wrap_does_not_replace_a_valid_sample(tmp_path: Path):
    assert store_result(tmp_path, {"shell": "first"}, True) is True
    assert store_result(tmp_path, {"shell": "failed"}, False) is False
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["shell"] == "first"


def test_previous_qualification_hashes_remain():
    for relative, digest in LOCKED.items():
        if relative.endswith(".py"):
            assert hashlib.sha256(Path("src/sensor_sim/scene_gen", relative).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(Path("assets/candidates/index.json").read_bytes()).hexdigest() == INDEX_SHA256
    dataset = Path("dataset/dataset_v1")
    files = [path for path in dataset.rglob("*") if path.is_file()]
    assert len(files) == DATASET_V1["files"]
    assert sum(path.stat().st_size for path in files) == DATASET_V1["bytes"]

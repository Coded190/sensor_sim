"""Shape identity and Stokes checks on the installed Mitsuba 3.9.1 build."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.indoor_mitsuba import (
    boundary_report,
    grid_shapes,
    identity_buffer,
    identity_inventory,
    load_xml_shapes,
    optical_control_difference,
    render_stokes,
    shared_bsdf_xml,
    stokes_scene,
)
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES


def _diffuse(value: float) -> dict:
    return {"type": "diffuse", "reflectance": {"type": "rgb", "value": [value, value, value]}}


def test_identical_materials_merge_and_identity_does_not():
    shapes = grid_shapes(10)
    report = boundary_report(shapes)
    assert report["scene_dict_shapes"] == 10
    assert report["loaded_shapes"] == 1
    assert report["loaded_ids"] == [""]
    assert report["identity_shapes"] == 10
    assert report["loaded_vertex_count"] == 240
    slots = {(item["logical_object"], item["component"], item["material_slot"], item["instance"]) for item in report["identity"]}
    assert len(slots) == 10
    optical = optical_control_difference(shapes[:4], spp=4, seed=19)
    assert optical["within_repeatability"]
    assert optical["repeatability"]["variant"] == "cuda_ad_spectral_polarized"


def test_eight_identity_cases(tmp_path: Path):
    paths = shared_bsdf_xml(tmp_path)
    shared = load_xml_shapes(paths["shared_xml"])
    separate = load_xml_shapes(paths["separate_xml"])
    assert len(shared) == 1 and shared[0]["id"] == ""
    assert len(separate) == 1 and separate[0]["id"] == ""
    multi = [
        {"shape_id": "body", "logical_object": "desk", "component": "top", "material_slot": "wood", "instance": "0", "center": [0.0, 0.0, 0.2], "scale": [0.2, 0.1, 0.02], "bsdf": _diffuse(0.4)},
        {"shape_id": "leg", "logical_object": "desk", "component": "leg", "material_slot": "wood", "instance": "0", "center": [0.0, 0.0, 0.08], "scale": [0.02, 0.02, 0.08], "bsdf": _diffuse(0.4)},
    ]
    multi_report = boundary_report(multi)
    assert multi_report["identity_shapes"] == 2
    assert {item["component"] for item in multi_report["identity"]} == {"top", "leg"}
    slots = [
        {"shape_id": "seat", "logical_object": "chair", "component": "seat", "material_slot": "fabric", "instance": "0", "center": [0.0, 0.0, 0.4], "scale": [0.15, 0.15, 0.02], "bsdf": _diffuse(0.2)},
        {"shape_id": "frame", "logical_object": "chair", "component": "frame", "material_slot": "metal", "instance": "0", "center": [0.0, 0.0, 0.15], "scale": [0.02, 0.02, 0.15], "bsdf": _diffuse(0.8)},
    ]
    slot_report = boundary_report(slots)
    assert slot_report["loaded_shapes"] == 2
    assert {item["material_slot"] for item in slot_report["identity"]} == {"fabric", "metal"}
    instances = grid_shapes(2)
    instances[0]["logical_object"] = "cup"
    instances[1]["logical_object"] = "cup"
    instances[0]["instance"] = "a"
    instances[1]["instance"] = "b"
    instance_report = boundary_report(instances)
    assert instance_report["identity_shapes"] == 2
    assert {item["instance"] for item in instance_report["identity"]} == {"a", "b"}
    for count in (10, 60):
        shapes = grid_shapes(count)
        report = boundary_report(shapes)
        assert report["loaded_shapes"] == 1
        assert report["identity_shapes"] == count
        image = identity_buffer(shapes, width=36, height=36)
        visible = set(int(value) for value in np.unique(image) if int(value) != 0)
        assert visible == set(range(1, count + 1))
        assert image.dtype == np.int32


def test_named_stokes_exr(tmp_path: Path):
    report = render_stokes(stokes_scene((32, 32)), tmp_path / "stokes.exr", spp=16, seed=7)
    assert report["channel_names"] == list(CHANNEL_NAMES)
    assert report["exr_names"] == list(CHANNEL_NAMES)
    assert report["aov_names"] == list(CHANNEL_NAMES[3:])
    assert report["dtype"] == "float32"
    assert report["shape"] == [32, 32, 15]
    assert report["s3_sign_independently_remeasured"] is False
    assert report["channel_mean"]["S0.R"] > 0.0
    assert abs(report["channel_mean"]["S1.R"]) > 1e-4
    assert "ch00" not in report["channel_names"]

"""Checks for the Infinigen-to-Mitsuba validation milestone."""

from __future__ import annotations

import numpy as np
import pytest

from sensor_sim.scene_gen.geometry_compare import comparison_report
from sensor_sim.scene_gen.infinigen_ingest import measure_semantic_frame
from sensor_sim.scene_gen.light_units import area_radiance_from_watts, point_intensity_from_watts, polarization_role
from sensor_sim.scene_gen.indoor_contact import measure_support


def _triangle(a, b, c):
    return [(0, a), (1, b), (2, c)]


def _mesh(vertices, faces):
    # faces are lists of local indices into vertices; stored as corners with vertex index only
    return {"vertices": vertices, "corners": [[(index,) for index in face] for face in faces]}


def test_sofa_forward_points_from_backrest_to_seat():
    vertices = [
        [-0.4, 0.3, 0.45], [0.4, 0.3, 0.45], [0.4, 0.9, 0.45], [-0.4, 0.9, 0.45],
        [-0.4, 0.05, 0.45], [0.4, 0.05, 0.45], [0.4, 0.05, 1.1], [-0.4, 0.05, 1.1],
    ]
    faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
    frame = measure_semantic_frame(_mesh(vertices, faces), "sofa")
    assert frame["status"] == "measured"
    assert frame["validation"] == "backrest_to_seat"
    assert frame["forward"][1] > 0.9
    assert abs(frame["forward"][2]) < 1e-6
    right = np.cross(frame["forward"], frame["up"])
    assert np.dot(right, frame["right"]) > 0.999


def test_ambiguous_mesh_stays_unknown():
    frame = measure_semantic_frame(_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]), "sofa")
    assert frame["status"] == "unknown"


def test_point_and_area_conversion_use_watts():
    assert point_intensity_from_watts(4.0 * np.pi) == pytest.approx(1.0)
    assert area_radiance_from_watts(np.pi, 1.0) == pytest.approx(1.0)
    assert point_intensity_from_watts(10.0) != pytest.approx(10.0)


def test_diffuse_is_not_called_polarization_accurate():
    role = polarization_role("diffuse")
    assert role["polarization"] == "depolarizing"
    assert role["can_polarize"] is False
    assert polarization_role("dielectric")["can_polarize"] is True


def test_shifted_mesh_fails_the_stated_tolerance():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    same = comparison_report(source, source.copy())
    assert same["passed"] is True
    moved = comparison_report(source, source + np.array([0.0, 0.0, 0.5]))
    assert moved["passed"] is False
    assert moved["max_m"] == pytest.approx(0.5)


def test_missing_material_spec_fails_the_full_scene(tmp_path: Path):
    from sensor_sim.scene_gen.mitsuba_scene_package import build_scene

    mesh = tmp_path / "a.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    manifest = {
        "schema": "infinigen_mitsuba_scene_v1",
        "camera": {"fov_deg": 40, "origin": [0, -2, 1], "target": [0, 0, 0], "width": 8, "height": 8},
        "lights": [],
        "assets": [{"id": "a", "mesh": "a.obj", "regions": [{"slot": "missing", "mesh": "a.obj", "spec": {}}], "split_meshes": False}],
        "instances": [{"id": "a", "asset_id": "a", "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "regions": [{"slot": "missing", "mesh": "a.obj", "spec": {}}]}],
    }
    with pytest.raises(RuntimeError, match="missing_material_spec"):
        build_scene(manifest, tmp_path)


def test_inventory_requires_an_exclusion_reason():
    from sensor_sim.scene_gen.mitsuba_scene_package import scene_inventory

    with pytest.raises(RuntimeError, match="exclusion_without_cause"):
        scene_inventory({"instances": [{"id": "scatter", "category": "NatureShelfTrinkets", "excluded": True}], "lights": []})


def test_separation_report_uses_triangles_after_the_box_test():
    from sensor_sim.scene_gen.placement_separation import separation_report

    clear = separation_report([
        {"instance_id": "a", "vertices": np.array([[0, 0, 0], [1, 1, 1]]), "faces": np.array([[0, 0, 0]])},
        {"instance_id": "b", "vertices": np.array([[3, 0, 0], [4, 1, 1]]), "faces": np.array([[0, 0, 0]])},
    ])
    assert clear["passed"] is True
    overlap = separation_report([
        {"instance_id": "a", "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), "faces": np.array([[0, 1, 2]])},
        {"instance_id": "b", "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), "faces": np.array([[0, 1, 2]])},
    ])
    assert overlap["passed"] is False


def test_seed11_sofa_gap_is_rejected():
    """The fast-scene gap is min sofa Z minus max floor Z, 0.516 m.

    Support uses the floor polygon and the sofa contact vertices. A 1 mm
    limit rejects this gap. The number is the measurement already stored for
    seed 11, not a new placement.
    """
    floor = np.array([[0.0, 0.0, 0.128], [1.0, 0.0, 0.128], [1.0, 1.0, 0.128], [0.0, 1.0, 0.128]])
    sofa = np.array([[0.2, 0.2, 0.128 + 0.516], [0.4, 0.2, 0.128 + 0.516], [0.3, 0.4, 0.128 + 0.516]])
    faces = np.array([[0, 1, 2]])
    report = measure_support(sofa, faces, floor)
    assert report["passed"] is False
    assert report["gap_m"] == pytest.approx(0.516, abs=1e-6)


def test_nonzero_radius_point_is_not_converted_as_isotropic():
    from sensor_sim.scene_gen.light_units import classify_area_emitter, classify_point_emitter

    soft = classify_point_emitter(4.0 * np.pi, 0.05, [1.0, 0.0, 1.0])
    assert soft["type"] == "sphere"
    assert soft["included"] is True
    assert soft["conversion"] == "soft_sphere_lambertian_power_over_area_pi"
    assert "isotropic_power_over_4pi" not in soft["conversion"]
    area = 4.0 * np.pi * 0.05 * 0.05
    assert soft["radiance"] == pytest.approx((4.0 * np.pi) / (area * np.pi))
    point = classify_point_emitter(4.0 * np.pi, 0.0, [1.0, 0.0, 1.0])
    assert point["included"] is True
    assert point["intensity"] == pytest.approx(1.0)
    assert point["conversion"] == "isotropic_power_over_4pi"


def test_area_shape_and_two_sided_conversion():
    from sensor_sim.scene_gen.light_units import classify_area_emitter

    one = classify_area_emitter(np.pi, shape="SQUARE", size=1.0, size_y=1.0, scale=(1.0, 1.0, 1.0), two_sided=False, spread_rad=np.pi, position=[0.0, 0.0, 2.0])
    two = classify_area_emitter(np.pi, shape="SQUARE", size=1.0, size_y=1.0, scale=(1.0, 1.0, 1.0), two_sided=True, spread_rad=np.pi, position=[0.0, 0.0, 2.0])
    disk = classify_area_emitter(np.pi * np.pi, shape="DISK", size=2.0, size_y=2.0, scale=(1.0, 1.0, 1.0), two_sided=False, spread_rad=np.pi, position=[0.0, 0.0, 2.0])
    origin = classify_area_emitter(np.pi, shape="SQUARE", size=1.0, size_y=1.0, scale=(1.0, 1.0, 1.0), two_sided=False, spread_rad=np.pi, position=[0.0, 0.0, 0.0])
    assert one["radiance"] == pytest.approx(1.0)
    assert two["radiance"] == pytest.approx(0.5)
    assert two["conversion"] == "two_sided_lambertian_power_over_two_area_pi"
    assert disk["area_m2"] == pytest.approx(np.pi)
    assert disk["radiance"] == pytest.approx(1.0)
    assert origin["included"] is False
    assert origin["reason"] == "light_transform_at_origin"


def test_matte_slot_is_not_forced_to_pplastic(tmp_path):
    from sensor_sim.scene_gen.infinigen_package import _spec_for_slot

    slot = {"name": "plaster", "maps": {}, "Pr": 0.95, "d": 1.0, "Pm": 0.0, "Ni": 1.5, "Kd": [0.8, 0.8, 0.75], "bump_strength": 1.0}
    spec = _spec_for_slot(
        slot,
        material_id="plaster",
        texture_root=tmp_path,
        package_root=tmp_path,
        convention={"flip_green": False, "orientation": "opengl_tangent"},
    )
    assert spec["family"] == "diffuse"
    assert spec["family"] != "pplastic"


def test_rough_glass_slot_stays_rough(tmp_path):
    from sensor_sim.scene_gen.infinigen_package import _spec_for_slot

    slot = {"name": "glass_pane", "maps": {}, "Pr": 0.4, "d": 0.2, "Pm": 0.0, "Ni": 1.5, "Kd": [1, 1, 1], "bump_strength": 1.0}
    spec = _spec_for_slot(
        slot,
        material_id="glass",
        texture_root=tmp_path,
        package_root=tmp_path,
        convention={"flip_green": False, "orientation": "opengl_tangent"},
    )
    assert spec["family"] == "roughdielectric"


def test_television_and_door_frames_are_measured():
    vertices = [[-0.4, 0.0, 0.2], [0.4, 0.0, 0.2], [0.4, 0.0, 1.2], [-0.4, 0.0, 1.2]]
    faces = [[0, 1, 2], [0, 2, 3]]
    television = measure_semantic_frame(_mesh(vertices, faces), "TV")
    door = measure_semantic_frame(_mesh(vertices, faces), "LiteDoor")
    assert television["status"] == "measured"
    assert door["status"] == "measured"
    assert television["forward"][1] < -0.9


def test_sofa_facing_away_fails_the_room_relation():
    from sensor_sim.scene_gen.infinigen_ingest import evaluate_room_relationships

    report = evaluate_room_relationships(
        [
            {"id": "sofa", "category": "Sofa", "origin": [0.0, 0.0, 0.0], "forward": [0.0, 1.0, 0.0], "frame_status": "measured"},
            {"id": "tv", "category": "TV", "origin": [0.0, -2.0, 0.5], "forward": [0.0, 1.0, 0.0], "frame_status": "measured"},
            {"id": "door", "category": "LiteDoor", "origin": [-2.0, 0.0, 0.0], "forward": [1.0, 0.0, 0.0], "frame_status": "measured", "bounds": [[-2.2, -0.2, 0.0], [-1.8, 0.2, 2.0]]},
        ],
        floor_center=[0.0, 0.0, 0.0],
    )
    sofa = next(item for item in report["relations"] if item["name"] == "sofa_toward_display")
    assert sofa["passed"] is False
    assert report["passed"] is False


def test_vertex_cluster_stays_inside_the_cell_diagonal():
    from sensor_sim.scene_gen.mesh_cluster import cluster_surface

    grid = np.stack(np.meshgrid(np.linspace(0, 0.02, 21), np.linspace(0, 0.02, 21)), axis=-1).reshape(-1, 2)
    vertices = np.column_stack([grid, np.zeros(len(grid))])
    faces = []
    side = 21
    for row in range(side - 1):
        for column in range(side - 1):
            base = row * side + column
            faces.append([base, base + 1, base + side])
    report = cluster_surface(vertices, np.asarray(faces), 0.005)
    assert report["max_move_m"] <= report["bound_m"] + 1e-9
    assert report["bound_m"] == pytest.approx(0.005 * np.sqrt(3.0))
    assert report["output_faces"] < report["input_faces"]
    assert report["output_vertices"] < report["input_vertices"]


def test_construction_rejects_an_instance_without_a_parent(tmp_path):
    import json
    from sensor_sim.scene_gen.mitsuba_scene_package import construction_report

    mesh = tmp_path / "a.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    manifest = {
        "schema": "infinigen_mitsuba_scene_v1",
        "variant": "cuda_ad_spectral_polarized",
        "camera": {"id": "cam", "fov_deg": 40, "origin": [0, -2, 1], "target": [0, 0, 0], "width": 8, "height": 8},
        "lights": [],
        "assets": [{"id": "a", "mesh": "a.obj"}],
        "instances": [{"id": "a", "asset_id": "a", "matrix": np.eye(4).tolist(), "regions": [{"slot": "s", "mesh": "a.obj", "spec": {}}]}],
    }
    path = tmp_path / "scene_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = construction_report(path)
    assert report["passed"] is False
    assert any(item.endswith(":parent") for item in report["incomplete_instances"])

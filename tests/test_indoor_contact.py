"""Known gaps, penetrations, and support-graph failures."""

from __future__ import annotations

import numpy as np
import pytest

from sensor_sim.scene_gen.asset_validate import triangulate
from sensor_sim.scene_gen.asset_package import PackageError
from sensor_sim.scene_gen.indoor_contact import (
    MAX_PENETRATION_M,
    MAX_SUPPORT_GAP_M,
    MEASUREMENT_TOLERANCE_M,
    measure_support,
    require_clearance_resource,
    support_graph_ok,
    unintended_intersection,
    world_vertices,
)
from sensor_sim.scene_gen.indoor_transform import translation_matrix
from sensor_sim.scene_gen.structural import box_mesh, build_table, cone_mesh, sphere_mesh


FLOOR = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)


def _box_at(z_value: float, xy=(0.4, 0.4), size=0.2):
    vertices, polygons = box_mesh((xy[0], xy[1], z_value), (xy[0] + size, xy[1] + size, z_value + size))
    return vertices, polygons


def test_floor_table_shelf_and_tolerances():
    resting, faces = _box_at(0.0)
    contact = measure_support(resting, faces, FLOOR)
    assert contact["passed"]
    assert abs(contact["gap_m"]) <= MEASUREMENT_TOLERANCE_M
    assert contact["penetration_m"] <= MEASUREMENT_TOLERANCE_M
    lifted, faces = _box_at(0.0004)
    lifted_contact = measure_support(lifted, faces, FLOOR)
    assert lifted_contact["passed"]
    assert abs(lifted_contact["gap_m"] - 0.0004) <= MEASUREMENT_TOLERANCE_M
    too_high, faces = _box_at(0.002)
    assert measure_support(too_high, faces, FLOOR)["reason"] == "gap"
    shallow, faces = _box_at(-0.00005)
    shallow_contact = measure_support(shallow, faces, FLOOR)
    assert shallow_contact["passed"]
    assert abs(shallow_contact["penetration_m"] - 0.00005) <= MEASUREMENT_TOLERANCE_M
    deep, faces = _box_at(-0.0002)
    assert measure_support(deep, faces, FLOOR)["reason"] == "penetration"
    outside, faces = _box_at(0.0, xy=(3.0, 3.0))
    assert measure_support(outside, faces, FLOOR)["reason"] == "outside_support_polygon"
    wrong = np.array([[0.0, 0.0, 0.5], [2.0, 0.0, 0.5], [2.0, 2.0, 0.5], [0.0, 2.0, 0.5]])
    assert measure_support(resting, faces, wrong)["passed"] is False

    table = build_table(17011)
    matrix = translation_matrix([0.5, 0.5, 0.0])
    for piece in table["pieces"]:
        if not piece["id"].startswith("leg_"):
            continue
        world = world_vertices(piece["vertices"], matrix)
        leg = measure_support(world, piece["polygons"], FLOOR)
        assert leg["passed"], piece["id"]
        assert leg["gap_m"] <= MEASUREMENT_TOLERANCE_M

    sphere_v, sphere_p = sphere_mesh(0.08, 8, 16)
    sphere_world = world_vertices(sphere_v, translation_matrix([0.6, 0.6, 0.08]))
    curved = measure_support(sphere_world, sphere_p, FLOOR)
    assert curved["passed"]
    assert abs(curved["gap_m"]) <= MEASUREMENT_TOLERANCE_M
    cone_v, cone_p = cone_mesh(0.06, 0.12, 24)
    cone_world = world_vertices(cone_v, translation_matrix([1.0, 1.0, 0.0]))
    assert measure_support(cone_world, cone_p, FLOOR)["passed"]


def test_overlap_cycle_and_proxy_rules():
    floating, faces = _box_at(0.05)
    # The floating box overlaps a tall solid's AABB in z, but it does not touch the z=0 polygon.
    solid_v, solid_p = box_mesh((0.4, 0.4, -0.2), (0.6, 0.6, 0.1))
    assert measure_support(floating, faces, FLOOR)["reason"] == "gap"
    low = solid_v.min(axis=0)
    high = solid_v.max(axis=0)
    object_low = floating.min(axis=0)
    object_high = floating.max(axis=0)
    assert np.all(object_low <= high) and np.all(low <= object_high)
    buried, buried_faces = _box_at(-0.02)
    hit = unintended_intersection(
        {"vertices": buried, "faces": triangulate(buried_faces)},
        {"vertices": solid_v, "faces": triangulate(solid_p)},
    )
    assert hit["unintended"]
    assert support_graph_ok([("floor", "table"), ("table", "cup")])
    assert support_graph_ok([("floor", "table"), ("floor", "shelf")]) is True
    assert support_graph_ok([("table", "cup"), ("cup", "table")]) is False
    require_clearance_resource(100, True, "abc", "abc")
    with pytest.raises(PackageError, match="collision_hash"):
        require_clearance_resource(100, True, "abc", "def")
    with pytest.raises(PackageError, match="clearance_proxy_required"):
        require_clearance_resource(CLEARANCE, False, "abc", "abc")
    assert MAX_SUPPORT_GAP_M == 0.001
    assert MAX_PENETRATION_M == 0.0001


CLEARANCE = 2501

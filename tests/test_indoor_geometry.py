"""Profile distinctions, structural openings, and deterministic procedural meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_validate import orient_outward, triangulate, validate_mesh
from sensor_sim.scene_gen.mesh_io import read_obj, write_obj
from sensor_sim.scene_gen.meshcheck import mesh_issues
from sensor_sim.scene_gen.structural import (
    box_mesh,
    build_prop,
    build_room,
    build_shelf,
    build_table,
    package_procedural_catalog,
)


def _quad():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    return vertices, [[0, 1, 2, 3]]


def test_profiles_distinguish_constructed_meshes():
    quad_v, quad_p = _quad()
    assert validate_mesh(quad_v, quad_p, "open_surface")["passed"]
    assert "boundary_edge" in validate_mesh(quad_v, quad_p, "closed_solid")["issues"]

    cube_v, cube_p = box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    closed = validate_mesh(cube_v, cube_p, "closed_solid")
    assert closed["passed"]
    assert closed["polygon_count"] == 6
    assert closed["triangle_count"] == 12
    solid = validate_mesh(cube_v, cube_p, "structural_solid", thickness_axis=[0.0, 0.0, 1.0])
    assert solid["passed"]
    assert solid["thickness_m"] == 1.0

    missing = validate_mesh(cube_v, cube_p[:-1], "closed_solid")
    assert "boundary_edge" in missing["issues"]

    edge_v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    edge = validate_mesh(edge_v, [[0, 1, 2], [0, 1, 3], [0, 1, 4]], "open_surface")
    assert "non_manifold_edge" in edge["issues"]

    vertex_v = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    vertex = validate_mesh(vertex_v, [[0, 1, 2], [0, 3, 4]], "open_surface")
    assert "non_manifold_vertex" in vertex["issues"]

    other_v, other_p = box_mesh((3.0, 0.0, 0.0), (4.0, 1.0, 1.0))
    stacked = np.vstack([cube_v, other_v])
    shifted = [[index + 8 for index in polygon] for polygon in other_p]
    both = cube_p + shifted
    named = validate_mesh(
        stacked,
        both,
        "multi_component",
        components=[
            {"id": "a", "profile": "closed_solid", "polygon_indexes": list(range(6))},
            {"id": "b", "profile": "closed_solid", "polygon_indexes": list(range(6, 12))},
        ],
    )
    assert named["passed"]
    unnamed = validate_mesh(
        stacked,
        both,
        "multi_component",
        components=[{"id": "a", "profile": "closed_solid", "polygon_indexes": list(range(6))}],
    )
    assert "unnamed_component" in unnamed["issues"]

    reversed_faces = [list(reversed(polygon)) for polygon in cube_p]
    inverted = validate_mesh(cube_v, reversed_faces, "closed_solid")
    assert "inverted_volume" in inverted["issues"]
    repaired = validate_mesh(cube_v, orient_outward(cube_v, reversed_faces), "closed_solid")
    assert repaired["passed"]

    one_flip = [list(polygon) for polygon in cube_p]
    one_flip[0] = list(reversed(one_flip[0]))
    assert "inconsistent_winding" in validate_mesh(cube_v, one_flip, "closed_solid")["issues"]

    degenerate = validate_mesh(quad_v, [[0, 1, 1]], "open_surface")
    assert "degenerate" in degenerate["issues"]

    thin_v, thin_p = box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 0.001))
    thin = validate_mesh(thin_v, thin_p, "structural_solid", thickness_axis=[0.0, 0.0, 1.0])
    assert "thickness" in thin["issues"]

    floor_v, floor_p = box_mesh((0.0, 0.0, 0.0), (4.0, 4.0, 0.02))
    assert validate_mesh(floor_v, floor_p, "structural_solid", thickness_axis=[0.0, 0.0, 1.0])["passed"]
    wide_v, wide_p = box_mesh((0.0, 0.0, 0.0), (8.0, 8.0, 0.02))
    assert validate_mesh(wide_v, wide_p, "structural_solid", thickness_axis=[0.0, 0.0, 1.0])["passed"]
    assert "bounds" in mesh_issues(wide_v, triangulate(wide_p), closed=True)

    tiny_v, tiny_p = box_mesh((0.0, 0.0, 0.0), (0.004, 0.004, 0.00002))
    assert "thickness" in validate_mesh(tiny_v, tiny_p, "structural_solid", thickness_axis=[0.0, 0.0, 1.0])["issues"]
    huge_v, huge_p = box_mesh((0.0, 0.0, 0.0), (4000.0, 4000.0, 20.0))
    assert "extent" in validate_mesh(huge_v, huge_p, "structural_solid", thickness_axis=[0.0, 0.0, 1.0])["issues"]

    loose_v = np.vstack([cube_v, [[5.0, 5.0, 5.0]]])
    assert "loose_vertex" in validate_mesh(loose_v, cube_p, "closed_solid")["issues"]

    normals = []
    for polygon in cube_p:
        geometric = np.zeros(3)
        for index, current in enumerate(polygon):
            start = cube_v[current]
            stop = cube_v[polygon[(index + 1) % len(polygon)]]
            geometric[0] += (start[1] - stop[1]) * (start[2] + stop[2])
            geometric[1] += (start[2] - stop[2]) * (start[0] + stop[0])
            geometric[2] += (start[0] - stop[0]) * (start[1] + stop[1])
        normals.append(geometric / np.linalg.norm(geometric))
    assert validate_mesh(cube_v, cube_p, "closed_solid", normals=np.array(normals))["passed"]
    assert "normal" in validate_mesh(cube_v, cube_p, "closed_solid", normals=-np.array(normals))["issues"]


def test_room_openings_and_repeatable_bytes(tmp_path: Path):
    room = build_room()
    interior = room["interior"]
    assert interior["size_m"] == [4.0, 3.2, 2.5]
    assert interior["origin_m"] == [0.12, 0.12, 0.05]
    ids = [piece["id"] for piece in room["pieces"]]
    assert "south_header" in ids
    assert "north_sill" in ids
    assert "south_sill" not in ids
    floor = next(piece for piece in room["pieces"] if piece["id"] == "floor")
    wall = next(piece for piece in room["pieces"] if piece["id"] == "west")
    assert validate_mesh(floor["vertices"], floor["polygons"], "structural_solid", thickness_axis=[0.0, 0.0, 1.0])["passed"]
    assert validate_mesh(wall["vertices"], wall["polygons"], "structural_solid", thickness_axis=[1.0, 0.0, 0.0])["passed"]
    first = write_obj(floor["vertices"], floor["polygons"], name="floor")
    second = write_obj(floor["vertices"], floor["polygons"], name="floor")
    assert first == second
    reread_v, reread_p = read_obj(first)
    assert reread_p == floor["polygons"]
    assert np.allclose(reread_v, floor["vertices"])
    again = build_room()
    assert write_obj(again["pieces"][0]["vertices"], again["pieces"][0]["polygons"], name="floor") == first

    crate_a = write_obj(build_prop("crate", 3)["pieces"][0]["vertices"], build_prop("crate", 3)["pieces"][0]["polygons"], name="crate")
    crate_b = write_obj(build_prop("crate", 3)["pieces"][0]["vertices"], build_prop("crate", 3)["pieces"][0]["polygons"], name="crate")
    crate_c = write_obj(build_prop("crate", 4)["pieces"][0]["vertices"], build_prop("crate", 4)["pieces"][0]["polygons"], name="crate")
    assert crate_a == crate_b
    assert crate_a != crate_c
    for seed in (3, 4):
        piece = build_prop("crate", seed)["pieces"][0]
        assert validate_mesh(piece["vertices"], piece["polygons"], "structural_solid", thickness_axis=[0.0, 0.0, 1.0])["passed"]
    assert build_table(8)["pieces"][0]["id"] == "top"
    assert len(build_shelf(8)["support_surfaces"]) == 2

    first_catalog = package_procedural_catalog(tmp_path / "a", seed=17011)
    second_catalog = package_procedural_catalog(tmp_path / "b", seed=17011)
    assert [item["package_sha256"] for item in first_catalog] == [item["package_sha256"] for item in second_catalog]
    assert all(item["geometry_passed"] for item in first_catalog)
    assert {item["asset_id"] for item in first_catalog} >= {
        "room_floor",
        "room_wall_south",
        "room_wall_north",
        "room_ceiling",
        "table_rect",
        "shelf_two",
        "prop_crate",
        "prop_cup",
        "prop_book",
        "fixture_sphere",
        "fixture_cone",
    }

"""OBJ face-corner UVs, seams, and geometry-only compatibility."""

from __future__ import annotations

import numpy as np
import pytest

from sensor_sim.scene_gen.mesh_io import ObjError, read_mesh, read_obj, validate_uv, write_mesh, write_obj


def _seam_mesh():
    return {
        "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        "texcoords": np.array([[0.0, 0.0], [0.45, 0.0], [0.45, 0.45], [0.0, 0.45], [0.55, 0.55], [1.0, 0.55], [1.0, 1.0]], dtype=np.float64),
        "normals": np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        "corners": [
            [(0, 0, 0), (1, 1, 0), (3, 3, 0)],
            [(1, 4, 0), (2, 5, 0), (3, 6, 0)],
        ],
    }


def test_geometry_only_round_trip_matches_the_legacy_reader():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    payload = write_obj(vertices, [[0, 1, 2]], name="triangle")
    reread, faces = read_obj(payload)
    assert faces == [[0, 1, 2]]
    assert np.allclose(reread, vertices)
    assert b"vt " not in payload


def test_uv_seam_round_trip_keeps_two_coordinates_for_one_vertex():
    mesh = _seam_mesh()
    payload = write_mesh(mesh, name="seam")
    reread = read_mesh(payload)
    assert reread["corners"][0][0][0] == 0 and reread["corners"][1][0][0] == 1
    assert reread["corners"][0][0][1] != reread["corners"][1][2][1]
    assert np.allclose(reread["texcoords"][reread["corners"][0][0][1]], mesh["texcoords"][0])
    assert reread["corners"][0][0][2] == 0
    geometry, faces = read_obj(payload)
    assert faces[0] == [0, 1, 3]
    assert geometry.shape == (4, 3)


def test_negative_indices_and_missing_uvs():
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n"
    vertices, faces = read_obj(text.encode())
    assert faces == [[0, 1, 2]]
    assert len(vertices) == 3
    bare = read_mesh(text.encode())
    assert bare["texcoords"].shape == (0, 2)
    assert bare["corners"][0][0][1] is None


def test_out_of_range_face_reference_fails():
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n"
    with pytest.raises(ObjError, match="index_out_of_range"):
        read_mesh(text.encode())


def test_uv_validation_reports_texel_density_and_rejects_overlap():
    mesh = _seam_mesh()
    report = validate_uv(mesh, texture_resolution=(64, 64), island_padding=0.01)
    assert report["texel_density_px_per_m"] > 0.0
    assert report["texture_resolution"] == [64, 64]
    overlap = _seam_mesh()
    overlap["texcoords"] = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    with pytest.raises(ObjError, match="overlapping_uv_islands"):
        validate_uv(overlap, texture_resolution=(32, 32), island_padding=0.0)
    quad = {
        "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        "texcoords": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        "normals": np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
        "corners": [[(0, 0, 0), (2, 2, 0), (1, 1, 0)], [(0, 0, 0), (3, 3, 0), (2, 2, 0)]],
    }
    assert validate_uv(quad, texture_resolution=(64, 64), island_padding=0.0)["triangle_count"] == 2
    with pytest.raises(ObjError, match="insufficient_island_padding"):
        validate_uv(mesh, texture_resolution=(64, 64), island_padding=0.25)

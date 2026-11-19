"""Reject malformed meshes before they are rendered."""

from __future__ import annotations

import numpy as np

from sensor_sim.film.shape_mesh import edge_face_counts


def _areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-15)


def inconsistent_edges(faces: np.ndarray) -> int:
    """Undirected edges whose two directions are not one each."""
    directed: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        for start, stop in ((0, 1), (1, 2), (2, 0)):
            key = (int(face[start]), int(face[stop]))
            directed[key] = directed.get(key, 0) + 1
    bad = 0
    seen = set()
    for first, second in directed:
        if first > second:
            continue
        pair = (first, second)
        if pair in seen:
            continue
        seen.add(pair)
        forward = directed.get((first, second), 0)
        backward = directed.get((second, first), 0)
        if forward != 1 or backward != 1:
            bad += 1
    return bad


# An object smaller than the geometry comparison tolerance cannot be validated.
MIN_EXTENT_M = 1e-4


def mesh_issues(vertices: np.ndarray, faces: np.ndarray, *, closed: bool, normals: np.ndarray | None = None) -> list[str]:
    """Problems that should stop a sample. An open mesh may have boundary edges."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    issues = []
    if faces.size == 0 or vertices.size == 0:
        return ["empty"]
    if not np.isfinite(vertices).all():
        issues.append("non_finite")
    areas = _areas(vertices, faces)
    if np.any(areas < 1e-14):
        issues.append("degenerate")
    counts = edge_face_counts(faces)
    if closed:
        if any(count != 2 for count in counts.values()):
            issues.append("not_closed")
        if any(count > 2 for count in counts.values()):
            issues.append("non_manifold")
        if inconsistent_edges(faces) > 0:
            issues.append("inverted_or_inconsistent_winding")
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if float(np.max(extent)) > 2.0 or float(np.max(extent)) <= 0.0:
        issues.append("bounds")
    if float(np.max(extent)) < MIN_EXTENT_M:
        issues.append("tiny")
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        if not np.isfinite(normals).all():
            issues.append("non_finite_normals")
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths < 1e-8):
            issues.append("zero_normals")
        finite = lengths > 1e-8
        if np.any(finite) and float(np.max(np.abs(lengths[finite] - 1.0))) > 1e-2:
            issues.append("non_unit_normals")
    return issues


def require_clean(vertices: np.ndarray, faces: np.ndarray, *, closed: bool, name: str) -> None:
    issues = mesh_issues(vertices, faces, closed=closed)
    if issues:
        raise RuntimeError(f"{name} mesh failed geometry checks: {issues}")

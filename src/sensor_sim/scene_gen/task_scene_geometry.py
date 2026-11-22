"""Convert an exported Infinigen OBJ into canonical Z-up meshes.

The packaged render mesh and the collision mesh are different files. A convex
hull is not described as the render mesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from sensor_sim.scene_gen.mesh_io import read_mesh
from sensor_sim.scene_gen.scene_manifest import exported_y_up_to_canonical


def load_exported_obj(path: Path) -> dict:
    """Read an OBJ and convert Y-up export vertices into the canonical frame."""
    mesh = read_mesh(path)
    converted = np.asarray([exported_y_up_to_canonical(np.eye(4)) @ np.array([*vertex, 1.0]) for vertex in mesh["vertices"]], dtype=np.float64)[:, :3]
    faces = []
    for corners in mesh["corners"]:
        indexes = [corner[0] for corner in corners]
        if len(indexes) == 3:
            faces.append(indexes)
        elif len(indexes) == 4:
            faces.append([indexes[0], indexes[1], indexes[2]])
            faces.append([indexes[0], indexes[2], indexes[3]])
        else:
            for offset in range(1, len(indexes) - 1):
                faces.append([indexes[0], indexes[offset], indexes[offset + 1]])
    return {
        "vertices": converted,
        "faces": np.asarray(faces, dtype=np.int64),
        "face_materials": list(mesh.get("face_materials") or []),
        "texcoords": mesh["texcoords"],
        "normals": mesh["normals"],
        "corners": mesh["corners"],
    }


def bounds(vertices: np.ndarray) -> dict:
    array = np.asarray(vertices, dtype=np.float64)
    low = array.min(axis=0)
    high = array.max(axis=0)
    return {"min_m": low.tolist(), "max_m": high.tolist(), "size_m": (high - low).tolist()}


def tabletop_mask(vertices: np.ndarray, band_m: float = 0.02) -> np.ndarray:
    """Vertices on the highest horizontal band. That band is the tabletop, not the whole asset."""
    height = float(np.max(vertices[:, 2]))
    return vertices[:, 2] >= height - band_m


def box_mesh(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array(
        [
            [low[0], low[1], low[2]],
            [high[0], low[1], low[2]],
            [high[0], high[1], low[2]],
            [low[0], high[1], low[2]],
            [low[0], low[1], high[2]],
            [high[0], low[1], high[2]],
            [high[0], high[1], high[2]],
            [low[0], high[1], high[2]],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
        dtype=np.int64,
    )
    return corners, faces


def table_collision(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """A tabletop slab plus a narrower leg box. Decorative detail is not kept."""
    mask = tabletop_mask(vertices)
    top = vertices[mask]
    top_low = top.min(axis=0)
    top_high = top.max(axis=0)
    slab_low = np.array([top_low[0], top_low[1], max(float(top_low[2]), float(top_high[2]) - 0.04)])
    slab_high = top_high.copy()
    below = vertices[~mask]
    if len(below) == 0:
        below = vertices
    center = (top_low[:2] + top_high[:2]) / 2.0
    half = (top_high[:2] - top_low[:2]) / 2.0 * 0.25
    leg_low = np.array([center[0] - half[0], center[1] - half[1], float(vertices[:, 2].min())])
    leg_high = np.array([center[0] + half[0], center[1] + half[1], float(slab_low[2])])
    slab_v, slab_f = box_mesh(slab_low, slab_high)
    leg_v, leg_f = box_mesh(leg_low, np.maximum(leg_high, leg_low + 1e-4))
    vertices_out = np.vstack([slab_v, leg_v])
    faces_out = np.vstack([slab_f, leg_f + len(slab_v)])
    report = {"method": "primitive_proxy", "tabletop_vertex_count": int(mask.sum()), "slab_min_m": slab_low.tolist(), "slab_max_m": slab_high.tolist()}
    return vertices_out, faces_out, report


def convex_collision(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hull = ConvexHull(np.asarray(vertices, dtype=np.float64))
    remap = {int(old): new for new, old in enumerate(hull.vertices)}
    faces = np.array([[remap[int(index)] for index in face] for face in hull.simplices], dtype=np.int64)
    return np.asarray(vertices, dtype=np.float64)[hull.vertices], faces


def surface_distance(source: np.ndarray, target_vertices: np.ndarray, target_faces: np.ndarray, *, sample_count: int = 4000, seed: int = 0) -> dict:
    """Distance from source points to the closest point on target triangles."""
    points = np.asarray(source, dtype=np.float64)
    if len(points) > sample_count:
        generator = np.random.default_rng(seed)
        points = points[generator.choice(len(points), sample_count, replace=False)]
    distances = np.array([_point_mesh_distance(point, target_vertices, target_faces) for point in points], dtype=np.float64)
    return {
        "p50_m": float(np.percentile(distances, 50)),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(distances.max()),
    }


def symmetric_surface_distance(render_v, render_f, collision_v, collision_f) -> dict:
    forward = surface_distance(render_v, collision_v, collision_f)
    backward = surface_distance(collision_v, render_v, render_f)
    return {
        "render_to_collision": forward,
        "collision_to_render": backward,
        "max_m": max(forward["max_m"], backward["max_m"]),
    }


def _point_mesh_distance(point: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> float:
    best = float("inf")
    for face in faces:
        triangle = vertices[face]
        best = min(best, _point_triangle(point, triangle))
    return best


def _point_triangle(point: np.ndarray, triangle: np.ndarray) -> float:
    ab = triangle[1] - triangle[0]
    ac = triangle[2] - triangle[0]
    ap = point - triangle[0]
    d1 = float(ab @ ap)
    d2 = float(ac @ ap)
    if d1 <= 0 and d2 <= 0:
        return float(np.linalg.norm(ap))
    bp = point - triangle[1]
    d3 = float(ab @ bp)
    d4 = float(ac @ bp)
    if d3 >= 0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return float(np.linalg.norm(point - (triangle[0] + (d1 / (d1 - d3)) * ab)))
    cp = point - triangle[2]
    d5 = float(ab @ cp)
    d6 = float(ac @ cp)
    if d6 >= 0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return float(np.linalg.norm(point - (triangle[0] + (d2 / (d2 - d6)) * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return float(np.linalg.norm(point - (triangle[1] + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (triangle[2] - triangle[1]))))
    denom = 1.0 / (va + vb + vc)
    return float(np.linalg.norm(point - (triangle[0] + ab * (vb * denom) + ac * (vc * denom))))


def write_textured_obj(path: Path, vertices: np.ndarray, corners: list, texcoords: np.ndarray, normals: np.ndarray, face_materials: list[str]) -> None:
    """Write the converted mesh and keep the exported UVs, normals, and material groups."""
    lines = [f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}" for vertex in vertices]
    for texcoord in np.asarray(texcoords):
        lines.append(f"vt {texcoord[0]:.8f} {texcoord[1]:.8f}")
    rotation = exported_y_up_to_canonical(np.eye(4))[:3, :3]
    for normal in np.asarray(normals):
        converted = rotation @ normal
        lines.append(f"vn {converted[0]:.8f} {converted[1]:.8f} {converted[2]:.8f}")
    active = None
    for index, face in enumerate(corners):
        material = face_materials[index] if index < len(face_materials) else None
        if material and material != active:
            lines.append(f"usemtl {material}")
            active = material
        tokens = []
        for vertex, texcoord, normal in face:
            tokens.append(f"{vertex + 1}/{texcoord + 1}/{normal + 1}" if texcoord is not None and normal is not None else str(vertex + 1))
        lines.append("f " + " ".join(tokens))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, *, materials: list[str] | None = None) -> None:
    lines = []
    for vertex in vertices:
        lines.append(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}")
    active = None
    for index, face in enumerate(faces):
        material = materials[index] if materials and index < len(materials) else None
        if material and material != active:
            lines.append(f"usemtl {material}")
            active = material
        lines.append(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

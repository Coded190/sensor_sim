"""Deterministic OBJ read and write for indoor assets.

Coordinates are meters. Floats are written with nine digits after the decimal.
The writer emits no timestamps and no random identifiers.

``read_obj`` still returns positions and vertex-index faces so existing callers
keep working. ``read_mesh`` keeps face-corner texture and normal indices,
including UV seams where one geometric vertex uses more than one ``vt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OBJ_DECIMALS = 9


class ObjError(ValueError):
    """An OBJ face reference is missing, out of range, or inconsistent."""


def quantize(values: np.ndarray, decimals: int = OBJ_DECIMALS) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.round(array, decimals)


def write_obj(vertices: np.ndarray, polygons: list[list[int]], *, name: str = "mesh") -> bytes:
    """Serialize one mesh. Vertex order, face order, and winding are preserved."""
    vertices = quantize(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    lines = [f"# sensor_sim indoor obj 1 name={name}", f"o {name}"]
    for vertex in vertices:
        lines.append(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}")
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("a face needs at least three vertices")
        indexes = " ".join(str(int(index) + 1) for index in polygon)
        lines.append(f"f {indexes}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_mesh(mesh: dict, *, name: str = "mesh") -> bytes:
    """Write positions, optional UVs, optional normals, and face-corner indices.

    A corner is ``(vertex, texcoord or None, normal or None)``. Seams are
    preserved because texture indices are not tied one-to-one to vertices.
    """
    vertices = quantize(np.asarray(mesh["vertices"], dtype=np.float64))
    texcoords = np.asarray(mesh.get("texcoords", np.zeros((0, 2))), dtype=np.float64)
    normals = np.asarray(mesh.get("normals", np.zeros((0, 3))), dtype=np.float64)
    if texcoords.size:
        texcoords = quantize(texcoords)
    if normals.size:
        normals = quantize(normals)
    lines = [f"# sensor_sim indoor obj 1 name={name}", f"o {name}"]
    for vertex in vertices:
        lines.append(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}")
    for texcoord in texcoords.reshape(-1, 2):
        lines.append(f"vt {texcoord[0]:.9f} {texcoord[1]:.9f}")
    for normal in normals.reshape(-1, 3):
        lines.append(f"vn {normal[0]:.9f} {normal[1]:.9f} {normal[2]:.9f}")
    face_materials = list(mesh.get("face_materials") or [])
    active_material = None
    for face_index, face in enumerate(mesh["corners"]):
        if len(face) < 3:
            raise ObjError("reason=face_size")
        material_name = face_materials[face_index] if face_index < len(face_materials) else None
        if material_name and material_name != active_material:
            lines.append(f"usemtl {material_name}")
            active_material = material_name
        tokens = []
        for corner in face:
            vertex, texcoord, normal = corner
            _check_index(int(vertex), len(vertices), "vertex")
            token = str(int(vertex) + 1)
            if texcoord is None and normal is None:
                tokens.append(token)
                continue
            if texcoord is None:
                tokens.append(f"{token}//{int(normal) + 1}")
                _check_index(int(normal), len(normals), "normal")
                continue
            _check_index(int(texcoord), len(texcoords), "texcoord")
            if normal is None:
                tokens.append(f"{token}/{int(texcoord) + 1}")
            else:
                _check_index(int(normal), len(normals), "normal")
                tokens.append(f"{token}/{int(texcoord) + 1}/{int(normal) + 1}")
        lines.append("f " + " ".join(tokens))
    return ("\n".join(lines) + "\n").encode("utf-8")


def read_obj(path: Path | bytes | str) -> tuple[np.ndarray, list[list[int]]]:
    """Read vertices and polygonal faces. Texture and normal indices are not returned.

    The geometric vertex index of each corner is preserved, including negative
    OBJ indices. Call ``read_mesh`` when the UV seam must be kept.
    """
    mesh = read_mesh(path)
    polygons = [[int(corner[0]) for corner in face] for face in mesh["corners"]]
    return mesh["vertices"], polygons


def read_mesh(path: Path | bytes | str) -> dict:
    """Read positions, texture coordinates, normals, and face-corner indices."""
    if isinstance(path, bytes):
        text = path.decode("utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    normals: list[list[float]] = []
    corners: list[list[tuple[int, int | None, int | None]]] = []
    face_materials: list[str] = []
    active_material = "default"
    for raw in text.splitlines():
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "v":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif kind == "vt":
            texcoords.append([float(parts[1]), float(parts[2])])
        elif kind == "vn":
            normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif kind == "usemtl" and len(parts) >= 2:
            active_material = parts[1]
        elif kind == "f":
            face = []
            for token in parts[1:]:
                face.append(_corner(token, len(vertices), len(texcoords), len(normals)))
            if len(face) < 3:
                raise ObjError("reason=face_size")
            corners.append(face)
            face_materials.append(active_material)
    vertex_array = np.asarray(vertices, dtype=np.float64) if vertices else np.zeros((0, 3), dtype=np.float64)
    texcoord_array = np.asarray(texcoords, dtype=np.float64) if texcoords else np.zeros((0, 2), dtype=np.float64)
    normal_array = np.asarray(normals, dtype=np.float64) if normals else np.zeros((0, 3), dtype=np.float64)
    return {
        "vertices": vertex_array,
        "texcoords": texcoord_array,
        "normals": normal_array,
        "corners": corners,
        "face_materials": face_materials,
    }


def validate_uv(mesh: dict, *, texture_resolution: tuple[int, int], island_padding: float = 0.002) -> dict:
    """Reject missing, invalid, degenerate, or overlapping UV islands.

    Texel density is the mean of ``texture_edge_pixels / world_edge_metres``
    over the three edges of every UV triangle. A round trip does not prove
    this density is enough for the texture.
    """
    texcoords = np.asarray(mesh["texcoords"], dtype=np.float64)
    if texcoords.size == 0 or any(corner[1] is None for face in mesh["corners"] for corner in face):
        raise ObjError("reason=missing_uv")
    triangles = []
    for face in mesh["corners"]:
        for index in range(1, len(face) - 1):
            triangles.append((face[0], face[index], face[index + 1]))
    areas = []
    densities = []
    grid = np.zeros((128, 128), dtype=np.int32)
    triangle_uvs = []
    for number, triangle in enumerate(triangles, start=1):
        uv = np.array([texcoords[int(corner[1])] for corner in triangle], dtype=np.float64)
        xyz = np.array([mesh["vertices"][int(corner[0])] for corner in triangle], dtype=np.float64)
        uv_area = _triangle_area(uv)
        if uv_area < 1e-12:
            raise ObjError("reason=degenerate_uv_island")
        areas.append(uv_area)
        densities.extend(_edge_density(xyz, uv, texture_resolution))
        _stamp(grid, uv, number)
        triangle_uvs.append(uv)
    if np.any(grid == -1):
        raise ObjError("reason=overlapping_uv_islands")
    islands = _islands(triangles, triangle_uvs)
    _padding(islands, island_padding)
    width, height = texture_resolution
    return {
        "uv_method_recorded_by_caller": True,
        "island_padding": float(island_padding),
        "triangle_count": len(triangles),
        "uv_area": float(sum(areas)),
        "texel_density_px_per_m": float(np.mean(densities)) if densities else 0.0,
        "texture_resolution": [int(width), int(height)],
        "definition": "Texel density is the mean of texture-edge pixels divided by the world edge length in metres.",
    }


def _check_index(index: int, count: int, kind: str) -> None:
    if count <= 0 or index < 0 or index >= count:
        raise ObjError(f"reason=index_out_of_range kind={kind} index={index} count={count}")


def _resolve(index: int, count: int, kind: str) -> int:
    resolved = index - 1 if index > 0 else count + index
    _check_index(resolved, count, kind)
    return resolved


def _corner(token: str, vertex_count: int, texcoord_count: int, normal_count: int) -> tuple[int, int | None, int | None]:
    pieces = token.split("/")
    if len(pieces) > 3 or pieces[0] == "":
        raise ObjError(f"reason=face_token token={token}")
    vertex = _resolve(int(pieces[0]), vertex_count, "vertex")
    texcoord = None
    normal = None
    if len(pieces) >= 2 and pieces[1] != "":
        texcoord = _resolve(int(pieces[1]), texcoord_count, "texcoord")
    if len(pieces) == 3 and pieces[2] != "":
        normal = _resolve(int(pieces[2]), normal_count, "normal")
    return vertex, texcoord, normal


def _triangle_area(points: np.ndarray) -> float:
    if points.shape[1] == 2:
        edge_u = points[1] - points[0]
        edge_v = points[2] - points[0]
        return float(abs(edge_u[0] * edge_v[1] - edge_u[1] * edge_v[0]) * 0.5)
    return float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) * 0.5)


def _edge_density(xyz: np.ndarray, uv: np.ndarray, resolution: tuple[int, int]) -> list[float]:
    width, height = resolution
    found = []
    for start, stop in ((0, 1), (1, 2), (2, 0)):
        world = float(np.linalg.norm(xyz[stop] - xyz[start]))
        texels = np.array([(uv[stop, 0] - uv[start, 0]) * width, (uv[stop, 1] - uv[start, 1]) * height])
        pixels = float(np.linalg.norm(texels))
        if world > 1e-12:
            found.append(pixels / world)
    return found


def _stamp(grid: np.ndarray, uv: np.ndarray, number: int) -> None:
    """Mark pixels whose centers are strictly inside this UV triangle.

    A shared edge is not an overlap. A pixel is marked only when its center is
    inside the triangle, so neighboring triangles in one island can share a
    boundary without being rejected.
    """
    height, width = grid.shape
    pixels = np.stack([uv[:, 0] * (width - 1), (1.0 - uv[:, 1]) * (height - 1)], axis=1)
    minimum = np.floor(pixels.min(axis=0)).astype(int)
    maximum = np.ceil(pixels.max(axis=0)).astype(int)
    for y_index in range(max(0, int(minimum[1])), min(height, int(maximum[1]) + 1)):
        for x_index in range(max(0, int(minimum[0])), min(width, int(maximum[0]) + 1)):
            if not _strict_inside(np.array([x_index + 0.5, y_index + 0.5]), pixels):
                continue
            if grid[y_index, x_index] not in (0, number):
                grid[y_index, x_index] = -1
            else:
                grid[y_index, x_index] = number


def _strict_inside(point: np.ndarray, triangle: np.ndarray) -> bool:
    """True when the point is inside the triangle and not on its boundary."""

    def side(start, stop):
        return (stop[0] - start[0]) * (point[1] - start[1]) - (stop[1] - start[1]) * (point[0] - start[0])

    signs = [side(triangle[0], triangle[1]), side(triangle[1], triangle[2]), side(triangle[2], triangle[0])]
    return all(value > 0.0 for value in signs) or all(value < 0.0 for value in signs)


def _islands(triangles, triangle_uvs) -> list[tuple[float, float, float, float]]:
    parent = list(range(len(triangles)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        parent[find(left)] = find(right)

    edges: dict[tuple[int, int], int] = {}
    for index, triangle in enumerate(triangles):
        indices = [int(corner[1]) for corner in triangle]
        for start, stop in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((indices[start], indices[stop])))
            if key in edges:
                union(index, edges[key])
            else:
                edges[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(triangles)):
        groups.setdefault(find(index), []).append(index)
    boxes = []
    for members in groups.values():
        points = np.vstack([triangle_uvs[index] for index in members])
        low = points.min(axis=0)
        high = points.max(axis=0)
        boxes.append((float(low[0]), float(low[1]), float(high[0]), float(high[1])))
    return boxes


def _padding(islands: list[tuple[float, float, float, float]], padding: float) -> None:
    for index, left in enumerate(islands):
        for right in islands[index + 1 :]:
            gap_x = max(left[0], right[0]) - min(left[2], right[2])
            gap_y = max(left[1], right[1]) - min(left[3], right[3])
            separated = gap_x > 0.0 or gap_y > 0.0
            nearest = max(gap_x, 0.0) + max(gap_y, 0.0)
            if separated and nearest < padding and gap_x < padding and gap_y < padding:
                raise ObjError(f"reason=insufficient_island_padding gap={nearest}")

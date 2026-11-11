"""Spherical-cap shell and inward-facing room box.

The cap area fraction on a unit sphere is ``(1 - cos(theta_max)) / 2``.
Outer-face normals point away from the origin. Inner-face normals point
toward the origin. Rim normals point toward increasing polar angle, out
of the dielectric volume.
"""

from __future__ import annotations

import numpy as np

FLOOR_GAP = 1e-3


def theta_max(coverage: float) -> float:
    """Polar half-angle whose cap covers ``coverage`` of the sphere."""
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    cos_theta = 1.0 - 2.0 * float(coverage)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("film axis has zero length")
    direction = direction / norm
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    tangent = np.cross(helper, direction)
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(direction, tangent)
    return direction, tangent, bitangent


def _direction(axis, tangent, bitangent, theta: float, phi: float) -> np.ndarray:
    return (
        np.cos(theta) * axis
        + np.sin(theta) * (np.cos(phi) * tangent + np.sin(phi) * bitangent)
    )


def _align(vertices: np.ndarray, tri: tuple[int, int, int], desired: np.ndarray) -> tuple[int, int, int]:
    a, b, c = tri
    normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
    if float(np.dot(normal, desired)) < 0.0:
        return (a, c, b)
    return (a, b, c)


def cap_shell(
    radius: float,
    thickness: float,
    coverage: float,
    axis: tuple[float, float, float] | np.ndarray,
    *,
    n_rings: int = 16,
    n_phi: int = 48,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, np.ndarray]:
    """Closed dielectric cap. Vertices are scaled after the sphere is built."""
    radius = float(radius)
    thickness = float(thickness)
    if radius <= 0.0 or thickness <= 0.0 or thickness >= radius:
        raise ValueError(f"need 0 < thickness < radius, got thickness={thickness} radius={radius}")
    polar = theta_max(coverage)
    direction, tangent, bitangent = _basis(np.asarray(axis, dtype=np.float64))
    phis = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    thetas = np.linspace(polar / int(n_rings), polar, int(n_rings))

    def surface(shell_radius: float) -> np.ndarray:
        pole = shell_radius * direction
        rings = []
        for theta in thetas:
            ring = [_direction(direction, tangent, bitangent, float(theta), float(phi)) * shell_radius for phi in phis]
            rings.append(ring)
        return np.vstack([pole[None, :], np.asarray(rings, dtype=np.float64).reshape(-1, 3)])

    outer = surface(radius)
    inner = surface(radius - thickness)
    vertices = np.vstack([outer, inner])
    inner_offset = outer.shape[0]
    n_phi = int(n_phi)

    def ring_index(base: int, ring: int, phi_index: int) -> int:
        return base + 1 + ring * n_phi + (phi_index % n_phi)

    faces: list[tuple[int, int, int]] = []
    groups = {"outer": [], "inner": [], "rim": []}

    def add(group: str, tri: tuple[int, int, int], desired: np.ndarray) -> None:
        aligned = _align(vertices, tri, desired)
        groups[group].append(len(faces))
        faces.append(aligned)

    for shell, base, sign in (("outer", 0, 1.0), ("inner", inner_offset, -1.0)):
        pole = base
        for phi_index in range(n_phi):
            a = ring_index(base, 0, phi_index)
            b = ring_index(base, 0, phi_index + 1)
            centroid = (vertices[pole] + vertices[a] + vertices[b]) / 3.0
            add(shell, (pole, a, b), sign * centroid)
        for ring in range(int(n_rings) - 1):
            for phi_index in range(n_phi):
                a = ring_index(base, ring, phi_index)
                b = ring_index(base, ring, phi_index + 1)
                c = ring_index(base, ring + 1, phi_index)
                d = ring_index(base, ring + 1, phi_index + 1)
                centroid = (vertices[a] + vertices[b] + vertices[c]) / 3.0
                add(shell, (a, c, b), sign * centroid)
                centroid = (vertices[b] + vertices[c] + vertices[d]) / 3.0
                add(shell, (b, c, d), sign * centroid)

    last = int(n_rings) - 1
    for phi_index, phi in enumerate(phis):
        o0 = ring_index(0, last, phi_index)
        o1 = ring_index(0, last, phi_index + 1)
        i0 = ring_index(inner_offset, last, phi_index)
        i1 = ring_index(inner_offset, last, phi_index + 1)
        phi_mid = float(phi + (phis[1] - phis[0]) / 2.0 if n_phi > 1 else phi)
        desired = _direction(direction, tangent, bitangent, polar + 0.01, phi_mid)
        desired = desired - direction * np.dot(desired, direction)
        add("rim", (o0, i0, o1), desired)
        add("rim", (o1, i0, i1), desired)

    scale_array = np.asarray(scale, dtype=np.float64)
    vertices = vertices * scale_array
    face_array = np.asarray(faces, dtype=np.int64)
    return {
        "vertices": vertices,
        "faces": face_array,
        "groups": {name: np.asarray(index, dtype=np.int64) for name, index in groups.items()},
    }


def uv_sphere(
    radius: float,
    *,
    n_rings: int = 32,
    n_phi: int = 64,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, np.ndarray]:
    """Closed UV sphere centered at the origin. Faces point outward."""
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    thetas = np.linspace(0.0, np.pi, int(n_rings) + 1)
    phis = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    vertices = []
    for index, theta in enumerate(thetas):
        if index == 0 or index == len(thetas) - 1:
            vertices.append([0.0, 0.0, radius * np.cos(theta)])
            continue
        for phi in phis:
            vertices.append(
                [
                    radius * np.sin(theta) * np.cos(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(theta),
                ]
            )
    vertices = np.asarray(vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    pole_s = 0
    pole_n = vertices.shape[0] - 1
    first_ring = 1
    faces = []

    def index(ring: int, phi_index: int) -> int:
        return first_ring + ring * int(n_phi) + (phi_index % int(n_phi))

    for phi_index in range(int(n_phi)):
        faces.append(_align(vertices, (pole_s, index(0, phi_index + 1), index(0, phi_index)), vertices[pole_s]))
    for ring in range(int(n_rings) - 2):
        for phi_index in range(int(n_phi)):
            a = index(ring, phi_index)
            b = index(ring, phi_index + 1)
            c = index(ring + 1, phi_index)
            d = index(ring + 1, phi_index + 1)
            faces.append(_align(vertices, (a, b, c), vertices[a]))
            faces.append(_align(vertices, (b, d, c), vertices[b]))
    last = int(n_rings) - 2
    for phi_index in range(int(n_phi)):
        faces.append(_align(vertices, (pole_n, index(last, phi_index), index(last, phi_index + 1)), vertices[pole_n]))
    return {"vertices": vertices, "faces": np.asarray(faces, dtype=np.int64)}


def reference_points(
    radius: float,
    coverage: float,
    axis: tuple[float, float, float] | np.ndarray,
    *,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    floor_z: float = 0.0,
) -> dict[str, np.ndarray]:
    """One film point, one uncovered ball point, and one floor point."""
    polar = theta_max(coverage)
    direction, tangent, bitangent = _basis(np.asarray(axis, dtype=np.float64))
    scale_array = np.asarray(scale, dtype=np.float64)
    film = _direction(direction, tangent, bitangent, polar * 0.3, 0.0) * float(radius)
    # Stay on the front hemisphere when the cap does not cover it.
    uncovered_theta = min(polar + 0.05, np.pi * 0.5 - 0.02)
    if uncovered_theta <= polar:
        uncovered_theta = min(np.pi * 0.5 + 0.2, polar + 0.15)
    uncovered = _direction(direction, tangent, bitangent, float(uncovered_theta), np.pi) * float(radius)
    return {
        "film": film * scale_array,
        "uncovered": uncovered * scale_array,
        "floor": np.array([0.2, 0.2, floor_z + 1e-4], dtype=np.float64),
        "uncovered_visible": bool(uncovered_theta > polar and uncovered_theta < np.pi * 0.5),
    }


def room_box(low: np.ndarray, high: np.ndarray) -> dict[str, np.ndarray]:
    """Six faces. Normals point into the box so a camera inside sees them."""
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    x0, y0, z0 = low
    x1, y1, z1 = high
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    # (a, b, c) listed so the right-hand normal points inward.
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # floor +Z
            [4, 6, 5],
            [4, 7, 6],  # ceiling -Z
            [0, 4, 5],
            [0, 5, 1],  # y0 +Y
            [3, 2, 6],
            [3, 6, 7],  # y1 -Y
            [0, 3, 7],
            [0, 7, 4],  # x0 +X
            [1, 5, 6],
            [1, 6, 2],  # x1 -X
        ],
        dtype=np.int64,
    )
    return {"vertices": vertices, "faces": faces}


def floor_z(radius: float, scale_z: float, gap: float = FLOOR_GAP) -> float:
    """Ball center is the origin. The floor sits ``gap`` meters under the ball."""
    return -float(radius) * float(scale_z) - float(gap)


# Fixed open-front room. The camera looks toward +Y, so -X is left and +Y is the back wall.
CORNELL_LOW = np.array([-0.3, -0.3, 0.0])
CORNELL_HIGH = np.array([0.3, 0.3, 0.6])
PLATFORM_SIZE = np.array([0.12, 0.12, 0.04])
PLATFORM_REFLECTANCE = (0.76, 0.64, 0.08)
DRAGON_REFLECTANCE = (0.70, 0.05, 0.04)
DRAGON_HEIGHT = 0.12
DRAGON_CENTER_XY = (0.18, 0.0)
WALL_COLORS = {
    "wall-left": (0.63, 0.06, 0.04),
    "wall-right": (0.15, 0.45, 0.09),
    "wall-back": (0.15, 0.25, 0.65),
}


def cornell_shell(low: np.ndarray | None = None, high: np.ndarray | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Open-front room. The face toward -Y is omitted so the camera looks in."""
    box = room_box(CORNELL_LOW if low is None else low, CORNELL_HIGH if high is None else high)
    groups = {
        "floor": (0, 2),
        "ceiling": (2, 4),
        "wall-back": (6, 8),
        "wall-left": (8, 10),
        "wall-right": (10, 12),
    }
    return {
        name: {"vertices": box["vertices"], "faces": box["faces"][start:stop]}
        for name, (start, stop) in groups.items()
    }


def solid_box(low: np.ndarray, high: np.ndarray) -> dict[str, np.ndarray]:
    """Closed box. Normals point outward."""
    mesh = room_box(low, high)
    return {"vertices": mesh["vertices"], "faces": np.flip(mesh["faces"], axis=1).copy()}


def platform_mesh(floor_z: float = 0.0) -> dict[str, np.ndarray]:
    """Pedestal centered on the floor. ``PLATFORM_SIZE`` is width, depth, height."""
    width, depth, height = PLATFORM_SIZE
    low = np.array([-width / 2.0, -depth / 2.0, float(floor_z)])
    high = np.array([width / 2.0, depth / 2.0, float(floor_z) + height])
    return solid_box(low, high)


def ball_center_offset(radius: float, scale_z: float, gap: float, floor_z: float = 0.0) -> np.ndarray:
    """Move a sphere built at the origin up onto the platform, leaving ``gap`` under it."""
    lift = float(floor_z) + float(PLATFORM_SIZE[2]) + float(gap) + float(radius) * float(scale_z)
    return np.array([0.0, 0.0, lift], dtype=np.float64)


def write_obj(path, vertices: np.ndarray, faces: np.ndarray, *, smooth: bool = True) -> None:
    """Wavefront file. Face indices are 1-based.

    ``smooth`` writes one area-weighted normal per vertex. A box needs
    ``smooth=False``: shared corners would mix the face normals, and a
    shading normal that points through the floor makes Mitsuba drop that edge.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if smooth:
        used = np.unique(faces.reshape(-1))
        remap = np.full(vertices.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.shape[0])
        vertices = vertices[used]
        faces = remap[faces]
        normals = np.zeros_like(vertices)
        for face in faces:
            normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
            normals[face] += normal
    else:
        corners = vertices[faces]
        face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        vertices = corners.reshape(-1, 3)
        normals = np.repeat(face_normals, 3, axis=0)
        faces = np.arange(vertices.shape[0], dtype=np.int64).reshape(-1, 3)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0.0] = 1.0
    normals = normals / lengths[:, None]
    lines = ["# generated by sensor_sim.film.geometry"]
    for vertex in vertices:
        lines.append(f"v {vertex[0]:.10f} {vertex[1]:.10f} {vertex[2]:.10f}")
    for normal in normals:
        lines.append(f"vn {normal[0]:.10f} {normal[1]:.10f} {normal[2]:.10f}")
    for face in faces:
        corners = " ".join(f"{int(index) + 1}//{int(index) + 1}" for index in face)
        lines.append(f"f {corners}")
    path.write_text("\n".join(lines) + "\n")

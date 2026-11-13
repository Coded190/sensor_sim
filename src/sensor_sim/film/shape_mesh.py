"""Filmed shapes for the held-out normal experiment.

Each object is a diffuse solid with a closed dielectric shell 20 µm outside
it. Spheres are a minority of the training list and are absent from the test
list. The catalog is fixed by seed 0, so two machines build the same 60 shapes.
"""

from __future__ import annotations

import numpy as np

from sensor_sim.film.geometry import _align

FILM_THICKNESS = 2e-5
SOLID_GAP = 1e-5
MIN_FILM_FRACTION = 0.15
TRAIN_COUNTS = {
    "sphere": 4,
    "ellipsoid": 8,
    "rounded_box": 8,
    "capsule": 6,
    "torus": 6,
    "cylinder": 6,
    "cone": 6,
    "bumpy": 6,
}
TEST_COUNTS = {
    "ellipsoid": 2,
    "rounded_box": 2,
    "bumpy": 2,
    "capsule": 1,
    "torus": 1,
    "cylinder": 1,
    "cone": 1,
}


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals. A zero-length sum becomes +Z."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    corners = vertices[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(vertices)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    missing = lengths[:, 0] < 1e-15
    normals = np.divide(normals, np.maximum(lengths, 1e-15))
    normals[missing] = np.array([0.0, 0.0, 1.0])
    return normals


def _unique_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undirected edges and how many faces share each one."""
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    if edges.shape[0] == 0:
        return edges, np.zeros(0, dtype=np.int64)
    change = np.ones(edges.shape[0], dtype=bool)
    change[1:] = np.any(edges[1:] != edges[:-1], axis=1)
    starts = np.flatnonzero(change)
    counts = np.diff(np.append(starts, edges.shape[0]))
    return edges[change], counts


def max_sagitta(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Largest circular-arc sagitta of an edge, ``length * turning_angle / 8``.

    The turning angle is the angle between the area-weighted normals at the
    two endpoints. On a smooth patch this is the chord sagitta of that arc.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    normals = vertex_normals(vertices, faces)
    edges, _counts = _unique_edges(faces)
    if edges.shape[0] == 0:
        return 0.0
    cosine = np.clip(np.sum(normals[edges[:, 0]] * normals[edges[:, 1]], axis=1), -1.0, 1.0)
    turning = np.arccos(cosine)
    length = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    return float(np.max(length * turning / 8.0))


def is_watertight(faces: np.ndarray) -> bool:
    """Every undirected edge belongs to exactly two faces."""
    _edges, counts = _unique_edges(faces)
    return bool(counts.size) and bool(np.all(counts == 2))


def edge_face_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    """How many faces share each undirected edge."""
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        for start, stop in ((0, 1), (1, 2), (2, 0)):
            first, second = int(face[start]), int(face[stop])
            key = (first, second) if first < second else (second, first)
            counts[key] = counts.get(key, 0) + 1
    return counts


def ring_count(radius: float, thickness: float = FILM_THICKNESS) -> int:
    """Segments around a circle whose chord sagitta stays under ``thickness``."""
    curved = max(float(radius), 1e-4)
    # The 1.25 factor covers triangle edges that are longer than the generating chord.
    needed = int(np.ceil(1.25 * 2.0 * np.pi * np.sqrt(curved / (8.0 * float(thickness))))) + 8
    return int(np.clip(needed, 64, 192))


def _signed_power(value: float, exponent: float) -> float:
    return float(np.sign(value) * (abs(value) ** exponent))


def _latlong(sample, n_rings: int, n_phi: int) -> dict[str, np.ndarray]:
    """Closed surface. ``sample(theta, phi)`` uses theta 0 at +Z and theta pi at -Z."""
    thetas = np.linspace(0.0, np.pi, int(n_rings) + 1)
    phis = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    rings = []
    for index, theta in enumerate(thetas):
        if index == 0 or index == len(thetas) - 1:
            rings.append(np.asarray([sample(float(theta), 0.0)], dtype=np.float64))
            continue
        rings.append(np.asarray([sample(float(theta), float(phi)) for phi in phis], dtype=np.float64))
    return _stitch(rings)


def _stitch(rings: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Connect consecutive rings. A one-vertex ring is a pole."""
    vertices = np.vstack(rings)
    offsets = []
    cursor = 0
    for ring in rings:
        offsets.append(cursor)
        cursor += int(ring.shape[0])
    faces: list[tuple[int, int, int]] = []
    for ring_index in range(len(rings) - 1):
        above = rings[ring_index]
        below = rings[ring_index + 1]
        above_at = offsets[ring_index]
        below_at = offsets[ring_index + 1]
        if above.shape[0] == 1 and below.shape[0] > 1:
            count = int(below.shape[0])
            for step in range(count):
                tri = (above_at, below_at + (step + 1) % count, below_at + step)
                desired = below[step]
                faces.append(_align(vertices, tri, desired))
        elif below.shape[0] == 1 and above.shape[0] > 1:
            count = int(above.shape[0])
            for step in range(count):
                tri = (below_at, above_at + step, above_at + (step + 1) % count)
                desired = above[step]
                faces.append(_align(vertices, tri, desired))
        elif above.shape[0] == below.shape[0]:
            count = int(above.shape[0])
            for step in range(count):
                a = above_at + step
                b = above_at + (step + 1) % count
                c = below_at + step
                d = below_at + (step + 1) % count
                faces.append(_align(vertices, (a, c, b), vertices[a]))
                faces.append(_align(vertices, (b, c, d), vertices[b]))
        else:
            raise RuntimeError(f"cannot stitch rings of {above.shape[0]} and {below.shape[0]}")
    return {"vertices": vertices, "faces": np.asarray(faces, dtype=np.int64)}


def _scale_to_radius(mesh: dict[str, np.ndarray], radius: float) -> dict[str, np.ndarray]:
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    peak = float(np.max(np.linalg.norm(vertices, axis=1)))
    if peak <= 0.0:
        raise RuntimeError("mesh has no extent")
    return {"vertices": vertices * (float(radius) / peak), "faces": mesh["faces"]}


def sphere_mesh(radius: float, n: int) -> dict[str, np.ndarray]:
    return _latlong(lambda theta, phi: _sphere_point(theta, phi, radius), n, n)


def _sphere_point(theta: float, phi: float, radius: float) -> np.ndarray:
    return np.array(
        [
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ],
        dtype=np.float64,
    )


def ellipsoid_mesh(radii: tuple[float, float, float], n: int) -> dict[str, np.ndarray]:
    rx, ry, rz = (float(value) for value in radii)

    def sample(theta: float, phi: float) -> np.ndarray:
        point = _sphere_point(theta, phi, 1.0)
        return np.array([point[0] * rx, point[1] * ry, point[2] * rz], dtype=np.float64)

    return _latlong(sample, n, n)


def rounded_box_mesh(radii: tuple[float, float, float], exponent: float, n: int) -> dict[str, np.ndarray]:
    """Superellipsoid. Exponent 2 is an ellipsoid. Larger exponents are boxier."""
    power = 2.0 / float(exponent)
    rx, ry, rz = (float(value) for value in radii)

    def sample(theta: float, phi: float) -> np.ndarray:
        sine = np.sin(theta)
        return np.array(
            [
                rx * _signed_power(sine, power) * _signed_power(np.cos(phi), power),
                ry * _signed_power(sine, power) * _signed_power(np.sin(phi), power),
                rz * _signed_power(np.cos(theta), power),
            ],
            dtype=np.float64,
        )

    return _latlong(sample, n, n)


def bumpy_mesh(radii: tuple[float, float, float], amplitude: float, n: int) -> dict[str, np.ndarray]:
    """Ellipsoid plus a smooth quadrupole. ``amplitude`` is a fraction of the local radius.

    ``sin(theta)^2 cos(2 phi)`` is zero at both poles, so the wave does not
    crease the latitude fan. A wave that stays finite at the pole makes the
    offset shell thicker than the chord sagitta.
    """
    base = ellipsoid_mesh(radii, n)
    vertices = base["vertices"]
    lengths = np.linalg.norm(vertices, axis=1, keepdims=True)
    direction = vertices / np.maximum(lengths, 1e-12)
    theta = np.arccos(np.clip(direction[:, 2], -1.0, 1.0))
    phi = np.arctan2(direction[:, 1], direction[:, 0])
    wave = 1.0 + float(amplitude) * (np.sin(theta) ** 2) * np.cos(2.0 * phi)
    return {"vertices": vertices * wave[:, None], "faces": base["faces"]}


def capsule_mesh(radius: float, length: float, n: int) -> dict[str, np.ndarray]:
    """Cylinder of ``length`` with hemispherical caps of ``radius``."""
    radius = float(radius)
    half = float(length) / 2.0
    n_cap = max(8, int(n) // 4)
    n_phi = int(n)
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    rings: list[np.ndarray] = [np.array([[0.0, 0.0, -half - radius]], dtype=np.float64)]
    for theta in np.linspace(np.pi - np.pi / (2 * n_cap), np.pi / 2.0, n_cap, endpoint=False):
        local = np.array([_sphere_point(float(theta), float(phi), radius) for phi in phis])
        local[:, 2] -= half
        rings.append(local)
    for z_value in np.linspace(-half, half, max(2, n_cap)):
        ring = np.stack([radius * np.cos(phis), radius * np.sin(phis), np.full(n_phi, z_value)], axis=1)
        rings.append(ring)
    for theta in np.linspace(np.pi / 2.0 - np.pi / (2 * n_cap), 0.0, n_cap, endpoint=False):
        local = np.array([_sphere_point(float(theta), float(phi), radius) for phi in phis])
        local[:, 2] += half
        rings.append(local)
    rings.append(np.array([[0.0, 0.0, half + radius]], dtype=np.float64))
    return _stitch(rings)


def torus_mesh(major: float, minor: float, n: int) -> dict[str, np.ndarray]:
    major = float(major)
    minor = float(minor)
    n_major = int(n)
    n_minor = max(16, int(n) // 2)
    majors = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
    minors = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
    vertices = []
    desired = []
    for major_angle in majors:
        for minor_angle in minors:
            center = np.array([np.cos(major_angle), np.sin(major_angle), 0.0])
            tube = np.cos(minor_angle) * center + np.array([0.0, 0.0, np.sin(minor_angle)])
            vertices.append((major + minor * np.cos(minor_angle)) * center + np.array([0.0, 0.0, minor * np.sin(minor_angle)]))
            desired.append(tube)
    vertices = np.asarray(vertices, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    faces = []

    def index(major_index: int, minor_index: int) -> int:
        return (major_index % n_major) * n_minor + (minor_index % n_minor)

    for major_index in range(n_major):
        for minor_index in range(n_minor):
            a = index(major_index, minor_index)
            b = index(major_index, minor_index + 1)
            c = index(major_index + 1, minor_index)
            d = index(major_index + 1, minor_index + 1)
            faces.append(_align(vertices, (a, b, c), desired[a]))
            faces.append(_align(vertices, (b, d, c), desired[b]))
    return {"vertices": vertices, "faces": np.asarray(faces, dtype=np.int64)}


def _profile_mesh(points: list[tuple[float, float]], n_phi: int) -> dict[str, np.ndarray]:
    """Revolve a (z, r) profile. A zero radius is a pole."""
    cleaned = [points[0]]
    for point in points[1:]:
        if abs(point[0] - cleaned[-1][0]) + abs(point[1] - cleaned[-1][1]) < 1e-9:
            continue
        cleaned.append(point)
    rings = []
    for z_value, ring_radius in cleaned:
        if ring_radius <= 1e-8:
            rings.append(np.array([[0.0, 0.0, z_value]], dtype=np.float64))
            continue
        phis = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
        rings.append(
            np.stack(
                [
                    ring_radius * np.cos(phis),
                    ring_radius * np.sin(phis),
                    np.full(int(n_phi), z_value),
                ],
                axis=1,
            )
        )
    return _stitch(rings)


def cylinder_mesh(radius: float, height: float, n: int) -> dict[str, np.ndarray]:
    """Closed cylinder. Both rims are quarter-circle fillets so the offset shell stays shut."""
    radius = float(radius)
    half = float(height) / 2.0
    fillet = min(0.22 * radius, 0.30 * half)
    steps = max(8, int(n) // 12)
    points = [(-half, 0.0), (-half, radius - fillet)]
    for index in range(1, steps + 1):
        angle = 0.5 * np.pi * index / steps
        points.append((-half + fillet * (1.0 - np.cos(angle)), radius - fillet + fillet * np.sin(angle)))
    z_lo = -half + fillet
    z_hi = half - fillet
    wall_steps = max(2, int(n) // 8)
    for index in range(1, wall_steps):
        points.append((z_lo + (z_hi - z_lo) * index / wall_steps, radius))
    for index in range(1, steps + 1):
        angle = 0.5 * np.pi * index / steps
        points.append((z_hi + fillet * np.sin(angle), radius - fillet + fillet * np.cos(angle)))
    points.append((half, 0.0))
    return _profile_mesh(points, int(n))


def cone_mesh(radius: float, height: float, n: int) -> dict[str, np.ndarray]:
    """Closed cone. The tip is a sphere cap and the base rim is a fillet, both tangent to the side."""
    radius = float(radius)
    height = float(height)
    fillet = min(0.12 * radius, 0.08 * height, 0.2 * radius, 0.15 * height)
    # phi is the angle from +Z to the tip radius that meets the side at a right angle.
    # The side then leans out at semi-angle atan(radius / height).
    phi = float(np.clip(np.arctan2(height, radius), 0.55, 1.25))
    sin_phi = float(np.sin(phi))
    cos_phi = float(np.cos(phi))
    tip_z = height - fillet
    tangent_z = tip_z + fillet * cos_phi
    tangent_r = fillet * sin_phi
    direction_z = -sin_phi
    direction_r = cos_phi
    # Fillet center sits on z = fillet and one fillet-radius inside the side.
    on_plane = (fillet - tangent_z) / direction_z
    radius_on_plane = tangent_r + on_plane * direction_r
    center_r = None
    for sign in (1.0, -1.0):
        candidate = tangent_r + (sign * fillet - (fillet - tangent_z) * cos_phi) / sin_phi
        if fillet * 0.25 < candidate < radius_on_plane - 1e-6:
            center_r = float(candidate)
            break
    if center_r is None:
        raise RuntimeError("cone fillet does not fit inside the side")
    # Outward normal of the side is (cos phi, sin phi) in (z, r).
    base_z = fillet + fillet * cos_phi
    base_r = center_r + fillet * sin_phi
    if not (0.0 < base_z < tangent_z):
        raise RuntimeError("cone side meets the base fillet on the wrong side of the tip")
    steps = max(8, int(n) // 12)
    points = [(0.0, 0.0), (0.0, center_r)]
    for index in range(1, steps + 1):
        # Arc from straight down (pi) to the side normal (phi).
        angle = np.pi + (phi - np.pi) * index / steps
        points.append((fillet + fillet * np.cos(angle), center_r + fillet * np.sin(angle)))
    side_steps = max(4, int(n) // 8)
    for index in range(1, side_steps):
        fraction = index / side_steps
        points.append((base_z + (tangent_z - base_z) * fraction, base_r + (tangent_r - base_r) * fraction))
    for index in range(1, steps + 1):
        angle = phi + (0.0 - phi) * index / steps
        points.append((tip_z + fillet * np.cos(angle), fillet * np.sin(angle)))
    points = [(z_value - height / 2.0, ring_radius) for z_value, ring_radius in points]
    return _profile_mesh(points, int(n))


def mesh_for_family(family: str, spec: dict, radius: float) -> dict[str, np.ndarray]:
    """Unit-scale family mesh, then scaled so its farthest vertex is ``radius``."""
    tight = min(float(radius), float(spec.get("minor_radius", radius)))
    count = ring_count(tight)
    # A rounded box concentrates curvature on its edges. 256 rings keeps the
    # chord sagitta under 20 µm for an 8 cm part without a global subdivision.
    if family == "rounded_box":
        count = 256
    if family == "sphere":
        mesh = sphere_mesh(1.0, count)
    elif family == "ellipsoid":
        mesh = ellipsoid_mesh(tuple(spec["radii"]), count)
    elif family == "rounded_box":
        mesh = rounded_box_mesh(tuple(spec["radii"]), float(spec["exponent"]), count)
    elif family == "capsule":
        mesh = capsule_mesh(float(spec["tube_radius"]), float(spec["body_length"]), count)
    elif family == "torus":
        mesh = torus_mesh(float(spec["major"]), float(spec["minor"]), count)
    elif family == "cylinder":
        mesh = cylinder_mesh(float(spec["tube_radius"]), float(spec["body_length"]), count)
    elif family == "cone":
        mesh = cone_mesh(float(spec["tube_radius"]), float(spec["body_length"]), count)
    elif family == "bumpy":
        mesh = bumpy_mesh(tuple(spec["radii"]), float(spec["amplitude"]), count)
    else:
        raise ValueError(f"unknown shape family {family}")
    return _scale_to_radius(mesh, radius)


def subdivide(mesh: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split every triangle into four. Midpoints stay on the current chords."""
    vertices = [row.copy() for row in np.asarray(mesh["vertices"], dtype=np.float64)]
    midpoint: dict[tuple[int, int], int] = {}

    def middle(first: int, second: int) -> int:
        key = (first, second) if first < second else (second, first)
        found = midpoint.get(key)
        if found is None:
            found = len(vertices)
            midpoint[key] = found
            vertices.append(0.5 * (vertices[first] + vertices[second]))
        return found

    faces = []
    for face in np.asarray(mesh["faces"], dtype=np.int64):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
        faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
    return {"vertices": np.asarray(vertices, dtype=np.float64), "faces": np.asarray(faces, dtype=np.int64)}


def refine_until(mesh: dict[str, np.ndarray], thickness: float = FILM_THICKNESS, levels: int = 3) -> dict[str, np.ndarray]:
    """Subdivide while an edge sagitta is at least the film thickness."""
    refined = mesh
    for _level in range(int(levels)):
        if max_sagitta(refined["vertices"], refined["faces"]) < float(thickness):
            return refined
        refined = subdivide(refined)
    return refined


def solid_and_film(mesh: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Diffuse solid inset by 10 µm and a 20 µm dielectric shell outside the base surface."""
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    faces = np.asarray(mesh["faces"], dtype=np.int64)
    normals = vertex_normals(vertices, faces)
    solid = {"vertices": vertices - SOLID_GAP * normals, "faces": faces.copy()}
    outer = vertices + FILM_THICKNESS * normals
    count = vertices.shape[0]
    inner_faces = np.stack([faces[:, 0], faces[:, 2], faces[:, 1]], axis=1) + count
    film = {
        "vertices": np.vstack([outer, vertices]),
        "faces": np.vstack([faces, inner_faces]),
    }
    return solid, film


def place_mesh(vertices: np.ndarray, yaw: float, center_xy: tuple[float, float]) -> np.ndarray:
    """Yaw about Z and rest the lowest vertex 1 mm above z = 0."""
    cosine, sine = np.cos(float(yaw)), np.sin(float(yaw))
    rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    placed = np.asarray(vertices, dtype=np.float64) @ rotation.T
    placed[:, 0] += float(center_xy[0])
    placed[:, 1] += float(center_xy[1])
    placed[:, 2] += 0.001 - float(placed[:, 2].min())
    return placed


def build_shape(spec: dict) -> dict:
    """World-space solid, film, and a point on the film's camera-facing side."""
    mesh = refine_until(mesh_for_family(spec["family"], spec, float(spec["radius"])))
    sagitta = max_sagitta(mesh["vertices"], mesh["faces"])
    if sagitta >= FILM_THICKNESS:
        raise RuntimeError(
            f"{spec['family']} sagitta {sagitta:.3e} m is not under the {FILM_THICKNESS:.3e} m film"
        )
    solid, film = solid_and_film(mesh)
    center_xy = (float(spec["x"]), float(spec["y"]))
    solid["vertices"] = place_mesh(solid["vertices"], float(spec["yaw"]), center_xy)
    film["vertices"] = place_mesh(film["vertices"], float(spec["yaw"]), center_xy)
    outer = film["vertices"][: mesh["vertices"].shape[0]]
    probe = outer[int(np.argmin(outer[:, 1]))]
    center = solid["vertices"].mean(axis=0)
    camera = camera_for_mesh(outer)
    fraction = frame_fraction(outer, camera)
    if fraction < MIN_FILM_FRACTION:
        camera = camera_for_mesh(outer, fill=0.94)
        fraction = frame_fraction(outer, camera)
    if fraction < MIN_FILM_FRACTION:
        raise RuntimeError(
            f"{spec['family']} projected hull covers {fraction:.3f} of the frame, below {MIN_FILM_FRACTION}"
        )
    return {
        "solid": solid,
        "film": film,
        "probe": probe,
        "center": center,
        "sagitta_m": sagitta,
        "camera": camera,
        "frame_fraction": fraction,
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def project_ndc(vertices: np.ndarray, origin, target, fov_deg: float = 55.0) -> np.ndarray:
    """Mitsuba perspective NDC for a square film. (0.5, 0.5) is the principal point.

    ``fov_deg`` is the horizontal field of view, which is also the vertical
    field on a square film. Image right is decreasing camera X, and image up
    is decreasing NDC y, matching ``sensor.projection_transform``.
    """
    origin = np.asarray(origin, dtype=np.float64)
    forward = _unit(np.asarray(target, dtype=np.float64) - origin)
    cam_x = _unit(np.cross(np.array([0.0, 0.0, 1.0]), forward))
    cam_y = _unit(np.cross(forward, cam_x))
    relative = np.asarray(vertices, dtype=np.float64) - origin
    depth = relative @ forward
    if np.any(depth <= 1e-5):
        raise RuntimeError("a mesh vertex is on or behind the camera")
    scale = 0.5 / np.tan(np.deg2rad(float(fov_deg)) / 2.0)
    ndc_x = 0.5 - scale * (relative @ cam_x) / depth
    ndc_y = 0.5 - scale * (relative @ cam_y) / depth
    return np.stack([ndc_x, ndc_y], axis=1)


def _hull_area(points: np.ndarray) -> float:
    """Area of the convex hull. Points are Nx2."""
    points = np.asarray(points, dtype=np.float64)
    points = np.unique(np.round(points, decimals=8), axis=0)
    if points.shape[0] < 3:
        return 0.0
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        return float((first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower: list[np.ndarray] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if hull.shape[0] < 3:
        return 0.0
    x_coord, y_coord = hull[:, 0], hull[:, 1]
    return 0.5 * abs(float(np.dot(x_coord, np.roll(y_coord, 1)) - np.dot(y_coord, np.roll(x_coord, 1))))


def frame_fraction(vertices: np.ndarray, camera: dict) -> float:
    """Convex-hull area of the mesh in NDC, divided by the frame area of 1.

    For these solids the silhouette is the hull. A torus seen from the side
    hides its hole, so the hull is the silhouette there as well.
    """
    ndc = project_ndc(vertices, camera["position"], camera["target"], float(camera["fov_deg"]))
    return _hull_area(ndc)


def camera_for_mesh(vertices: np.ndarray, fov_deg: float = 55.0, fill: float = 0.84) -> dict:
    """Look at the mesh from -Y and slightly above, so the longer image side fills ``fill``."""
    vertices = np.asarray(vertices, dtype=np.float64)
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    center = 0.5 * (low + high)
    extent = float(np.max(high - low))
    lift = 0.06 * extent
    lo = max(extent * 0.35, 1e-3)
    hi = max(extent * 30.0, lo * 4.0)
    best = hi
    for _step in range(28):
        distance = 0.5 * (lo + hi)
        origin = center + np.array([0.0, -distance, lift])
        ndc = project_ndc(vertices, origin, center, fov_deg)
        span = float(np.max(np.abs(ndc - 0.5)) * 2.0)
        if span > fill:
            lo = distance
        else:
            hi = distance
            best = distance
    origin = center + np.array([0.0, -best, lift])
    return {
        "position": [float(value) for value in origin],
        "target": [float(value) for value in center],
        "up": [0.0, 0.0, 1.0],
        "fov_deg": float(fov_deg),
    }


def _family_spec(family: str, local: np.random.Generator) -> dict:
    if family == "sphere":
        return {}
    if family in ("ellipsoid", "rounded_box", "bumpy"):
        radii = tuple(float(value) for value in local.uniform(0.62, 1.0, size=3))
        extra = {"radii": radii}
        if family == "rounded_box":
            # Exponent 2 is an ellipsoid. Staying under 3 keeps the edge
            # curvature coarse enough that a 192-ring mesh clears 20 µm.
            extra["exponent"] = float(local.uniform(2.55, 2.95))
        if family == "bumpy":
            extra["amplitude"] = float(local.uniform(0.035, 0.055))
        return extra
    if family == "torus":
        # The camera sees the ring from the side. A thin tube cannot cover
        # 0.15 of a square frame, so the tube is a large fraction of the ring.
        minor = float(local.uniform(0.55, 0.75))
        return {"major": 1.0, "minor": minor, "minor_radius": minor}
    if family == "cone":
        tube = float(local.uniform(0.60, 0.95))
        length = float(local.uniform(1.00, 1.55))
    elif family == "capsule":
        tube = float(local.uniform(0.45, 0.80))
        length = float(local.uniform(0.70, 1.50))
    else:
        tube = float(local.uniform(0.50, 0.85))
        length = float(local.uniform(1.05, 1.90))
    return {"tube_radius": tube, "body_length": length, "minor_radius": tube * 0.35}


def shape_catalog(seed: int = 0) -> list[dict]:
    """50 training shapes and 10 held-out shapes. Test seeds are not training seeds."""
    rng = np.random.default_rng(int(seed))
    catalog = []
    for split, counts in (("train", TRAIN_COUNTS), ("test", TEST_COUNTS)):
        index = 1
        for family, count in counts.items():
            for _repeat in range(count):
                instance_seed = int(rng.integers(1, 2**31 - 1))
                local = np.random.default_rng(instance_seed)
                spec = _family_spec(family, local)
                spec.update(
                    {
                        "split": split,
                        "index": index,
                        "family": family,
                        "folder": f"{index:03d}_{family}",
                        "instance_seed": instance_seed,
                        "radius": float(local.uniform(0.03, 0.08)),
                        "yaw": float(local.uniform(0.0, 2.0 * np.pi)),
                        "x": float(local.uniform(-0.15, 0.15)),
                        "y": float(local.uniform(-0.05, 0.12)),
                    }
                )
                catalog.append(spec)
                index += 1
    assert_catalog(catalog)
    return catalog


def assert_catalog(catalog: list[dict]) -> None:
    """The split the held-out run is allowed to render."""
    train = [item for item in catalog if item["split"] == "train"]
    test = [item for item in catalog if item["split"] == "test"]
    if len(train) != 50 or len(test) != 10:
        raise RuntimeError(f"expected 50 train and 10 test shapes, got {len(train)} and {len(test)}")
    if any(item["family"] == "sphere" for item in test):
        raise RuntimeError("the test split contains a sphere")
    train_seeds = {int(item["instance_seed"]) for item in train}
    test_seeds = {int(item["instance_seed"]) for item in test}
    if train_seeds & test_seeds:
        raise RuntimeError("a test seed also appears in train")
    if len(train_seeds) != 50 or len(test_seeds) != 10:
        raise RuntimeError("shape seeds are not unique")
    for split, items, expected in (("train", train, TRAIN_COUNTS), ("test", test, TEST_COUNTS)):
        found: dict[str, int] = {}
        for item in items:
            found[item["family"]] = found.get(item["family"], 0) + 1
        if found != expected:
            raise RuntimeError(f"{split} family counts {found} do not match {expected}")

"""Separate conformal shell specifications and geometry checks.

The target mesh is never rewritten. A finite-thickness shell has named outer,
inner, and sidewall face groups. An air gap between the shell and an opaque
substrate means the inner face is film-to-air, not a validated film-to-substrate
interface.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.narrow import _point_triangles, mesh_clearance

logger = logging.getLogger("sensor_sim.scene_gen.plastic_wrap")

SCHEMA_VERSION = "wrap_spec_v1"
THICKNESS_TOLERANCE_M = 2e-4
OFFSET_TOLERANCE_M = 1.5e-3


class WrapError(ValueError):
    """The shell failed a geometry or interface check."""


def make_wrap_spec(wrap_id: str, target_id: str, mode: str, *, offset_m: float, thickness_m: float, seed: int, wrinkle_amplitude_m: float = 0.0, wrinkle_frequency: float = 8.0, subdivision: int = 2, film_ior: float = 1.5, substrate_family: str = "diffuse") -> dict:
    if mode not in {"partial", "full"}:
        raise WrapError(f"reason=mode value={mode}")
    if offset_m <= 0.0 or thickness_m <= 0.0:
        raise WrapError("reason=non_positive_length")
    if wrinkle_amplitude_m < 0.0:
        raise WrapError("reason=wrinkle")
    return {
        "schema_version": SCHEMA_VERSION,
        "wrap_id": wrap_id,
        "target_id": target_id,
        "mode": mode,
        "offset_m": float(offset_m),
        "thickness_m": float(thickness_m),
        "subdivision": int(subdivision),
        "wrinkle_amplitude_m": float(wrinkle_amplitude_m),
        "wrinkle_frequency_per_m": float(wrinkle_frequency),
        "seed": int(seed),
        "units": "m",
        "boundary": "open_recorded" if mode == "partial" else "closed_required",
        "interfaces": interface_records(film_ior, substrate_family),
    }


def interface_records(film_ior: float, substrate_family: str) -> dict:
    """Name both boundaries. Opaque substrates are not a volumetric exterior medium."""
    opaque = substrate_family not in {"dielectric", "roughdielectric", "thindielectric"}
    inner = {
        "name": "inner",
        "intended": "plastic_to_substrate",
        "int_ior": float(film_ior),
        "ext_ior": None if opaque else float(film_ior),
        "normal": "points_out_of_the_film_toward_the_target",
        "mitsuba_convention": "int_ior is the medium opposite the geometric normal; ext_ior is the medium the normal points into",
        "status": "not_validated_opaque_substrate" if opaque else "dielectric_volume_only",
        "reason": "An opaque substrate is a surface BSDF, not a volumetric IOR, so a dielectric inner face cannot be a film-to-substrate interface without inventing a medium.",
    }
    return {
        "outer": {
            "name": "outer",
            "intended": "air_to_plastic",
            "int_ior": float(film_ior),
            "ext_ior": 1.0,
            "normal": "points_out_of_the_film_into_air",
            "mitsuba_type": "dielectric",
            "status": "assigned",
        },
        "inner": inner,
        "sidewall": {
            "name": "sidewall",
            "intended": "film_edge_in_air",
            "int_ior": float(film_ior),
            "ext_ior": 1.0,
            "mitsuba_type": "dielectric",
            "status": "assigned_partial_boundary",
            "note": "Solidify sidewalls and the open sheet boundary use the air-to-plastic dielectric. They are not the substrate interface.",
        },
    }


def _faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    return triangles


def _normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = _faces(vertices, faces)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(normal, axis=1)
    bad = length < 1e-12
    if np.any(bad):
        raise WrapError(f"reason=degenerate_faces count={int(np.count_nonzero(bad))}")
    return normal / length[:, None]


def classify_faces(vertices: np.ndarray, faces: np.ndarray, target_center: np.ndarray) -> dict:
    """Split faces into outer, inner, and sidewall by the normal's direction from the target center."""
    normals = _normals(vertices, faces)
    centroids = _faces(vertices, faces).mean(axis=1)
    outward = centroids - np.asarray(target_center, dtype=np.float64)
    outward = outward / np.linalg.norm(outward, axis=1)[:, None]
    alignment = np.sum(normals * outward, axis=1)
    groups = {"outer": np.nonzero(alignment > 0.45)[0], "inner": np.nonzero(alignment < -0.45)[0], "sidewall": np.nonzero(np.abs(alignment) <= 0.45)[0]}
    return {"normals": normals, "alignment": alignment, "groups": {name: faces[index] for name, index in groups.items()}, "counts": {name: int(len(index)) for name, index in groups.items()}}


def _triangulate(faces) -> np.ndarray:
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append([int(face[0]), int(face[1]), int(face[2])])
        elif len(face) == 4:
            triangles.append([int(face[0]), int(face[1]), int(face[2])])
            triangles.append([int(face[0]), int(face[2]), int(face[3])])
        else:
            raise WrapError(f"reason=face_size size={len(face)}")
    return np.asarray(triangles, dtype=np.int64)


def shell_report(shell: dict, target: dict, spec: dict) -> dict:
    """Geometry checks. Optical film-to-substrate status stays not validated when a clearance remains."""
    vertices = np.asarray(shell["vertices"], dtype=np.float64)
    faces = _triangulate(shell["faces"])
    if vertices.size == 0 or faces.size == 0 or not np.isfinite(vertices).all():
        raise WrapError("reason=empty_or_nonfinite")
    target_vertices = np.asarray(target["vertices"], dtype=np.float64)
    center = target_vertices.mean(axis=0)
    groups = classify_faces(vertices, faces, center)
    if groups["counts"]["outer"] == 0 or groups["counts"]["inner"] == 0:
        raise WrapError(f"reason=missing_interface counts={groups['counts']}")
    clearance = mesh_clearance({"vertices": vertices, "faces": faces}, {"vertices": target_vertices, "faces": _triangulate(target["faces"])})
    minimum = float(clearance["clearance_m"])
    if minimum < spec["offset_m"] - OFFSET_TOLERANCE_M:
        raise WrapError(f"reason=target_intersection clearance_m={minimum}")
    outer = groups["groups"]["outer"]
    inner = groups["groups"]["inner"]
    outer_centroids = vertices[outer].mean(axis=1)
    distances = _point_triangles(outer_centroids, vertices[inner]).min(axis=1)
    thickness_m = float(np.median(distances))
    if abs(thickness_m - spec["thickness_m"]) > max(THICKNESS_TOLERANCE_M, 0.5 * spec["thickness_m"]):
        raise WrapError(f"reason=thickness measured_m={thickness_m} requested_m={spec['thickness_m']}")
    air_gap = minimum > 1e-4
    report = {
        "passed_geometry": True,
        "mode": spec["mode"],
        "counts": groups["counts"],
        "minimum_clearance_m": minimum,
        "offset_m": spec["offset_m"],
        "thickness_m": thickness_m,
        "requested_thickness_m": spec["thickness_m"],
        "air_gap": air_gap,
        "optical_film_to_substrate": "not_validated" if air_gap or spec["interfaces"]["inner"]["status"].startswith("not_validated") else "assigned",
        "interfaces": spec["interfaces"],
        "boundary": spec["boundary"],
    }
    logger.info(
        "plastic_wrap event=shell wrap_id=%s mode=%s clearance_m=%.6f thickness_m=%.6f optical=%s",
        spec["wrap_id"],
        spec["mode"],
        minimum,
        thickness_m,
        report["optical_film_to_substrate"],
    )
    return report


def mesh_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    payload = np.asarray(vertices, dtype=np.float64).tobytes() + np.asarray(faces, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def store_result(directory: Path, payload: dict, valid: bool) -> bool:
    """A failed sample must not replace a result already marked valid."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "valid.json"
    if not valid and marker.is_file() and json_valid(marker):
        logger.info("plastic_wrap event=keep_valid path=%s", directory)
        return False
    (directory / "result.json").write_text(json_dumps(payload), encoding="utf-8")
    if valid:
        marker.write_text("{\"valid\": true}\n", encoding="utf-8")
    return True


def json_valid(path: Path) -> bool:
    return "true" in Path(path).read_text(encoding="utf-8")


def json_dumps(payload: dict) -> str:
    import json
    return json.dumps(payload, indent=2, default=str) + "\n"

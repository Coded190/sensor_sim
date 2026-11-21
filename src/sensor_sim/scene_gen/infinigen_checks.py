"""Qualification checks for imported Infinigen geometry. Each failure has its own reason."""

from __future__ import annotations

import numpy as np


class InfinigenCheckError(RuntimeError):
    """One planted or measured check failed."""


def _gap(first: np.ndarray, second: np.ndarray) -> float:
    low_a, high_a = first.min(axis=0), first.max(axis=0)
    low_b, high_b = second.min(axis=0), second.max(axis=0)
    delta = np.maximum(low_b - high_a, low_a - high_b)
    return float(np.max(delta))


def require_sofa_near_television(sofa: np.ndarray, television: np.ndarray, limit_m: float = 3.0) -> None:
    if _gap(sofa, television) > limit_m:
        raise InfinigenCheckError("reason=sofa_away_from_television")


def require_monitor_toward_user(screen_normal: np.ndarray, user: np.ndarray, screen_origin: np.ndarray) -> None:
    direction = np.asarray(user, dtype=np.float64) - np.asarray(screen_origin, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    if float(np.dot(np.asarray(screen_normal, dtype=np.float64), direction)) < 0.2:
        raise InfinigenCheckError("reason=monitor_away_from_user")


def require_monitor_upright(screen_up: np.ndarray) -> None:
    if float(np.asarray(screen_up, dtype=np.float64)[2]) < 0.5:
        raise InfinigenCheckError("reason=upside_down_monitor")


def require_chair_near_desk(chair: np.ndarray, desk: np.ndarray, limit_m: float = 0.8) -> None:
    if _gap(chair, desk) > limit_m:
        raise InfinigenCheckError("reason=chair_away_from_desk")


def require_door_clear(door_sweep: np.ndarray, blocker: np.ndarray) -> None:
    if _gap(door_sweep, blocker) < 0.0:
        raise InfinigenCheckError("reason=blocked_refrigerator_door")


def require_appliance_access(appliance: np.ndarray, obstacle: np.ndarray, access: np.ndarray) -> None:
    if _gap(appliance, obstacle) < 0.0 and _gap(access, obstacle) < 0.0:
        raise InfinigenCheckError("reason=inaccessible_microwave")


def require_doorway_clear(doorway: np.ndarray, blocker: np.ndarray) -> None:
    if _gap(doorway, blocker) < 0.0:
        raise InfinigenCheckError("reason=blocked_doorway")


def require_supported(vertices: np.ndarray, floor_z: float, limit_m: float = 0.05) -> None:
    gap = float(vertices[:, 2].min() - floor_z)
    if gap > limit_m:
        raise InfinigenCheckError("reason=unsupported_object")


def require_separated(first: np.ndarray, second: np.ndarray, minimum_m: float = 0.0) -> None:
    if _gap(first, second) < minimum_m:
        raise InfinigenCheckError("reason=intersection")


def require_texture(roles: set[str], required: str) -> None:
    if required not in roles:
        raise InfinigenCheckError("reason=missing_required_texture")


def require_color_space(record: dict) -> None:
    role = record.get("role")
    space = record.get("color_space")
    if role == "base_color" and space != "srgb":
        raise InfinigenCheckError("reason=wrong_color_space")
    if role in {"alpha_mitsuba", "normal_opengl", "roughness_blender"} and space != "non_color":
        raise InfinigenCheckError("reason=wrong_color_space")


def require_uvs(uv_count: int) -> None:
    if uv_count <= 0:
        raise InfinigenCheckError("reason=missing_uvs")


def require_material_assignment(names: list[str]) -> None:
    if not names or any(name in {"", "default"} for name in names):
        raise InfinigenCheckError("reason=missing_material_assignment")


def require_frame_measured(validation: str) -> None:
    if "measured" not in validation:
        raise InfinigenCheckError("reason=bad_frame")


def require_regions(source_slots: int, exported_slots: int) -> None:
    if exported_slots < source_slots:
        raise InfinigenCheckError("reason=dropped_material_region")


def require_normal_convention(flip_green: bool, bake_green: str) -> None:
    expected = bake_green == "NEG_Y"
    if flip_green != expected:
        raise InfinigenCheckError("reason=wrong_normal_convention")


def require_displacement_realized(method: str, mesh_displacement: bool) -> None:
    if method in {"DISPLACEMENT", "BOTH"} and not mesh_displacement:
        raise InfinigenCheckError("reason=displacement_not_in_mesh")


def require_scale(measured: float, expected: float, tolerance: float = 0.02) -> None:
    if abs(measured - expected) > tolerance * max(expected, 1e-6):
        raise InfinigenCheckError("reason=wrong_scale")


def require_instance_transform(measured: np.ndarray, expected: np.ndarray, tolerance: float = 1e-4) -> None:
    if np.max(np.abs(np.asarray(measured) - np.asarray(expected))) > tolerance:
        raise InfinigenCheckError("reason=wrong_instance_transform")


def require_camera_sees_room(forward: np.ndarray, furniture: np.ndarray, opening: np.ndarray) -> None:
    """Reject a camera whose forward ray meets the opening before any furniture."""
    origin = np.zeros(3)
    furniture_distance = float(np.dot(furniture - origin, forward))
    opening_distance = float(np.dot(opening - origin, forward))
    if opening_distance > 0.0 and (furniture_distance < 0.0 or opening_distance + 0.2 < furniture_distance):
        raise InfinigenCheckError("reason=camera_sees_opening")

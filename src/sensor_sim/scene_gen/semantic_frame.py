"""Explicit local frames for orientation-sensitive assets.

``right`` is ``forward × up``. A frame that is missing, degenerate, or not
right-handed is rejected. Facing tests use these vectors, not the bounding box.
"""

from __future__ import annotations

import numpy as np


class FrameError(ValueError):
    """The semantic frame does not match the required axes or the mesh."""


def _unit(vector, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise FrameError(f"reason=axis name={name}")
    length = float(np.linalg.norm(array))
    if length < 1e-8:
        raise FrameError(f"reason=degenerate name={name}")
    return array / length


def require_frame(frame: dict) -> dict:
    """Reject a frame whose axes are not unit, orthogonal, and right-handed."""
    semantic = frame.get("semantic") if "semantic" in frame else frame
    if not isinstance(semantic, dict):
        raise FrameError("reason=missing_frame")
    up = _unit(semantic.get("up"), "up")
    forward = _unit(semantic.get("forward"), "forward")
    right = _unit(semantic.get("right"), "right")
    if abs(float(np.dot(up, forward))) > 1e-3 or abs(float(np.dot(up, right))) > 1e-3 or abs(float(np.dot(forward, right))) > 1e-3:
        raise FrameError("reason=not_orthogonal")
    expected = np.cross(forward, up)
    if float(np.dot(expected, right)) < 0.999:
        raise FrameError("reason=not_right_handed")
    if "origin_m" not in semantic or "frame_version" not in semantic:
        raise FrameError("reason=missing_origin")
    return semantic


def make_frame(forward, up=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0), **regions) -> dict:
    forward_axis = _unit(forward, "forward")
    up_axis = _unit(up, "up")
    right_axis = np.cross(forward_axis, up_axis)
    if float(np.linalg.norm(right_axis)) < 1e-8:
        raise FrameError("reason=parallel_axes")
    right_axis = right_axis / np.linalg.norm(right_axis)
    semantic = {
        "up": [float(value) for value in up_axis],
        "forward": [float(value) for value in forward_axis],
        "right": [float(value) for value in right_axis],
        "origin_m": [float(value) for value in origin],
        "frame_version": "semantic_frame_v1",
    }
    semantic.update(regions)
    require_frame({"semantic": semantic})
    return semantic


def screen_agrees(screen_normal, face_normal, *, minimum_dot: float = 0.985) -> dict:
    """The declared screen normal must match the screen surface, not the long box axis."""
    declared = _unit(screen_normal, "screen_normal")
    measured = _unit(face_normal, "face_normal")
    dot = float(np.dot(declared, measured))
    return {"dot": dot, "minimum_dot": minimum_dot, "passed": dot >= minimum_dot, "definition": "dot of declared screen_normal and the mean geometric normal of the screen component"}


def sofa_forward_from_regions(back_centroid, seat_front_centroid) -> np.ndarray:
    """Forward points from the backrest toward the front of the seat."""
    direction = np.asarray(seat_front_centroid, dtype=np.float64) - np.asarray(back_centroid, dtype=np.float64)
    direction[2] = 0.0
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise FrameError("reason=sofa_forward")
    return direction / length

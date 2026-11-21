"""Declarative relations for the qualification composer."""

from __future__ import annotations

RELATIONS = (
    "supported_on",
    "inside",
    "against_wall",
    "faces_toward",
    "screen_faces",
    "back_to",
    "front_clearance",
    "adjacent_to",
    "between",
    "aligned_with",
    "symmetric_with",
    "grouped_with",
    "near",
    "keep_clear_of",
)


class CompositionError(ValueError):
    """A relation record is incomplete."""


def relation(kind: str, source: str, target: str, *, tolerance: float, hard: bool, reference: str, validator: str) -> dict:
    if kind not in RELATIONS:
        raise CompositionError(f"reason=relation value={kind}")
    if tolerance <= 0.0:
        raise CompositionError("reason=tolerance")
    return {
        "type": kind,
        "source": source,
        "target": target,
        "tolerance": float(tolerance),
        "priority": "hard" if hard else "soft",
        "reference_frame": reference,
        "validator": validator,
    }

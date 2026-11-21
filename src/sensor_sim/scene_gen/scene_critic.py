"""Geometric, material, and semantic checks. A hard failure is not overridden by an image."""

from __future__ import annotations

import numpy as np

from sensor_sim.scene_gen.composition_solver import hard_failures


def mesh_report(mesh: dict) -> dict:
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    edges = []
    degenerate = 0
    for face in mesh["corners"]:
        points = vertices[[int(corner[0]) for corner in face]]
        if len(points) < 3:
            degenerate += 1
            continue
        area = float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) * 0.5)
        if area < 1e-10:
            degenerate += 1
        for start, stop in ((0, 1), (1, 2), (2, 0)):
            if stop < len(points):
                edges.append(float(np.linalg.norm(points[stop] - points[start])))
    return {
        "finite": bool(np.isfinite(vertices).all()) if len(vertices) else False,
        "degenerate_faces": int(degenerate),
        "triangle_count": int(sum(max(len(face) - 2, 0) for face in mesh["corners"])),
        "edge_length_mean_m": float(np.mean(edges)) if edges else 0.0,
        "edge_length_max_m": float(np.max(edges)) if edges else 0.0,
        "passed": bool(len(vertices) and np.isfinite(vertices).all() and degenerate == 0),
        "definition": "triangle_count is recorded and is not the pass condition. passed requires finite vertices and no degenerate faces.",
    }


def scene_report(solved: dict, *, required: list[str]) -> dict:
    failures = hard_failures(solved["results"])
    present = {item["id"] for item in solved["instances"]}
    missing = [name for name in required if name not in present]
    hard = [{"reason": item["failure_reason"], "type": item["type"], "source": item["source"]} for item in failures]
    if missing:
        hard.append({"reason": "missing_required", "type": "required", "source": ",".join(missing)})
    spacing = []
    positions = [np.asarray(item["position"], dtype=np.float64) for item in solved["instances"]]
    for index, origin in enumerate(positions):
        others = [float(np.linalg.norm(origin - other)) for other_index, other in enumerate(positions) if other_index != index]
        if others:
            spacing.append(min(others))
    return {
        "valid": not hard and bool(solved["valid"]),
        "hard_failures": hard,
        "soft": {
            "minimum_spacing_m": float(min(spacing)) if spacing else None,
            "object_count": len(solved["instances"]),
            "definition": "minimum_spacing_m is the smallest horizontal-and-vertical center distance between two instances. It is not a clearance.",
        },
    }

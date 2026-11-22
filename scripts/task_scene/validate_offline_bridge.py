"""Aggregate offline-bridge reports. This process does not import Isaac Lab."""

from __future__ import annotations

import json
from pathlib import Path

TASK = Path("/home/cmoore/sensor_sim/dataset/task_scene")
REPORTS = TASK / "reports"


def _load(name: str) -> dict:
    path = REPORTS / name
    if not path.is_file():
        return {"status": "not_run", "path": str(path)}
    document = json.loads(path.read_text(encoding="utf-8"))
    document["path"] = str(path)
    return document


def _gate(document: dict, key: str | None = None) -> str:
    if document.get("status") == "not_run":
        return "not_run"
    if key is None:
        return document.get("status", "not_run")
    return document.get("gates", {}).get(key, document.get("status", "not_run"))


def main() -> int:
    static = _load("offline_bridge_static.json")
    trajectory = _load("offline_bridge_trajectory.json")
    replay = _load("offline_bridge_mitsuba_replay.json")
    static_state = static.get("status", "not_run")
    results = {
        "asset_library_validation": "not_run",
        "asset_retrieval": "not_run",
        "shared_connector": "not_run",
        "isaac_composite_usd": _gate(static, "composite_usd") if static_state != "not_run" else "not_run",
        "isaac_initial_scene": _gate(static, "static_pose") if static_state != "not_run" else "not_run",
        "mitsuba_initial_scene": replay.get("status", "not_run"),
        "static_geometry_parity": _gate(static, "composite_usd") if static_state != "not_run" else "not_run",
        "camera_parity": _gate(static, "camera") if static_state != "not_run" else "not_run",
        "isaac_physics": trajectory.get("status", "not_run"),
        "state_serialization": trajectory.get("status", "not_run"),
        "mitsuba_snapshot_reconstruction": replay.get("status", "not_run"),
        "dynamic_state_parity": replay.get("status", "not_run"),
    }
    if static_state == "failed":
        for key in ("isaac_physics", "state_serialization", "mitsuba_snapshot_reconstruction", "dynamic_state_parity"):
            if results[key] == "not_run":
                results[key] = "blocked"
    if trajectory.get("status") == "failed":
        for key in ("mitsuba_snapshot_reconstruction", "dynamic_state_parity"):
            if results[key] == "not_run":
                results[key] = "blocked"
    summary = {
        "milestone": "offline_isaac_mitsuba_bridge",
        "so_arm101_gate_d": "blocked",
        "results": results,
        "static_report": static.get("path"),
        "trajectory_report": trajectory.get("path"),
        "replay_report": replay.get("path"),
        "manifest_sha256": static.get("manifest_sha256"),
        "catalog_sha256": static.get("catalog_sha256"),
        "run_hash": static.get("run_hash"),
    }
    path = REPORTS / "offline_bridge_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

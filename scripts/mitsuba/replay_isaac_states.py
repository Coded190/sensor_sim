"""Rebuild Mitsuba from saved Isaac snapshots. This does not rerun physics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.indoor_mitsuba import render_stokes, write_rgb_png
from sensor_sim.scene_gen.mitsuba_scene_package import build_mitsuba_scene
from sensor_sim.scene_gen.scene_state import read_jsonl, translation_rotation_error, validate_snapshot
from sensor_sim.scene_gen.task_scene_connector import project_pinhole, resolve_scene_files

TASK = Path("/home/cmoore/sensor_sim/dataset/task_scene")
MANIFEST = TASK / "scenes" / "pick_can" / "scene_manifest.json"
CATALOG = TASK / "library" / "catalog.json"
LIBRARY = TASK / "library"
NAMED = ("t0", "t1", "t2")


def main() -> int:
    trajectory_path = TASK / "reports" / "offline_bridge_trajectory.json"
    report = {"status": "not_run", "snapshots": {}}
    if not trajectory_path.is_file():
        report.update({"status": "blocked", "error": "reason=trajectory_report_missing"})
        _write(report)
        return 2
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if trajectory.get("status") != "passed":
        report.update({"status": "blocked", "error": f"reason=trajectory_not_passed status={trajectory.get('status')}"})
        _write(report)
        return 2
    isaac_dir = Path(trajectory["output"])
    snapshots, sequence_hash = read_jsonl(isaac_dir / "states.jsonl")
    if sequence_hash != trajectory["sequence_sha256"]:
        report.update({"status": "failed", "error": "reason=sequence_hash_mismatch"})
        _write(report)
        return 1
    resolved = resolve_scene_files(MANIFEST, CATALOG, LIBRARY)
    config = {
        "manifest_path": str(MANIFEST),
        "catalog_path": str(CATALOG),
        "library_root": str(LIBRARY),
        "manifest_sha256": resolved["manifest_sha256"],
    }
    out = isaac_dir.parent / "mitsuba"
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    for name in NAMED:
        snapshot = next(item for item in snapshots if item["snapshot_id"] == name)
        validate_snapshot(
            snapshot,
            expected_manifest=resolved["manifest_sha256"],
            expected_catalog=resolved["catalog_sha256"],
            expected_ids={item["instance_id"] for item in resolved["instances"]},
        )
        overridden = _apply(resolved, snapshot)
        built = build_mitsuba_scene({}, TASK, config, resolved=overridden)
        exr = out / f"{name}.exr"
        png = out / f"{name}.png"
        stokes = render_stokes(built["scene"], exr, 8, 1)
        write_rgb_png(exr, png)
        measurements = _compare(resolved, overridden, snapshot)
        if not measurements["within_tolerance"]:
            failures.append(name)
        report["snapshots"][name] = {
            "png": str(png),
            "exr": str(exr),
            "stokes_shape": stokes["shape"],
            "measurements": measurements,
            "content_hash": snapshot["content_hash"],
        }
    report.update(
        {
            "status": "failed" if failures else "passed",
            "failed_snapshots": failures,
            "manifest_sha256": resolved["manifest_sha256"],
            "catalog_sha256": resolved["catalog_sha256"],
            "scene_resolution_sha256": resolved["scene_resolution_sha256"],
            "sequence_sha256": sequence_hash,
            "variant": "cuda_ad_spectral_polarized",
            "output": str(out),
        }
    )
    _write(report)
    print(json.dumps({"status": report["status"], "failed_snapshots": failures}, indent=2))
    return 0 if report["status"] == "passed" else 1


def _apply(resolved: dict, snapshot: dict) -> dict:
    by_id = {item["instance_id"]: item for item in snapshot["instances"]}
    updated = json.loads(json.dumps(resolved))
    for instance in updated["instances"]:
        saved = by_id[instance["instance_id"]]
        if saved["dynamic"]:
            instance["world_from_asset"] = saved["world_from_asset"]
    updated["source_snapshot_id"] = snapshot["snapshot_id"]
    return updated


def _compare(original: dict, consumed: dict, snapshot: dict) -> dict:
    rows = []
    saved = {item["instance_id"]: item for item in snapshot["instances"]}
    used = {item["instance_id"]: item for item in consumed["instances"]}
    within = True
    for instance_id, item in saved.items():
        translation, rotation = translation_rotation_error(
            np.asarray(item["world_from_asset"]),
            np.asarray(used[instance_id]["world_from_asset"]),
        )
        within = within and translation <= 0.001 and rotation <= 0.1
        rows.append({"instance_id": instance_id, "translation_error_m": translation, "rotation_error_deg": rotation})
    camera = original["camera"]
    forward = np.asarray(camera["world_from_camera"], dtype=np.float64)[:3, 2]
    origin = np.asarray(camera["world_from_camera"], dtype=np.float64)[:3, 3]
    pixel = project_pinhole(np.asarray(camera["world_from_camera"], dtype=np.float64), origin + forward, camera)
    pixel_error = float(np.linalg.norm(np.asarray(pixel) - np.array([camera["cx"], camera["cy"]])))
    within = within and pixel_error <= 1.0
    return {"instances": rows, "reprojection_error_px": pixel_error, "within_tolerance": within}


def _write(report: dict) -> None:
    path = TASK / "reports" / "offline_bridge_mitsuba_replay.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

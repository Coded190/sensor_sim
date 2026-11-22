"""Run the fixed-timestep can slide and write scene_state_snapshot_v1 records.

The static scene must already have passed. This script reloads those USD files.
It does not spawn a robot, teleport the can after the run starts, or overwrite
an existing trajectory directory.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from sensor_sim.scene_gen.bridge_physics import (
    DURATION_S,
    INITIAL_VELOCITY_M_S,
    SAMPLE_EVERY_STEPS,
    SNAPSHOTS_S,
    TIMESTEP_S,
    physics_document,
)
from sensor_sim.scene_gen.scene_state import pose_to_world_matrix, translation_rotation_error, write_jsonl, write_snapshot
from sensor_sim.scene_gen.task_scene_connector import pose_wxyz, resolve_scene_files

TASK = Path("/home/cmoore/sensor_sim/dataset/task_scene")
MANIFEST = TASK / "scenes" / "pick_can" / "scene_manifest.json"
CATALOG = TASK / "library" / "catalog.json"
LIBRARY = TASK / "library"


def main() -> int:
    static_path = TASK / "reports" / "offline_bridge_static.json"
    report = {"status": "not_run"}
    try:
        if not static_path.is_file():
            report.update({"status": "blocked", "error": "reason=static_report_missing"})
            _write(report, None)
            return 2
        static = json.loads(static_path.read_text(encoding="utf-8"))
        if static.get("status") != "passed":
            report.update({"status": "blocked", "error": f"reason=static_not_passed status={static.get('status')}"})
            _write(report, None)
            return 2
        out = Path(static["output"]) / "isaac"
        if (out / "states.jsonl").exists():
            report.update({"status": "blocked", "error": f"reason=trajectory_exists path={out}"})
            _write(report, out)
            return 2
        out.mkdir(parents=True, exist_ok=True)
        resolved = resolve_scene_files(MANIFEST, CATALOG, LIBRARY)
        if resolved["manifest_sha256"] != static["manifest_sha256"]:
            raise RuntimeError("reason=manifest_hash_changed")
        snapshots, settle = _run(resolved, static)
        sequence_hash = write_jsonl(out / "states.jsonl", snapshots)
        named = {}
        for snapshot in snapshots:
            if snapshot["snapshot_id"] in SNAPSHOTS_S:
                named[snapshot["snapshot_id"]] = write_snapshot(out / f"{snapshot['snapshot_id']}.json", snapshot)
        t0 = next(item for item in snapshots if item["snapshot_id"] == "t0")
        t2 = next(item for item in snapshots if item["snapshot_id"] == "t2")
        can0 = _instance(t0, "can_0")
        can2 = _instance(t2, "can_0")
        moved, _rotation = translation_rotation_error(np.asarray(can0["world_from_asset"]), np.asarray(can2["world_from_asset"]))
        report = {
            "status": "passed" if moved >= 0.01 and settle["supported"] else "failed",
            "movement_m": moved,
            "settle": settle,
            "sequence_sha256": sequence_hash,
            "snapshot_ids": list(named),
            "physics": physics_document(),
            "manifest_sha256": resolved["manifest_sha256"],
            "catalog_sha256": resolved["catalog_sha256"],
            "scene_resolution_sha256": resolved["scene_resolution_sha256"],
            "output": str(out),
            "versions": static.get("versions"),
        }
        if moved < 0.01:
            report["error"] = "reason=motion_below_10mm"
        if not settle["supported"]:
            report["error"] = "reason=can_did_not_settle_on_table"
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    _write(report, Path(report["output"]) if report.get("output") else None)
    print(json.dumps({key: report.get(key) for key in ("status", "error", "movement_m", "sequence_sha256")}, indent=2))
    return 0 if report.get("status") == "passed" else 1


def _run(resolved: dict, static: dict) -> tuple[list[dict], dict]:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=TIMESTEP_S, device="cuda:0", gravity=(0.0, 0.0, -9.81)))
    by_id = {item["instance_id"]: item for item in resolved["instances"]}
    table_pos, table_quat = pose_wxyz(np.asarray(by_id["table_0"]["world_from_asset"], dtype=np.float64))
    can_pos, can_quat = pose_wxyz(np.asarray(by_id["can_0"]["world_from_asset"], dtype=np.float64))
    spawn_from_usd(
        "/World/Table",
        sim_utils.UsdFileCfg(usd_path=static["assets"]["table_0"]["usd"]["usd_path"]),
        translation=tuple(table_pos),
        orientation=tuple(table_quat),
    )
    can = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Can",
            spawn=sim_utils.UsdFileCfg(
                usd_path=static["assets"]["can_0"]["usd"]["usd_path"],
                mass_props=schemas_cfg.MassPropertiesCfg(mass=float(static["assets"]["can_0"]["mass_kg"])),
                rigid_props=schemas_cfg.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(can_pos), rot=tuple(can_quat)),
        )
    )
    sim.reset()
    settle = _settle(sim, can, by_id["can_0"]["world_from_asset"])
    _reset_pose(can, can_pos, can_quat)
    velocity = torch.tensor([INITIAL_VELOCITY_M_S + [0.0, 0.0, 0.0]], device=sim.device, dtype=torch.float32)
    can.write_root_velocity_to_sim(velocity)
    can.update(sim.get_physics_dt())
    snapshots = [_record(resolved, static, can, 0, 0.0, "t0")]
    steps = int(round(DURATION_S / TIMESTEP_S))
    named = {int(round(time_s / TIMESTEP_S)): name for name, time_s in SNAPSHOTS_S.items() if time_s > 0}
    for step in range(1, steps + 1):
        sim.step(render=False)
        can.update(sim.get_physics_dt())
        if step % SAMPLE_EVERY_STEPS == 0:
            time_s = step * TIMESTEP_S
            snapshot_id = named.get(step, f"f{step}")
            snapshots.append(_record(resolved, static, can, step, time_s, snapshot_id))
    return snapshots, settle


def _settle(sim, can, manifest_pose) -> dict:
    can.update(sim.get_physics_dt())
    start = can.data.root_pos_w[0].clone()
    for _ in range(int(0.5 / TIMESTEP_S)):
        sim.step(render=False)
        can.update(sim.get_physics_dt())
    end = can.data.root_pos_w[0]
    fell = bool(float(end[2]) < 0.2)
    translation, rotation = translation_rotation_error(np.asarray(manifest_pose), pose_to_world_matrix(start.tolist(), can.data.root_quat_w[0].tolist()))
    return {
        "start_z_m": float(start[2]),
        "end_z_m": float(end[2]),
        "fell_through": fell,
        "initial_translation_error_m": translation,
        "initial_rotation_error_deg": rotation,
        "supported": (not fell) and float(end[2]) > 0.7,
        "contact": {
            "method": "vertical_support_not_named_contact_pair",
            "note": "ContactSensor requires PhysxContactReportAPI. This run records support from the can height instead of a named contact pair.",
        },
    }


def _reset_pose(can, position, quaternion) -> None:
    pose = torch.tensor([position + quaternion], device=can.device, dtype=torch.float32)
    zero = torch.zeros((1, 6), device=can.device, dtype=torch.float32)
    can.write_root_pose_to_sim(pose)
    can.write_root_velocity_to_sim(zero)


def _record(resolved, static, can, step: int, time_s: float, snapshot_id: str) -> dict:
    by_id = {item["instance_id"]: item for item in resolved["instances"]}
    can_pose = pose_to_world_matrix(can.data.root_pos_w[0].tolist(), can.data.root_quat_w[0].tolist())
    linear = can.data.root_lin_vel_w[0].tolist()
    angular = can.data.root_ang_vel_w[0].tolist()
    instances = []
    for instance_id, instance in by_id.items():
        dynamic = instance_id == "can_0"
        instances.append(
            {
                "instance_id": instance_id,
                "asset_id": instance["asset_id"],
                "package_sha256": instance["package_hash"],
                "world_from_asset": can_pose if dynamic else instance["world_from_asset"],
                "linear_velocity_m_s": linear if dynamic else [0.0, 0.0, 0.0],
                "angular_velocity_rad_s": angular if dynamic else [0.0, 0.0, 0.0],
                "dynamic": dynamic,
            }
        )
    return {
        "snapshot_id": snapshot_id,
        "scene_id": resolved["scene_id"],
        "manifest_sha256": resolved["manifest_sha256"],
        "catalog_sha256": resolved["catalog_sha256"],
        "scene_resolution_sha256": resolved["scene_resolution_sha256"],
        "frame_index": step // SAMPLE_EVERY_STEPS,
        "physics_step": step,
        "time_s": time_s,
        "timestep_s": TIMESTEP_S,
        "backend": {"name": "isaaclab", **static["versions"]},
        "coordinate_convention": {"up": "z", "units": "m", "quaternion": "wxyz", "handedness": "right"},
        "instances": instances,
    }


def _instance(snapshot: dict, instance_id: str) -> dict:
    return next(item for item in snapshot["instances"] if item["instance_id"] == instance_id)


def _write(report: dict, out: Path | None) -> None:
    text = json.dumps(report, indent=2) + "\n"
    path = TASK / "reports" / "offline_bridge_trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "trajectory.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)

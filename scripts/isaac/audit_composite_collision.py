"""Rebuild can and table USDs with collision schemas removed from the visual meshes.

Run with the Isaac 3.11 interpreter after Kit has a chance to start. This script
does not overwrite the passed object-only bridge USDs. A PhysX log line that
cooks a visual mesh fails the stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/home/cmoore/sensor_sim")
TASK = ROOT / "dataset" / "task_scene"
MANIFEST = TASK / "scenes" / "pick_can" / "scene_manifest.json"
CATALOG = TASK / "library" / "catalog.json"
LIBRARY = TASK / "library"
MANIFEST_SHA = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
OUT = TASK / "robot_collision" / MANIFEST_SHA / "schema_removal_v1"
OUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "kit.log"
PROCESS_START_S = time.time()
sys.argv += ["--kit_args", f"--/log/file={LOG_PATH} --/log/level=info"]

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import numpy as np
import torch

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material
from pxr import UsdGeom

from bridge_usd import bind_friction, closed_volume_m3, compare_render_geometry, compose_asset, convert_obj, describe_stage, mesh_points
from sensor_sim.scene_gen.collision_audit import schema_failures, visual_cook_failures
from sensor_sim.scene_gen.bridge_physics import DENSITY_KG_M3
from sensor_sim.scene_gen.scene_state import pose_to_world_matrix
from sensor_sim.scene_gen.task_scene_connector import pose_wxyz, resolve_scene_files

ISAAC_VERSION_FILE = Path("/home/cmoore/moore/isaac_so_arm101/.venv/lib/python3.11/site-packages/isaacsim/VERSION")
DROP_M = 0.02
TIMESTEP_S = 1.0 / 120.0
DROP_S = 2.0
SETTLE_S = 0.5
SETTLE_TOLERANCE_M = 0.005
# Settle-test assumption only. This is not the slide-test friction and not a measured material.
STATIC_FRICTION = 0.6
DYNAMIC_FRICTION = 0.5
RESTITUTION = 0.0
FRICTION_COMBINE = "average"


def main() -> int:
    report: dict = {
        "status": "failed",
        "stage": "collision_cleanup",
        "command": " ".join(sys.argv),
        "interpreter": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "output": str(OUT),
        "preserved_bridge_usd": {
            "can": str(TASK / "isaac_assets/can/9d5419cae8b43dcf637233756a060c311df04aea59a46afb6077a4666b2dc9e2/asset.usd"),
            "table": str(TASK / "isaac_assets/table_dining/1d28aed9d1366d2f64f417abaff443d916b778c3960dea731f139f7114d42976/asset.usd"),
        },
    }
    try:
        report["versions"] = _versions()
        resolved = resolve_scene_files(MANIFEST, CATALOG, LIBRARY)
        report["manifest_sha256"] = resolved["manifest_sha256"]
        report["catalog_sha256"] = resolved["catalog_sha256"]
        report["scene_resolution_sha256"] = resolved["scene_resolution_sha256"]
        assets = {}
        for instance in resolved["instances"]:
            assets[instance["instance_id"]] = _build_asset(instance)
        report["assets"] = assets
        schema_errors = []
        for instance_id, asset in assets.items():
            role = "can" if asset["asset_id"] == "can" else "table"
            schema_errors.extend(f"{instance_id}:{item}" for item in schema_failures(asset["stage"], role=role))
            if not asset["geometry"]["within_1mm"]:
                schema_errors.append(f"{instance_id}:reason=render_geometry_mismatch")
        report["schema_failures"] = schema_errors
        report["physics"] = _drop(resolved, assets)
        log_text = _read_logs()
        (OUT / "physx_cook.log").write_text(log_text, encoding="utf-8")
        cook = visual_cook_failures(log_text)
        report["visual_cook_lines"] = cook
        report["cook_log_bytes"] = len(log_text)
        report["status"] = "passed" if not schema_errors and not cook and report["physics"]["passed"] else "failed"
        if not log_text:
            report["status"] = "failed"
            report["error"] = "reason=cook_log_missing"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    text = json.dumps(report, indent=2) + "\n"
    (OUT / "robot_collision_cleanup.json").write_text(text, encoding="utf-8")
    report_path = TASK / "reports" / "robot_collision_cleanup.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    print(json.dumps({"status": report["status"], "error": report.get("error"), "schema_failures": report.get("schema_failures"), "visual_cook_lines": report.get("visual_cook_lines"), "physics_passed": (report.get("physics") or {}).get("passed")}, indent=2))
    return 0 if report["status"] == "passed" else 1


def _versions() -> dict:
    physx = "unavailable"
    try:
        import omni.physx.bindings._physx as bindings

        physx = str(getattr(bindings, "PHYSX_VERSION", "imported"))
    except Exception as exc:
        physx = f"unavailable:{type(exc).__name__}"
    return {
        "isaaclab_file": isaaclab.__file__,
        "isaaclab": isaaclab.__version__,
        "isaacsim": ISAAC_VERSION_FILE.read_text(encoding="utf-8").strip() if ISAAC_VERSION_FILE.is_file() else "missing",
        "physx": physx,
        "physx_schema_package": "omni.usd.schema.physx-107.3.26+107.3.3",
    }


def _build_asset(instance: dict) -> dict:
    asset_dir = OUT / instance["asset_id"]
    render_obj = Path(instance["render_path"])
    collision_obj = Path(instance["collision_path"])
    approximation = "convexHull" if instance["asset_id"] == "can" else "triangle"
    render_usd = convert_obj(render_obj, asset_dir / "render_convert", collision="none")
    collision_usd = convert_obj(collision_obj, asset_dir / "collision_convert", collision=approximation)
    mass = None
    volume = None
    if instance["asset_id"] == "can":
        volume = closed_volume_m3(collision_obj)
        if not volume["valid"]:
            raise RuntimeError("reason=can_volume_invalid")
        mass = float(volume["volume_m3"]) * DENSITY_KG_M3
        volume["mass_kg"] = mass
        volume["provenance"] = "convex_mesh_volume_times_simulation_default_density"
    destination = asset_dir / "asset.usd"
    compose_asset(render_usd, collision_usd, destination, dynamic=instance["asset_id"] == "can", mass_kg=mass, approximation=approximation)
    bind_friction(destination, STATIC_FRICTION, DYNAMIC_FRICTION, RESTITUTION, FRICTION_COMBINE)
    stage = describe_stage(destination)
    return {
        "asset_id": instance["asset_id"],
        "package_hash": instance["package_hash"],
        "render_obj_sha256": hashlib.sha256(render_obj.read_bytes()).hexdigest(),
        "collision_obj_sha256": hashlib.sha256(collision_obj.read_bytes()).hexdigest(),
        "render_geometry_hash": instance["render_geometry_hash"],
        "collision_geometry_hash": instance["collision_geometry_hash"],
        "render_path": str(render_obj),
        "collision_path": str(collision_obj),
        "mass_kg": mass,
        "volume": volume,
        "stage": stage,
        "geometry": compare_render_geometry(render_obj, destination),
    }


def _drop(resolved: dict, assets: dict) -> dict:
    """Release the can 20 mm above the manifest pose and require a 0.5 s settle.

    Settled position is the mean root position over the last 0.5 s. The tolerance
    is the maximum distance of a sample in that window from the mean. Support
    penetration is the table collision mesh maximum z minus the can collision
    mesh minimum z at the settled pose. Positive means overlap.
    """
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=TIMESTEP_S, device="cuda:0", gravity=(0.0, 0.0, -9.81)))
    by_id = {item["instance_id"]: item for item in resolved["instances"]}
    table_pos, table_quat = pose_wxyz(np.asarray(by_id["table_0"]["world_from_asset"], dtype=np.float64))
    can_pos, can_quat = pose_wxyz(np.asarray(by_id["can_0"]["world_from_asset"], dtype=np.float64))
    spawn_from_usd(
        "/World/Table",
        sim_utils.UsdFileCfg(usd_path=assets["table_0"]["stage"]["usd_path"]),
        translation=tuple(table_pos),
        orientation=tuple(table_quat),
    )
    can = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Can",
            spawn=sim_utils.UsdFileCfg(
                usd_path=assets["can_0"]["stage"]["usd_path"],
                activate_contact_sensors=True,
                mass_props=schemas_cfg.MassPropertiesCfg(mass=float(assets["can_0"]["mass_kg"])),
                rigid_props=schemas_cfg.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(can_pos), rot=tuple(can_quat)),
        )
    )
    spawn_rigid_body_material(
        "/World/PhysicsMaterial",
        RigidBodyMaterialCfg(static_friction=STATIC_FRICTION, dynamic_friction=DYNAMIC_FRICTION, restitution=RESTITUTION, friction_combine_mode=FRICTION_COMBINE),
    )
    contact = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/Can",
            update_period=0.0,
            history_length=2,
            track_contact_points=True,
            track_air_time=False,
            max_contact_data_count_per_prim=8,
            filter_prim_paths_expr=["/World/Table"],
        )
    )
    sim.reset()
    table_start = _world_matrix("/World/Table")
    release = list(can_pos)
    release[2] += DROP_M
    pose = torch.tensor([release + can_quat], device=can.device, dtype=torch.float32)
    can.write_root_pose_to_sim(pose)
    can.write_root_velocity_to_sim(torch.zeros((1, 6), device=can.device, dtype=torch.float32))
    can.update(TIMESTEP_S)
    positions = []
    forces = []
    steps = int(round(DROP_S / TIMESTEP_S))
    for _ in range(steps):
        sim.step(render=False)
        can.update(TIMESTEP_S)
        contact.update(TIMESTEP_S, force_recompute=True)
        positions.append([float(value) for value in can.data.root_pos_w[0].tolist()])
        force = contact.data.net_forces_w
        forces.append(float(torch.linalg.norm(force).item()) if force is not None else 0.0)
    window = np.asarray(positions[-int(round(SETTLE_S / TIMESTEP_S)) :], dtype=np.float64)
    mean = window.mean(axis=0)
    deviation = np.linalg.norm(window - mean, axis=1)
    settled_matrix = pose_to_world_matrix(mean.tolist(), can.data.root_quat_w[0].tolist())
    penetration = _penetration(assets, settled_matrix, by_id["table_0"]["world_from_asset"])
    table_end = _world_matrix("/World/Table")
    table_delta = float(np.max(np.abs(np.asarray(table_end) - np.asarray(table_start))))
    settled = bool(float(deviation.max()) <= SETTLE_TOLERANCE_M)
    supported = bool(mean[2] > 0.7 and abs(penetration["overlap_m"]) <= SETTLE_TOLERANCE_M)
    contacted = bool(max(forces[-int(round(SETTLE_S / TIMESTEP_S)) :]) > 0.0)
    return {
        "passed": bool(settled and supported and contacted and table_delta <= 1e-5 and mean[2] >= 0.2),
        "definitions": {
            "drop_height_m": DROP_M,
            "settle_window_s": SETTLE_S,
            "settle_tolerance_m": SETTLE_TOLERANCE_M,
            "settled_position_m": "mean root position over the last settle_window_s",
            "settle_deviation_m": "max distance of a window sample from settled_position_m",
            "overlap_m": "table collision max z minus can collision min z at the settled pose; positive is overlap",
            "friction": "settle-test assumption, not a measured coefficient and not the grasp setting",
        },
        "timestep_s": TIMESTEP_S,
        "static_friction": STATIC_FRICTION,
        "dynamic_friction": DYNAMIC_FRICTION,
        "restitution": RESTITUTION,
        "friction_combine_mode": FRICTION_COMBINE,
        "release_m": release,
        "settled_position_m": mean.tolist(),
        "settle_deviation_m": float(deviation.max()),
        "settled": settled,
        "supported": supported,
        "contact_force_norm_max_n": max(forces) if forces else 0.0,
        "contact_during_settle": contacted,
        "penetration": penetration,
        "table_matrix_delta": table_delta,
        "fell_through": bool(mean[2] < 0.2),
        "position_trace_m": positions[::10],
    }


def _penetration(assets: dict, can_matrix: np.ndarray, table_matrix: list) -> dict:
    can_local = mesh_points(Path(assets["can_0"]["stage"]["usd_path"]), "/AssetRoot/Collision")
    table_local = mesh_points(Path(assets["table_0"]["stage"]["usd_path"]), "/AssetRoot/Collision")
    can_world = _apply(can_matrix, can_local)
    table_world = _apply(np.asarray(table_matrix, dtype=np.float64), table_local)
    table_top = float(table_world[:, 2].max())
    can_bottom = float(can_world[:, 2].min())
    return {
        "table_collision_max_z_m": table_top,
        "can_collision_min_z_m": can_bottom,
        "overlap_m": table_top - can_bottom,
    }


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (matrix @ homo.T).T[:, :3]


def _world_matrix(path: str) -> list:
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    matrix = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def _read_logs() -> str:
    """Read only logs written after this process started, so an older cook warning cannot fail the new USD."""
    chunks = []
    candidates = [LOG_PATH]
    candidates.extend(Path.home().glob(".nvidia-omniverse/logs/**/*.log"))
    for path in candidates:
        if not path.is_file():
            continue
        if path.stat().st_mtime < PROCESS_START_S - 1.0:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.append(f"\n# log={path}\n{text[-2_000_000:]}")
    return "\n".join(chunks)


if __name__ == "__main__":
    raise SystemExit(main())

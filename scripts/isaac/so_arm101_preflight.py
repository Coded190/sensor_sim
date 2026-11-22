"""Verify the SO-ARM101 articulation and independent finger-pad contacts.

Run with the Isaac 3.11 interpreter. The imported ``SO_ARM101_CFG`` is copied
before ``spawn.activate_contact_sensors`` is set. The robot project is not edited.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import sys
import traceback
import xml.etree.ElementTree as ElementTree
from pathlib import Path

ROOT = Path("/home/cmoore/sensor_sim")
TASK = ROOT / "dataset" / "task_scene"
OUT = TASK / "robot_preflight" / "so_arm101_v1"
OUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "kit.log"
sys.argv += ["--kit_args", f"--/log/file={LOG_PATH} --/log/level=info"]

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import numpy as np
import torch

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from pxr import PhysxSchema, UsdPhysics

from isaac_so_arm101.robots.trs_so101.so_arm101 import SO_ARM101_CFG

from sensor_sim.scene_gen.gripper_geometry import axis_aligned_bounds, noise_threshold, pinch_measurement, point_in_bounds, principal_axis
from sensor_sim.scene_gen.task_scene_connector import quat_wxyz_to_matrix

ROBOT_PY = Path("/home/cmoore/moore/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/so_arm101.py")
URDF = Path("/home/cmoore/moore/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf")
CAN_USD = TASK / "robot_collision/463d082c30e0e892a1842c696082799b28cbaeb02111cc9ee5e74c0d42e1db72/schema_removal_v1/can/asset.usd"
TABLE_USD = TASK / "robot_collision/463d082c30e0e892a1842c696082799b28cbaeb02111cc9ee5e74c0d42e1db72/schema_removal_v1/table_dining/asset.usd"
ISAAC_VERSION_FILE = Path("/home/cmoore/moore/isaac_so_arm101/.venv/lib/python3.11/site-packages/isaacsim/VERSION")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TIMESTEP_S = 1.0 / 120.0
CAN_MASS_KG = 0.0023561118122472045


def main() -> int:
    report: dict = {"status": "failed", "stage": "so_arm101_preflight", "interpreter": sys.executable, "pythonpath": os.environ.get("PYTHONPATH", "")}
    try:
        report["versions"] = _versions()
        report["urdf_sha256"] = hashlib.sha256(URDF.read_bytes()).hexdigest()
        report["config_py_sha256"] = hashlib.sha256(ROBOT_PY.read_bytes()).hexdigest()
        report["source_config"] = _config_record(SO_ARM101_CFG)
        robot_cfg = copy.deepcopy(SO_ARM101_CFG)
        if robot_cfg is SO_ARM101_CFG:
            raise RuntimeError("reason=deepcopy_returned_global")
        if not hasattr(robot_cfg.spawn, "activate_contact_sensors"):
            raise RuntimeError("reason=spawn_has_no_activate_contact_sensors")
        robot_cfg.spawn.activate_contact_sensors = True
        if SO_ARM101_CFG.spawn.activate_contact_sensors:
            raise RuntimeError("reason=global_config_mutated")
        robot_cfg.prim_path = "/World/Robot"
        robot_cfg.init_state.pos = (0.0, 0.0, 0.95)
        report["local_config"] = _config_record(robot_cfg)
        sim, robot, can, fixed_sensor, moving_sensor = _build(robot_cfg)
        sim.reset()
        robot.update(TIMESTEP_S)
        report["runtime"] = _runtime(robot)
        report["contact_report_api"] = _contact_api(robot)
        if not report["contact_report_api"]["gripper_link"] or not report["contact_report_api"]["moving_jaw_so101_v1_link"]:
            raise RuntimeError(f"reason=contact_report_api_missing detail={report['contact_report_api']}")
        collisions = _urdf_collisions()
        report["gripper_sweep"] = _sweep_gripper(sim, robot, collisions)
        reference = max(report["gripper_sweep"], key=lambda item: item["separation_m"])
        report["reference_aperture"] = reference
        report["tcp"] = _tcp(sim, robot, collisions, reference)
        report["noise"] = _noise(sim, robot, can, fixed_sensor, moving_sensor)
        fixed_threshold = report["noise"]["fixed"]["threshold_n"]
        moving_threshold = report["noise"]["moving"]["threshold_n"]
        report["cases"] = {
            "fixed_only": _contact_case(sim, robot, can, fixed_sensor, moving_sensor, collisions, "fixed", fixed_threshold, moving_threshold, reference["command_rad"]),
            "moving_only": _contact_case(sim, robot, can, fixed_sensor, moving_sensor, collisions, "moving", fixed_threshold, moving_threshold, reference["command_rad"]),
            "bilateral": _contact_case(sim, robot, can, fixed_sensor, moving_sensor, collisions, "both", fixed_threshold, moving_threshold, reference["command_rad"]),
        }
        report["status"] = "passed" if all(item["passed"] for item in report["cases"].values()) else "failed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    text = json.dumps(report, indent=2, default=_json_default) + "\n"
    (OUT / "so_arm101_preflight.json").write_text(text, encoding="utf-8")
    (TASK / "reports" / "so_arm101_preflight.json").write_text(text, encoding="utf-8")
    print(json.dumps({"status": report["status"], "error": report.get("error"), "cases": {key: value.get("passed") for key, value in (report.get("cases") or {}).items()}}, indent=2))
    return 0 if report["status"] == "passed" else 1


def _versions() -> dict:
    physx = "imported"
    try:
        import omni.physx.bindings._physx as bindings

        physx = str(getattr(bindings, "PHYSX_VERSION", "imported"))
    except Exception as exc:
        physx = f"unavailable:{type(exc).__name__}"
    return {
        "isaaclab_file": isaaclab.__file__,
        "isaaclab": isaaclab.__version__,
        "isaacsim": ISAAC_VERSION_FILE.read_text(encoding="utf-8").strip(),
        "physx": physx,
    }


def _config_record(cfg) -> dict:
    spawn = cfg.spawn
    actuators = {}
    for name, actuator in cfg.actuators.items():
        actuators[name] = {
            "class": type(actuator).__name__,
            "joint_names_expr": list(actuator.joint_names_expr),
            "stiffness": actuator.stiffness,
            "damping": actuator.damping,
            "effort_limit_sim": actuator.effort_limit_sim,
            "velocity_limit_sim": actuator.velocity_limit_sim,
        }
    return {
        "urdf_path": str(spawn.asset_path),
        "fix_base": bool(spawn.fix_base),
        "merge_fixed_joints": bool(spawn.merge_fixed_joints),
        "replace_cylinders_with_capsules": bool(spawn.replace_cylinders_with_capsules),
        "collision_from_visuals": bool(spawn.collision_from_visuals),
        "collider_type": str(spawn.collider_type),
        "activate_contact_sensors": bool(spawn.activate_contact_sensors),
        "init_pos": list(cfg.init_state.pos),
        "init_rot_wxyz": list(cfg.init_state.rot),
        "init_joint_pos": dict(cfg.init_state.joint_pos),
        "init_joint_vel": dict(cfg.init_state.joint_vel),
        "actuators": actuators,
    }


def _build(robot_cfg):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=TIMESTEP_S, device="cuda:0", gravity=(0.0, 0.0, -9.81)))
    spawn_from_usd("/World/Table", sim_utils.UsdFileCfg(usd_path=str(TABLE_USD)), translation=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))
    can = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Can",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(CAN_USD),
                activate_contact_sensors=True,
                mass_props=schemas_cfg.MassPropertiesCfg(mass=CAN_MASS_KG),
                rigid_props=schemas_cfg.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 1.2), rot=(1.0, 0.0, 0.0, 0.0)),
        )
    )
    robot = Articulation(robot_cfg)
    fixed = ContactSensor(_sensor_cfg("/World/Robot/gripper_link"))
    moving = ContactSensor(_sensor_cfg("/World/Robot/moving_jaw_so101_v1_link"))
    return sim, robot, can, fixed, moving


def _sensor_cfg(prim_path: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=prim_path,
        update_period=0.0,
        history_length=8,
        track_contact_points=True,
        track_air_time=True,
        force_threshold=1.0e-4,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=["/World/Can"],
    )


def _runtime(robot) -> dict:
    joint_ids, joint_names = robot.find_joints(JOINTS, preserve_order=True)
    if list(joint_names) != JOINTS:
        raise RuntimeError(f"reason=joint_names names={joint_names}")
    limits = robot.data.joint_pos_limits[0, joint_ids].detach().cpu().tolist()
    bodies = list(robot.body_names)
    return {
        "is_fixed_base": bool(robot.is_fixed_base),
        "joint_names": list(joint_names),
        "joint_ids": [int(value) for value in joint_ids],
        "joint_pos_limits_rad": limits,
        "joint_pos_rad": robot.data.joint_pos[0, joint_ids].detach().cpu().tolist(),
        "joint_vel_rad_s": robot.data.joint_vel[0, joint_ids].detach().cpu().tolist(),
        "body_names": bodies,
        "root_pos_w": robot.data.root_pos_w[0].detach().cpu().tolist(),
        "root_quat_wxyz": robot.data.root_quat_w[0].detach().cpu().tolist(),
        "gripper_frame_link_in_bodies": "gripper_frame_link" in bodies,
    }


def _contact_api(robot) -> dict:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    found = {}
    for name in ("gripper_link", "moving_jaw_so101_v1_link", "gripper_frame_link"):
        matches = [prim for prim in stage.Traverse() if prim.GetName() == name]
        found[name] = [
            {
                "path": str(prim.GetPath()),
                "contact_report": prim.HasAPI(PhysxSchema.PhysxContactReportAPI),
                "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
            }
            for prim in matches
        ]
    flags = {name: bool(items and items[0]["contact_report"]) for name, items in found.items()}
    flags["prims"] = found
    return flags


def _urdf_collisions() -> dict:
    root = ElementTree.parse(URDF).getroot()
    wanted = {
        "gripper_link": "assets/wrist_roll_follower_so101_v1.stl",
        "moving_jaw_so101_v1_link": "assets/moving_jaw_so101_v1.stl",
    }
    parsed = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name not in wanted:
            continue
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None or mesh.get("filename") != wanted[name]:
                continue
            origin = collision.find("origin")
            xyz = [float(item) for item in (origin.get("xyz") if origin is not None else "0 0 0").split()]
            rpy = [float(item) for item in (origin.get("rpy") if origin is not None else "0 0 0").split()]
            parsed[name] = {"xyz": xyz, "rpy": rpy, "points": _load_stl(URDF.parent / mesh.get("filename"))}
    if set(parsed) != set(wanted):
        raise RuntimeError(f"reason=urdf_collision_missing found={sorted(parsed)}")
    return parsed


def _load_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    points = np.zeros((count * 3, 3), dtype=np.float64)
    for index in range(count):
        values = struct.unpack_from("<12f", data, 84 + index * 50)
        points[index * 3 : index * 3 + 3] = np.asarray(values[3:], dtype=np.float64).reshape(3, 3)
    return points


def _sweep_gripper(sim, robot, collisions) -> list[dict]:
    gripper_id = robot.find_joints(["gripper"])[0][0]
    lower, upper = robot.data.joint_pos_limits[0, gripper_id].detach().cpu().tolist()
    targets = [lower + 0.02, 0.0, 0.5, 1.0, upper - 0.02]
    measured = []
    for target in targets:
        _command_gripper(sim, robot, gripper_id, target, steps=120)
        achieved = float(robot.data.joint_pos[0, gripper_id].item())
        pads, _, _ = _pads(robot, collisions)
        measured.append({"command_rad": target, "achieved_rad": achieved, "separation_m": pads["separation_m"], "closing_axis_world": pads["closing_axis_world"]})
    return measured


def _command_gripper(sim, robot, gripper_id: int, target: float, steps: int) -> None:
    position = robot.data.joint_pos.clone()
    position[0, gripper_id] = target
    robot.set_joint_position_target(position)
    for _ in range(steps):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(TIMESTEP_S)
        robot.set_joint_position_target(position)


def _pads(robot, collisions) -> tuple[dict, np.ndarray, np.ndarray]:
    fixed, moving = _jaw_world(robot, collisions)
    origin = _body_matrix(robot, "gripper_link")[:3, 3]
    tip = fixed[int(np.argmax(np.linalg.norm(fixed - origin, axis=1)))]
    return pinch_measurement(fixed, moving, tip - origin), fixed, moving


def _jaw_world(robot, collisions) -> tuple[np.ndarray, np.ndarray]:
    """Distal jaw surfaces. The global closest points sit at the knuckle, not the pads."""
    fixed = _link_points(robot, "gripper_link", collisions["gripper_link"])
    moving = _link_points(robot, "moving_jaw_so101_v1_link", collisions["moving_jaw_so101_v1_link"])
    origin = _body_matrix(robot, "gripper_link")[:3, 3]
    tip = fixed[int(np.argmax(np.linalg.norm(fixed - origin, axis=1)))]
    direction = tip - origin
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)

    def distal(points: np.ndarray) -> np.ndarray:
        along = (points - origin) @ direction
        return points[along >= np.quantile(along, 0.60)]

    return distal(fixed), distal(moving)


def _link_points(robot, body_name: str, collision: dict) -> np.ndarray:
    body_id = robot.find_bodies([body_name])[0][0]
    matrix = quat_wxyz_to_matrix(
        robot.data.body_quat_w[0, body_id].detach().cpu().tolist(),
        robot.data.body_pos_w[0, body_id].detach().cpu().tolist(),
    )
    local = _rpy_matrix(collision["rpy"], collision["xyz"]) @ _homogenize(collision["points"]).T
    return (matrix @ local).T[:, :3]


def _tcp(sim, robot, collisions, reference: dict) -> dict:
    """TCP is the midpoint of the facing pad centers, stored on gripper_link at this aperture."""
    gripper_id = robot.find_joints(["gripper"])[0][0]
    _command_gripper(sim, robot, gripper_id, reference["command_rad"], steps=60)
    pads, fixed_world, moving_world = _pads(robot, collisions)
    midpoint = 0.5 * (np.asarray(pads["fixed_face_center_m"]) + np.asarray(pads["moving_face_center_m"]))
    gripper_matrix = _body_matrix(robot, "gripper_link")
    local = np.linalg.inv(gripper_matrix) @ np.array([*midpoint, 1.0])
    fixed_local = _local_points(np.linalg.inv(gripper_matrix), fixed_world[pads["fixed_face_index"]])
    moving_matrix = _body_matrix(robot, "moving_jaw_so101_v1_link")
    moving_local = _local_points(np.linalg.inv(moving_matrix), moving_world[pads["moving_face_index"]])
    approach = principal_axis(fixed_world)
    frame = _frame_link_error(robot, midpoint)
    return {
        "parent_link": "gripper_link",
        "reference_command_rad": reference["command_rad"],
        "reference_achieved_rad": reference["achieved_rad"],
        "translation_in_gripper_link_m": local[:3].tolist(),
        "world_m": midpoint.tolist(),
        "approach_axis_world": approach["axis"],
        "closing_axis_world": pads["closing_axis_world"],
        "fixed_pad_bounds_link": axis_aligned_bounds(fixed_local),
        "moving_pad_bounds_link": axis_aligned_bounds(moving_local),
        "gripper_frame_link": frame,
        "uses_gripper_frame_link": bool(frame["matches_tcp"]),
    }


def _frame_link_error(robot, tcp_world: np.ndarray) -> dict:
    bodies = list(robot.body_names)
    if "gripper_frame_link" not in bodies:
        return {"present": False, "matches_tcp": False, "reason": "merged_or_absent"}
    matrix = _body_matrix(robot, "gripper_frame_link")
    error = float(np.linalg.norm(matrix[:3, 3] - tcp_world))
    return {"present": True, "translation_error_m": error, "matches_tcp": error <= 0.005, "match_tolerance_m": 0.005}


def _noise(sim, robot, can, fixed_sensor, moving_sensor) -> dict:
    _place_can(can, [2.0, 0.0, 1.2])
    fixed_samples = []
    moving_samples = []
    for _ in range(int(round(5.0 / TIMESTEP_S))):
        robot.write_data_to_sim()
        can.write_data_to_sim()
        sim.step(render=False)
        robot.update(TIMESTEP_S)
        can.update(TIMESTEP_S)
        fixed_sensor.update(TIMESTEP_S, force_recompute=True)
        moving_sensor.update(TIMESTEP_S, force_recompute=True)
        fixed_samples.append(_filtered_force(fixed_sensor))
        moving_samples.append(_filtered_force(moving_sensor))
    return {"duration_s": 5.0, "fixed": noise_threshold(np.asarray(fixed_samples)), "moving": noise_threshold(np.asarray(moving_samples))}


def _contact_case(sim, robot, can, fixed_sensor, moving_sensor, collisions, mode: str, fixed_threshold: float, moving_threshold: float, open_command: float) -> dict:
    gripper_id = robot.find_joints(["gripper"])[0][0]
    _command_gripper(sim, robot, gripper_id, open_command, steps=90)
    pads, fixed_world, moving_world = _pads(robot, collisions)
    axis = np.asarray(pads["closing_axis_world"], dtype=np.float64)
    fixed_center = np.asarray(pads["fixed_face_center_m"], dtype=np.float64)
    moving_center = np.asarray(pads["moving_face_center_m"], dtype=np.float64)
    half = _can_support(axis)
    fixed_local = _local_points(np.linalg.inv(_body_matrix(robot, "gripper_link")), fixed_world[pads["fixed_face_index"]])
    moving_local = _local_points(np.linalg.inv(_body_matrix(robot, "moving_jaw_so101_v1_link")), moving_world[pads["moving_face_index"]])
    bounds = {"fixed": axis_aligned_bounds(fixed_local, margin_m=0.01), "moving": axis_aligned_bounds(moving_local, margin_m=0.01)}
    attempts = []
    offsets = [0.0004, 0.001]
    finger_shifts = [0.0]
    fingertip = fixed_center - _body_matrix(robot, "gripper_link")[:3, 3]
    fingertip = fingertip / max(float(np.linalg.norm(fingertip)), 1e-9)
    matched = False
    for finger_shift in finger_shifts:
        for offset in offsets:
            if mode == "fixed":
                center = fixed_center + axis * (half - offset)
            elif mode == "moving":
                center = moving_center - axis * (half - offset)
            else:
                center = 0.5 * (fixed_center + moving_center)
            center = center + fingertip * finger_shift
            rotation = _quat_align_z(axis)
            root = center - _rotate(rotation, np.array([0.0, 0.0, 0.0146145]))
            _place_can(can, root.tolist(), rotation.tolist())
            samples = _sample_contact(sim, robot, can, fixed_sensor, moving_sensor, steps=60)
            classified = _classify(samples, fixed_threshold, moving_threshold)
            localized = _localize_samples(robot, samples, bounds, mode, fixed_threshold, moving_threshold)
            attempts.append({"offset_m": offset, "finger_shift_m": finger_shift, "can_center_m": center.tolist(), "classification": classified, "pad_localization": localized})
            if _case_matches(mode, classified) and localized["passed"]:
                matched = True
                break
        if matched:
            break
    passed = matched
    if mode == "both" and not passed:
        closed, extra = _close_until_bilateral(sim, robot, can, fixed_sensor, moving_sensor, collisions, gripper_id, fixed_threshold, moving_threshold, open_command)
        closed_pads, closed_fixed, closed_moving = _pads(robot, collisions)
        bounds = {
            "fixed": axis_aligned_bounds(_local_points(np.linalg.inv(_body_matrix(robot, "gripper_link")), closed_fixed[closed_pads["fixed_face_index"]]), margin_m=0.01),
            "moving": axis_aligned_bounds(_local_points(np.linalg.inv(_body_matrix(robot, "moving_jaw_so101_v1_link")), closed_moving[closed_pads["moving_face_index"]]), margin_m=0.01),
        }
        extra["pad_localization"] = _localize_sensors(robot, fixed_sensor, moving_sensor, bounds, mode)
        attempts.append(extra)
        passed = closed and extra["pad_localization"]["passed"]
    return {"mode": mode, "passed": bool(passed), "attempts": attempts, "separation_m": pads["separation_m"]}


def _can_support(axis: np.ndarray) -> float:
    """Half-extent along the closing axis after local +Z is aligned with that axis.

    The packaged can collision mesh is centered in x and y and runs from z=0 to
    about 0.0292 m, so the half-height about the mesh center is 0.0146 m.
    """
    del axis
    return 0.0146145


def _sample_contact(sim, robot, can, fixed_sensor, moving_sensor, steps: int) -> dict:
    fixed_forces = []
    moving_forces = []
    fixed_points = []
    moving_points = []
    for _ in range(steps):
        robot.write_data_to_sim()
        can.write_data_to_sim()
        sim.step(render=False)
        robot.update(TIMESTEP_S)
        can.update(TIMESTEP_S)
        fixed_sensor.update(TIMESTEP_S, force_recompute=True)
        moving_sensor.update(TIMESTEP_S, force_recompute=True)
        can.write_root_velocity_to_sim(torch.zeros((1, 6), device=can.device, dtype=torch.float32))
        fixed_forces.append(_filtered_force(fixed_sensor))
        moving_forces.append(_filtered_force(moving_sensor))
        fixed_points.append(_contact_point(fixed_sensor))
        moving_points.append(_contact_point(moving_sensor))
    return {"fixed_force_n": fixed_forces, "moving_force_n": moving_forces, "fixed_point_w": fixed_points, "moving_point_w": moving_points}


def _classify(samples: dict, fixed_threshold: float, moving_threshold: float) -> dict:
    fixed = np.asarray(samples["fixed_force_n"], dtype=np.float64)
    moving = np.asarray(samples["moving_force_n"], dtype=np.float64)
    fixed_on = fixed > fixed_threshold
    moving_on = moving > moving_threshold
    both = _longest_true(fixed_on & moving_on)
    return {
        "fixed_contact": bool(np.any(fixed_on)),
        "moving_contact": bool(np.any(moving_on)),
        "fixed_only": bool(np.any(fixed_on) and not np.any(moving_on)),
        "moving_only": bool(np.any(moving_on) and not np.any(fixed_on)),
        "bilateral_steps": both,
        "bilateral_s": both * TIMESTEP_S,
        "fixed_max_n": float(fixed.max()),
        "moving_max_n": float(moving.max()),
    }


def _case_matches(mode: str, classified: dict) -> bool:
    if mode == "fixed":
        return classified["fixed_only"]
    if mode == "moving":
        return classified["moving_only"]
    return classified["bilateral_s"] >= 0.05


def _longest_true(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _closing_sign(sim, robot, collisions, gripper_id: int, open_command: float) -> float:
    """Return the joint-command sign that reduces the measured pad separation."""
    _command_gripper(sim, robot, gripper_id, open_command, steps=40)
    opened = _pads(robot, collisions)[0]["separation_m"]
    _command_gripper(sim, robot, gripper_id, open_command + 0.08, steps=40)
    nudged = _pads(robot, collisions)[0]["separation_m"]
    _command_gripper(sim, robot, gripper_id, open_command, steps=40)
    return 1.0 if nudged < opened else -1.0


def _close_until_bilateral(sim, robot, can, fixed_sensor, moving_sensor, collisions, gripper_id: int, fixed_threshold: float, moving_threshold: float, open_command: float) -> tuple[bool, dict]:
    sign = _closing_sign(sim, robot, collisions, gripper_id, open_command)
    position = robot.data.joint_pos.clone()
    current = float(position[0, gripper_id].item())
    lower = float(robot.data.joint_pos_limits[0, gripper_id, 0].item())
    upper = float(robot.data.joint_pos_limits[0, gripper_id, 1].item())
    rate = 0.2 * sign
    steps = int(round(3.0 / TIMESTEP_S))
    samples_fixed = []
    samples_moving = []
    for _ in range(steps):
        current = float(np.clip(current + rate * TIMESTEP_S, lower + 0.02, upper - 0.02))
        position[0, gripper_id] = current
        robot.set_joint_position_target(position)
        robot.write_data_to_sim()
        can.write_data_to_sim()
        sim.step(render=False)
        robot.update(TIMESTEP_S)
        can.update(TIMESTEP_S)
        fixed_sensor.update(TIMESTEP_S, force_recompute=True)
        moving_sensor.update(TIMESTEP_S, force_recompute=True)
        samples_fixed.append(_filtered_force(fixed_sensor))
        samples_moving.append(_filtered_force(moving_sensor))
        both = np.asarray(samples_fixed) > fixed_threshold
        both = both & (np.asarray(samples_moving) > moving_threshold)
        if _longest_true(both) * TIMESTEP_S >= 0.05:
            return True, {"method": "controlled_close", "closing_sign": sign, "achieved_rad": current, "bilateral_s": _longest_true(both) * TIMESTEP_S}
    return False, {"method": "controlled_close", "closing_sign": sign, "achieved_rad": current, "bilateral_s": 0.0}


def _localize_samples(robot, samples: dict, bounds: dict, mode: str, fixed_threshold: float = 0.0, moving_threshold: float = 0.0) -> dict:
    fixed_point = _last_active(samples["fixed_point_w"], samples["fixed_force_n"], fixed_threshold)
    moving_point = _last_active(samples["moving_point_w"], samples["moving_force_n"], moving_threshold)
    fixed_inside = _point_requirement(mode, "fixed", robot, "gripper_link", fixed_point, bounds["fixed"])
    moving_inside = _point_requirement(mode, "moving", robot, "moving_jaw_so101_v1_link", moving_point, bounds["moving"])
    return {
        "passed": bool(fixed_inside and moving_inside),
        "fixed_point_inside_pad": fixed_inside,
        "moving_point_inside_pad": moving_inside,
        "fixed_point_w": fixed_point,
        "moving_point_w": moving_point,
    }


def _point_requirement(mode: str, finger: str, robot, body_name: str, point, region: dict) -> bool:
    required = mode == "both" or mode == finger
    if point is None:
        return not required
    return point_in_bounds(_to_link(robot, body_name, point), region)


def _localize_sensors(robot, fixed_sensor, moving_sensor, bounds: dict, mode: str) -> dict:
    return _localize_samples(
        robot,
        {
            "fixed_point_w": [_contact_point(fixed_sensor)],
            "moving_point_w": [_contact_point(moving_sensor)],
            "fixed_force_n": [_filtered_force(fixed_sensor)],
            "moving_force_n": [_filtered_force(moving_sensor)],
        },
        bounds,
        mode,
    )


def _last_active(points: list, forces: list, threshold: float):
    for point, force in zip(reversed(points), reversed(forces)):
        if point is not None and float(force) > threshold:
            return point
    return None


def _to_link(robot, body_name: str, point) -> np.ndarray:
    inverse = np.linalg.inv(_body_matrix(robot, body_name))
    homo = np.array([float(point[0]), float(point[1]), float(point[2]), 1.0])
    return (inverse @ homo)[:3]


def _last_finite(points: list):
    for point in reversed(points):
        if point is not None:
            return point
    return None


def _filtered_force(sensor) -> float:
    matrix = sensor.data.force_matrix_w
    if matrix is None or matrix.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(matrix).detach().cpu().item())


def _contact_point(sensor):
    points = sensor.data.contact_pos_w
    if points is None or points.numel() == 0:
        return None
    value = points.detach().cpu().numpy().reshape(-1, 3)[0]
    if not np.all(np.isfinite(value)):
        return None
    return value.tolist()


def _place_can(can, position: list[float], quaternion: list[float] | None = None) -> None:
    rotation = [1.0, 0.0, 0.0, 0.0] if quaternion is None else quaternion
    pose = torch.tensor([position + rotation], device=can.device, dtype=torch.float32)
    can.write_root_pose_to_sim(pose)
    can.write_root_velocity_to_sim(torch.zeros((1, 6), device=can.device, dtype=torch.float32))
    can.write_data_to_sim()


def _quat_align_z(axis: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternion rotating local +Z onto ``axis``."""
    target = np.asarray(axis, dtype=np.float64)
    target = target / np.linalg.norm(target)
    source = np.array([0.0, 0.0, 1.0])
    cross = np.cross(source, target)
    dot = float(np.dot(source, target))
    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _rotate(quaternion: np.ndarray, point: np.ndarray) -> np.ndarray:
    matrix = quat_wxyz_to_matrix(quaternion.tolist())
    return matrix[:3, :3] @ point


def _body_matrix(robot, body_name: str) -> np.ndarray:
    body_id = robot.find_bodies([body_name])[0][0]
    return quat_wxyz_to_matrix(
        robot.data.body_quat_w[0, body_id].detach().cpu().tolist(),
        robot.data.body_pos_w[0, body_id].detach().cpu().tolist(),
    )


def _local_points(inverse: np.ndarray, points: np.ndarray) -> np.ndarray:
    homo = np.concatenate([np.asarray(points, dtype=np.float64), np.ones((len(points), 1))], axis=1)
    return (inverse @ homo.T).T[:, :3]


def _rpy_matrix(rpy: list[float], xyz: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = xyz
    return matrix


def _homogenize(points: np.ndarray) -> np.ndarray:
    return np.concatenate([points, np.ones((len(points), 1))], axis=1)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    raise SystemExit(main())

"""Headless Isaac Lab feasibility run for the packaged table and can.

This file is executed with the Isaac Lab interpreter. It does not import Isaac
from the sensor_sim project environment.
"""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaac_so_arm101.robots.trs_so101.so_arm101 import SO_ARM101_CFG

ROOT = Path("/home/cmoore/sensor_sim/dataset/task_scene")
REPORT = ROOT / "reports" / "gate_d_preflight.json"
SIM_DEFAULTS = {
    "purpose": "simulation_only",
    "can_density_kg_m3": {"value": 400.0, "units": "kg/m^3", "provenance": "default", "method": "assumed plastic density, not a measurement"},
    "friction": {"value": 0.8, "units": "1", "provenance": "default", "method": "isaac_contact_default_for_this_gate"},
    "restitution": {"value": 0.0, "units": "1", "provenance": "default", "method": "no_bounce_assumption"},
    "table": "static_collision_mesh",
}


def convert(obj: Path, collision: str) -> str:
    props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    mesh_props = schemas_cfg.ConvexHullPropertiesCfg() if collision == "convex" else schemas_cfg.TriangleMeshSimplificationPropertiesCfg()
    converter = MeshConverter(MeshConverterCfg(asset_path=str(obj), collision_props=props, mesh_collision_props=mesh_props, make_instanceable=False))
    return converter.usd_path


def main() -> None:
    report = {"launch": False, "joints": [], "gripper": {}, "settling": {}, "grasp": {}, "sim_defaults": SIM_DEFAULTS}
    try:
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
        sim.set_camera_view([1.4, -1.4, 1.6], [0.0, -0.2, 0.85])
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        table_obj = ROOT / "library" / "table_dining" / "1.0.0" / "geometry" / "collision.obj"
        can_obj = ROOT / "library" / "can" / "1.0.0" / "geometry" / "collision.obj"
        try:
            report["collision_usd"] = {"table": convert(table_obj, "mesh"), "can": convert(can_obj, "convex")}
            report["collision_usd"]["rigid_body_on_converted_root"] = False
            report["collision_usd"]["note"] = "MeshConverter wrote a USD, and UsdFileCfg did not apply a rigid body on the root prim. Physics bodies below are cuboids fitted to the collision-mesh bounds, not the collision mesh itself."
        except Exception as exc:
            report["collision_usd"] = {"error": f"{type(exc).__name__}: {exc}"}
        table_cfg = RigidObjectCfg(
            prim_path="/World/Table",
            spawn=sim_utils.CuboidCfg(
                size=(1.098, 0.941, 0.02),
                rigid_props=schemas_cfg.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.836)),
        )
        can_cfg = RigidObjectCfg(
            prim_path="/World/Can",
            spawn=sim_utils.CuboidCfg(
                size=(0.113, 0.047, 0.029),
                rigid_props=schemas_cfg.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
                collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=schemas_cfg.MassPropertiesCfg(density=400.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.04, -0.18, 0.87)),
        )
        robot_cfg = SO_ARM101_CFG.replace(prim_path="/World/Robot")
        robot_cfg.init_state.pos = (0.25, -0.42, 0.85)
        table = RigidObject(table_cfg)
        can = RigidObject(can_cfg)
        robot = Articulation(robot_cfg)
        sim.reset()
        report["launch"] = True
        names = list(robot.joint_names)
        report["joints"] = names
        report["expected_joints_present"] = all(name in names for name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"])
        gripper_index = names.index("gripper")
        arm_indexes = [index for index, name in enumerate(names) if name != "gripper"]
        targets = robot.data.joint_pos.clone()
        targets[:, gripper_index] = 0.5
        for _ in range(30):
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
        report["gripper"]["opened_to"] = float(robot.data.joint_pos[0, gripper_index])
        targets[:, gripper_index] = 0.0
        for _ in range(30):
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
        report["gripper"]["closed_to"] = float(robot.data.joint_pos[0, gripper_index])
        report["gripper"]["moved"] = abs(report["gripper"]["opened_to"] - report["gripper"]["closed_to"]) > 0.05
        start = can.data.root_pos_w.clone()
        for _ in range(120):
            sim.step()
            can.update(sim.get_physics_dt())
            table.update(sim.get_physics_dt())
            robot.update(sim.get_physics_dt())
        end = can.data.root_pos_w.clone()
        report["settling"] = {"start_z": float(start[0, 2]), "end_z": float(end[0, 2]), "fell_through": bool(end[0, 2] < 0.2)}
        controller = DifferentialIKController(DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"), num_envs=1, device="cuda:0")
        body_index = robot.find_bodies("gripper_link")[0][0]
        jacobian_index = body_index - 1 if robot.is_fixed_base else body_index
        goal = can.data.root_pos_w.clone()
        goal[:, 2] += 0.08
        ee_quat = robot.data.body_quat_w[:, body_index, :]
        controller.set_command(goal, ee_quat=ee_quat)
        for _ in range(80):
            jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_index, :, arm_indexes]
            ee_pose = robot.data.body_pos_w[:, body_index, :]
            ee_quat = robot.data.body_quat_w[:, body_index, :]
            joint_pos = robot.data.joint_pos[:, arm_indexes]
            delta = controller.compute(ee_pose, ee_quat, jacobian, joint_pos)
            targets[:, arm_indexes] = joint_pos + delta
            targets[:, gripper_index] = 0.5
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            can.update(sim.get_physics_dt())
        report["grasp"]["ee_after_move"] = robot.data.body_pos_w[0, body_index].tolist()
        report["grasp"]["can_after_move"] = can.data.root_pos_w[0].tolist()
        report["grasp"]["controller"] = "DifferentialIKController position dls, fixed-base jacobian column body_index-1"
        report["status"] = "preflight_ran"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    simulation_app.close()


if __name__ == "__main__":
    main()

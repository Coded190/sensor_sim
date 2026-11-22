"""Load the packaged table and can in Isaac Lab and record static parity.

Run with the Isaac 3.11 interpreter. This file starts the Kit app before
importing pxr. It does not spawn a robot and it does not edit the manifest.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np
import torch

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material
from bridge_usd import bind_friction, closed_volume_m3, compare_render_geometry, compose_asset, convert_obj, inspect_stage
from sensor_sim.scene_gen.bridge_physics import DENSITY_KG_M3, physics_document, run_hash
from sensor_sim.scene_gen.scene_state import pose_to_world_matrix, translation_rotation_error
from sensor_sim.scene_gen.task_scene_connector import pose_wxyz, project_pinhole, resolve_scene_files

ROOT = Path("/home/cmoore/sensor_sim")
TASK = ROOT / "dataset" / "task_scene"
MANIFEST = TASK / "scenes" / "pick_can" / "scene_manifest.json"
CATALOG = TASK / "library" / "catalog.json"
LIBRARY = TASK / "library"
ISAAC_VERSION_FILE = Path("/home/cmoore/moore/isaac_so_arm101/.venv/lib/python3.11/site-packages/isaacsim/VERSION")


def versions() -> dict:
    physx = "unavailable"
    try:
        import omni.physx.bindings._physx as bindings

        physx = str(getattr(bindings, "PHYSX_VERSION", "imported"))
    except Exception as exc:
        physx = f"unavailable:{type(exc).__name__}"
    return {
        "isaaclab": isaaclab.__version__,
        "isaacsim": ISAAC_VERSION_FILE.read_text(encoding="utf-8").strip() if ISAAC_VERSION_FILE.is_file() else "missing",
        "physx": physx,
    }


def main() -> int:
    report: dict = {"status": "failed", "gates": {}}
    try:
        resolved = resolve_scene_files(MANIFEST, CATALOG, LIBRARY)
        backend = versions()
        digest = run_hash(resolved["manifest_sha256"], resolved["catalog_sha256"], backend)
        out = TASK / "offline_bridge" / resolved["manifest_sha256"] / digest
        finished = out / "static" / "offline_bridge_static.json"
        if finished.is_file():
            report["status"] = "blocked"
            report["error"] = f"reason=output_exists path={out}"
            _write(report, out=None)
            return 2
        if out.exists():
            incomplete = out.with_name(out.name + "_incomplete")
            if incomplete.exists():
                report["status"] = "blocked"
                report["error"] = f"reason=incomplete_output_exists path={incomplete}"
                _write(report, out=None)
                return 2
            out.rename(incomplete)
        out.mkdir(parents=True)
        report["output"] = str(out)
        report["versions"] = backend
        report["manifest_sha256"] = resolved["manifest_sha256"]
        report["catalog_sha256"] = resolved["catalog_sha256"]
        report["scene_resolution_sha256"] = resolved["scene_resolution_sha256"]
        report["run_hash"] = digest
        assets = {}
        for instance in resolved["instances"]:
            asset_dir = TASK / "isaac_assets" / instance["asset_id"] / instance["package_hash"]
            render_usd = convert_obj(Path(instance["render_path"]), asset_dir / "render_convert", collision="none")
            approximation = "convexHull" if instance["asset_id"] == "can" else "triangle"
            collision_usd = convert_obj(Path(instance["collision_path"]), asset_dir / "collision_convert", collision=approximation)
            mass = None
            volume = None
            if instance["asset_id"] == "can":
                volume = closed_volume_m3(Path(instance["collision_path"]))
                if volume["valid"]:
                    mass = float(volume["volume_m3"]) * DENSITY_KG_M3
                    volume["mass_kg"] = mass
                    volume["provenance"] = "convex_mesh_volume_times_simulation_default_density"
                else:
                    mass = 0.05
                    volume["mass_kg"] = mass
                    volume["provenance"] = "explicit_simulation_mass_volume_invalid"
            destination = asset_dir / "asset.usd"
            inspection = compose_asset(
                render_usd,
                collision_usd,
                destination,
                dynamic=instance["asset_id"] == "can",
                mass_kg=mass,
                approximation=approximation,
            )
            bind_friction(destination, 0.05, 0.05, 0.0, "average")
            inspection = inspect_stage(destination)
            geometry = compare_render_geometry(Path(instance["render_path"]), destination)
            assets[instance["instance_id"]] = {
                "asset_id": instance["asset_id"],
                "package_hash": instance["package_hash"],
                "render_geometry_hash": instance["render_geometry_hash"],
                "collision_geometry_hash": instance["collision_geometry_hash"],
                "usd": inspection,
                "geometry": geometry,
                "volume": volume,
                "mass_kg": mass,
            }
            if inspection["up_axis"] != "Z" or abs(float(inspection["meters_per_unit"]) - 1.0) > 1e-6:
                raise RuntimeError(f"reason=stage_convention up={inspection['up_axis']} mpu={inspection['meters_per_unit']}")
            if not geometry["within_1mm"]:
                raise RuntimeError(f"reason=render_geometry_mismatch instance={instance['instance_id']} extent_error_m={geometry['extent_error_m']} max_m={geometry['surface_distance']['max_m']}")
        report["assets"] = assets
        report["gates"]["composite_usd"] = "passed"
        _simulate_static(resolved, assets, report, out)
        report["status"] = "passed" if report["gates"].get("camera") == "passed" and report["gates"].get("static_pose") == "passed" else "failed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    out_path = Path(report["output"]) if report.get("output") else None
    _write(report, out_path)
    print(json.dumps({key: report.get(key) for key in ("status", "error", "run_hash", "gates")}, indent=2))
    return 0 if report["status"] == "passed" else 1


def _simulate_static(resolved: dict, assets: dict, report: dict, out: Path) -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
    sim.set_camera_view([1.4, -1.4, 1.6], [0.0, -0.2, 0.85])
    by_id = {item["instance_id"]: item for item in resolved["instances"]}
    table = by_id["table_0"]
    can = by_id["can_0"]
    table_pos, table_quat = pose_wxyz(np.asarray(table["world_from_asset"], dtype=np.float64))
    can_pos, can_quat = pose_wxyz(np.asarray(can["world_from_asset"], dtype=np.float64))
    spawn_from_usd(
        "/World/Table",
        sim_utils.UsdFileCfg(usd_path=assets["table_0"]["usd"]["usd_path"]),
        translation=tuple(table_pos),
        orientation=tuple(table_quat),
    )
    can_cfg = RigidObjectCfg(
        prim_path="/World/Can",
        spawn=sim_utils.UsdFileCfg(
            usd_path=assets["can_0"]["usd"]["usd_path"],
            mass_props=schemas_cfg.MassPropertiesCfg(mass=float(assets["can_0"]["mass_kg"])),
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(can_pos), rot=tuple(can_quat)),
    )
    can_body = RigidObject(can_cfg)
    spawn_rigid_body_material(
        "/World/PhysicsMaterial",
        RigidBodyMaterialCfg(static_friction=0.05, dynamic_friction=0.05, restitution=0.0, friction_combine_mode="average"),
    )
    sim.reset()
    can_body.update(sim.get_physics_dt())
    measured = pose_to_world_matrix(can_body.data.root_pos_w[0].tolist(), can_body.data.root_quat_w[0].tolist())
    translation, rotation = translation_rotation_error(np.asarray(can["world_from_asset"]), np.asarray(measured))
    report["static_pose"] = {
        "can_translation_error_m": translation,
        "can_rotation_error_deg": rotation,
        "table_command_m": table_pos,
        "can_command_m": can_pos,
        "can_measured_m": can_body.data.root_pos_w[0].tolist(),
    }
    report["gates"]["static_pose"] = "passed" if translation <= 0.001 and rotation <= 0.1 else "failed"
    camera_report = _capture_camera(sim, resolved["camera"], out)
    report["camera"] = camera_report
    report["gates"]["camera"] = "passed" if camera_report["intrinsics_within_one_pixel"] and camera_report["reprojection_error_px"] <= 1.0 else "failed"
    report["physics"] = physics_document()
    if report["gates"]["static_pose"] != "passed" or report["gates"]["camera"] != "passed":
        raise RuntimeError(f"reason=static_parity gates={report['gates']}")


def _capture_camera(sim, camera: dict, out: Path) -> dict:
    """Read the runtime intrinsics and save an RGB frame.

    ``AppLauncher(enable_cameras=True)`` is required. GLFW may warn that it
    has no display. The render product still returned an image on this machine.
    """
    position, quaternion = pose_wxyz(np.asarray(camera["world_from_camera"], dtype=np.float64))
    sensor = Camera(
        CameraCfg(
            prim_path="/World/Camera",
            update_period=0.0,
            height=int(camera["height"]),
            width=int(camera["width"]),
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=1.0, clipping_range=(0.01, 20.0)),
        )
    )
    sim.reset()
    sensor.set_world_poses(
        torch.tensor([position], device=sim.device),
        torch.tensor([quaternion], device=sim.device),
        convention="ros",
    )
    intrinsic = torch.tensor(
        [[[camera["fx"], 0.0, camera["cx"]], [0.0, camera["fy"], camera["cy"]], [0.0, 0.0, 1.0]]],
        device=sim.device,
    )
    sensor.set_intrinsic_matrices(intrinsic, focal_length=1.0)
    sim.render()
    sensor.update(sim.get_physics_dt())
    matrices = sensor.data.intrinsic_matrices[0].detach().cpu().numpy()
    intrinsics_ok = abs(float(matrices[0, 0]) - float(camera["fx"])) <= 1.0 and abs(float(matrices[1, 1]) - float(camera["fy"])) <= 1.0
    intrinsics_ok = intrinsics_ok and abs(float(matrices[0, 2]) - float(camera["cx"])) <= 1.0 and abs(float(matrices[1, 2]) - float(camera["cy"])) <= 1.0
    forward = np.asarray(camera["world_from_camera"], dtype=np.float64)[:3, 2]
    origin = np.asarray(camera["world_from_camera"], dtype=np.float64)[:3, 3]
    pixel = project_pinhole(np.asarray(camera["world_from_camera"], dtype=np.float64), origin + forward, camera)
    frames = out / "static"
    frames.mkdir(parents=True, exist_ok=True)
    rgb_path = frames / "isaac_rgb.png"
    depth_path = frames / "isaac_depth.npy"
    from PIL import Image

    rgb = sensor.data.output["rgb"][0].detach().cpu().numpy()
    Image.fromarray(rgb[..., :3].astype(np.uint8)).save(rgb_path)
    np.save(depth_path, sensor.data.output["distance_to_image_plane"][0].detach().cpu().numpy())
    return {
        "runtime_intrinsic": matrices.tolist(),
        "declared_fx_fy_cx_cy": [camera["fx"], camera["fy"], camera["cx"], camera["cy"]],
        "intrinsics_within_one_pixel": bool(intrinsics_ok),
        "known_point_pixel": [float(pixel[0]), float(pixel[1])],
        "reprojection_error_px": float(np.linalg.norm(np.asarray(pixel) - np.array([camera["cx"], camera["cy"]]))),
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "convention": "ros",
    }


def _write(report: dict, out: Path | None) -> None:
    text = json.dumps(report, indent=2) + "\n"
    target = TASK / "reports" / "offline_bridge_static.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    if out is not None:
        (out / "static" ).mkdir(parents=True, exist_ok=True)
        (out / "static" / "offline_bridge_static.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)

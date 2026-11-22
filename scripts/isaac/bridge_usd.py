"""Compose one Isaac USD per packaged asset.

This module imports pxr and Isaac Lab. Import it only after ``AppLauncher`` has
started. ``MeshConverter`` converts one OBJ. The composite file is built here:
one root, a visible render child, and a hidden collision child.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from sensor_sim.scene_gen.mesh_io import read_mesh
from sensor_sim.scene_gen.task_scene_geometry import surface_distance


def convert_obj(obj_path: Path, usd_dir: Path, *, collision: str | None) -> str:
    """Convert one OBJ. Rigid-body and mass stay unset so they are not baked onto the child."""
    usd_dir.mkdir(parents=True, exist_ok=True)
    collision_props = None
    mesh_props = None
    if collision == "convexHull":
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
        mesh_props = schemas_cfg.ConvexHullPropertiesCfg()
    elif collision == "triangle":
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
        mesh_props = schemas_cfg.TriangleMeshPropertiesCfg(weld_tolerance=0.0)
    elif collision == "none":
        collision_props = None
    converter = MeshConverter(
        MeshConverterCfg(
            asset_path=str(obj_path),
            usd_dir=str(usd_dir),
            usd_file_name="converted.usd",
            force_usd_conversion=True,
            make_instanceable=False,
            collision_props=collision_props,
            mesh_collision_props=mesh_props,
            translation=(0.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )
    )
    if collision is None:
        _strip_collision(converter.usd_path)
    return converter.usd_path


def compose_asset(
    render_usd: str,
    collision_usd: str,
    destination: Path,
    *,
    dynamic: bool,
    mass_kg: float | None,
    approximation: str,
) -> dict:
    """Write ``/AssetRoot/Visual`` and ``/AssetRoot/Collision`` into one USD file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/AssetRoot")
    stage.SetDefaultPrim(root.GetPrim())
    visual = stage.DefinePrim("/AssetRoot/Visual", "Xform")
    visual.GetReferences().AddReference(render_usd)
    collision = stage.DefinePrim("/AssetRoot/Collision", "Xform")
    collision.GetReferences().AddReference(collision_usd)
    UsdGeom.Imageable(collision).MakeInvisible()
    # A disabled CollisionAPI is still cooked. The visual subtree must have the schemas removed.
    _remove_collision_schemas(stage, "/AssetRoot/Visual")
    _set_collision_enabled(stage, "/AssetRoot/Collision", enabled=True, approximation=approximation)
    if dynamic:
        if mass_kg is None or mass_kg <= 0:
            raise RuntimeError("reason=dynamic_mass_missing")
        body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(False)
        mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
        mass.CreateMassAttr(float(mass_kg))
    stage.GetRootLayer().Save()
    return inspect_stage(destination)


def inspect_stage(path: Path) -> dict:
    stage = Usd.Stage.Open(str(path))
    default = stage.GetDefaultPrim()
    prims = []
    for prim in stage.Traverse():
        prims.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                "mass": prim.HasAPI(UsdPhysics.MassAPI),
                "collision": prim.HasAPI(UsdPhysics.CollisionAPI),
                "mesh_collision": prim.HasAPI(UsdPhysics.MeshCollisionAPI),
                "triangle_mesh": prim.HasAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI),
                "convex_hull": prim.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI),
            }
        )
    return {
        "usd_path": str(path),
        "usd_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "default_prim": str(default.GetPath()) if default else None,
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "prims": prims,
    }


def describe_stage(path: Path) -> dict:
    """Schema, bounds, and transform record used by the collision audit.

    Approximation tokens are the USD ``physics:approximation`` values. They are
    not ``UrdfFileCfg.collider_type`` strings.
    """
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])
    prims = []
    for prim in stage.Traverse():
        imageable = UsdGeom.Imageable(prim) if prim.IsA(UsdGeom.Imageable) else None
        local_bound = None
        world_bound = None
        if prim.IsA(UsdGeom.Boundable):
            try:
                local_range = UsdGeom.Boundable(prim).ComputeLocalBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_).GetRange()
                if not local_range.IsEmpty():
                    local_bound = [list(local_range.GetMin()), list(local_range.GetMax())]
            except Exception:
                local_bound = None
            try:
                world = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if not world.IsEmpty():
                    world_bound = [list(world.GetMin()), list(world.GetMax())]
            except Exception:
                world_bound = None
        approximation = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        collision_enabled = None
        enabled_attr = prim.GetAttribute("physics:collisionEnabled")
        if enabled_attr.IsValid() and enabled_attr.HasAuthoredValue():
            collision_enabled = bool(enabled_attr.Get())
        mass_kg = None
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_kg = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            mass_kg = None if mass_kg is None else float(mass_kg)
        local_matrix = None
        if prim.IsA(UsdGeom.Xformable):
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local_matrix = [[float(matrix[row][col]) for col in range(4)] for row in range(4)]
        prims.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "applied_schemas": [str(schema) for schema in prim.GetAppliedSchemas()],
                "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                "mass_kg": mass_kg,
                "collision": prim.HasAPI(UsdPhysics.CollisionAPI),
                "mesh_collision": prim.HasAPI(UsdPhysics.MeshCollisionAPI),
                "collision_enabled": collision_enabled,
                "approximation": approximation,
                "collision_schemas": _collision_schema_names(prim),
                "purpose": str(imageable.GetPurposeAttr().Get()) if imageable is not None else None,
                "visibility": str(imageable.ComputeVisibility()) if imageable is not None else None,
                "local_bound": local_bound,
                "world_bound": world_bound,
                "world_matrix": local_matrix,
            }
        )
    default = stage.GetDefaultPrim()
    return {
        "usd_path": str(path),
        "usd_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "default_prim": str(default.GetPath()) if default else None,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "prims": prims,
    }


def mesh_points(stage_path: Path, root_prefix: str) -> np.ndarray:
    stage = Usd.Stage.Open(str(stage_path))
    points = []
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root_prefix) or prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        array = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in array:
            transformed = transform.Transform(Gf.Vec3d(*point.tolist()))
            points.append([transformed[0], transformed[1], transformed[2]])
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def compare_render_geometry(obj_path: Path, usd_path: Path) -> dict:
    """Compare the visible USD mesh with the source render OBJ. Extents must agree within 1 mm."""
    mesh = read_mesh(obj_path)
    source = np.asarray(mesh["vertices"], dtype=np.float64)
    converted = mesh_points(usd_path, "/AssetRoot/Visual")
    source_extent = source.max(axis=0) - source.min(axis=0)
    converted_extent = converted.max(axis=0) - converted.min(axis=0)
    source_center = source.mean(axis=0)
    converted_center = converted.mean(axis=0)
    extent_error = np.abs(source_extent - converted_extent)
    center_error = np.abs(source_center - converted_center)
    faces = []
    for corners in mesh["corners"]:
        indexes = [corner[0] for corner in corners]
        for offset in range(1, len(indexes) - 1):
            faces.append([indexes[0], indexes[offset], indexes[offset + 1]])
    distance = surface_distance(converted, source, np.asarray(faces, dtype=np.int64), sample_count=min(400, len(converted)), seed=0)
    return {
        "source_vertex_count": int(len(source)),
        "usd_vertex_count": int(len(converted)),
        "source_extent_m": source_extent.tolist(),
        "usd_extent_m": converted_extent.tolist(),
        "extent_error_m": extent_error.tolist(),
        "center_error_m": center_error.tolist(),
        "surface_distance": distance,
        "within_1mm": bool(np.all(extent_error <= 0.001) and np.all(center_error <= 0.001) and distance["max_m"] <= 0.001),
    }


def closed_volume_m3(obj_path: Path) -> dict:
    mesh = read_mesh(obj_path)
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    faces = []
    for corners in mesh["corners"]:
        indexes = [corner[0] for corner in corners]
        for offset in range(1, len(indexes) - 1):
            faces.append([indexes[0], indexes[offset], indexes[offset + 1]])
    triangles = np.asarray(faces, dtype=np.int64)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    volume = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    bbox = float(np.prod(extent))
    valid = bool(np.isfinite(volume) and volume > 0 and volume < bbox)
    return {"volume_m3": abs(volume) if valid else None, "signed_volume_m3": volume, "bbox_volume_m3": bbox, "valid": valid}


def bind_friction(stage_path: Path, static: float, dynamic: float, restitution: float, combine: str) -> None:
    stage = Usd.Stage.Open(str(stage_path))
    material = UsdShade.Material.Define(stage, "/AssetRoot/PhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(static))
    api.CreateDynamicFrictionAttr(float(dynamic))
    api.CreateRestitutionAttr(float(restitution))
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr(combine)
    physx.CreateRestitutionCombineModeAttr(combine)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/AssetRoot/Collision") and prim.GetTypeName() == "Mesh":
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()


def _strip_collision(usd_path: str) -> None:
    stage = Usd.Stage.Open(usd_path)
    _remove_collision_schemas(stage, "/")
    stage.GetRootLayer().Save()


def _collision_schema_names(prim: Usd.Prim) -> list[str]:
    names = []
    for schema in prim.GetAppliedSchemas():
        text = str(schema)
        if "Collision" in text or "Cook" in text:
            names.append(text)
    return names


def _collision_api_types() -> list[type]:
    found: list[type] = [UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI]
    for name in dir(PhysxSchema):
        if "Collision" not in name and "Cook" not in name:
            continue
        candidate = getattr(PhysxSchema, name)
        if isinstance(candidate, type):
            found.append(candidate)
    return found


def _remove_collision_schemas(stage: Usd.Stage, prefix: str) -> list[str]:
    """Remove collision and cooking APIs under ``prefix``. Do not only disable them.

    The removal is authored on the root layer, including an explicit ``apiSchemas``
    deletion, so a schema that arrived through a reference does not stay composed.
    """
    removed: list[str] = []
    layer = stage.GetRootLayer()
    stage.SetEditTarget(layer)
    api_types = _collision_api_types()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prefix != "/" and not path.startswith(prefix):
            continue
        if prim.GetTypeName() == "Mesh" and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            removed.append(f"{path} RigidBodyAPI")
        if prim.GetTypeName() == "Mesh" and prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)
            removed.append(f"{path} MassAPI")
        names = _collision_schema_names(prim)
        for api_type in api_types:
            if prim.HasAPI(api_type):
                prim.RemoveAPI(api_type)
        for name in list(names):
            prim.RemoveAppliedSchema(name)
        if names:
            _delete_applied_schemas(layer, prim.GetPath(), names)
            removed.append(f"{path} {','.join(names)}")
        attr = prim.GetAttribute("physics:collisionEnabled")
        if attr.IsValid():
            attr.Block()
    return removed


def _delete_applied_schemas(layer: Sdf.Layer, path: Sdf.Path, names: list[str]) -> None:
    spec = layer.GetPrimAtPath(path)
    if spec is None:
        spec = Sdf.CreatePrimInLayer(layer, path)
    existing = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else Sdf.TokenListOp()
    deleted = list(existing.deletedItems or [])
    for name in names:
        if name not in deleted:
            deleted.append(name)
    listop = Sdf.TokenListOp()
    if existing.explicitItems:
        listop.explicitItems = [item for item in existing.explicitItems if item not in names]
    if existing.prependedItems:
        listop.prependedItems = [item for item in existing.prependedItems if item not in names]
    if existing.appendedItems:
        listop.appendedItems = [item for item in existing.appendedItems if item not in names]
    listop.deletedItems = deleted
    spec.SetInfo("apiSchemas", listop)


def _set_collision_enabled(stage: Usd.Stage, prefix: str, *, enabled: bool, approximation: str | None) -> None:
    if not enabled:
        _remove_collision_schemas(stage, prefix)
        return
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(prefix) or prim.GetTypeName() != "Mesh":
            continue
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr(True)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        if approximation == "convexHull":
            mesh_collision.CreateApproximationAttr("convexHull")
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
        elif approximation == "triangle":
            mesh_collision.CreateApproximationAttr("none")
            PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"reason=rigid_body_on_geometry prim={prim.GetPath()}")

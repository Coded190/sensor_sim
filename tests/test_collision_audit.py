"""Schema and PhysX-log checks for the visual/collision split."""

from sensor_sim.scene_gen.collision_audit import schema_failures, visual_cook_failures


def _prim(path: str, **overrides) -> dict:
    prim = {
        "path": path,
        "type": "Xform",
        "applied_schemas": [],
        "collision_schemas": [],
        "rigid_body": False,
        "mass_kg": None,
        "collision": False,
        "mesh_collision": False,
        "collision_enabled": None,
        "approximation": None,
    }
    prim.update(overrides)
    return prim


def _can_stage() -> dict:
    return {
        "default_prim": "/AssetRoot",
        "up_axis": "Z",
        "meters_per_unit": 1.0,
        "prims": [
            _prim("/AssetRoot", type="Xform", rigid_body=True, mass_kg=0.002356),
            _prim("/AssetRoot/Visual"),
            _prim("/AssetRoot/Visual/geometry/mesh", type="Mesh"),
            _prim(
                "/AssetRoot/Collision/geometry/mesh",
                type="Mesh",
                applied_schemas=["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxConvexHullCollisionAPI"],
                collision_schemas=["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxConvexHullCollisionAPI"],
                collision=True,
                mesh_collision=True,
                approximation="convexHull",
            ),
        ],
    }


def test_clean_can_passes():
    assert schema_failures(_can_stage(), role="can") == []


def test_disabled_collision_api_on_visual_fails():
    stage = _can_stage()
    stage["prims"][2]["applied_schemas"] = ["PhysicsCollisionAPI"]
    stage["prims"][2]["collision"] = True
    stage["prims"][2]["collision_enabled"] = False
    failures = schema_failures(stage, role="can")
    assert any("visual_collision_schema" in item for item in failures)


def test_table_requires_triangle_mesh_token_none():
    stage = {
        "default_prim": "/AssetRoot",
        "up_axis": "Z",
        "meters_per_unit": 1.0,
        "prims": [
            _prim("/AssetRoot"),
            _prim("/AssetRoot/Visual/geometry/mesh", type="Mesh"),
            _prim(
                "/AssetRoot/Collision/geometry/mesh",
                type="Mesh",
                applied_schemas=["PhysicsCollisionAPI", "PhysxTriangleMeshCollisionAPI"],
                collision=True,
                approximation="none",
            ),
        ],
    }
    assert schema_failures(stage, role="table") == []
    stage["prims"][0]["rigid_body"] = True
    assert any("table_rigid_body" in item for item in schema_failures(stage, role="table"))


def test_visual_convex_cook_line_fails():
    log = "PhysicsUSD: falling back to convexHull approximation: /World/Can/Visual/geometry/mesh\n"
    assert visual_cook_failures(log)
    assert visual_cook_failures("cooking finished for /World/Can/Collision/geometry/mesh\n") == []

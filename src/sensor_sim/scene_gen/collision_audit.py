"""Acceptance checks for the can and table collision split.

These functions do not import Isaac or USD. The Isaac audit script builds the
stage description and the PhysX log text, then calls these checks.
"""

from __future__ import annotations


def visual_cook_failures(log_text: str) -> list[str]:
    """Return PhysX log lines that cook a visual render mesh.

    A convex-hull fallback that names the visual mesh is a failure even when
    the simulation looks stable. The line match is case-sensitive on ``Visual``
    because that is the composed prim name.
    """
    failures = []
    for line in log_text.splitlines():
        if "Visual" not in line:
            continue
        lowered = line.lower()
        if "convex" in lowered or "cook" in lowered or "parse collision" in lowered:
            failures.append(line.strip())
    return failures


def schema_failures(description: dict, *, role: str) -> list[str]:
    """Return schema violations for one composite USD description.

    ``role`` is ``can`` or ``table``. The can root is the only rigid body.
    Visual prims must not carry a collision or cooking schema. The can collider
    uses the USD approximation token ``convexHull``. The table collider uses
    token ``none`` together with a triangle-mesh collision schema.
    """
    if role not in {"can", "table"}:
        raise ValueError(f"reason=unknown_role role={role}")
    prims = description.get("prims") or []
    failures: list[str] = []
    rigid = [prim for prim in prims if prim.get("rigid_body")]
    root = description.get("default_prim")
    if role == "can":
        if len(rigid) != 1:
            failures.append(f"reason=rigid_body_count count={len(rigid)}")
        elif root is not None and rigid[0].get("path") != root:
            failures.append(f"reason=rigid_body_not_root path={rigid[0].get('path')}")
        elif rigid[0].get("mass_kg") is None:
            failures.append("reason=root_mass_missing")
    elif rigid:
        failures.append(f"reason=table_rigid_body paths={[prim.get('path') for prim in rigid]}")
    collision_meshes = []
    for prim in prims:
        path = str(prim.get("path", ""))
        schemas = [str(item) for item in prim.get("applied_schemas") or []]
        collision_names = [str(item) for item in prim.get("collision_schemas") or []]
        if "/Visual" in path:
            bad = [name for name in schemas + collision_names if "Collision" in name or "Cook" in name]
            if bad or prim.get("collision") or prim.get("mesh_collision"):
                failures.append(f"reason=visual_collision_schema path={path} schemas={sorted(set(bad))}")
            if prim.get("collision_enabled") is True:
                failures.append(f"reason=visual_collision_enabled path={path}")
            if prim.get("rigid_body"):
                failures.append(f"reason=visual_rigid_body path={path}")
        if "/Collision" in path and prim.get("type") == "Mesh":
            collision_meshes.append(prim)
            if prim.get("rigid_body"):
                failures.append(f"reason=collision_child_rigid_body path={path}")
    if len(collision_meshes) != 1:
        failures.append(f"reason=collision_mesh_count count={len(collision_meshes)}")
    elif role == "can" and collision_meshes[0].get("approximation") != "convexHull":
        failures.append(f"reason=can_approximation token={collision_meshes[0].get('approximation')}")
    elif role == "table":
        token = collision_meshes[0].get("approximation")
        schemas = [str(item) for item in collision_meshes[0].get("applied_schemas") or []]
        triangle = any("TriangleMesh" in name for name in schemas)
        if token != "none" or not triangle:
            failures.append(f"reason=table_approximation token={token} triangle={triangle}")
    if str(description.get("up_axis")) != "Z":
        failures.append(f"reason=up_axis value={description.get('up_axis')}")
    meters = description.get("meters_per_unit")
    if meters is None or abs(float(meters) - 1.0) > 1e-6:
        failures.append(f"reason=meters_per_unit value={meters}")
    return failures

# Connector gaps

Each row is a behavior the bridge needs and the upstream tool does not provide by itself. The audit statuses are below. The offline-bridge run updated the ones named in `docs/task_scene_offline_bridge_report.md`.

## Status after the offline bridge run

- Catalog and retrieval: `compensated`. `tests/test_asset_library.py` and `dataset/task_scene/library/catalog.json`.
- Shared pose and Mitsuba rebuild: `compensated` for object states. `scripts/mitsuba/replay_isaac_states.py` rendered `t0`, `t1`, and `t2`. Shape `to_world` is still a rebuild. Stokes is still not a polarization qualification.
- Composite USD: `compensated` for the hierarchy and the 1 mm geometry check. PhysX still cooks the can render mesh under the dynamic root as a convex hull. That exclusion is `open`.
- Camera: `compensated` for this square, centered pinhole. Runtime intrinsics matched. A general `cx`/`cy` offset remains `unsupported` by Isaac Lab 2.3.0 and the connector rejects it.
- Named finger or table contact pairs: `open`. The physics run recorded a net contact force of `0.023` N after `activate_contact_sensors` was enabled. A filtered table-versus-can pair was not required for the object-only slide.
- SO-ARM101 grasp: `blocked`. Not part of this run.

## Indoors OBJ is Y-up and has no package contract

- Required behavior: Z-up meter packages with a stable hash, a render mesh, and a separate collision mesh.
- Current capability: Indoors `generate_individual_assets` writes a Y-up OBJ and baked textures. The code license is BSD-3-Clause.
- Missing behavior: `asset_package_v1`, `asset_extension_v1`, collision role, and a redistribution grant for the generated mesh.
- Compensation: `exported_y_up_to_canonical` during packaging. The catalog keeps `redistribution_status=unresolved` and `material_fidelity=approximated`.
- Test: `tests/test_asset_library.py` and `verify_package` on the two live packages.
- Status: `partial`. Packaging exists. The catalog is the new retrieval surface.

## Mitsuba has no USD import and no live shape-transform update

- Required behavior: render the packaged render OBJ at a canonical pose, then at later Isaac poses.
- Current capability: Mitsuba 3.9.1 loads OBJ and accepts `to_world` at scene build. `scene.traverse()` on this build exposes `sensor.to_world` and not shape `to_world`. A shape-group instance cannot override the group's material.
- Missing behavior: USD import, in-place shape updates, and a polarization-qualified BSDF for baked Indoors maps.
- Compensation: one shape per instance, rebuilt from the connector for every snapshot. Stokes stays an experimental product. The constant light stays `relative`.
- Test: `tests/test_task_scene_connector.py` and `scripts/mitsuba/replay_isaac_states.py`.
- Status: `partial`.

## MeshConverter does not build the visual/collision pair

- Required behavior: one root, a visible render child with collision off, and a hidden collision child with collision on. Rigid body and mass on the dynamic root only.
- Current capability: `MeshConverter` converts one file. `rigid_props`, `mass_props`, and collision fields default to `None` and then add nothing. `usd_dir=None` writes `/tmp/IsaacLab`. The first converter pass is documented as Y-up centimeters before the stage is set to Z-up meters.
- Missing behavior: a shared root for two source OBJs, and a guarantee that identity scale preserves OBJ vertices.
- Compensation: compose `/AssetRoot/Visual` and `/AssetRoot/Collision` in a persistent USD. Compare bounds and surface samples with the source OBJ. Stop if the stage is not Z-up meters.
- Test: `scripts/isaac/bridge_render_scene.py` and `offline_bridge_static.json`.
- Status: `open`.

## Isaac camera intrinsics are not a general pinhole

- Required behavior: the manifest `fx`, `fy`, `cx`, `cy`, and the pinhole axes, within 1 pixel.
- Current capability: `Camera.set_intrinsic_matrices` assumes square pixels and a centered optical center. Pose conventions are `opengl` (forward -Z, up +Y), `ros` (forward +Z, up -Y), and `world` (forward +X, up +Z).
- Missing behavior: a general intrinsic matrix. The manifest camera happens to be square and centered (`fx = fy = 120`, `cx = 80`, `cy = 60` on a 160 by 120 image). The pinhole frame matches the ROS description.
- Compensation: pass the pose as ROS. Read the runtime intrinsics. Stop the camera gate if they disagree. Do not edit the manifest camera.
- Test: `tests/test_task_scene_connector.py` for the projection, and the Isaac camera readback.
- Status: `open`.

## PhysX dynamic bodies do not accept a concave triangle mesh

- Required behavior: the can collides as a dynamic convex body. The table collides as a static triangle mesh of the packaged proxy, including the space under the top.
- Current capability: PhysX cooks a dynamic convex hull and a static triangle mesh. It does not accept a dynamic concave triangle mesh.
- Missing behavior: knowledge of which packaged file is the proxy and which is the hull.
- Compensation: the catalog `collision_method` selects `convexHull` for the can and a static triangle mesh for the table. A hull that fills the volume under the tabletop fails the bounds check.
- Test: the USD inspection in the static report, then the gravity settle.
- Status: `open`.

## Isaac quaternion and velocity layouts are caller obligations

- Required behavior: canonical `world_from_asset` and world-frame linear and angular velocity.
- Current capability: `write_root_pose_to_sim` takes `(x, y, z, w, x, y, z)`. `write_root_velocity_to_sim` takes linear then angular velocity, in the world frame, at the center of mass.
- Missing behavior: a conversion from our matrix into that pose, and a statement that the velocity is the center of mass rather than the mesh origin.
- Compensation: one tested matrix-to-`wxyz` function. The state record stores that convention in the snapshot. The initial velocity is applied with `write_root_velocity_to_sim` before the first step.
- Test: `tests/test_scene_state.py` and `tests/test_task_scene_connector.py`.
- Status: `open`.

## Task retrieval is a different catalog

- Required behavior: deterministic lookup of these packages by id, category, and tag.
- Current capability: `task_retrieve.py` ranks text and may call an embedding backend.
- Missing behavior: package-hash checks and collision-role filters.
- Compensation: `asset_library.py`. `task_retrieve.py` is not called.
- Test: `tests/test_asset_library.py`.
- Status: `open`.

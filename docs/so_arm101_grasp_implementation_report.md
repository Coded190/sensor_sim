# SO-ARM101 grasp implementation

The object-only offline bridge stays `passed`. The grasp gate stays `blocked` until Stages 1–7 pass. This file is the running record. Hashes below were recomputed from the live tree at the start of this cycle. They agree with the bridge report. They were not copied in as the expected values.

## Stage 0 — freeze

Status: `passed`.

- Branch `main`, HEAD `d1cc0c7f0edfb84ea4f0b6bf049639d8b6c7efee`, tracking `origin/main`.
- `git status --short --branch` showed the existing task-scene implementation as uncommitted. No commit was created.
- Table package `1d28aed9d1366d2f64f417abaff443d916b778c3960dea731f139f7114d42976`. Sidecar `dataset/task_scene/library/table_dining/1.0.0.asset_extension.json` SHA-256 `5d1835f66f82ee33a58963f1a75b4e88d1d3619027a85d8f3df00b7dd42e95ca`.
- Can package `9d5419cae8b43dcf637233756a060c311df04aea59a46afb6077a4666b2dc9e2`. Sidecar `dataset/task_scene/library/can/1.0.0.asset_extension.json` SHA-256 `ef9c8e4fbd2347f4705225c76287377059e5016efe8de449308b97242ffd129f`.
- `pick_can` manifest file SHA-256 `463d082c30e0e892a1842c696082799b28cbaeb02111cc9ee5e74c0d42e1db72`.
- Catalog file SHA-256 `81bbec185fba599da99709a0174741822fd61569312f25dc7798107fcaafe49e`.
- Scene-resolution SHA-256 `6632efb4a5cb8e4eac2f55416a4d561d47c4b7ae5e9df0dca3d6cf117180b02b`.
- Object-only run directory `dataset/task_scene/offline_bridge/463d082c30e0e892a1842c696082799b28cbaeb02111cc9ee5e74c0d42e1db72/cf9347d7129020054a6e1c0a139120eb72440c20155487ec2464699e98d95dd7`. `states.jsonl` SHA-256 `9ad0f4a0265ef34b590d5871ab5c52f92963173805d7c5e0c1f537be29af1d3e`.
- Passed composite USDs, not overwritten: table `ded93e81144ed34554cb2fb37a6263201bdd5f3dbe52386e06695dde06b16c4f`, can `3e45d81f9e0cb84a82925797c432394bd91ab7c9a932de02731ffb98f1a8e010`.
- SO-ARM101 URDF SHA-256 `2f3269d80f84ab0040a8160b34e99bf16c8f3f197eb989e1a6afee6567725c61`. `so_arm101.py` SHA-256 `cf2ebf4167a9d390d9949f1df3018d68a6b429cf4de6fc1c23793bb54201a06c`.
- Isaac interpreter `/home/cmoore/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11`.
- Isaac `PYTHONPATH`: `/home/cmoore/moore/isaac_so_arm101/.venv/lib/python3.11/site-packages/isaaclab/source/isaaclab:/home/cmoore/moore/isaac_so_arm101/.venv/lib/python3.11/site-packages:/home/cmoore/moore/isaac_so_arm101/src:/home/cmoore/sensor_sim/src:/home/cmoore/sensor_sim/scripts/isaac`.
- On-disk Isaac Lab package file `.../isaaclab/source/isaaclab/isaaclab/__init__.py`. `extension.toml` package version `0.47.2`, which is what `isaaclab.__version__` reads. The robot project pin is `isaaclab[all,isaacsim]==2.3.0`.
- Isaac Sim `VERSION` file `5.1.0-rc.19+release.26219.9c81211b.gl`.
- PhysX schema package directory `omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353`. A separate `PHYSX_VERSION` symbol was not exposed in the earlier bridge run.
- Mitsuba `3.9.1`. Import-time variant is `None` until `require_variant()` selects `cuda_ad_spectral_polarized`.

The saved can composite still has `PhysicsCollisionAPI` on `/AssetRoot/Visual/geometry/mesh` with collision enabled left false and no mesh-collision API. That is the schema the next stage removes. The collision child has the convex-hull API. The table visual mesh has the same disabled collision API.

### Tests

Focused bridge tests, both `LD_LIBRARY_PATH` and `PYTHONPATH` unset, project `.venv`:

`python -m pytest tests/test_asset_library.py tests/test_task_scene_connector.py tests/test_scene_state.py tests/test_task_scene_composer.py tests/test_task_scene_contracts.py tests/test_episode_writer.py -q`

28 passed in 0.68 s.

Full project pytest, same sanitized environment:

`python -m pytest -q`

346 passed, 2 failed, in 209.96 s. Both failures are `tests/test_phase4.py` opening the missing `outputs/phase3/frames/000/stokes/render.exr`. That file was not created. The three earlier Blender failures, which came from an inherited Isaac `LD_LIBRARY_PATH`, did not recur.

`tests/test_collision_audit.py` was added after that full run and passed separately: 4 passed. It checks the schema predicate and the visual-mesh cook-log rule. It does not import Isaac.

## Stage 1 — visual collision schemas

Status: `passed`.

Command, Isaac interpreter, and `PYTHONPATH` are in `dataset/task_scene/reports/robot_collision_cleanup.json`. Isaac Lab `__file__` is the package path above and `__version__` is `0.47.2`. PhysX reported `imported`. The run did not overwrite the passed bridge USDs. Their hashes after the run are still `3e45d81f9e0cb84a82925797c432394bd91ab7c9a932de02731ffb98f1a8e010` and `ded93e81144ed34554cb2fb37a6263201bdd5f3dbe52386e06695dde06b16c4f`.

New USDs are under `dataset/task_scene/robot_collision/463d082c30e0e892a1842c696082799b28cbaeb02111cc9ee5e74c0d42e1db72/schema_removal_v1/`.

The can visual mesh `/AssetRoot/Visual/geometry/mesh` has no applied schemas, no collision API, and `collision_enabled` unset. The can collision mesh keeps `PhysicsCollisionAPI`, `PhysicsMeshCollisionAPI`, `PhysxConvexHullCollisionAPI`, and `PhysxCollisionAPI`, with USD approximation token `convexHull`. The table visual mesh has no collision schema. The table collision mesh uses approximation token `none` and `PhysxTriangleMeshCollisionAPI`. The can root is the only rigid body, with mass `0.0023561118122472045` kg from the closed collision-mesh volume times the simulation density 400. That mass is not a measured can mass. Render geometry stayed within 1 mm of the packaged OBJs. The maximum surface distance was `1.68e-9` m for the can.

The drop released the can 20 mm above the manifest pose, at z `0.865598` m. Settled position is the mean root position over the last 0.5 s at 120 Hz. That mean was `(-0.037886, -0.180225, 0.845598)` m. The maximum distance of a sample in that window from the mean was `6.1e-8` m. The tolerance was 5 mm. Overlap is the table collision mesh maximum z minus the can collision mesh minimum z at the settled pose. It was `2.9e-7` m, so the can rested on the packaged table collider rather than floating or penetrating. The table world matrix did not change. The can did not fall through.

`net_forces_w` on the can was nonzero during the settle window. The run maximum of that norm was `0.185` N, which includes the landing impact. The can weight at this mass is about `0.023` N. PhysX logged `GPU contact filter for collider '/World/Table' is not supported`, so this force is the can body's net normal force, not a filtered pair report. The visual mesh has no collision schema, and the collision child is the only can collider, so that contact is carried by the packaged collision mesh.

The Kit log for this process has no `Parse collision` line and no convex-cook line naming the visual mesh. `visual_cook_lines` is empty. A quiet log was not treated as sufficient by itself. The schema inspection and the settle measurement are the other two required checks.

## Stage 2 — SO-ARM101 identity and finger contacts

Status: `failed`.

Command: `scripts/isaac/so_arm101_preflight.py` with the Isaac interpreter and the same `PYTHONPATH` as Stage 1. Report: `dataset/task_scene/reports/so_arm101_preflight.json`. Earlier attempts are kept beside it as `so_arm101_preflight_attempt1` through `attempt6`. The robot package was not edited. `SO_ARM101_CFG.spawn.activate_contact_sensors` is `False` in the source and stayed `False` on the imported object. A `copy.deepcopy` set it `True` on the nested URDF spawn.

Verified on the live config: `fix_base=True`, `merge_fixed_joints=True`, `replace_cylinders_with_capsules=True`, `collision_from_visuals=False`, `collider_type=convex_hull`. The six joints resolve by name to articulation ids 0 through 5 in the order `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. `robot.is_fixed_base` is true. `gripper_frame_link` is still an articulation body, so the importer did not merge that fixed joint away. Its origin is 34.7 mm from the measured TCP, outside the 5 mm match tolerance, so the TCP is the pad midpoint expressed in `gripper_link` and is not redefined when the jaw moves.

PhysX contact-report API is on `/World/Robot/gripper_link` and `/World/Robot/moving_jaw_so101_v1_link`. Both are rigid bodies. The no-contact run was 5.0 s at 120 Hz, 600 samples per sensor. Every sample was exactly 0 N, so the application threshold is strict positive force. `ContactSensorCfg.force_threshold` was left at `1e-4` N for air-time tracking only and was not used as this threshold.

Pinch gap is the distance between the inner surfaces of the distal 40 percent of each jaw, measured along the jaw-to-jaw direction perpendicular to the finger. At a commanded gripper angle of 1.0 rad the achieved angle was 0.974 rad and that gap was 32.5 mm. The can collision mesh is 29.2 mm along its shortest axis, so the total clearance at the widest measured opening is 3.3 mm. That is already below the 4 mm clearance required for a later grasp.

Fixed finger only, at 0.4 mm overlap: fixed-finger filtered force 0.0058 N, moving finger 0 N, and the fixed contact point lies inside the pad bounds. That case passed.

Moving finger only and bilateral contact did not pass. Placing the can on the moving pad, or between the pads, produced forces of several newtons on both bodies for at most 0.017 s. The 0.05 s bilateral requirement was not met, and those contact points were not both inside the pad regions. The can is heavier than the clearance, so it is pushed out of the 3.3 mm gap. The velocity of the can was zeroed after each calibration step so it would not fall. Gravity on the can was disabled for the same reason. Those are calibration choices, not a grasp.

## Later stages

Stages 3–7 are `blocked`. They depend on independent pad contact from both fingers and on a jaw opening that clears the can. The grasp gate remains `blocked`. The object-only bridge remains `passed`.

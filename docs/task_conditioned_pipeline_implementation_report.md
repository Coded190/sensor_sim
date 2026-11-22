# Task-conditioned pipeline implementation report

HEAD at the start of this work was `d1cc0c7f0edfb84ea4f0b6bf049639d8b6c7efee` on `main`, even with `origin/main`. The protected file hashes listed in the architecture audit are unchanged. Nothing was committed or pushed. `assets/candidates/index.json` was not edited. Dataset v1 was not edited. The pinned Indoors commit was not changed.

Status words: `implemented_and_tested`, `implemented_not_qualified`, `partial`, `planned`, `unsupported`, `blocked`.

## Gate summary

| Gate | Required evidence | Status | Blocking issue |
| --- | --- | --- | --- |
| Stage 0 | Live audit | implemented_and_tested | None. Audit is `docs/task_conditioned_pipeline_architecture_audit.md`. |
| Stage 1 | Versioned schemas and tests | implemented_and_tested | None. 18 focused tests passed. |
| Gate A | Two real Infinigen packages | implemented_and_tested | Generated-mesh redistribution license remains `unresolved`. Table leg proxy is coarse. |
| Gate B | Deterministic one-object scene | implemented_and_tested | `reachable_from` is `not_evaluated`. |
| Gate C | Real Mitsuba render and portable replay | implemented_and_tested | Shape transforms are applied by rebuilding. Stokes is not a polarization qualification. |
| Gate D | Physics pick-and-place with the collision meshes | blocked | Differential IK did not reach the can. Converted collision USD did not become a rigid body, so the bodies that did simulate were cuboids fitted to the collision bounds. |
| Gate E | Episode from a real Isaac trajectory and three Mitsuba snapshots of those states | blocked | No successful grasp trajectory exists to record. |
| Expansion | Five assets and one-at-a-time variations | planned | Not started. Gates D and E did not pass. |

## What was built

- `src/sensor_sim/scene_gen/asset_extension.py` validates `asset_extension_v1` against the package id, version, and hash. Render and collision paths must differ.
- `src/sensor_sim/scene_gen/scene_manifest.py` validates `scene_manifest_v1`, rigid transforms, and the Blender, Mitsuba, and USD conversions. Mitsuba 3.9.1 on this machine looks along camera +Z. The pinhole frame is rolled 180 degrees about forward so that +Y-down becomes +Y-up. That was checked against `ScalarTransform4f.look_at`.
- `src/sensor_sim/scene_gen/task_scene_request.py` rejects unknown categories, bad sizes, and `facing`, `attached_to`, `aligned_with`, and `reachable_from`.
- `src/sensor_sim/scene_gen/task_scene_composer.py` resolves one support and one object, writes a per-check report, and does not treat support as reachability.
- `build_mitsuba_scene` was added to `mitsuba_scene_package.py`. `build_scene` and `replay_infinigen_scene` were not removed.
- `src/sensor_sim/dataset/episode_writer.py` is a separate JSON record. It does not call LeRobot and does not modify Dataset v1.
- `scripts/isaac/gate_d_preflight.py` runs in the Isaac environment. `isaac_scene_adapter.py` was not added, because the feasibility run did not grasp the object.

## Gate A

Commands, from `/home/cmoore/infinigen_src/indoors-stable`, interpreter `/home/cmoore/infinigen_envs/indoors/bin/python`:

- `python -m infinigen_examples.generate_individual_assets -o /home/cmoore/sensor_sim/dataset/task_scene/generation/table_dining_seed7 -f TableDiningFactory -n 1 -D 7 -s --export obj -r none -G --export_texture_res 1024`
- The same command for `CanFactory`, seed 11, output `.../can_seed11`

Both exited 0. Source blends:

- `dataset/task_scene/source_blends/table_dining.blend` SHA-256 `00ad37f0c3ca645d0ae275030600eba55171f2cc5ef7077656c60eab80444c10`
- `dataset/task_scene/source_blends/can.blend` SHA-256 `a3577a42f1f9bbafe30a492c6e169e20dec864db41ccb56a865e1857694a1150`

Packages, written by `scripts/infinigen/package_task_asset.py` with the project `.venv` (exit 0):

- `table_dining` `1.0.0` hash `1d28aed9d1366d2f64f417abaff443d916b778c3960dea731f139f7114d42976`. Size about 1.098 x 0.941 x 0.846 m. Tabletop vertices are at most 2.7 mm from the slab collider (p50 0.12 mm, p95 1.8 mm). The leg box is up to 119 mm from the render surface. That is recorded and is not called an accurate leg collider.
- `can` `1.0.0` hash `9d5419cae8b43dcf637233756a060c311df04aea59a46afb6077a4666b2dc9e2`. Size about 0.113 x 0.047 x 0.029 m. Convex-hull max surface distance 1.6 mm.

Upstream code license is BSD-3-Clause. Generated mesh redistribution is `unresolved`. The can is a size candidate for the gripper, not a demonstrated grasp. The packaged render is `dataset/task_scene/scenes/pick_can/render.png`.

## Gate B

`compose_scene` on the two packages produced scene hash `826294842441006b1aa624b91328930bc37141aa64eb45657abd08228ebf4d3b`. The validation file is `dataset/task_scene/scenes/pick_can/validation.json`. No check in that file has status `fail`. `reachable_from` is `not_evaluated`. Fixture tests show that changing the layout seed moves the object and leaves the camera, light, material profile, and asset id unchanged, and that a different selection seed changes only the chosen can when two cans are in the library.

## Gate C

Command: `cd /home/cmoore/sensor_sim && .venv/bin/python scripts/infinigen/run_task_scene_gates.py`. Exit 0. Mitsuba 3.9.1, variant `cuda_ad_spectral_polarized`, Stokes shape `[120, 160, 15]`. The linear EXR mean absolute RGB difference against the same render with the can omitted is `0.00048055677325464785`. Every sampled can vertex projects inside the image. A copied library under `dataset/task_scene/scenes/pick_can/portable` reloads both instance ids. Rebuilding the scene after adding 0.2 m to the can's x translation moves the can center by 0.200 m and leaves the table center unchanged. `scene.traverse()` exposed `sensor.to_world` and not the shape transforms, so the pose change is a rebuild. That limitation is in `dataset/task_scene/scenes/pick_can/mitsuba_conversion.json`. The constant light is `relative`, not calibrated. Stokes output is a render product only.

## Gate D

Interpreter: `/home/cmoore/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11`. `PYTHONPATH` included the Isaac Lab source tree, the Isaac site-packages, and `/home/cmoore/moore/isaac_so_arm101/src`. The project's `.venv/bin/python` symlink is broken, so that launcher was not used. Isaac Sim reported GPU RTX 5080 and driver 580.178.04. The pin in the robot project is `isaaclab[all,isaacsim]==2.3.0`.

`dataset/task_scene/reports/gate_d_preflight.json` records:

- Launch succeeded.
- Joints present: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
- Gripper moved from 0.385 rad to 0.086 rad under joint-position targets. The existing lift config uses 0.5 open and 0.0 close.
- A cuboid fitted to the can's collision bounds settled at z 0.8605 m and did not fall through a kinematic table slab.
- `MeshConverter` wrote `/tmp/IsaacLab/usd_20261121_100416_4606/collision.usd` and `/tmp/IsaacLab/usd_20261121_100416_6387/collision.usd`. `UsdFileCfg` did not apply a rigid body on the converted root. The simulated bodies were cuboids, not those meshes.
- Differential IK ran. The gripper link ended near `(0.264, -0.333, 0.948)` m. The can stayed at `(-0.040, -0.180, 0.861)` m. The arm did not reach the can, did not close on it, and did not lift it.

Gate D is `blocked`. No `isaac_scene_adapter.py` was added. A headless viewport was not available (`GLFW initialization failed`), so there is no Isaac image of the robot. Simulation-only numbers are in `configs/task_scene/isaac_gate_d_sim_defaults.yaml` and are not in the asset extension.

## Gate E

`episode_record_v1` round-trips, rejects a decreasing timestamp, and keeps joint targets in `action` rather than in `joint_position`. Those tests passed. They use a synthetic fixture. They are not a successful robot episode. LeRobot `v3.0` (`add_frame`, `save_episode`, parquet) was not called, because that package is not in the project environment and there is no grasp trajectory to export. Three Mitsuba snapshots of Isaac states were not rendered.

## Tests

Command, working directory `/home/cmoore/sensor_sim`, interpreter `.venv/bin/python`, `PYTHONPATH` unset:

`python -m pytest tests/test_episode_writer.py tests/test_task_scene_contracts.py tests/test_task_scene_composer.py -q`

Exit 0. `18 passed`.

Unexecuted: the full pytest suite, including the two existing `tests/test_phase4.py` failures for the missing Phase 3 EXR. Those files were not created. A robot success test against the 5 mm / 0.5 s settle rule was not passed. Bottle, bowl, cup, and the remaining table variants were not generated.

## Offline bridge

The object-state bridge is reported in `docs/task_scene_offline_bridge_report.md`. Isaac and Mitsuba shared the packaged table and can. PhysX moved the can `0.0399` m and Mitsuba rebuilt that trajectory. SO-ARM101 Gate D stays `blocked`. The grasp is the next implementation, and it was not run.

## Smallest next action

Make the converted collision USD a rigid body at the prim Isaac Lab's `RigidObject` resolves, spawn that mesh instead of the fitted cuboid, and drive the existing differential IK until the gripper link reaches the can before any close-and-lift sequence is claimed.

## Verification index

This section is the per-stage map for checking the work. Paths below are absolute. Status words are the same as the table at the top of this report. The robot embodiment in the Isaac preflight is the SO-ARM101, loaded from `SO_ARM101_CFG` in `/home/cmoore/moore/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/so_arm101.py`. That config points at `/home/cmoore/moore/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf`. The preflight recorded the six URDF joints `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, and `gripper`, and it commanded the body `gripper_link`. SO-ARM100 was not instantiated.

### Stage 0 — done

Code and documents:

- `/home/cmoore/sensor_sim/docs/task_conditioned_pipeline_architecture_audit.md`

How it works: the audit was written from the live tree before the new modules were added. It records existing function signatures, the Indoors individual-asset command, the Isaac interpreter that actually imports (the project Isaac venv symlink is broken), and the LeRobot `v3.0` dataset API. It does not generate geometry or launch Isaac.

Result: the audit file itself. There is no image for this stage. Focused tests that were already in the tree (`tests/test_asset_package.py`, `tests/test_task_scenes.py`, `tests/test_tabletop_layout.py`, `tests/test_infinigen_default.py`, `tests/test_expansion_parallel.py`) were 32 passed. Protected modules listed in the audit were not edited.

### Stage 1 — done

Code:

- `/home/cmoore/sensor_sim/src/sensor_sim/scene_gen/asset_extension.py`
- `/home/cmoore/sensor_sim/src/sensor_sim/scene_gen/scene_manifest.py`
- `/home/cmoore/sensor_sim/src/sensor_sim/scene_gen/task_scene_request.py`
- `/home/cmoore/sensor_sim/src/sensor_sim/scene_gen/task_scene_geometry.py`
- `/home/cmoore/sensor_sim/docs/task_conditioned_pipeline_schemas.md`
- `/home/cmoore/sensor_sim/tests/test_task_scene_contracts.py`

How it works: `validate_extension` checks `asset_extension_v1` against the package id, version, and SHA-256, and it requires separate render and collision paths. `validate_rigid` accepts a 4×4 right-handed rigid matrix (column vectors, last row `[0, 0, 0, 1]`, determinant about +1). `canonical_to_mitsuba` is identity because the canonical frame is already Z-up meters. `pinhole_camera_to_mitsuba` rolls the stored +Y-down pinhole frame 180 degrees about forward (`diag(-1, -1, 1)`), because Mitsuba 3.9.1 on this machine looks along camera +Z with +Y up. `domain_seed` is the first 8 bytes of SHA-256 of `"{master}:{domain}"` modulo `2^31-1`. Unsupported relations (`facing`, `attached_to`, `aligned_with`, `reachable_from`) fail closed.

Result: schema document plus `18 passed` from `tests/test_episode_writer.py`, `tests/test_task_scene_contracts.py`, and `tests/test_task_scene_composer.py`. No image.

### Gate A — done, with two disclosed limits

Code:

- `/home/cmoore/sensor_sim/scripts/infinigen/package_task_asset.py`
- `/home/cmoore/sensor_sim/src/sensor_sim/scene_gen/task_scene_geometry.py` (`table_collision`, `convex_collision`, `surface_distance`)

Directories:

- `/home/cmoore/sensor_sim/dataset/task_scene/generation/table_dining_seed7/`
- `/home/cmoore/sensor_sim/dataset/task_scene/generation/can_seed11/`
- `/home/cmoore/sensor_sim/dataset/task_scene/source_blends/`
- `/home/cmoore/sensor_sim/dataset/task_scene/library/table_dining/1.0.0/`
- `/home/cmoore/sensor_sim/dataset/task_scene/library/can/1.0.0/`
- `/home/cmoore/sensor_sim/dataset/task_scene/staging/`

Results:

- `/home/cmoore/sensor_sim/dataset/task_scene/source_blends/table_dining.blend`
- `/home/cmoore/sensor_sim/dataset/task_scene/source_blends/can.blend`
- `/home/cmoore/sensor_sim/dataset/task_scene/library/table_dining/1.0.0.asset_extension.json`
- `/home/cmoore/sensor_sim/dataset/task_scene/library/can/1.0.0.asset_extension.json`
- `/home/cmoore/sensor_sim/dataset/task_scene/reports/table_dining_qualification.json`
- `/home/cmoore/sensor_sim/dataset/task_scene/reports/can_qualification.json`
- Table diffuse atlas: `/home/cmoore/sensor_sim/dataset/task_scene/generation/table_dining_seed7/export_007/TableDiningFactory(7)_spawn_asset(7)_DIFFUSE.png`
- Can diffuse atlas: `/home/cmoore/sensor_sim/dataset/task_scene/generation/can_seed11/export_011/CanFactory(11)_spawn_asset(11)_DIFFUSE.png`

How it works: Indoors `generate_individual_assets` ran `TableDiningFactory` seed 7 and `CanFactory` seed 11. `package_task_asset.py` shifts the exported mesh so its minimum z is 0, writes a separate collision mesh (table: slab plus a narrower leg box, method `primitive_proxy`; can: convex hull), copies the OBJ, MTL, and PNGs into the package, and writes the `asset_extension_v1` sidecar outside the package directory. Surface distance samples up to 4000 points. Tabletop acceptance uses the tabletop band against the slab, not the legs.

Images: the two DIFFUSE files are texture atlases baked by Indoors, not photographs of the assembled objects. The assembled view is the Gate C render.

Limits that stay open: generated-mesh redistribution is `unresolved` (Indoors code is BSD-3-Clause; that license is not a redistribution license for the meshes). The table leg box is up to 119 mm from the render surface and is not an accurate leg collider. Tabletop render-to-collision max is 2.7 mm. Can render-to-collision max is 1.6 mm.

### Gate B — done

Code:

- `/home/cmoore/sensor_sim/src/sensor_sim/scene_gen/task_scene_composer.py`
- `/home/cmoore/sensor_sim/scripts/infinigen/run_task_scene_gates.py`

Directories and results:

- `/home/cmoore/sensor_sim/dataset/task_scene/scenes/pick_can/scene_manifest.json` (SHA-256 `826294842441006b1aa624b91328930bc37141aa64eb45657abd08228ebf4d3b`)
- `/home/cmoore/sensor_sim/dataset/task_scene/scenes/pick_can/validation.json`

How it works: `compose_scene` validates the request, selects one support and one object, checks the extension hash, places the table at the identity, and places the can on the support frame from the layout-domain seed with a 0.05 m margin. `measure_support` uses a 1 mm gap limit and a 0.1 mm penetration limit. Appearance, camera, and light each use their own domain seed. The camera looks from about `(0.9, -1.3, 1.5)` toward `(0, 0, 0.4)` with `look_at_y_down`. A failed check raises `ComposerError`.

Result: validation has no `fail` row. `reachable_from` is `not_evaluated` because reachability requires the robot. No image of its own; the composed scene is the Gate C PNG.

### Gate C — done

Code:

- `build_mitsuba_scene` in `/home/cmoore/sensor_sim/src/sensor_sim/scene_gen/mitsuba_scene_package.py`
- `/home/cmoore/sensor_sim/scripts/infinigen/run_task_scene_gates.py`

Results:

- `/home/cmoore/sensor_sim/dataset/task_scene/scenes/pick_can/render.png`
- `/home/cmoore/sensor_sim/dataset/task_scene/scenes/pick_can/render.exr`
- `/home/cmoore/sensor_sim/dataset/task_scene/scenes/pick_can/render_without_object.png`
- `/home/cmoore/sensor_sim/dataset/task_scene/scenes/pick_can/render_without_object.exr`
- `/home/cmoore/sensor_sim/dataset/task_scene/scenes/pick_can/mitsuba_conversion.json`
- `/home/cmoore/sensor_sim/dataset/task_scene/scenes/pick_can/gate_c_summary.json`
- `/home/cmoore/sensor_sim/dataset/task_scene/scenes/pick_can/portable/`

How it works: the runner loads the packaged OBJs (already Z-up), composes the manifest, and builds one Mitsuba scene per instance. The variant is `cuda_ad_spectral_polarized`. The film is Stokes, then a PNG is written from the RGB channels. A second render omits the can. Visibility is the mean absolute difference of the linear EXR RGB, not the 8-bit PNG. A copy of the library under `portable/` is reloaded. Moving the can by 0.2 m in x and rebuilding moves only the can.

What the images show: `render.png` is a dark rectangular top, two pale legs, and a small light object on the top. `render_without_object.png` is the same table with that object absent. The measured linear EXR mean absolute RGB difference is `0.00048055677325464785`. Every sampled can vertex projects inside the 160×120 image. Stokes shape is `(120, 160, 15)`. That Stokes film is a render product. It is not a polarization qualification. The constant emitter value is relative, not a calibrated radiance. `scene.traverse()` does not expose shape `to_world`, so a pose change is a scene rebuild.

### Gate D — did not pass

Code that ran:

- `/home/cmoore/sensor_sim/scripts/isaac/gate_d_preflight.py`
- `/home/cmoore/sensor_sim/configs/task_scene/isaac_gate_d_sim_defaults.yaml`

Robot code, not modified:

- `/home/cmoore/moore/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/so_arm101.py`
- `/home/cmoore/moore/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf`

Result:

- `/home/cmoore/sensor_sim/dataset/task_scene/reports/gate_d_preflight.json` (`status` is `preflight_ran`)

How it works: the script launches Isaac Lab headless with `/home/cmoore/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11` and imports `SO_ARM101_CFG` only. It converts the packaged collision OBJs with `MeshConverter`, then, because those USD roots had no rigid body, it spawned a kinematic cuboid for the table `(1.098, 0.941, 0.02)` m and a dynamic cuboid for the can `(0.113, 0.047, 0.029)` m at density 400 kg/m³. The arm base is at `(0.25, -0.42, 0.85)` m. Gripper joint targets are 0.5 rad then 0.0 rad. `DifferentialIKController` is position / damped least squares, and `set_command` is given the current end-effector quaternion because position mode still requires it. The Jacobian column is `body_index - 1` on the fixed base, and the body name is `gripper_link`.

Why it did not pass: the required evidence is a physics pick-and-place of the collision meshes. The converted USD did not become a rigid body (`rigid_body_on_converted_root` is false), so the bodies that settled were the fitted cuboids. The gripper link ended near `(0.264, -0.333, 0.948)` m while the can stayed near `(-0.040, -0.180, 0.861)` m, about 0.35 m away. The arm did not reach, close on, lift, or place the can. The 5 mm / 0.5 s settle rule was not met. `isaac_scene_adapter.py` was not added. GLFW initialization failed in the headless session, so there is no Isaac PNG of the robot. The USD files under `/tmp/IsaacLab/usd_20261121_100416_4606/collision.usd` and `/tmp/IsaacLab/usd_20261121_100416_6387/collision.usd` are temporary and may not persist.

### Gate E — did not finish

Code:

- `/home/cmoore/sensor_sim/src/sensor_sim/dataset/__init__.py`
- `/home/cmoore/sensor_sim/src/sensor_sim/dataset/episode_writer.py`
- `/home/cmoore/sensor_sim/tests/test_episode_writer.py`

How it works: `write_episode` validates `episode_record_v1` (monotonic time, frame indexes, matching joint and action lengths, finite values) and writes `episode.json` plus `episode.sha256` by renaming a `.partial` directory. Joint targets stay in `action`. It does not call LeRobot and it does not write Dataset v1.

Why it did not finish: the tests use a synthetic fixture. No real SO-ARM101 trajectory was available, so no episode file was written from Isaac states, and the three Mitsuba snapshots of those states were not rendered. LeRobot at `/home/cmoore/moore/lerobot` (`CODEBASE_VERSION` `v3.0`) was not called.

### Expansion — not started

Bottle, bowl, cup, a second table, one-parameter variations, and extra task types were not generated. The plan stops expansion until Gates D and E pass. That stop was kept.

## What the next implementation plan has to revisit

Done and usable as the starting point: Stage 0 audit, Stage 1 contracts and their tests, Gate A packages for one `TableDiningFactory` and one `CanFactory`, Gate B manifest `826294842441006b1aa624b91328930bc37141aa64eb45657abd08228ebf4d3b`, and Gate C Mitsuba render of that manifest. The embodiment for any robot work is the SO-ARM101 URDF above, with the six joints recorded in the preflight.

Not done, and the reason each one stopped:

1. Gate D collision meshes. `UsdFileCfg` did not put a rigid body on the `MeshConverter` root, so the script substituted cuboids. Next step: apply a rigid body on the prim `RigidObject` resolves, spawn that mesh, and drop the cuboid substitution before any grasp is claimed.
2. Gate D reach. Differential IK on `gripper_link` stopped about 0.35 m from the can. Next step: fix the fixed-base Jacobian column or the base placement until the gripper link reaches the can, then run the close, lift, transport, and release sequence to the 5 mm / 0.5 s settle rule. Only after that sequence passes should `isaac_scene_adapter.py` be added.
3. Gate D image. Headless GLFW failed, so there is no robot PNG. Next step: a viewport or an offscreen capture of the SO-ARM101 at the grasp pose.
4. Gate E. The writer is tested on a fixture only. Next step: record one episode from the successful SO-ARM101 states and render the initial, intermediate, and final poses in Mitsuba within 1 mm and 0.1 degree. LeRobot export stays out until that episode exists.
5. Expansion. Not started, because D and E did not pass.
6. Still unresolved, and not to be inferred: generated-mesh redistribution license; table leg proxy (119 mm); `reachable_from` left `not_evaluated`; Stokes film is not a polarization qualification; the constant light is relative; shape pose updates rebuild the Mitsuba scene. The two existing `tests/test_phase4.py` failures for the missing Phase 3 EXR were not touched. The full pytest suite was not re-run. Nothing from this slice was committed.

## Offline scene-state bridge

A later cycle connected the same table and can packages to Isaac Lab and Mitsuba without a robot. The report is `docs/task_scene_offline_bridge_report.md`. PhysX moved the can `0.0399` m. Mitsuba rebuilt `t0`, `t1`, and `t2` from `scene_state_snapshot_v1`. The composite USD replaced the cuboid fallback for that bridge. SO-ARM101 Gate D stays `blocked`. The grasp is the next implementation and was not run.

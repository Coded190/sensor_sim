# Offline Isaac Lab and Mitsuba bridge

The milestone is the transfer of one canonical scene and a real PhysX trajectory into Mitsuba. It is not a grasp, and it does not change the SO-ARM101 Gate D status. That gate stays `blocked`.

## Result

| Gate | Status | Evidence |
| --- | --- | --- |
| Asset library validation | passed | `tests/test_asset_library.py` against the live packages |
| Asset retrieval | passed | Table category returns `table_dining`. Can category returns `can`. |
| Shared connector | passed | `tests/test_task_scene_connector.py`. Scene-resolution hash `6632efb4a5cb8e4eac2f55416a4d561d47c4b7ae5e9df0dca3d6cf117180b02b`. |
| Isaac composite USD | passed | `/AssetRoot` with `Visual` and `Collision` children. Stage up axis Z, metersPerUnit 1. |
| Isaac initial scene | passed | Can translation error `9.72e-8` m, rotation error `9.30e-5` deg. |
| Mitsuba initial scene | passed | `t0.png` is the manifest pose rebuilt from the connector. |
| Static geometry parity | passed | Render-mesh extent error about `1e-8` m. Surface-distance max about `4e-8` m for the table and `2e-9` m for the can. Tolerance was 1 mm. |
| Camera parity | passed | Runtime intrinsics were exactly `fx = fy = 120`, `cx = 80`, `cy = 60`. Forward-point reprojection error is numerical noise. |
| Isaac physics | passed | The can moved `0.0399` m from `t0` to `t2` and stayed at z `0.845598` m. |
| State serialization | passed | Sequence SHA-256 `9ad0f4a0265ef34b590d5871ab5c52f92963173805d7c5e0c1f537be29af1d3e`. 31 samples at 60 Hz. |
| Mitsuba snapshot reconstruction | passed | `t0`, `t1`, and `t2` rendered. Stokes shape `(120, 160, 15)`. |
| Dynamic state parity | passed | Mitsuba received the saved matrices. The largest rotation residual was `9.4e-6` deg. |

Exit code 0 from Kit is not the pass. The JSON `status` fields are.

## Commands

Project tests, `PYTHONPATH` unset, interpreter `/home/cmoore/sensor_sim/.venv/bin/python`:

`python -m pytest tests/test_asset_library.py tests/test_task_scene_connector.py tests/test_scene_state.py tests/test_task_scene_composer.py tests/test_task_scene_contracts.py tests/test_episode_writer.py -q`

28 passed.

Isaac interpreter `/home/cmoore/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11`. `PYTHONPATH` was the Isaac Lab source tree, the Isaac site-packages, `isaac_so_arm101/src`, this repo's `src`, and `scripts/isaac`.

- `scripts/isaac/bridge_render_scene.py`
- `scripts/isaac/bridge_simulate_offline.py`

Mitsuba replay, project `.venv`, `PYTHONPATH` unset:

`python scripts/mitsuba/replay_isaac_states.py`

Versions: Isaac Lab extension `0.47.2` from the `isaaclab[all,isaacsim]==2.3.0` pin, Isaac Sim `5.1.0-rc.19+release.26219.9c81211b.gl`, Mitsuba `3.9.1`, variant `cuda_ad_spectral_polarized`. PhysX was imported. The schema package on disk is `omni.usd.schema.physx-107.3.26`. A separate PhysX version string was not exposed as `PHYSX_VERSION`.

Hashes:

- Manifest file SHA-256 `463d082c30e0e892a1842c696082799b28cbaeb02111cc9ee5e74c0d42e1db72`. The canonical JSON hash remains `826294842441006b1aa624b91328930bc37141aa64eb45657abd08228ebf4d3b`. The bridge uses the file bytes.
- Catalog file SHA-256 `81bbec185fba599da99709a0174741822fd61569312f25dc7798107fcaafe49e`.
- `table_dining` package `1d28aed9d1366d2f64f417abaff443d916b778c3960dea731f139f7114d42976`.
- `can` package `9d5419cae8b43dcf637233756a060c311df04aea59a46afb6077a4666b2dc9e2`.
- Run hash `cf9347d7129020054a6e1c0a139120eb72440c20155487ec2464699e98d95dd7`.

## What the pictures show

Isaac RGB, Mitsuba `t0`, and Mitsuba `t2`:

- `/home/cmoore/sensor_sim/dataset/task_scene/offline_bridge/463d082c30e0e892a1842c696082799b28cbaeb02111cc9ee5e74c0d42e1db72/cf9347d7129020054a6e1c0a139120eb72440c20155487ec2464699e98d95dd7/static/isaac_rgb.png`
- `.../mitsuba/t0.png`
- `.../mitsuba/t2.png`

The Mitsuba frames show the packaged table render mesh and the can. Between `t0` and `t2` the can moves across the top. That is the saved PhysX motion, not a second layout. Isaac's RGB is much darker. The same source textures are not the same pixels. RGB equality was not a gate.

The can's x position in the state file is `-0.0378` m at `t0`, `-0.0204` m at `t1` (0.10 s), and `0.0022` m at `t2` (0.40 s). z stays `0.845598` m, which is the support height. The initial velocity was `0.20` m/s along world +X and had fallen to about `0.0038` m/s by `t2`. Friction `0.05` and restitution `0` are test settings, not measured materials.

The can mass used in simulation is `0.002356` kg. It is the closed-mesh volume `5.890e-6` m³ times the simulation-default density `400` kg/m³. The bounding-box volume is `1.574e-4` m³, so this is the mesh's enclosed volume, not a measured solid-can mass.

## Limits that stay open

- Generated-mesh redistribution is `unresolved`. The packages are not fully qualified for manipulation.
- The table collision mesh is still the slab-plus-leg proxy. Legs differ from the render surface by up to `0.119` m. The visible mesh is the render OBJ. Extent agreement was about `1e-8` m.
- PhysX logged that the can's render mesh, which sits under the dynamic root, was cooked as a convex hull because a triangle mesh cannot be dynamic. The packaged collision child is also a convex hull. The render mesh was not fully excluded from cooking. The can did not fall through the table.
- GLFW reported that it could not open a display. The render product still wrote the RGB and a depth array. That warning is not a missing image.
- Stokes output is not a polarization qualification. The light is relative.
- `reachable_from` was not evaluated. No robot was spawned.
- The full project pytest was run with `PYTHONPATH` unset. The first run also inherited `LD_LIBRARY_PATH` from the Isaac USD libraries used earlier in this session. That run was 343 passed and 5 failed. Three of those failures were Blender loading `libusd_ms.so` and missing a MaterialX symbol. Rerunning those three tests with `LD_LIBRARY_PATH` unset passed. The other two failures are the existing `tests/test_phase4.py` cases for the missing `outputs/phase3/frames/000/stokes/render.exr`. That EXR was not created.

## Next implementation: SO-ARM101 grasp

This section does not run the robot and it does not mark Gate D passed. Gate D stays `blocked`.

The bridge now supplies the catalog, `task_scene_connector.py`, the Z-up asset-root convention, the composite USD hierarchy, separate render and collision meshes, `scene_state_snapshot_v1`, and a Mitsuba replay of object states. The camera and geometry numbers above are the tolerances a later robot replay should meet: 1 mm and 0.1°.

The grasp still has to prove all of this before it can pass:

- Use `SO_ARM101_CFG` and `/home/cmoore/moore/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf`. The six joints are `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, and `gripper`. Do not build the arm from cuboids.
- Reuse the composite USD. Do not return to the fitted cuboids in `scripts/isaac/gate_d_preflight.py`.
- Keep the five arm joints on differential IK and command the gripper separately. Finite-difference the Jacobian before changing the fixed-base `body_id - 1` rule. Convert targets into the robot base frame.
- Measure the TCP against `gripper_link` and `gripper_frame_link`. Do not assume either origin is the center of the finger pads.
- Pass a reach-only test at 5 mm and 3°, then both finger contacts, a physical lift of at least 50 mm, transport, release inside the manifest target, and a settle within 5 mm for 0.5 s. No teleport and no fixed joint on the can.
- Put joint positions, joint velocities, gripper state, and the TCP pose in a new schema version or a versioned extension. Do not change the meaning of `scene_state_snapshot_v1`.
- Replay that real trajectory in Mitsuba at the same 1 mm and 0.1° tolerances.

`isaac_scene_adapter.py`, LeRobot export, and real-time synchronization stay deferred until that replay exists.

# Offline bridge audit

This audit was read from the live tree before the catalog, connector, and Isaac bridge runners were added. Status words are `implemented_and_tested`, `implemented_not_integration_tested`, `partial`, `planned`, `unsupported`, or `blocked`.

## Baseline

- Branch `main`, tracking `origin/main`. HEAD `d1cc0c7f0edfb84ea4f0b6bf049639d8b6c7efee`.
- `git status --short --branch` at the start of this cycle showed the task-scene modules and reports as uncommitted. Those files are the previous task-scene prototype. This audit did not revert them.
- Project interpreter `/home/cmoore/sensor_sim/.venv/bin/python`, Python 3.12.3. Mitsuba 3.9.1. The default variant was `None` until a script calls `require_variant()`.
- Isaac interpreter `/home/cmoore/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11`. Isaac Lab source is `/home/cmoore/moore/isaac_so_arm101/.venv/lib/python3.11/site-packages/isaaclab/source/isaaclab`. The robot project pins `isaaclab[all,isaacsim]==2.3.0`. The installed extension.toml version is `0.47.2`. Isaac Sim `VERSION` is `5.1.0-rc.19+release.26219.9c81211b.gl`. The PhysX USD schema package on disk is `omni.usd.schema.physx-107.3.26+107.3.3`. A Kit launch was not required to read those files. The runtime PhysX version is recorded by the bridge scripts when they launch.
- Indoors commit `b11700ebac77695d8f53538705cfdc0cd890c54d`. Local patch `/home/cmoore/sensor_sim/dataset/qualification/asset_realism/reports/indoors_local_current.patch` SHA-256 `2730888215eddd934318b4298a1effd00a7e3ebe4ea4660fd30718d632e0fe31`.

## Package and scene hashes

`verify_package` recomputed these package hashes from the package bytes:

- `table_dining` `1.0.0`: `1d28aed9d1366d2f64f417abaff443d916b778c3960dea731f139f7114d42976`
- `can` `1.0.0`: `9d5419cae8b43dcf637233756a060c311df04aea59a46afb6077a4666b2dc9e2`

Sidecar file SHA-256:

- `/home/cmoore/sensor_sim/dataset/task_scene/library/table_dining/1.0.0.asset_extension.json`: `5d1835f66f82ee33a58963f1a75b4e88d1d3619027a85d8f3df00b7dd42e95ca`
- `/home/cmoore/sensor_sim/dataset/task_scene/library/can/1.0.0.asset_extension.json`: `ef9c8e4fbd2347f4705225c76287377059e5016efe8de449308b97242ffd129f`

`pick_can/scene_manifest.json` file SHA-256 `463d082c30e0e892a1842c696082799b28cbaeb02111cc9ee5e74c0d42e1db72`. The canonical JSON hash from `manifest_hash` is `826294842441006b1aa624b91328930bc37141aa64eb45657abd08228ebf4d3b`. The bridge uses the file-byte hash. The two values differ because the file is pretty-printed.

Gate C render `/home/cmoore/sensor_sim/dataset/task_scene/scenes/pick_can/render.png` SHA-256 `3f974fcd69444a6a2cdc961d5235e6c301894a2ac12e6355d96c5a47c33e4932`. Conversion report `mitsuba_conversion.json` SHA-256 `a363e17a0be80a6488e2fda69308f43e68da11a8e0ee52a9ec847d621513d403`.

The manifest instances already name `asset_id`, `asset_version`, and `package_sha256` for `table_0` / `table_dining` and `can_0` / `can`. A binding sidecar is not required for this scene. The connector still checks those fields against the catalog.

Camera `camera_0` is a pinhole, 160 by 120, `fx = fy = 120`, `cx = 80`, `cy = 60`. Pixels are square and the principal point is the image center. That is the case Isaac Lab 2.3.0 documents as representable. The bridge still reads the runtime intrinsics and stops if they disagree. It does not edit this camera.

## What already exists

| Piece | Status | Live evidence |
| --- | --- | --- |
| `asset_package_v1` and `verify_package` | implemented_and_tested | Both task packages verify. The digest is SHA-256 of canonical `{"files": [{path, sha256}]}`. |
| `asset_extension_v1` | implemented_and_tested | Sidecars match the package id, version, and hash. Render and collision paths differ. |
| `scene_manifest_v1` | implemented_and_tested | Rigid-transform checks and the Mitsuba pinhole roll are tested in `tests/test_task_scene_contracts.py`. |
| `task_scene_composer.py` | implemented_and_tested | Category selection exists, but the caller passes asset dicts. There is no catalog file. |
| `task_retrieve.py` | implemented_and_tested | It ranks text over a different catalog and can use an embedding backend. It is not the asset-package retrieval API. |
| `build_mitsuba_scene` | implemented_not_integration_tested | It reads `package_path/geometry/render.obj` from the manifest directly. Gate C rendered `pick_can` before this bridge. The connector is not in that path yet. |
| `canonical_to_usd` | implemented_and_tested | It maps canonical +Z to USD +Y. `MeshConverter` then builds a stage with `upAxis = Z` and `metersPerUnit = 1`. Using the Y-up helper as the Isaac simulation pose would change `world_from_asset`. |
| Gate D preflight | blocked | `scripts/isaac/gate_d_preflight.py` converts the collision OBJs, then spawns cuboids. The grasp did not happen. |
| Asset catalog | planned | No `catalog.json` at audit time. |
| Shared connector and `scene_state_snapshot_v1` | planned | Not in the tree at audit time. |
| SO-ARM101 pick-and-place | blocked | Not part of this cycle. |

## Prior USD files

These files still exist:

- `/tmp/IsaacLab/usd_20261121_100416_4606/collision.usd` (2011 bytes) and `config.yaml`
- `/tmp/IsaacLab/usd_20261121_100416_6387/collision.usd` (7075 bytes) and `config.yaml`

`pxr` from the Isaac Sim USD libs did not import in the uv Python 3.11 outside Kit (`libpython3.11.so.1.0` was not found). A full schema walk was not possible in this audit. The converter config beside each USD is readable and is the configuration that produced the file:

- Both set `usd_dir: null`, so the output stayed under `/tmp/IsaacLab`.
- Both set `mass_props: null` and `rigid_props: null`.
- The table mesh collision config names `PhysxTriangleMeshSimplificationCollisionAPI`.
- The can mesh collision config names `PhysxConvexHullCollisionAPI`.
- Translation, rotation, and scale are identity.

Printable tokens in the crate include `upAxis`, `Z`, `geometry`, and `Xform`. The table file's tokens include `triangle`. The can file's tokens include `convexHull`. A `RigidBody` token was not found in the table crate's short printable set. That is consistent with `rigid_props: null`, and it is not a substitute for a Kit stage walk. `gate_d_preflight.json` sets `rigid_body_on_converted_root` to false in Python without opening the stage. This audit does not treat that JSON field as a USD inspection.

The installed `MeshConverter._convert_asset` writes rigid-body and mass APIs on the root only when those configs are not `None`. Collision APIs go on the mesh prim. The comment in that function says the first asset-converter pass is Y-up and centimeters, then the stage is set to Z-up and meters. Identity translation, rotation, and scale do not by themselves prove the vertex positions stayed in the OBJ frame. The bridge compares USD bounds and surface samples with the source OBJ and stops if they disagree by more than 1 mm.

## Qualification limits copied from the live reports

Table render-to-collision max 0.002722 m. Collision-to-render max 0.118552 m. That second number is the leg proxy. Can render-to-collision max 0.001561 m. Both reports set `license_status` to `unresolved` and `distance_within_tolerance` to true under their own tolerances (tabletop 0.02 m, object 0.01 m). Neither report is a manipulation qualification. Density 400 kg/m³ is only in `configs/task_scene/isaac_gate_d_sim_defaults.yaml` and is labeled a default.

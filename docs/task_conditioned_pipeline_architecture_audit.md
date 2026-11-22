# Task-conditioned pipeline architecture audit

Recorded before the task-scene contracts were added. The public README is not used as evidence. This checkout is the authority.

## Repository

- Root: `/home/cmoore/sensor_sim`
- Branch: `main`, tracking `origin/main`, ahead/behind `+0 -0`
- HEAD: `d1cc0c7f0edfb84ea4f0b6bf049639d8b6c7efee`
- `git status --short --branch`: `## main...origin/main`
- `git status --porcelain=v2 --branch`: branch oid, head `main`, upstream `origin/main`, no changed entries
- Submodules: none (`git submodule status` printed nothing)
- Uncommitted work: none. Nothing was reset or cleaned.

Protected file SHA-256 at this HEAD, verified unchanged at the start of implementation:

- `src/sensor_sim/scene_gen/task_replay.py` `d9e6e1ac4103cb59c67d077977815632239ec94637365609fdf5c2f1fec1bcbf`
- `src/sensor_sim/scene_gen/tabletop_assets.py` `d47c24dff6b096a5295b3a31520fee1431b085966da3ae9cbee06a858706a990`
- `src/sensor_sim/scene_gen/interior_assets.py` `1c53b4bfb6223f0f6a17d862974bc937b3aa6f2e8c36f7bfc7cb760c8bc4e133`
- `src/sensor_sim/scene_gen/interior_scenes.py` `317fc479108e068b886846794d821ac96548afa846f9423f6bc6e6e24149e57f`
- `src/sensor_sim/scene_gen/semantic_composition.py` `6927712eeec7b616692c8a1a926a2195820aa404c2864a9371e240e7c8746043`
- `src/sensor_sim/scene_gen/room_schema.py` `626d19794ca73cebbd22f14e15754998a9e871294056dfd4e20651685efb4a88`
- `src/sensor_sim/scene_gen/room_generator.py` `75abb14ed8569ac17fc8d9232102e9d04c7c83b67518d01cb9f2094f8ea4b118`
- `src/sensor_sim/scene_gen/indoor_mitsuba.py` `dfbd5e9019be0d512f787ce2ec1ebffd78cf60cf52fff427d00a8a1e6c26d2b3`
- `src/sensor_sim/scene_gen/generate.py` `78ee05ab9afe551436b0f35836880e8c6b1680e8c559f642083269ff3931ddf8`
- `src/sensor_sim/scene_gen/indoor_cli.py` `e4828c9ff829b2bad8394ba5271e4a368b47cacd8d4c5f26d6e24074b9d3f10f`
- `assets/candidates/index.json` `081bf07291a4cc3f8f2af1fe27b8f8beb0164153100244aefa25df1ba209c157`

Generated outputs already on disk live under `dataset/qualification/asset_realism/source_runs/`. Those are full-room Indoors solves, not reusable individual asset packages. `assets/library/` does not exist.

The last full pytest log, `dataset/qualification/asset_realism/reports/pytest_full_livingroom.log`, says `2 failed, 314 passed`. Those two failures are the pre-existing missing Phase 3 EXR in `tests/test_phase4.py`. That log predates this audit and was not re-run as the full suite here. A focused pre-change run of `tests/test_asset_package.py`, `tests/test_task_scenes.py`, `tests/test_tabletop_layout.py`, `tests/test_infinigen_default.py`, and `tests/test_expansion_parallel.py` was started and is recorded in the implementation report if it finished.

## Asset package contract

`asset_package.py` schema version is integer `1`. `prepare_manifest` and `verify_package` reject any key outside `_MANIFEST_KEYS` with `reason=unknown_manifest_key`. Allowed keys are `schema_version`, `exclusion_policy_version`, `asset_id`, `version`, `package_sha256`, `provenance`, `files`, `geometry`, `materials`, `textures`, `collision`, `support_surfaces`, `contact_regions`, `components`, `transform`, `units`, and `frame`.

`package_digest` is the SHA-256 of canonical JSON `{"files": [{"path", "sha256"}, ...]}` sorted by path, and that list includes `asset.json`'s own body hash. `verify_package` recomputes file hashes from bytes, rejects symlinks through `sha256_regular`, rejects absolute paths and `..` in `normalize_payload_path`, and rejects any file inside the package that is not listed and not excluded. Exclusions are `admission_report.json`, names ending in `.blend1`, `.log`, or `.tmp`, and top-level `logs`, `cache`, `caches`, `tmp`, and `temp`.

A sidecar inside the package directory would be `unexpected_file` unless it is listed in `asset.json`. The new extension therefore lives outside the package directory and stores the package hash. Collision is an allowed manifest group, but nothing in this module checks that a collision mesh differs from the render mesh. `frame`, `support_surfaces`, and `contact_regions` are stored and passed through `reject_nonfinite`. They are not given the semantic-frame rules this pipeline needs.

Tests: `tests/test_asset_package.py` covers canonical bytes, path rejection, symlinks, and hash mismatch. Status: `implemented_and_tested` for the v1 package. Status of a separate extension: `planned` at audit time.

## Module signatures that the composer may call

Imports of these modules succeed. Signatures below are from the source, not from comments.

| Function | Lines | Signature | What it does | Tests | Status |
| --- | --- | --- | --- | --- | --- |
| `task_retrieve.retrieve` | `task_retrieve.py` 110 | `(catalog: Path, query: str, *, top_k: int = 10, require_renderer: bool = False) -> dict` | TF-IDF rank over catalog manifests. Rejects a bad package inside `_reject_reason`. | `tests/test_task_scenes.py::test_retrieval_filters_and_tie_break` | `implemented_and_tested` for catalog text. No Infinigen asset library exists to retrieve. |
| `tabletop_layout.place_tabletop` | `tabletop_layout.py` 40 | `(catalog, selections, support_id, support_surface, *, seed, parent_pose, minimum_separation_m=0.02, edge_margin_m=0.03, clutter=0.0) -> dict` | Places selections on one support polygon from the asset package. | `tests/test_tabletop_layout.py` | `implemented_and_tested` on project tabletop assets, not on Infinigen packages. |
| `task_layout.resolve` | `task_layout.py` 34 | `(catalog: Path, selections: list[dict], relations: list[dict], *, seed: int) -> dict` | Poses and evaluates relations. | `tests/test_task_scenes.py` | `partial` |
| `indoor_contact.measure_support` | `indoor_contact.py` 38 | `(vertices, faces, polygon, *, gap_limit_m=MAX_SUPPORT_GAP_M, penetration_limit_m=MAX_PENETRATION_M) -> dict` | Gap, penetration, and polygon containment. | Used by `task_layout` tests and Infinigen support reports | `implemented_and_tested` |
| `placement_separation.separation_report` | `placement_separation.py` 124 | `(loaded, placements, *, support_pairs, minimum_m=0.0) -> dict` | Triangle intersection after an AABB test. | `tests/test_placement_separation.py` | `implemented_and_tested` |
| `infinigen_backend.run_generation` | `infinigen_backend.py` 88 | `(room, seed, output, *, fast, root=None, python=None) -> dict` | Runs `generate_indoors` coarse for a room name. | `tests/test_infinigen_default.py` routes without executing a solve | `implemented_not_qualified` as an individual-asset generator. It is a room solver. |
| `mitsuba_scene_package.build_scene` | `mitsuba_scene_package.py` 62 | `(manifest: dict, root: Path) -> dict` | Builds a Mitsuba dict from an Infinigen scene manifest. | `tests/test_infinigen_package.py::test_replay_rebuilds_a_frozen_scene` | `implemented_and_tested` for that manifest shape. It does not load `scene_manifest_v1`. |
| `mitsuba_scene_package.replay_infinigen_scene` | `mitsuba_scene_package.py` 209 | `(manifest_path, png, *, spp=4, seed=1, width=None, height=None) -> dict` | Construction report, then two renders. | same test | `implemented_and_tested` for the Infinigen package |

`task_layout._evaluate` (`task_layout.py` 107) measures `supported_on`, `inside`, `near`, `adjacent_to`, and `keep_clear_of`. `reachable_from` returns `passed=False`, `reason=robot_required`. `attached_to`, `aligned_with`, and `facing` return `passed=False`, `reason=unsupported_numeric_relation`. Any other type returns `unsupported_relation`. Those three unsupported relations are not treated as solved.

`indoor_generate.generate` accepts `kind=asset` for `sofa`, `chair`, `table`, `lamp`, and `cabinet`, then calls `run_generation`, which only builds a room coarse command. That is not individual asset generation. Status: `partial`.

`task_schema.py` validates `schema_version` integer `1` task documents and bundles. It is the older task bundle, not `task_scene_request_v1`. Status: `implemented_and_tested` for its own schema, `unsupported` as the new request.

No Isaac scene builder and no episode writer exist in `sensor_sim`. Both are `planned`.

## Composer reuse map

| Responsibility | Existing implementation | Reusable function | Missing work | Evidence |
| --- | --- | --- | --- | --- |
| Asset package resolution | v1 verify and hash | `asset_package.verify_package` | Sidecar that names the same hash | `tests/test_asset_package.py` |
| Asset metadata extension | `frame` and `support_surfaces` are free-form v1 fields | none for `asset_extension_v1` | New sidecar outside the package | No extension module |
| Task request validation | Integer schema task bundle | `task_schema.validate_task` does not accept the new request | `task_scene_request_v1` | `tests/test_task_scenes.py` |
| Object placement | Tabletop placement on a package support polygon | `tabletop_layout.place_tabletop` when the package already has that polygon | Placement from explicit semantic frames for the first Infinigen pair | `tests/test_tabletop_layout.py` |
| Support and contact | Gap and polygon test | `indoor_contact.measure_support` | Report every check with units and tolerances | `task_layout` and contact tests |
| Scene manifest | Infinigen manifest `infinigen_mitsuba_scene_v1` | `mitsuba_scene_package.load_manifest` for that file only | `scene_manifest_v1` and transform conversions | LivingRoom 21 manifest on disk |
| Mitsuba scene construction | Infinigen loader | `build_scene`, `replay_infinigen_scene` | Loader for the canonical manifest | `tests/test_infinigen_package.py` |
| Isaac scene construction | External lift scene uses a Nucleus table USD | none in `sensor_sim` | Feasibility run, then an adapter | `isaac_so_arm101` lift config |
| Episode writing | LeRobot `add_frame` / `save_episode` in another repo | none in `sensor_sim` | Versioned writer after the dataset API is used as-is | LeRobot source, not a written episode |

`tabletop_layout.place_tabletop` is the right function when a package already exposes the support polygon its tests use. The first Infinigen table does not have that package shape until Gate A builds it. The composer will place the first object by the semantic frames and call `measure_support` and `separation_report` for the numeric checks, instead of calling every layout module.

## Infinigen Indoors

- Checkout: `/home/cmoore/infinigen_src/indoors-stable`
- Commit: `b11700ebac77695d8f53538705cfdc0cd890c54d`
- Local modifications: `material_assignments.py`, `part_util.py`, `nature_shelf_trinkets/generate.py`, `populate.py`, `decorate.py`, `terrain/__init__.py`, plus an untracked `dataset/` directory
- Patch hashes: `indoors_local.patch` `2f98aa479d074cbfa508e51f92f7d725de5b2d8e0ee97780feb8a635381297e9`; `indoors_local_current.patch` `2730888215eddd934318b4298a1effd00a7e3ebe4ea4660fd30718d632e0fe31`; `indoors_utility_floor.patch` `8748e467579e8fba4b59132676c9ff1f1003592fbb1921101b5c2699ae659950`
- Interpreter: `/home/cmoore/infinigen_envs/indoors/bin/python`
- Individual CLI, from `--help` in that interpreter: `python -m infinigen_examples.generate_individual_assets`. Flags used later are only ones this help text lists: `-o`, `-f`, `-n`, `-D`, `-s`, `--export {fbx,obj,usdc,usda,stl,ply}`, `--export_texture_res` default 1024, `-r {image,video,none}`, `-G`.
- `TableDiningFactory` is `infinigen/assets/objects/tables/dining_table.py` line 193. `CanFactory` is `infinigen/assets/objects/tableware/can.py` line 26. The can samples width `log_uniform(0.05, 0.1)` m and height `0.5` to `2.5` times that width. That size is why it is the first grasp candidate. It is not a demonstrated grasp.
- Upstream code license at this checkout is BSD-3-Clause (`LICENSE`). Generated mesh redistribution is `unresolved`. The code license is not a license for the baked meshes.
- The exporter bakes textures at a finite resolution. The `.blend` remains the source evidence. The OBJ/MTL is what later adapters load.

## Mitsuba

Project environment `/home/cmoore/sensor_sim/.venv`. `mitsuba` version `3.9.1`. Variant `cuda_ad_spectral_polarized` is installed. Existing phase outputs and the two missing-EXR test failures stay untouched.

## Isaac Lab and SO-ARM101

Project: `/home/cmoore/moore/isaac_so_arm101`, commit `69104fe44ff0e40dfdee777b9f32888fd87a9f52`. `pyproject.toml` pins `isaaclab[all,isaacsim]==2.3.0`. This audit did not install or upgrade it and did not edit the project.

The venv launcher is broken. `.venv/bin/python` is a symlink to `/home/cirplab/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11`, and that path is not on this machine. `.venv/bin/isaaclab` has that same missing shebang. Import of the installed package did succeed with `/home/cmoore/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11` and `sys.path` pointed at the venv `site-packages`. Torch in that import is `2.7.0+cu128`. A simulation launch was not run. Import is not a launch. Status of launch: `unsupported` until Gate D uses a working interpreter.

Robot source: `src/isaac_so_arm101/robots/trs_so101/so_arm101.py` `SO_ARM101_CFG`, URDF `urdf/so_arm101.urdf`. Revolute joints and limits in radians:

- `shoulder_pan` -1.91986 to 1.91986
- `shoulder_lift` -1.74533 to 1.74533
- `elbow_flex` -1.69 to 1.69
- `wrist_flex` -1.65806 to 1.65806
- `wrist_roll` -2.74385 to 2.84121
- `gripper` -0.174533 to 1.74533, parent `gripper_link`, child `moving_jaw_so101_v1_link`

The gripper is one revolute joint, not a coupled pair. End-effector body in the config is the gripper link used by the existing lift task. `tasks/lift/lift_env_cfg.py` spawns a Nucleus Seattle table USD and a `RigidObjectCfg`. That scene is not the canonical manifest. `SO_ARM101_CFG` does not contain an IK solver or a grasp controller. The lift task is a manager-based RL config, not a scripted physics grasp.

## LeRobot

Checkout `/home/cmoore/moore/lerobot`, commit `9bc2df80bb2be79553e88946a3bfd1795f97726f`, describe `v0.5.1-24-g9bc2df80`. `CODEBASE_VERSION` in `src/lerobot/datasets/dataset_metadata.py` is `v3.0`.

`LeRobotDataset.add_frame(frame)` requires a `task` key and delegates to `DatasetWriter.add_frame`. `save_episode(episode_data=None, parallel_encoding=True)` encodes video, writes parquet, and updates metadata. `finalize()` flushes writers. Readback is `LeRobotDataset` in read mode. `sensor_sim` does not call this API. Dataset v1 in this project is a different store and will not be modified.

Episode storage is not chosen in this audit beyond this interface. Gate E will use a versioned internal record and will call LeRobot only if the recorded fields fit `add_frame` without inventing a second copy of its format.

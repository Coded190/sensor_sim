# Semantic scene composition

Branch `main`, HEAD `0609c61124005144060a8f9eb673d73510b57ff5`. No commit was created. The six scenes are qualification candidates. `visually_reviewed`, `admitted`, and `registered` were not set. `assets/candidates/index.json` was not modified. Nothing was trained, and no large dataset was generated.

The working tree was already dirty. `git status --short` still shows the same pre-existing edits to `.gitignore`, `README.md`, the film configs, `pyproject.toml`, eight film modules, `tests/test_heldout.py`, and `uv.lock`. `git diff --stat` is still 15 files, 933 insertions, 757 deletions. Those files were not edited for this milestone. `src/sensor_sim/scene_gen/` remains untracked.

Pytest before this milestone, after room diversity: 214 passed, 2 failed, in 190.04 s. The new file `tests/test_semantic_scenes.py` passed 10 tests in 2.97 s. Full pytest after this milestone: 224 passed, 2 failed, in 191.48 s. Both failures are `tests/test_phase4.py` raising `FileStream` while opening `outputs/phase3/frames/000/stokes/render.exr` from `src/sensor_sim/phase4.py:128`. That pair is unchanged. The missing Phase 3 EXR was not repaired.

## What was built

Six curated scenes, one render each, on the saved room-diversity shells. The shells were loaded. They were not rebuilt.

| Scene | Seed | Room | Run |
| --- | --- | --- | --- |
| Kitchen | 17401 | `room_standard` | `13566dca561f6af1` |
| Dining | 17402 | `room_spacious` | `424d03bd7f75e505` |
| Desk | 17403 | `room_compact` | `d2911facd429bc7f` |
| Workbench low | 17411 | `room_spacious` | `9f6ace8c60824574` |
| Workbench moderate | 17412 | `room_spacious` | `3a8e490512d174ed` |
| Workbench high | 17413 | `room_spacious` | `89e0e3a99524cf51` |

The renderer seed is 17421. Samples per pixel are 16. The variant is `cuda_ad_spectral_polarized`. A second placement seed, 17499, is used only by the unit test. It is not a seventh render.

Room spec hashes, first 16 hex characters: compact `852d97c515062c06`, standard `d724c849213f4e8d`, spacious `986232c566b2ccaf`. The spacious east wall has no opening. The workbench uses that wall.

## Frozen furniture

Sizes were frozen before the first package hash. A later zone-role correction is a new version. The 1.0.0 bytes were not edited.

| Asset | Version in use | Size m | Package sha256 |
| --- | --- | --- | --- |
| `furniture_workbench` | 1.0.1 | 1.80 × 0.80 × 0.90 | `49f1f30f0a7ff24f1475d781129d008baf322bbbf5ced9d9fbeb85f977ba1cdd` |
| `furniture_counter` | 1.0.1 | 1.60 × 0.60 × 0.90 | `d9a520421f6eeb46dbc1d8a39d0da44dfcb47ee23f84363434b4653c0d3d88e7` |
| `furniture_desk` | 1.0.0 | 1.40 × 0.70 × 0.75 | `0cbf55db92136b9cd523e432ac3d8813af1b31c85d72750ea28d12fcfee5ae0d` |

`furniture_workbench` 1.0.0 remains `ff20a9d6dd59df07ad91fc0b6347386f96c61067b683ffc63aa86ca6e918b749`. `furniture_counter` 1.0.0 remains `58c5473ffafb44ec5593848bd8552c6a6d243214ecae5c28c7c53afe031108e1`. The 1.0.1 `render.obj` bytes equal the 1.0.0 bytes. The 1.0.1 manifests only widen a zone role list:

- Workbench `small_parts` now allows `material`, because the wire spool is a material stored with the trays.
- Counter `storage` now allows `manipulation_target`, because the pot sits in that zone.

The workbench has four legs, one top, and one fixed lower shelf. The shelf is inset to `[0.08, 0.08, 0.28]`–`[1.72, 0.72, 0.31]`. Each leg is a 0.06 m square at a corner and stops at the underside of the top. The shelf does not intersect the legs. The top support polygon is the 1.80 × 0.80 m rectangle at z = 0.90 m. The legs and the shelf are not part of that occupancy.

World bounds after placement:

| Instance | AABB min m | AABB max m | Floor gap | Room-facing gap |
| --- | --- | --- | --- | --- |
| `workbench_1` | [4.65, 1.50, 0] | [5.45, 3.30, 0.90] | 0 | 4.65 m to the west wall |
| `counter_1` | [0.05, 1.10, 0] | [0.65, 2.70, 0.90] | 0 | 3.55 m to the east wall |
| `desk_1` | [2.45, 1.10, 0] | [3.15, 2.50, 0.75] | 0 | 2.45 m to the west wall |
| `table_1` | [2.15, 2.05, 0] | [3.35, 2.75, 0.74] | 0 | centered |
| dining `chair_1` | [2.54, 1.53, 0] | [2.96, 1.93, 0.88] | 0 | faces the table from the south |
| dining `chair_2` | [2.54, 2.87, 0] | [2.96, 3.27, 0.88] | 0 | faces the table from the north |
| desk `chair_1` | [2.59, 0.58, 0] | [3.01, 0.98, 0.88] | 0 | sits south of the desk |

The workbench long axis runs along Y, parallel to the east wall, with the 0.05 m wall gap on the east face. Window keepout clearance, using the room-diversity 0.20 m keepout, is 2.427 m to the north window and 4.45 m to the west window. The door corridor does not overlap the bench. The counter is on the west wall of the standard room. Its north-window clearance is 0.90 m and its east-window clearance is 3.351 m. The desk is clear of the compact south door and 0.731 m clear of the north-window keepout. The desk chair is in front of the desk and outside the door corridor in X.

## Zones

Rectangles are in the support surface's local frame, in meters, as `[x0, y0, x1, y1]`. Each object has one `zone_id`. `inside` is not used. The hollow assets have interior samples that fall outside every component box, but no object is posed inside another mesh.

Workbench, on the top only:

| Zone | Rectangle | Allowed roles | Low | Moderate adds | High adds |
| --- | --- | --- | --- | --- | --- |
| `assembly` | [0.05, 0.05, 0.70, 0.42] | tool, manipulation target, material | screwdriver, hammer |  | screwdriver 2, pliers 2 |
| `tools` | [0.74, 0.05, 1.24, 0.42] | tool | pliers, wrench | clamp, hex-key group |  |
| `staging` | [0.05, 0.46, 0.70, 0.75] | material, tool | wood offcut, bracket | cable | component block |
| `small_parts` | [1.28, 0.05, 1.75, 0.40] | container, tool, context, material | fastener tray, wire spool | tape roll | parts bin |
| `free_workspace` | [1.28, 0.46, 1.75, 0.75] | none | empty | empty | empty |

`free_workspace` is 0.47 × 0.29 m and contains no instance. Object bounds were checked against their own zone, and the zones do not overlap, so the free rectangle is empty measured geometry.

Kitchen, on the counter: `preparation` [0.05, 0.05, 0.70, 0.55] holds the board, spatula, plate, and spoon; `ingredients` [0.75, 0.05, 1.20, 0.55] holds the apple and bowl; `storage` [1.25, 0.05, 1.55, 0.55] holds the bottle and pot.

Dining, on `table_rect`: `place_a` and `place_b` each hold a plate, a bowl, and a spoon; `place_a` also holds the fork; `serving` holds the dish and folded cloth; `drinks` holds the bottle and cup. Two chairs face the table.

Desk: `main_work` holds the notebook and cable group; `input_devices` holds the keyboard and phone; `writing` holds the pen and marker.

## Clutter

Counts exclude the furniture. Union fraction is the 4 mm XY raster of support-frame triangles, counting overlaps once, divided by the support-polygon area. For the workbench that area is 1.80 × 0.80 = 1.44 m². Height does not add area. Footprint-area sum is the sum of per-object XY axis-aligned boxes in that same frame, so it can exceed the union. Free area is support area minus union area. The free-rectangle value is `[0.25, 0.25]` when any empty 0.25 × 0.25 m window exists on that raster, and `[0, 0]` otherwise. It is not the measured size of a larger empty region. Yaw standard deviation is the standard deviation of the placed yaw samples. Nearest clearance is the minimum mesh clearance among object pairs. Repeat count is the number of extra copies beyond one per asset.

| Level | Count | Unique assets | Union m² | Union fraction | Footprint sum m² | Free area m² | Yaw std rad | Nearest m | Repeats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low | 8 | 8 | 0.057104 | 0.039656 | 0.080131 | 1.382896 | 0.0462 | 0.0469 | none |
| moderate | 12 | 12 | 0.069776 | 0.048456 | 0.136765 | 1.370224 | 0.2014 | 0.0326 | none |
| high | 16 | 14 | 0.094080 | 0.065333 | 0.195708 | 1.345920 | 0.3166 | 0.0264 | pliers, screwdriver |

The fractions rise in the same order as the counts. They are small because the tools are small relative to 1.44 m². No extra workpiece was added to inflate them. Separation stays above 0.004 m. The high free-rectangle check passes. Yaw is drawn from a uniform range of ±0.08, ±0.30, and ±0.55 rad for low, moderate, and high. Hammer and pliers 2 keep yaw 0 so the pair stays in its slots. A sample that leaves its zone is retried at yaw 0.

Kitchen union fraction is 0.127883 on the 0.96 m² counter. Dining is 0.155848 on the 0.84 m² table. Desk is 0.082351 on the 0.98 m² desktop. Those three are moderate recipes, so scene type is not mixed with the workbench clutter ladder.

## Visibility

`visible_to_projected_ratio` is the close-camera pixel count divided by the pixel count of that same mesh rendered alone. The close camera is 1600 × 900. A ratio below 1 with a nonzero count is partial occlusion. The catalog `visible_pixels` value is the largest close-view count among the scenes that contain the asset.

On the high workbench the partial objects are `prop_pliers__1` at 0.481 (535 pixels), `prop_wire_spool__1` at 0.507 (718 pixels), and `prop_metal_bracket__1` at 0.949 (1480 pixels). The repeated tools are the second screwdriver and the second pliers.

`image_support_flag` is true when the shorter side of the visible pixel box is at least 32 px. Below 32 px the flag is false and the role is unchanged. On the high bench the screwdriver, wrench, spool, hex keys, cable, tape, and second pliers are under 32 px and remain tool, material, or context as designed. The fork, pen, marker, spoon, and cable group are context or tool by design and are also under 32 px in their close views.

## Camera, light, and renders

The close camera uses the same `_fit_origin` projection as the table camera. For a support whose long axis is Y, the camera stands on the room side of that axis instead of looking down its length. The clip limits are that room's interior, not the old 3.7 m table-room clip. Every close view has clip shift 0 and peak NDC 0.9259, which is `1 / 1.08`, the 8% margin. The three workbench levels share one origin, `[2.8024, 2.40, 1.884]`, one target, one ceiling light, and one renderer.

Each room has two unpolarized, uncalibrated area lights. The ceiling panel is inset 0.20 m from the walls, so it is 2.80 × 2.60 m in the compact room, 3.80 × 3.40 m in the standard room, and 5.10 × 4.40 m in the spacious room. It sits 0.08 m below that room’s ceiling and faces down. Its radiance is 2.2. A rectangle outside each existing window shines inward at radiance 5. There is no camera-side light. The earlier 0.80 × 0.50 m panel at radiance 40 lit only a patch of floor. The larger panel plus the window light is what the current renders use. The ceiling underside stays darker than the floor because the panel faces down. The openings have no glass.

The kitchen overview from the southwest corner missed the west counter, so that one overview was moved to `[3.65, 0.55, 1.50]`, still inside the room, with the counter and the east window in frame. The other overviews stay at `[0.45, 0.55, 1.50]`.

Each scene has `room.png`, `tabletop.png`, a Stokes EXR for each, `instance_ids.npy`, and `instances.json` under `render_ceiling_v1`. All 12 EXRs were reopened. Each is float32, finite, and has the 15 `CHANNEL_NAMES` in that order. The close-view mask shape matches the EXR. A valid EXR is the file contract. It is not a Fresnel proof. Stokes values were not clamped.

Before those renders, `diffuse`, `pplastic`, `conductor`, and `roughconductor` were loaded on `cuda_ad_spectral_polarized`. A plain `plastic` material raises. New packages use only those four types. The reused cup keeps its existing dielectric BSDF. Wood, paper, and cardboard assets are diffuse and labeled depolarizing references.

## Packages and replay

New packages live only in `dataset/qualification/semantic_scenes/catalog/`. Status is only in `semantic_catalog.json`. All 26 catalog entries are `renderer_validated`. Each one has `verify_package`, a passing geometry check, a close view with nonzero pixels, and an evidence EXR. No asset needed a standalone render. The screwdriver, pen, and spatula are laid on their side by a placement rotation. That does not change the package bytes.

Reused packages, unchanged, include `table_rect` `af388a7ce55f371fd572e9e82226c017a8f46502e6e787161fe253e4db17e135`, `prop_chair` `333c26c0adbdcc7485788eb7c07df301b78aa8ebe870c4f1102c3fd42e5c1350`, and the plate, bowl, spoon, bottle, apple, cup, phone, marker, and tape roll already in the candidate index.

Replay loads the saved recipe and bundle, checks the recipe hash and the room and semantic package hashes, and requires the rebuilt `scene.json` bytes to match. It does not mesh, retrieve, select furniture, recompute zones, repack, or move objects. It does not call task replay or room replay, and it does not modify the source. The same placement seed repeats the object matrices. Seed 17499 changes them and still passes support, separation, and the free-rectangle check.

## Protected inputs

These hashes are unchanged:

- `sfp_scene.py` `3287fc531e50069d3fb912292f3a3bc907fa1e80a39d1c1a30398a8940ea3ea0`
- `mitsuba_scene.py` `6bdd40ac8ed12a243b0693da78c87219f3c5d51d66466b8772cd42c89d9b0653`
- `scenes.py` `752e7c5c7aa067eaaa77559b307066a8c57d72a481bd8853c1a0194d80aa0022`
- `xml_patch.py` `6b8ced6842897aa462eb1d855d4f7fe8cf8702e9f796af5bebe497d4b3942e32`
- `generate.py` `78ee05ab9afe551436b0f35836880e8c6b1680e8c559f642083269ff3931ddf8`
- `cli.py` `29a5ddc431bc4fa13887afed2fab774f485331d7aa5e97d0732b7d42a806763f`
- `meshcheck.py` `051676b437f4ae4394d201b8470cf179ddeebbbb3d25f3b96479f3fbeb684198`
- `placement.py` `650e9b36e9fb4422877b4cb392b9d7253c258cc1385329be502661a9d6ca466d`
- `indoor_mitsuba.py` `dfbd5e9019be0d512f787ce2ec1ebffd78cf60cf52fff427d00a8a1e6c26d2b3`
- `structural.py` `6e839ad9debb3bcbc4b16c1016f1eac77b6af58e8f1dc93b49c5302e007db1a7`
- `room_generator.py` `75abb14ed8569ac17fc8d9232102e9d04c7c83b67518d01cb9f2094f8ea4b118`
- `room_schema.py` `626d19794ca73cebbd22f14e15754998a9e871294056dfd4e20651685efb4a88`
- `task_replay.py` `d9e6e1ac4103cb59c67d077977815632239ec94637365609fdf5c2f1fec1bcbf`
- `tabletop_assets.py` `d47c24dff6b096a5295b3a31520fee1431b085966da3ae9cbee06a858706a990`
- candidate index `081bf07291a4cc3f8f2af1fe27b8f8beb0164153100244aefa25df1ba209c157`

Dataset v1 still has 1533 files and 457265977 bytes. Its `run_manifest.json` hash is `9475a86ae204d9b2840b23e558f8043272ad8c6f6e689f00541688879b08c196`. The six task scenes, Scenes A–C, the 15 tabletop bundles, the three scene-quality bundles (`ed940a5f855d1588`, `b05e275699b4654c`, `2aa23b16d561407a`), and the six room bundles and 18 room packages match the hashes locked in `tests/test_semantic_scenes.py`.

## Limits

These are curated recipes. The same recipe and seed rebuild the same transforms. Arbitrary text does not invent assets. This is not language understanding, dynamics, or robot manipulation. Objects were not scaled, and the 0.004 m separation was not relaxed. A failed slot returns `failure.json` and does not publish a bundle. The gallery is `dataset/qualification/semantic_scenes/review/index.html`. Human visual review, admission, and registration are pending.

```bash
xdg-open dataset/qualification/semantic_scenes/review/index.html
```

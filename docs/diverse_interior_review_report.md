# Diverse interior visual acceptance

Six furnished rooms were generated for visual inspection. RGB is the review product. Nothing in this suite is marked `visually_reviewed`, `admitted`, or `registered`.

Open the portable copy on a local machine:

`local_review/diverse_interior_review.zip`

Unzip it and open `diverse_interior_review/index.html`. Each room starts with the primary interior overview, then the secondary view, then the task close-up. The links are relative. The zip contains that HTML file and 18 PNGs. It has no EXR, Stokes, mask, or npy files.

The same gallery on this machine is `dataset/qualification/diverse_interior_review/review/index.html`.

## 1. Repository state

- Branch `main`
- HEAD `c86aa702fa1c1810e36f0688cc74752afa085410`
- That commit already contains the scene-quality closure. It was not reverted.
- This milestone is uncommitted. `git status --short` shows only the three new tracked files below. `dataset/` is gitignored, so the rendered suite does not appear in git status.

## 2. Scenes, seeds, and room size

Interior size is width × depth × height in meters. The finished floor is z = 0. The door is on the south wall in every room.

| Scene | Run | Seed | Interior (m) | Change from the sketch |
| --- | --- | --- | --- | --- |
| kitchen | `273e0498be80e064` | 17511 | 5.4 × 4.6 × 2.6 | Depth is 4.6 m so the west cabinet run, oven, and north-window keepout fit. The sketch depth was 4.0 m. |
| dining | `1696312e0090dbee` | 17512 | 4.6 × 3.6 × 2.7 | Matches the sketch. |
| living | `3b89459abb2c71ad` | 17513 | 6.2 × 4.8 × 2.8 | Matches the sketch. |
| bedroom | `8b36c966eae82e55` | 17514 | 4.4 × 3.8 × 2.5 | Increased from 3.8 × 3.4 × 2.5 m so the bed, two nightstands, and wardrobe fit. Furniture was not scaled. |
| office | `8b4ba8d12669c9b9` | 17515 | 3.4 × 3.0 × 2.5 | Matches the sketch. |
| workshop | `f79b07e012bd7b11` | 17516 | 5.0 × 4.0 × 2.8 | Depth is 4.0 m so the reused 1.0.2 workbench clears the north-window keepout. The sketch depth was 3.6 m. |

The six sizes are all different. Floor, wall, and ceiling colors are set per room on the room specification before `package_profile`. Geometry still comes from `assemble`.

## 3. Inventories

Objects that sit on another surface are stored with role `object`, including the sink, cooktop, microwave, and lamps. The asset ids are the inventory. Counts below are those roles, not a judgment of whether an appliance is furniture.

Kitchen, seed 17511. Role counts: 7 furniture, 11 objects.

- Cabinets: `cabinet_a`, `cabinet_b`, `upper_a`, `upper_b`
- Appliances: `sink_1`, `oven_1`, `cooktop_1` on the oven, `microwave_1`, `fridge_1`
- Work surface: `island_1` with `lamp_island`, `plate_1`, `board_1`, `apple_1`, `spoon_1`, `fork_1`
- Cookware: `pot_1` and `spatula_1` on the cooktop

The oven package has separate `door` and `panel` components. The refrigerator package has `door_left`, `door_right`, `handle_left`, and `handle_right`. Those are separate mesh components, not a single box.

Dining, seed 17512. Role counts: 6 furniture, 9 objects.

- `table_1`, chairs `chair_1` through `chair_4`, `sideboard_1`, `lamp_side`
- Tableware: two plates, two bowls, `cup_1`, `bottle_1`, `fork_1`, `dish_1`

Living, seed 17513. Role counts: 6 furniture, 4 objects.

- `sofa_1`, `armchair_1`, `coffee_1`, `media_1`, `tv_1`, `shelf_1`, `lamp_1`
- On the coffee table: `book_1`, `cup_1`, `phone_1`

Bedroom, seed 17514. Role counts: 4 furniture, 3 objects.

- `bed_1` (frame, mattress, headboard, two pillows, blanket), `stand_a`, `stand_b`, `wardrobe_1`, `lamp_bed`
- `book_1` and `phone_1` on the second nightstand
- No clothing asset exists in the candidate or semantic catalogs, so none was invented.

Office, seed 17515. Role counts: 3 furniture, 7 objects.

- `desk_1`, `office_chair_1`, `shelf_1`, `monitor_1`, `keyboard_1`, `mouse_1`, `notebook_1`, `pen_1`, `phone_1`, `lamp_desk`
- The bookcase is the storage piece. There is no separate filing cabinet.

Workshop, seed 17516. Role counts: 1 furniture, 20 objects.

- Reused `furniture_workbench` 1.0.2, not a new mesh
- `lamp_bench` plus the high-clutter workbench set: work panel, fixture block, stock plate, two screwdrivers, two pliers, wrench, hammer, clamp, hex keys, cable, fastener tray, wire spool, wood offcut, metal bracket, parts bin, component block, and tape
- The right-hand free column of the 1.0.2 bench is the open work area

## 4. Gallery and report paths

- Gallery: `dataset/qualification/diverse_interior_review/review/index.html`
- Zip: `local_review/diverse_interior_review.zip` (34,921,660 bytes)
- This report: `docs/diverse_interior_review_report.md`
- Scene bundles: `dataset/qualification/diverse_interior_review/<scene>/<run>/`

## 5. RGB renders

Every review image is 1280 × 720. The renderer is `cuda_ad_spectral_polarized`, 16 samples per pixel, seed 17501. `write_rgb_png` writes the film channels R, G, and B with sRGB gamma. There is no additional exposure control and no new tone mapper.

Linear statistics below are measured on those three channels before the sRGB conversion. For each pixel the value is the mean of R, G, and B. `median` is the median of that image. `p99` is the 99th percentile. `dark` is the fraction of pixels whose mean is below 0.02. These are not a weighted luminance and they are not the PNG values.

| Scene | View | PNG | Median | p99 | Dark fraction |
| --- | --- | --- | --- | --- | --- |
| kitchen | primary | `.../kitchen/273e0498be80e064/render_rgb_v1/primary.png` | 0.2406 | 1.1479 | 0.160 |
| kitchen | secondary | `.../secondary.png` | 0.2555 | 1.0358 | 0.109 |
| kitchen | task | `.../task.png` | 0.2278 | 1.1148 | 0.225 |
| dining | primary | `.../dining/1696312e0090dbee/render_rgb_v1/primary.png` | 0.1913 | 0.9426 | 0.126 |
| dining | secondary | `.../secondary.png` | 0.1598 | 0.6513 | 0.132 |
| dining | task | `.../task.png` | 0.0820 | 0.7790 | 0.269 |
| living | primary | `.../living/3b89459abb2c71ad/render_rgb_v1/primary.png` | 0.1521 | 1.1553 | 0.172 |
| living | secondary | `.../secondary.png` | 0.1423 | 1.1713 | 0.183 |
| living | task | `.../task.png` | 0.1422 | 0.3845 | 0.232 |
| bedroom | primary | `.../bedroom/8b36c966eae82e55/render_rgb_v1/primary.png` | 0.1499 | 1.1255 | 0.127 |
| bedroom | secondary | `.../secondary.png` | 0.2072 | 1.1387 | 0.136 |
| bedroom | task | `.../task.png` | 0.1464 | 0.8711 | 0.201 |
| office | primary | `.../office/8b4ba8d12669c9b9/render_rgb_v1/primary.png` | 0.1347 | 0.9297 | 0.159 |
| office | secondary | `.../secondary.png` | 0.0934 | 0.9307 | 0.180 |
| office | task | `.../task.png` | 0.0810 | 0.9042 | 0.225 |
| workshop | primary | `.../workshop/f79b07e012bd7b11/render_rgb_v1/primary.png` | 0.2119 | 0.8305 | 0.232 |
| workshop | secondary | `.../secondary.png` | 0.1266 | 0.6116 | 0.199 |
| workshop | task | `.../task.png` | 0.1139 | 0.6542 | 0.207 |

The Stokes EXR is written next to each PNG because that is how `render_stokes` records the film. The gallery and the zip do not use it.

## 6. Cameras

An overview is accepted only when its origin is inside the room, the ray from the origin to the furniture centroid does not hit an opening before the centroid, the origin is at least 0.40 m from furniture, and a 32 × 18 ray grid is not mostly opening, floor, or one wall. The primary is the accepted view that sees the most furniture and is not crowded against one piece. The secondary is another accepted view at least 1.2 m away in the floor plane. The task camera frames the island, dining table, coffee table, bed, desk, or workbench from the side of that surface that has room.

Origins and targets are meters. Field of view is the horizontal Mitsuba `fov`.

| Scene | View | Origin | Target | fov (deg) |
| --- | --- | --- | --- | --- |
| kitchen | primary | [0.90, 0.90, 1.55] | [1.378, 2.697, 1.05] | 100 |
| kitchen | secondary | [4.20, 0.90, 1.55] | [1.378, 2.697, 1.05] | 55 |
| kitchen | task | [1.036, 2.30, 1.985] | [2.90, 2.30, 0.96] | 50 |
| dining | primary | [1.20, 0.90, 1.55] | [2.483, 2.029, 1.05] | 80 |
| dining | secondary | [1.20, 2.40, 1.55] | [2.483, 2.029, 1.05] | 100 |
| dining | task | [4.075, 1.80, 1.776] | [2.30, 1.80, 0.80] | 62 |
| living | primary | [0.90, 2.40, 1.55] | [3.037, 2.566, 1.05] | 100 |
| living | secondary | [0.90, 0.90, 1.55] | [3.037, 2.566, 1.05] | 100 |
| living | task | [5.088, 2.30, 1.278] | [3.60, 2.30, 0.46] | 50 |
| bedroom | primary | [3.50, 0.90, 1.55] | [2.545, 2.553, 1.05] | 100 |
| bedroom | secondary | [1.20, 0.90, 1.55] | [2.545, 2.553, 1.05] | 100 |
| bedroom | task | [3.976, 1.805, 1.567] | [2.145, 1.805, 0.56] | 74 |
| office | primary | [0.90, 0.90, 1.55] | [1.359, 1.758, 1.05] | 100 |
| office | secondary | [2.50, 0.90, 1.55] | [1.359, 1.758, 1.05] | 55 |
| office | task | [1.80, 0.528, 1.720] | [1.80, 2.20, 0.80] | 50 |
| workshop | primary | [4.10, 2.00, 1.55] | [4.55, 2.40, 1.05] | 80 |
| workshop | secondary | [2.50, 0.90, 1.55] | [4.55, 2.40, 1.05] | 55 |
| workshop | task | [2.433, 2.40, 2.104] | [4.55, 2.40, 0.94] | 50 |

The kitchen primary ray grid hits `cabinet_a`, `cabinet_b`, `upper_a`, `upper_b`, `oven_1`, `fridge_1`, and `island_1`.

## 7. Lighting and exposure

Each room has a ceiling rectangle, one rectangle outside each window, and a 0.35 m task lamp over the main surface. Radiance was chosen from a 160 × 90, 4-sample probe of the primary camera, at most three tries. If the third probe was still hot, one further reduction was applied before the 1280 × 720 renders. The values below are the ones used for those full images.

| Scene | Ceiling radiance | Ceiling center (m) | Ceiling size (m) | Window radiance | Lamp radiance | Lamp center (m) |
| --- | --- | --- | --- | --- | --- | --- |
| kitchen | 1.1054 | [2.70, 2.30, 2.52] | 4.60 × 3.80 | 0.2662 | 7.3695 | [2.90, 2.30, 1.51] |
| dining | 1.5300 | [2.30, 1.80, 2.62] | 3.80 × 2.80 | 0.8800 | 10.2000 | [2.30, 1.80, 1.34] |
| living | 1.1054 | [3.10, 2.40, 2.72] | 5.40 × 4.00 | 0.2662 | 7.3695 | [3.60, 2.30, 1.11] |
| bedroom | 1.1054 | [2.20, 1.90, 2.42] | 3.60 × 3.00 | 0.2662 | 7.3695 | [2.13, 2.01, 1.24] |
| office | 1.5300 | [1.70, 1.50, 2.42] | 2.60 × 2.20 | 0.8800 | 10.2000 | [1.80, 2.14, 1.33] |
| workshop | 1.3005 | [2.50, 2.00, 2.72] | 4.20 × 3.20 | 0.4840 | 8.6700 | [4.55, 2.40, 1.33] |

Exposure is the sRGB gamma inside `write_rgb_png`. It is not a separate exposure or white-balance control.

## 8. Geometry and composition checks

Hard checks passed for all six scenes: finite transforms, package hashes, floor or parent support within the existing 1 mm contact test, mesh separation of at least 0.004 m, no unintended furniture-wall intersection, openings clear, and the required assets present. A failure would have been written to `failure.json` and the scene would not have been rendered. None was written.

Warnings are recorded and are not pass/fail cutoffs. The workbench 10–30 percent band is not applied to these rooms.

Occupancy is the fraction of 0.02 m cells on the task support covered by the axis-aligned bounds of objects sitting on that support. The usable free rectangle is the largest empty rectangle on that grid whose shorter side is at least 0.20 m. Frustum membership is the count of furniture whose vertex centroid falls inside the primary view. It is not a pixel-visibility count.

| Scene | Task occupancy | Usable free rectangle (m) | Primary furniture centroids in frame |
| --- | --- | --- | --- |
| kitchen | 0.157 | 1.02 × 0.20 at [0.38, 0.00] | 5 |
| dining | 0.110 | 1.60 × 0.22 at [0.00, 0.68] | 4 |
| living | 0.107 | 0.30 × 0.60 at [0.00, 0.00] | 4 |
| bedroom | 0.000 | 1.92 × 1.34 at [0.06, 0.06] | 4 |
| office | 0.261 | 0.54 × 0.22 at [0.86, 0.00] | 3 |
| workshop | 0.343 | 0.44 × 0.46 at [1.36, 0.34] | 1 |

Bedroom occupancy is 0 because the task surface is the mattress and nothing is placed on it. The book and phone are on a nightstand. Workshop occupancy is the high-clutter bench. Its free rectangle is the open column on the right of the 1.0.2 bench, in the bench's local frame.

## 9. Replay and hashes

`replay_interior` loads `recipe.json` and `bundle.json`, checks that the recipe matches the bundle, checks every package hash, and rebuilds `scene.json` in a temporary directory. It does not remesh, retrieve, or call task replay or room replay, and it does not modify the saved files. All six replays returned `valid: true` and `remeshed: false`.

New furniture packages live only in `dataset/qualification/diverse_interior_review/catalog/`. Their status file says `technically_validated`, with `visually_reviewed`, `admitted`, and `registered` set false. Reused props and the workbench are referenced from the existing semantic catalog or `assets/candidates`. Those packages were not rewritten.

## 10. Tests

- `tests/test_diverse_interiors.py`: 6 passed in 0.42 s.
- Scene-generation modules (`test_diverse_interiors`, `test_semantic_scenes`, `test_room_diversity`, `test_scene_quality`, tabletop, task, indoor, asset package, and `test_scene_gen`): 111 passed in 136.39 s.
- Full `pytest`: 231 passed, 2 failed, 192.02 s.

The two failures are the existing pair:

- `tests/test_phase4.py::test_dolp_on_a_known_pixel_matches_polarization_maps`
- `tests/test_phase4.py::test_feature_stack_widths_and_order`

Both raise `FileStream` at `src/sensor_sim/phase4.py:128` because `outputs/phase3/frames/000/stokes/render.exr` is missing. That file was not created.

The previous full run, before this milestone, was 225 passed and the same 2 failures. The 6 new tests account for the increase to 231.

## 11. Files created

Tracked, uncommitted:

- `src/sensor_sim/scene_gen/interior_assets.py`
- `src/sensor_sim/scene_gen/interior_scenes.py`
- `tests/test_diverse_interiors.py`
- `docs/diverse_interior_review_report.md`

Untracked output, under `dataset/qualification/diverse_interior_review/`:

- `catalog/` for the new furniture and the six room shells
- one bundle directory per scene, including `recipe.json`, `bundle.json`, `scene.json`, and `render_rgb_v1/`
- `review/index.html`

Also `local_review/diverse_interior_review.zip`.

No protected module was edited. No dependency was added. Mitsuba, Dr.Jit, Blender, and CUDA pins were not changed.

## 12. Existing outputs

These stayed unchanged. The new tests check the locked hashes.

- `assets/candidates/index.json` `081bf07291a4cc3f8f2af1fe27b8f8beb0164153100244aefa25df1ba209c157`
- Room bundles `fdff86896dfce89eb822b7a46a101f73f25fd561950d20684155a91d5baaa4ec`
- Room specs `96d818140a4264c1c9bd3d9680bfb834f0c98e2bdcf56da9b10de27abd313936`
- Room packages `e07a4cc9b9873f5cf11b1f28abdeae2ac077e1c532e86ae8e7e32d7be343b181`
- Quality bundles `851173653a3e1a77a13fefa2ee3a25768d48abffdd9b7dc5db6bcfbf42ae6486`
- Tabletop bundles `5cf2d69950ac58908b8832481a3d9a95c1b8d49c8c14f15e0b312d7c7d2b231c`
- Candidate asset files `10634d43db2480bea85f42d6aad1f9892ecd90512dfec26a5c4b03ef7fcfa1e6`
- Dataset v1: 1533 files, 457,265,977 bytes
- `dataset/dataset_v1/run_manifest.json` `9475a86ae204d9b2840b23e558f8043272ad8c6f6e689f00541688879b08c196`
- `dataset/dataset_v1/audit.json` `2bc69f6f45849d30c7eee283da1093649c3560f3050d936105e479f779d0a95e`
- `dataset/dataset_v1/production_config.yaml` `9fac8b70d7e663427593db4fe871b1a3270621baeb07fa92da87a4efd51872d8`
- `room_schema.py`, `room_generator.py`, `indoor_mitsuba.py`, `task_replay.py`, and `tabletop_assets.py` match the hashes locked in `tests/test_semantic_scenes.py`

`dataset/qualification/semantic_scenes`, `room_diversity`, and `scene_quality_v2` were not rewritten.

## 13. Known visual limitations

I inspected the six primary PNGs and the task views.

- The kitchen primary shows the sink and faucet, lower and upper cabinets, the oven and cooktop, the island with food and a lamp, the north window, and the refrigerator on the right. The refrigerator is a tall two-door body. Its doors and carcass share one white material, so the split is the door geometry and the handles, not a color change. The sink is a basin, rim, and faucet, and it is simpler than a manufactured sink.
- Openings have no glass. A window is a hole plus a rectangle emitter outside it. The dining and office primaries include a bright window. The kitchen window in the primary is a gray aperture rather than a blown-out white one.
- Several primaries use a 100 degree horizontal field of view so the camera can see both sides of the room. That view also shows the ceiling emitter as a bright band along the top of the kitchen, living, and bedroom frames.
- Floor under and behind furniture stays dark. The ceiling panel faces down, and the path depth is 5, so those regions get little light. The dark region is shadow, not a missing floor.
- The images are noisy at 16 samples per pixel.
- The dining and office task views are dimmer than their overviews (median 0.082 and 0.081). The table and desk are still readable. The dining lamp sits on the sideboard, not over the table.
- There is no clothing prop and no knife prop in the existing catalogs. The bedroom uses a book and a phone. The kitchen uses a spoon, a fork, and a spatula.
- Block furniture is recognizable as the named object. It is not a manufactured model.

Visual review is pending. Do not treat these scenes as admitted.

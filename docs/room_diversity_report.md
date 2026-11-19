# Room diversity

Branch `main`, HEAD `0609c61124005144060a8f9eb673d73510b57ff5`. No commit was created. The six rooms are qualification candidates. `visually_reviewed`, `admitted`, and `registered` were not set. `assets/candidates/index.json` was not modified. Semantic scene composition was not started.

The working tree was already dirty. `git status --short` still shows the same pre-existing edits to `.gitignore`, `README.md`, the film configs, `pyproject.toml`, eight film modules, `tests/test_heldout.py`, and `uv.lock` (`git diff --stat`: 15 files, 933 insertions, 757 deletions). Those files were not edited again. `src/sensor_sim/scene_gen/` remains untracked.

Pytest before this milestone: 203 passed, 2 failed, in 188.93 s. Both failures are `tests/test_phase4.py` raising `FileStream` while opening `outputs/phase3/frames/000/stokes/render.exr` from `src/sensor_sim/phase4.py:128`.

## Coordinate convention

Documented in `src/sensor_sim/scene_gen/room_schema.py`. Z is up, units are meters, and the frame is right-handed. X increases east and Y increases north. Width, depth, and height are interior clear distances between finished faces.

The southwest interior floor corner is `[0, 0, 0]`. The finished floor is `z = 0`. The west face is `x = 0`, the east face is `x = width`, the south face is `y = 0`, the north face is `y = depth`, and the ceiling underside is `z = height`. The floor slab extends below `z = 0`. The ceiling slab extends above `z = height`. Wall thickness extends outward.

`build_room` still uses the exterior floor corner as its origin, with the finished floor at `z = 0.05`. The new assembler matches its 11 structural pieces after the shift `[0.12, 0.12, 0.05]`. Frame trim and the door leaf are not part of that comparison. `structural.py` was not edited. Its hash remains `6e839ad9debb3bcbc4b16c1016f1eac77b6af58e8f1dc93b49c5302e007db1a7`.

North and south walls include the corner returns. East and west walls stop against those returns. Shared faces are allowed. A second corner box is not generated. An opening that leaves its pier is rejected and is not moved.

## Profiles

Wall, floor, and ceiling thickness stay at the reference values, 0.12 m, 0.05 m, and 0.05 m. Compact was tried at 3.20 × 3.00 × 2.40 m. Both furniture arrangements passed the floor contact, 0.08 m furniture clearance, 0.75 m door corridor, and 0.20 m window keepout checks, so that size was frozen before any package hash was written. The 3.40 × 3.20 × 2.40 m alternative was not used.

| Profile | Interior m | Openings |
| --- | --- | --- |
| `room_compact` | 3.20 × 3.00 × 2.40 | south door 0.90 × 2.10 at offset 0.25; north window 1.00 × 0.90, sill 0.90, offset 0.80 |
| `room_standard` | 4.20 × 3.80 × 2.70 | south door; north window 1.40 × 1.10, sill 0.90; east window 0.80 × 1.00, sill 1.00 |
| `room_spacious` | 5.50 × 4.80 × 3.00 | south door; north window 1.80 × 1.20, sill 0.80; west window 1.00 × 1.20, sill 0.90 |

Exterior bounds are the interior box plus the outward thicknesses. For compact that is minimum `[-0.12, -0.12, -0.05]` and maximum `[3.32, 3.12, 2.45]`.

Each opening is a gap between wall boxes. Several rays through the clear aperture miss that wall. A control ray 0.05 m outside the aperture hits it. Frames are 0.02 m deep and 0.04 m wide on the interior face. They do not fill the hole. A static door leaf is included beside the opening when the pier is long enough. It does not swing. There is no glass. The backdrop setting is `none`, so an unlit opening is dark.

## Packages

Eighteen packages, six per profile, were written with `create_package` under `dataset/qualification/room_diversity/catalog/`. `publish_candidate` and `set_candidate_status` were not called. Status lives only in `room_catalog.json`. Every package is `renderer_validated` there after both arrangements of its profile produced a finite Stokes EXR. The production index hash is still `081bf07291a4cc3f8f2af1fe27b8f8beb0164153100244aefa25df1ba209c157`.

## Scenes

The place setting is `quality_controlled` seed 17201, run `ed940a5f855d1588`. Its bundle hash begins `41d63e6436c47b65`. Each object uses `M_table_new @ inverse(M_table_source) @ M_object_source`. Asset ids, versions, instance ids, and package hashes are the source values. The chair yaw is computed so local −Y, the seating direction, points at the table. It is not a fixed value of π.

Furniture clearance is the mesh distance for table–chair, table–shelf, and chair–shelf. The smallest measured value in these six scenes is the table–chair distance, 0.26 m. The limit is 0.08 m. The door corridor is the floor rectangle of the doorway, 0.90 m deep. The window keepout is the aperture extended 0.20 m into the room. Both were clear.

| Scene | Seed | Run |
| --- | --- | --- |
| `room_compact_central` | 17301 | `7f774e080dde15d6` |
| `room_compact_perimeter` | 17302 | `459e09531c7e8e52` |
| `room_standard_central` | 17303 | `0ea3fd7596033729` |
| `room_standard_perimeter` | 17304 | `764f63997834f6c5` |
| `room_spacious_central` | 17305 | `dd985a60befaf87d` |
| `room_spacious_perimeter` | 17306 | `4a6929a751f3eee8` |

Room replay rebuilds `scene.json` from the saved instances and relations and requires the same bytes. It does not call the mesher, retrieval, or placement, and it does not use `task_replay.replay`.

## Light and renders

The ceiling light is the 0.80 × 0.50 m rectangle, radiance 40, centered 0.02 m below that room’s ceiling underside, normal down. It fits every profile, so the size was not changed. It is unpolarized and not calibrated. Because it points down, the ceiling underside is lit only by bounce and reads dark at 16 samples per pixel. That dark band is not a gap in the mesh. Floor, wall, and ceiling bounds meet on `z = 0` and `z = height`.

`ceiling_plus_daylight_v1` lists one emitter per window: `daylight_window_north` on every profile, plus `daylight_window_east` on standard and `daylight_window_west` on spacious. Radiance is 8. Each plane sits 0.08 m outside the exterior face, aimed through the measured aperture. The six comparison scenes use `ceiling_v1` only. Daylight is an extra overview on the central arrangement, with one image per window.

The variant is `cuda_ad_spectral_polarized`. The overview is 512×384 at 58°. The tabletop camera is the scene-quality close camera at 1600×900 with an 8% support margin. Every EXR reopened as float32, finite, 15 channels in the existing `CHANNEL_NAMES` order, with an id mask of the same height and width where the tabletop view was rendered. A valid EXR is the file contract, not a Fresnel result.

## Tests

`tests/test_room_diversity.py` covers the interior origin, rejection of an opening that enters the pier, reference parity, distinct exterior bounds, rays on all four walls, furniture facing and the failure report for a room that cannot hold the set, table-local matrices, replay without meshing, the EXR contract, gallery links, and the locked task, indoor, tabletop, scene-quality, and candidate-index hashes.

Full pytest after this milestone: 214 passed, 2 failed, in 190.04 s. The two failures are the same Phase 3 tests and the same `FileStream` exception on `outputs/phase3/frames/000/stokes/render.exr`. The 11 new room tests are included in the 214.

## Limits

These are rectangular rooms. Openings are geometry, not textures. The door leaf does not articulate. A clear floor corridor is not a claim of human access or robot reach. The ceiling light is not calibrated radiometry.

```bash
xdg-open dataset/qualification/room_diversity/review/index.html
```

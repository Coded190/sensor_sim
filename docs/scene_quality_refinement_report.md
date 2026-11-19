# Scene quality refinement

Branch `main`, HEAD `0609c61124005144060a8f9eb673d73510b57ff5`. No commit was created. The three new pilots are qualification candidates. `visually_reviewed`, `admitted`, and `registered` were not set. Catalog statuses and package hashes were not changed. This pass does not generate new rooms and does not start semantic scene composition.

The working tree was already dirty before this milestone and was left that way. `git status --short` still shows modifications to `.gitignore`, `README.md`, `configs/film/rendering.yaml`, `configs/film/train.yaml`, `pyproject.toml`, eight modules under `src/sensor_sim/film/`, `tests/test_heldout.py`, and `uv.lock`. `git diff --stat` for that pre-existing set is 15 files, 933 insertions, and 757 deletions. `pyproject.toml` is 14 changed lines and `uv.lock` is 1008. Those film, project, and lockfile edits have no originating commit on this branch; they are uncommitted working-tree changes. They were not reverted and were not edited again here. `src/sensor_sim/scene_gen/` is untracked, so the new quality module does not appear as a separate tracked diff.

## What was held fixed

Protected hashes match the previous reports:

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

`RECIPES`, `ARRANGEMENTS`, `SEPARATION`, and `place_tabletop` were not changed. The 15 saved tabletop `bundle.json` files hash to the same values recorded in `dataset/qualification/scene_quality_v2/old_bundle_hashes.json` at the start of the pass. Scene A–C `scene.json`, the six task `scene.json` files, and `assets/candidates/index.json` (`081bf07291a4cc3f8f2af1fe27b8f8beb0164153100244aefa25df1ba209c157`) are unchanged. The index still has 25 `renderer_validated`, 7 `technically_validated`, and 1 `candidate` entries.

Pytest before this work was 191 passed and 2 failed. After it, the same two tests fail and 203 pass. Both failures are `tests/test_phase4.py` opening the missing file `outputs/phase3/frames/000/stokes/render.exr` from `src/sensor_sim/phase4.py:128`. That path and exception are the historical pair. The Phase 3 EXR was not created to make them pass.

## Coverage definition

`occupied_fraction` is not a sum of object areas. `_occupancy` stamps every triangle into a 4 mm boolean grid on the XY plane, counts cells that any triangle covers, and divides by the axis-aligned table support rectangle. The support is 1.2 m by 0.7 m, so `table_area_m2` is 0.840. Height does not add area. Overlapping projections count once. `union_m2` is that cell count times 0.004². `footprint_sum_m2` is a separate number: the sum of each object's XY convex-hull area. It can exceed `union_m2` when hulls overlap. `nearest_clearance_m` is the minimum mesh clearance among placed objects, ignoring the first object, which has no neighbor yet.

The nine saved non-comparison scenes were recomputed from their `scene.json` poses and the packaged meshes. They were not regenerated.

| Saved task | Seeds | Instances | Union fraction | Footprint sum |
| --- | --- | --- | --- | --- |
| `tabletop_polarization_controlled` | 17101, 17102, 17103 | 8 | 0.0667, 0.0669, 0.0668 | 0.057 m² |
| `tabletop_mixed_objects` | 17101, 17102, 17103 | 10 | 0.0666, 0.0668, 0.0668 | 0.058 m² |
| `tabletop_dense_clutter` | 17101, 17102, 17103 | 16 | 0.0959, 0.0961, 0.0960 | 0.084 m² |

For controlled seed 17101, each object's own raster area is plate 0.0153 m², bowl 0.0094 m², glossy bowl 0.0094 m², phone 0.0100 m², polished can 0.0036 m², rough can 0.0036 m², spoon 0.0022 m², and bottle 0.0025 m². Those eight areas sum to 0.056 m², and the scene union is also 0.056 m², so overlap is negligible. Controlled and mixed land on the same fraction because the mixed recipe replaces the second bowl and the rough can with smaller pieces. Sixteen small objects on 0.840 m² are still only about 10% covered.

## New placement

New recipes live in `src/sensor_sim/scene_gen/scene_quality.py`. They call `measure_support` and `mesh_clearance`, keep scale `[1, 1, 1]`, use the existing 0.015 m inset, and do not go below the old clutter separation of 0.008 m. Sampling stays inside the named zone. It does not fall back to the full table.

All three pilots use one new arrangement, `quality_south`. The table is at `[1.55, 1.70, 0.05]`, the shelf at `[3.20, 2.30, 0.05]`, and the chair is yawed by π about its footprint center to world xy `[2.15, 1.15]` with the feet at floor z = 0.05. The chair's local +Y backrest points to world −Y after that yaw, so the back is south of the seat and the seat faces the table, whose near edge is at y = 1.70. Floor contacts for the table, shelf, and chair all report gap 0 m. The room check reports zero unintended intersections. The shelf sits east of the north window (window x is 0.62 m to 1.82 m).

| Pilot | Seed | Run | Separation | Instances | Union fraction | Footprint sum | Nearest clearance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `quality_controlled` | 17201 | `ed940a5f855d1588` | 0.020 m | 10 | 0.1122 | 0.0956 m² | 0.0227 m |
| `quality_mixed` | 17202 | `b05e275699b4654c` | 0.012 m | 15 | 0.1644 | 0.1403 m² | 0.0258 m |
| `quality_dense` | 17203 | `2aa23b16d561407a` | 0.008 m | 24 | 0.2519 | 0.2152 m² | 0.0082 m |

The design bands were 10–15%, 15–25%, and 25–35% of 0.840 m². All three landed inside those bands. Dense is at the low end of its band. No new large surface (placemat, tray, or board) was packaged. The cup in the controlled scene is the existing `prop_cup` dielectric. It is not a new Fresnel test.

## Camera, masks, and pixels

The room overview stays 512×384 with a 58° field. Its origin is `[0.55, 0.48, 1.72]` and its target is `[2.35, 2.15, 0.75]`.

The first close-camera probe, before the field-of-view correction below, compared two resolutions of the controlled scene at 16 samples per pixel:

| Resolution | Render time | Spoon visible pixels | Spoon shorter side |
| --- | --- | --- | --- |
| 1024×768 | 1.24 s | 268 | 16 px |
| 1600×900 | 1.37 s | 363 | 18 px |

1600×900 was frozen because it added spoon pixels for about 0.13 s. An overhead view of the same scene took 0.54 s and shows the footprint layout, so it is kept for every pilot.

Mitsuba's perspective `fov` is the horizontal field (`x_fov`). The visibility rays had been treating that number as a vertical field, so mask pixels did not line up with the EXR. `_rays` and `_project_point` in `tabletop_render.py` now use the horizontal field. A check against `sensor.sample_ray` gives direction dot products of 1.0 at the center, the right edge, the top edge, and a corner. The saved validation counts in `dataset/qualification/tabletop_scenes/` were not recomputed.

After that correction, the close camera was refit so the projected support quadrilateral reaches NDC `1 / 1.08 = 0.926` on its tighter axis at 1600×900. That is an 8% extra margin around the support polygon, inside the requested 5–10% band. The origin is `[2.15, 0.486, 1.487]`, the target is `[2.15, 2.05, 0.830]`, and the field is 50°. Rays from that origin to the four table corners do not hit the chair. The final controlled close view, at the frozen 1600×900, was not compared again with 1024×768.

`visible_pixels` counts pixels whose nearest real-mesh hit is that instance. A nearer hit replaces the id when it is closer by more than 1e-6 m. An equal distance keeps the smaller id. Background is 0. `frame_fraction` divides by the full frame (1,440,000 pixels at 1600×900). `roi_fraction` divides pixels that are both that instance and inside the projected table-support quadrilateral by the ROI pixel count. The ROI is that quadrilateral, not a loose rectangle. The controlled close view has 368,822 ROI pixels. `shorter_pixel_side` is the smaller side of the visible pixel bounding box. A value under 32 px sets `low_spatial_support`. That 32 px flag is an observation marker for a later experiment. It is not a polarization cutoff, and it is not used to mark an object polarization-ready.

Controlled close view, ceiling light, seed 17201:

| Instance | Visible pixels | Unoccluded pixels | Shorter side | ROI fraction |
| --- | --- | --- | --- | --- |
| `prop_plate__1` | 7732 | 7732 | 66 | 0.0210 |
| `prop_plate__2` | 5172 | 5172 | 51 | 0.0140 |
| `prop_plate__3` | 5456 | 5456 | 52 | 0.0148 |
| `prop_plate__4` | 6762 | 6762 | 60 | 0.0183 |
| `prop_bowl__1` | 8793 | 8793 | 87 | 0.0238 |
| `prop_bowl__2` | 6641 | 6641 | 73 | 0.0180 |
| `prop_cup__1` | 15958 | 15958 | 123 | 0.0335 |
| `prop_apple__1` | 5892 | 5892 | 86 | 0.0160 |
| `prop_spoon__1` | 938 | 938 | 29 | 0.0025 |
| `prop_bottle__1` | 5970 | 5970 | 56 | 0.0037 |

The earlier wide tabletop view of controlled seed 17101 gave the spoon 151 pixels. This close view gives it 938. The shorter side is still 29 px because the handle is 30 mm wide, so the spoon remains under the 32 px observation flag. On the dense pilot, one spoon is 35 px and the other is 28 px. The bottle's ROI fraction is lower than its frame count because the neck projects outside the table-top quadrilateral. None of these tabletop objects are occluded by another object in the controlled close view (`visible_to_projected_ratio` is 1).

Each close view writes `tabletop_instance_ids.npy` as int32 with the same height and width as the EXR, plus `instances.json` (label, instance id, asset, version, package hash, category, material, transform, and the pixel counts). The false-color PNG is only a gallery preview. Per-instance RGB and Stokes crops are written from the visible bounding boxes. The full EXR is kept.

## Lighting

The ceiling emitter is unchanged: center `[2.12, 1.72, 2.40]` m, size 0.80 × 0.50 m, rotated so the normal points down, radiance 40, id `ceiling_v1`. It is unpolarized and not calibrated radiometry.

The north window is the existing hole in `room_wall_north`. A ray from `[1.22, 4.20, 1.45]` toward `[0, -1, 0]` misses the wall. A control ray from `[0.30, 4.20, 1.45]` hits at 0.760 m. The opening is real, so daylight was added.

`ceiling_plus_daylight_v1` is one more rectangle outside the wall, translated to `[1.22, 3.70, 1.45]`, rotated +90° about X, and scaled to a 1.00 × 0.80 m emitter. Its four corners stay at y = 3.70 m, which is outside the wall (the wall ends at y = 3.44 m), with x in [0.72, 1.72] and z in [1.05, 1.85], inside the opening. There is no camera-side light.

A room-view probe at 16 samples per pixel, before freezing the pair:

| Radiance | RGB mean | RGB 99th percentile |
| --- | --- | --- |
| 8 | 0.319 | 1.55 |
| 16 | 0.414 | 1.98 |
| 24 | 0.510 | 2.35 |

Radiance 60 was rendered once and discarded. That room had mean 0.940 and a 99th percentile of 4.50, and the window was blown out. Radiance 8 was frozen. The saved daylight room EXR matches the probe: mean 0.319, 99th percentile 1.553. The ceiling-only room is mean 0.223 and 99th percentile 0.751. The daylight pair uses the controlled pilot only. Camera, poses, materials, resolution, samples per pixel, and render seed 17211 stay the same. Only the emitter set changes.

## Renders and readback

Bundles and renders are under `dataset/qualification/scene_quality_v2/`. Each ceiling pilot has a room overview, a 1600×900 close view, an overhead view, an id mask, crops, an RGB PNG, and a Stokes EXR. The controlled pilot also has `render_daylight_v1` with the same three cameras. Render times were 3.47 s, 3.66 s, and 3.41 s for the three ceiling pilots, and 3.24 s for the daylight pair. Those times include visibility, not only the path tracer. The variant is `cuda_ad_spectral_polarized`. Samples per pixel are 16. The integrator is `stokes` around `path` with max depth 5.

All 12 EXRs were reopened. Each is float32, finite, 15 channels in the existing `CHANNEL_NAMES` order (`R, G, B`, then `S0`, `S1`, `S2`, `S3` for R, G, and B), and the id mask has the same height and width. A valid EXR means the file matches that contract. It is not a Fresnel or S3-sign measurement. S3 sign was not remeasured. Stokes values were not clamped. The bowl, can, and plate/apple comparisons were not re-rendered. The gallery links the existing files.

The room, close, overhead, daylight, mixed, and dense PNGs were inspected, along with the controlled id preview. The preview outlines the same objects as the RGB close view. The window is dark under the ceiling light and bright under daylight. The chair in the room overview has its back toward the south camera and its seat toward the table.

## Tests

`tests/test_scene_quality.py` covers deterministic grouped placement, the dining and work zones, the support inset, a dense pack above the old 0.096 fraction with clearance at least 0.008 m, chair yaw toward the table, the 8% camera margin, ROI counts, a fully hidden box at zero visible pixels, stable integer ids, mask shape equal to a tiny EXR, daylight corners outside the measured opening, reopening the 12 saved EXRs, gallery links, and the locked hashes and catalog statuses.

Full pytest: 203 passed, 2 failed, in 188.42 s. The two failures are the historical Phase 3 pair above. Tabletop, indoor, and task tests are included in that run and passed.

## Stop

Open the gallery with:

```bash
xdg-open dataset/qualification/scene_quality_v2/review/index.html
```

Room diversity and semantic scene composition were not started. This report does not claim general room generation, Fresnel correctness, dynamic stability, or robot reach. Human visual review is still pending.

# Dataset Generation Quality v1

Local HEAD before these edits is `bfdc9f829e681764f8f061ef5950b4a68f2ff344` on `main` (`Save the scene-generation pipeline and held-out film edits.`). That commit was not pushed: the push to `origin/main` was skipped. This quality-closure diff is uncommitted, as this milestone does not create another commit.

`dataset/` is gitignored, so the rendered qualification scenes stay on this machine.

## 1. Initial quality problems

The six semantic scenes already had support, separation, zone membership, room clearance, replay, and a 15-channel Stokes EXR contract. The workbench tops did not. Union fraction on the 1.44 m² top was low 0.0397, moderate 0.0485, and high 0.0653. The free-space check only reported whether some 0.25 × 0.25 m empty window existed. There was no split between hard failures and composition warnings, and no separate view of a grouping or of the open workspace.

Kitchen (0.128), dining (0.156), and desk (0.082) already passed support, clearance, and category checks. Their object layouts were left as they were.

## 2. Changes made

- Added `prop_work_panel` (0.52 × 0.30 × 0.018 m), `prop_fixture_block` (0.28 × 0.14 × 0.040 m), and `prop_stock_plate` (0.34 × 0.16 × 0.008 m).
- Republished the workbench zones as version 1.0.2. The mesh bytes match 1.0.0 and 1.0.1. The right-hand column is the open workspace. Tools sit beside the panel. The fixture is added at moderate clutter and the stock plate at high clutter.
- `largest_free_rectangle` is the maximum empty rectangle on the 4 mm raster. `usable_free_rectangle` is the maximum empty rectangle whose shorter side is at least 0.20 m.
- Added `src/sensor_sim/scene_gen/semantic_quality.py`. Hard failures reject a scene. Coverage and workspace size are warnings.
- Each scene now has overview, work-surface, grouping, and workspace views.

## 3. Changes deliberately not made

Kitchen, dining, and desk object poses were not rearranged. Room profiles were not added. Lighting stayed at the ceiling panel (inset 0.20 m, 0.08 m below the ceiling, radiance 2.2) plus window daylight (radiance 5). Protected renderer modules were not edited. Dataset v1 was not regenerated. The candidate index was not edited. No asset or scene was marked `visually_reviewed`, `admitted`, or `registered`. The missing Phase 3 EXR was not created.

## 4. Final scene inventory

| Scene | Run | Room | Seed | Objects | Union fraction | Usable free rectangle | Nearest clearance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen | `13566dca561f6af1` | standard 4.2 × 3.8 × 2.7 | 17401 | 8 | 0.1279 | 0.436 × 0.340 m at [0.936, 0.000] | 0.0282 m |
| dining | `424d03bd7f75e505` | spacious 5.5 × 4.8 × 3.0 | 17402 | 11 | 0.1558 | 0.344 × 0.372 m at [0.856, 0.328] | 0.0368 m |
| desk | `d2911facd429bc7f` | compact 3.2 × 3.0 × 2.4 | 17403 | 6 | 0.0824 | 0.268 × 0.700 m at [0.384, 0.000] | 0.1036 m |
| workbench low | `9f6ace8c60824574` | spacious | 17411 | 9 | 0.1494 | 0.488 × 0.800 m at [1.312, 0.000] | 0.0214 m |
| workbench moderate | `3a8e490512d174ed` | spacious | 17412 | 14 | 0.1865 | 0.464 × 0.800 m at [1.336, 0.000] | 0.0300 m |
| workbench high | `89e0e3a99524cf51` | spacious | 17413 | 19 | 0.2414 | 0.444 × 0.800 m at [1.356, 0.000] | 0.0147 m |

Union fraction is the 4 mm XY raster of support-frame triangles, overlaps counted once, divided by the support area. The workbench denominator is 1.80 × 0.80 = 1.44 m². Usable free rectangle is the largest empty rectangle on that raster whose shorter side is at least 0.20 m. Nearest clearance is the minimum mesh clearance among object pairs.

Workbench bands used as warnings, not hard cutoffs: low 10–15%, moderate 15–22%, high 22–30%. All three landed inside those bands and increase in that order. The open column is on the right of the bench, about 0.44–0.49 m by 0.80 m, which is large enough for a manipulation approach. High clutter still repeats the screwdriver and the pliers.

`bundle.json` sha256:

- kitchen `6ca68350435b6d0a0d830999242d7fe469567611c074f618897b2639390a2485`
- dining `c2235a69b1b9fd1c8c9de54c5b3cc224316044ed9d06170a81467ff7a148c811`
- desk `dfb2f1b73ad3cab3b86b528e79a30c5ca3c283c3b0e86b79ffb2f34926db261a`
- workbench low `4990753554c57c41d9e424323e8de65954b5bb68038f3dded5acfca5223a879c`
- workbench moderate `f39f90e4e6c5f5e0793bab28c409540de946af75444c84aeba507f1694d74e62`
- workbench high `18498b96d7f25e10dfd99b32914d1bc51631ff39a9f200044bff3bc2365a3801`

New package sha256: workbench 1.0.2 `41e6ab761e59fd2128ed10cd2b1af3e8e46d9371905443e2b7090bc72ff1406b`, work panel `527bec4394d14558a9b73f5489830392b41c1f55c86733df11869e7b98d91517`, fixture `05b69e7d324e72fa64ba73c7ebc1f87cdc94f0fb229807898fb03b2b37cfd0a3`, stock plate `17249e5955c3517e9b996740316fe1d3cd3bfcc83b1157b7a5fa3c99ffe7c2b9`.

## 5. Composition-quality results

All six `quality.json` files have `passed: true` and no warnings. Hard checks covered finite transforms, support relations within 1 mm, nearest clearance at or above 4 mm, package hashes, required category objects, and the `cuda_ad_spectral_polarized` camera and renderer records.

No candidate was rejected. The first moderate and high slot attempts failed clearance or zone containment. Those slots were moved and the published bundles are the passing layouts. The failing attempts were not written as scene bundles.

## 6. Render and EXR validation

Each scene has `room`, `tabletop`, `grouping`, and `workspace` PNG and EXR files under `render_ceiling_v1`, 24 EXRs in total. Renderer seed 17421, 16 samples per pixel, variant `cuda_ad_spectral_polarized`. The three workbench levels share the bench pose, the close camera, the grouping camera, the workspace camera, and the lights.

Reopened EXRs are float32, finite, and use the 15 `CHANNEL_NAMES` in that order. The close-view mask shape matches the image. A valid EXR is the file contract. It is not Fresnel proof and it does not establish the Stokes sign convention. That physical check remains outstanding. The existing analytical polarization test was not treated as a substitute.

## 7. Replay

`replay_semantic` reloads the saved recipe and bundle, checks room and semantic package hashes, and requires the rebuilt `scene.json` bytes to match. It does not mesh, retrieve, or move objects. The semantic tests monkeypatch `build_asset`, `create_package`, task replay, and room replay so a call would fail the test. Those tests passed.

## 8. Tests

Scene, asset, and replay modules (`test_semantic_scenes`, room diversity, scene quality, tabletop, indoor scenes and contact, task scenes and render): 63 passed in 114.52 s.

Full pytest: 225 passed, 2 failed, in 192.81 s. The failures are the historical pair:

- `tests/test_phase4.py::test_dolp_on_a_known_pixel_matches_polarization_maps`
- `tests/test_phase4.py::test_feature_stack_widths_and_order`

Both raise `FileStream` opening missing `outputs/phase3/frames/000/stokes/render.exr` from `src/sensor_sim/phase4.py:128`. No new failure appeared.

## 9. Known limitations

The ceiling panel faces down, so the ceiling underside and the strip of wall immediately behind a counter or bench stay darker than the work surface. Window openings have no glass. Several thin tools still have a shorter visible side under 32 px. That flag does not change their role. Stokes sign and Fresnel behavior are not validated here. Human visual review, admission, and registration are not set.

## 10. Files changed after the snapshot commit

- `src/sensor_sim/scene_gen/semantic_assets.py`
- `src/sensor_sim/scene_gen/semantic_composition.py`
- `src/sensor_sim/scene_gen/semantic_quality.py` (new)
- `tests/test_semantic_scenes.py`
- `docs/dataset_generation_quality_v1_report.md` (this file)
- qualification outputs under gitignored `dataset/qualification/semantic_scenes/`

## 11. Working tree

HEAD `bfdc9f829e681764f8f061ef5950b4a68f2ff344` on `main`. The quality-closure edits above are uncommitted. `origin/main` was not updated.

## 12. Protected inputs

Dataset v1, the candidate index, room-diversity bundles and packages, scene-quality bundles, the 15 tabletop bundles, the six task scenes, and Scenes A–C were checked by `tests/test_semantic_scenes.py` against the locked hashes. Those tests passed. Protected module hashes in that test also passed.

## 13. Status words

The six scenes are `generated`, `geometry_validated`, `renderer_validated`, and `replay_validated`. They are not `visually_reviewed`, `admitted`, or `registered`. The semantic catalog lists 29 packages as `renderer_validated` only. That automated status is not human approval.

## Freeze decision

The six-scene suite can be frozen as the Dataset Generation Quality v1 baseline for human review. It should not be used to start a large dataset until that review is done. Remaining blockers: human visual review of `dataset/qualification/semantic_scenes/review/index.html`, and a separate Stokes-sign and Fresnel validation. The missing Phase 3 EXR is unrelated and was left missing.

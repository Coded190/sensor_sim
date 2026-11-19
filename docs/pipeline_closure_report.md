# Pipeline closure report

Dataset v1 was not generated and no model was trained. `spp` is null. The pipeline is not ready for Dataset v1. The current decision and its measurements are in the Convergence v2 section below. The sections before that record the earlier closure cycle.

The evidence file is `dataset/qualification/pipeline_closure/evidence.json`. The baseline pytest, saved before the later edits in this cycle, is `dataset/qualification/pipeline_closure/baseline/pytest.txt`: 100 passed, 2 failed. The final pytest is `dataset/qualification/pipeline_closure/final_pytest.txt`: 104 passed, 2 failed. Both failures, before and after, are `tests/test_phase4.py` opening the missing file `outputs/phase3/frames/000/stokes/render.exr`. `git diff --check` reported no whitespace errors.

## What existed

The generator already exported Blender scenes, patched Mitsuba XML, rejected empty shape ids, and wrote unclamped Stokes EXRs. `ray_triangle_ids` is a Möller–Trumbore test on exported meshes and does not read the shape-index voter. The scoring in this milestone uses a separate copy, `reference_ray_ids`, which also does not call the voter.

## What changed

Publication now refuses a sample with no `run_id`. The dirty-tree hash includes hashes of untracked scene-generation source and config files, not only `git diff`. A read-only config view splits assets, scene, and sensor settings without changing `generate_sample`. Placement runs a narrow phase after the bounding-box test: containment, sampled surface clearance, and the 5 mm gap. Mesh checks reject tiny extents below `1e-4` m, non-finite normals, and zero-length normals. A third identical BSDF no longer becomes the same mask as the second copy.

## What passed a real export or render

Rebuilding seed `269837089` from the saved config reproduced the geometry seed `3201542270`. Loaded vertex positions matched with maximum error 0. Same-seed S0 p95 absolute difference was about `1e-8` on 5808 illuminated pixels. A different render seed moved S0 by median 0.0093 and p95 0.033. That second number is Monte Carlo variability.

Eight Blender scenes were exported. Interior pixels, defined as the 8-neighborhood sharing an id, were:

- Shared datablock: 639 and 634.
- Different transforms: 247 and 229.
- Same geometry, different BSDFs: 626 and 631.
- Different geometry, same BSDF: 626 and 262.
- Three copies: 359, 292, and 360.
- Partial occlusion: 1182 and 331.
- Identical vertices, two datablocks: 639 and 634.
- Hidden: front 2463, back 0 and the name `back` is in the hidden list.

A capsule with scale 1.2, yaw 0.7 rad, and an extra x translation matched Mitsuba to `6e-9` m. Depth residual against `near_clip / dot(ray, camera_forward)` had median `2.6e-8` m and max `1.1e-7` m, inside the 0.4 mm tolerance. Inverse-transpose normals were not tested.

Primary BSDFs loaded as SmoothDielectric eta 1.33, 1.5, and 1.7, copper and rough copper alpha 0.1, and pplastic eta 1.5, reflectance 0.5, alpha 0.1. The patched-versus-unpatched render comparison was not called a pass: Phase 5 did not produce a usable uncertainty.

EXR channels by name are R, G, B, then S0, S1, S2, and S3 as RGB. Fifteen float32 channels, no alpha. The five training views have shapes `(128,128,1)`, `(128,128,3)`, `(128,128,4)`, `(128,128,2)`, and `(128,128,7)`. Building a view did not change the EXR bytes. S1 remained negative.

The 128² visibility recount kept the 80-pixel rule. It was not moved.

## Convergence

The reference is the mean of four independent 1024-sample Stokes renders. DoLP and AoLP were not averaged as angles. The predeclared rule, fixed before the renders, is that reference uncertainty is at most 25 percent of each existing cap. For S0 relative p95 that limit is 0.0375. The multi-object scene's 1024-sample S0 p95 versus the mean was 0.061. Four 2048-sample renders of the same scene gave 0.044. The reference is not stable. The 0.15 cap was not loosened. `spp` stays null.

Because `spp` is null, the quota run and the integration batch were not started. Their status is `blocked`.

## Ready or not

The pipeline is not ready for Dataset v1. The blocker is polarization reference stability. Identity, the second export, and the capsule depth check passed on real Mitsuba loads. Quota preservation on a real CLI run, the integration-batch split leakage checks, and the patched-versus-unpatched radiance comparison were not run.

## Convergence v2

The pipeline is still **not ready for Dataset v1**. `configs/scenes/dataset_v1.yaml` stays at `spp: null`. Dataset generation is not authorized. The new evidence is under `dataset/qualification/convergence_v2/`. The earlier pipeline-closure files were not overwritten.

The historical reference statistic was recomputed from new EXRs and matches the saved JSON. Its definition, unchanged, is the maximum over seeds of the pixel 95th percentile of `abs(render - mean) / np.maximum(component_mean, 1e-4)` on pixels with mean S0 > 1e-4. That is the deviation of one render from the four-render mean. It is not the standard error of the mean. For S1, S2, and S3 the denominator is that component, not reference S0. At 1024 spp the reconstructed S0 value is 0.06061512 against the saved 0.06061513 (absolute difference 7.0e-9). The reconstructed S1 value is 1.04138106 against 1.04138161 (difference 5.4e-7). The tolerance fixed before the comparison was 1e-5. The 2048 spp S0 value differs from the saved 0.04412999 by 1.0e-7. `stable` remains false under that definition. Old candidate booleans that used the component denominator are superseded.

The corrected relative error is `abs(candidate - reference) / reference_S0` on pixels with object id > 0, a valid ground-truth mask, and reference S0 > 1e-4. Scalar Stokes are the mean of `component.R`, `component.G`, and `component.B`. On that definition the multi-object S1 p95 of the per-render deviation falls to 0.00270 at 1024 spp and 0.00210 at 2048 spp. The ratio 0.775 is close to the Monte Carlo factor `sqrt(1024/2048) = 0.707`. The historical S1 number is about 385 times the corrected one because the old denominator collapses where S1 is near zero or negative. The error that remains sits in the dimmest S0 quartile and on the small objects. Object 1 has a worst-seed S1 relative p95 of 0.129 on 179 pixels. The acceptance mask was not changed to exclude it. Stokes consistency violations on the illuminated object mask: 0 of 5844.

The prospective reference uses the sample standard deviation across seeds, with divisor `n - 1`, divided by `sqrt(n_seeds)` and then by reference S0. The reported upper bound is the 95th percentile of 200 seed-level bootstrap replicates of the spatial median and spatial p95. Pixels are not seed replicates. DoLP and AoLP are computed after the Stokes images are averaged. With eight seeds, 1101–1108, at 1024 spp, the multi-object S0 p95 upper bound is 0.0245 (limit 0.0375), S1 is 0.00169 (limit 0.02), DoLP is 0.00226 (limit 0.02), and AoLP is 1.23° (limit 2.5°) on 5844 pixels. The same bounds pass for dielectric IOR 1.33, 1.40, 1.50, 1.60, and 1.70, measured pplastic, copper, and rough copper. Every independent candidate seed 2101–2103 passes every one of those scenes at 256, 512, and 1024 spp. The lowest is 256. On the multi-object scene at 256 spp, seed 2101, S0 relative median is 0.0121 and p95 is 0.110 (n = 5844); S1 relative p95 is 0.00630; DoLP absolute p95 is 0.00924; AoLP p95 is 3.89° on 432 pixels where both DoLP values are at least 0.05. The noise floor was not subtracted. `spp` is not frozen because the generation contract has no approved scene count.

Identity was rescored on the existing eight Blender exports at 256×256, every interior pixel, with `reference_ray_ids`. Interior agreement is 1.0 on every scene (n from 476 to 2463). Fractional ids are 0. The hidden object `back` has 0 visible object-id pixels and remains in the metadata. Boundary agreement is 0.882 to 0.928. Material agreement on all ray hits, including boundary pixels, is 0.975 to 0.990.

Six separate capsule exports, x, y, z, yaw, uniform scale, and one combined pose, match loaded vertices to a maximum of 2.7e-9 to 5.1e-9 m. Depth residual against `near_clip / dot(ray, camera_forward)` has maximum 1.73e-7 m, inside the 0.4 mm tolerance. Camera-space shading normals match `world_normals_to_camera` to a maximum of about 2e-6 degrees. On capsule-id pixels only, the geometric normal is within 0.021 degrees of the nearest loaded face normal (n = 60 to 93). Inverse-transpose normals for nonuniform scale remain untested.

The narrow phase now reports triangle distance through a bounding-volume search, plus containment and the floor test at z = 0.001. The adversarial cases are unit tests: a piercing triangle, a 5.0 mm gap, a 4.0 mm gap, overlapping boxes with an 8.0 mm surface gap, floor contact, floor penetration, and a closed mesh inside another mesh. That is not a separate Blender placement scene.

Loaded dielectric eta values are 1.33000004, 1.39999998, 1.50000000, 1.60000002, and 1.70000005. Rough copper alpha is 0.10000000149. An opacity-1 mask and an `alpha_u`/`alpha_v` pair of 0.1 differ from the unpatched single-object controls by an S0 relative p95 of about 1.05e-7, with the same shape ids. Rough gold was not promoted.

The candidate EXR is float32, has the 15 named channels, has no alpha, and keeps signed S1 (minimum -0.0882, maximum 0.325 on the multi-object 256-spp seed 2101 file). Building training views did not change the historical rerun EXR bytes. Polarized leading RGB versus a path integrator, same camera, light, 128², box filter, 256 spp, and seed 2101, has median absolute error 1.49e-8 on 5941 pixels with S0 > 1e-4. That difference is at float32 rounding and is not a loader bug.

A real CLI run at 256 spp, seed 17, run id `7b254dd047a84e90adc369da3f2d9eb4`, requested 5 successes and published 5, with 0 failures. Wall time from the manifest timestamps 2026-11-17T22:37:48Z to 2026-11-17T22:39:13Z is 85 seconds, so 17 seconds per success including Blender export, the 256-spp render, and validation. Mean sample size is 6.53 MB (minimum 3.49 MB, maximum 11.4 MB). One sphere has 73 visible pixels, below the 80-pixel training rule, and was kept. The two-object scenes do not repeat an `asset_instance_id` and do not hide an object. Resume completed without rendering again. Changing spp to 128 was refused. The scene split was written. A material holdout was not requested and is not assessed. The 8+8 pilot is not this run.

`successful_scenes` stays null. The existing 64-scene figure remains a coverage floor, not an approved size. Per-family quotas are therefore unset. `generation_authorized` is false.

Protected-file hashes for `sfp_scene.py`, `mitsuba_scene.py`, and `scenes.py` match the pre-edit snapshot. The final test run is `dataset/qualification/convergence_v2/reports/pytest_final.txt`: 121 passed, 2 failed. Both failures are the pre-existing missing file `outputs/phase3/frames/000/stokes/render.exr`. `git diff --check` is clean, and the untracked `scene_gen` Python files have no tab or trailing-whitespace hits.

## Dataset v1 readiness

The project is still not ready to generate Dataset v1. `generation_authorized` remains false. `spp` is now 256, the lowest candidate that passed the corrected caps, and `successful_scenes` is still unset. The evidence is `dataset/qualification/dataset_v1_readiness/`.

The saved nine-scene candidate matrix and eight-seed references were recomputed from the per-seed numbers and pass. A new ten-scene CLI run, id `75d5acc52c304e73a6512850a6a43f25`, published every requested preset, including the five dielectric IORs, pplastic, copper, and rough copper. The repeated-sphere slot has two `sphere` asset ids and two different `asset_instance_id` values. The occlusion slot has 32 pixels of silhouette overlap and both objects remain visible. That run is quarantined and is not a training split.

Exported placements of 1, 2, 3, and 4 objects stay at least 5 mm apart, with floor contact at z = 0.001 m. Four identical dielectrics needed a distinct opacity form so Mitsuba would not merge the fourth shape. A 4 mm gap and a floor at z = 0 are rejected.

The loader now builds a separate normal-supervision mask. A 40-pixel sphere and a 36-pixel sphere receive no supervised normals. A 320-pixel sphere in the same scene does. The old metadata value of 73 pixels does not match that sample's object-id image, which contains 430 pixels of that id, and the loader rejects the mismatch.

Opacity-1 and `alpha_u`/`alpha_v` matched-seed differences stay under the predeclared 1e-5 absolute tolerance. Two coverage EXRs reopened in a fresh process keep the 15 float32 channels and are unchanged by the five training views.

The 12-, 64-, and 256-scene proposals are slot tables under `dataset_v1_contract_proposals/`. None is selected. The measured coverage rate is 15.5 seconds and 5.27 MB per success. With a 1.2 retry factor and a 1.5× contingency, the serial estimates are about 6 minutes, 33 minutes, and 2.2 hours. The test run after these changes is `dataset/qualification/dataset_v1_readiness/pytest_final.txt`: 126 passed, 2 failed, the same two missing-EXR failures. Protected-file hashes are unchanged.

## Dataset v1 generation

The 64-scene contract is now the production file. `successful_scenes` is 64, the quotas are approved, and `generation_authorized` is true. Schema 3 stores `object_id_pixel_count` separately from the legacy ownership field `visible_pixels`. The 80-pixel rule uses the image count. `load_sample` returns `normal_supervision_mask` beside the five unchanged training views.

The saved convergence matrices were recomputed and still pass at 256 spp. The historical component-denominator result remains `stable: false`. On 103 non-ground objects in existing 128×128 images, the image-count histogram has minimum 9, median 172, p10 62.2, p90 423.2, 0 fully hidden, 22 below 80, and 81 at or above 80. That histogram does not use the old ownership count. A new schema-3 hidden diagnostic keeps the rear object in metadata with 0 object-id pixels and 0 supervised normals.

Four identical dielectric shapes at 256 spp keep distinct ids. The XML assigns the first shape the unwrapped BSDF, the second float opacity 1.0, the third spectrum opacity 1, and the fourth RGB opacity 1 1 1. Each wrapper, compared with an isolated unwrapped control at the same seed, stays inside the frozen tolerances. The largest RGB absolute difference on the float-opacity control is 1.19e-7 on 5953 pixels with S0 > 1e-4.

The production run id is `0e0d3bc4916847b8857f42ed761208a9`. It published 64 samples from 80 attempts, with 16 failed attempts that were not added to `completed_ids`. Manifest timestamps 2026-11-18T03:41:00Z to 2026-11-18T06:20:32Z give 9572 seconds. The samples occupy 457,039,003 bytes. Every family and every non-gold preset appears 8 times. Object counts are 32, 16, 8, and 8. The eight occlusion fractions are 0.103 to 0.225. The scene-disjoint split is 48, 8, and 8, with each family and each preset at 6, 1, and 1. Materials and families appear in all three splits. An earlier incomplete run used cone radii of 0.036 m, which produced 77 object-id pixels, and was replaced by this run. It is stored under `dataset/dataset_v1_incomplete_radius` and is not in the production manifest.

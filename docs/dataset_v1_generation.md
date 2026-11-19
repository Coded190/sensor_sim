# Dataset v1 generation

This specification describes a future run. It does not authorize one. `configs/scenes/dataset_v1.yaml` keeps `spp: null` and `successful_scenes: null`.

## What was measured

Eight independent 1024-sample reference seeds on `cuda_ad_spectral_polarized` met the prospective uncertainty bounds for S0, S1, S2, S3, DoLP, and AoLP. The historical denominator `np.maximum(component_mean, 1e-4)` still fails, and that failure is unchanged. Against the corrected reference, candidate renders at 256, 512, and 1024 spp passed the unchanged caps on every required scene. The lowest passing value is 256 spp. It is recorded as `measured_candidate_spp` and is not a frozen generation setting until the remaining gates and an approved scene count exist.

## What a later run must use

- Mitsuba 3.9.1, Blender 4.5.9 LTS, add-on `8848ded5e5df575f2cd0a9c10061d3f6b7137905`.
- Variant `cuda_ad_spectral_polarized`, integrator Stokes over path, independent sampler, box filter, 128 by 128.
- The fixed camera and light in the YAML. Z up, meters, 5 mm separation, no dropped objects.
- Camera-space shading normals. Signed 15-channel float32 EXRs. Scalar Stokes are the mean of the RGB-converted channels.
- A new `run_id` and a new output directory. Resume refuses a changed config hash. Publication is audited before a sample enters `completed_ids`.
- Splits are built only from that run's audited completed ids. A material shared by a scene-disjoint split is not leakage. A material holdout does not share BSDF types.

## Scene count

`proposed_coverage_floor.successful_scenes` is 64 and is a coverage floor: one success for each family, each primary material group, and object counts 1 and 3. It is not an approved Dataset v1 size. Per-family and per-material quotas stay unset until that size is approved. This milestone does not invent the number.

## How the run will be checked

Requested, attempted, successful, and failed counts are recorded by stratum. A failure retries the same stratum. The command is the scene-gen CLI, not a fake publisher. Storage and time are estimated from the qualification renders at the frozen spp, including export, not from the 16-sample pilot alone.

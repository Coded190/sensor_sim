# Qualification report

Dataset v1 has not been generated and no model has been trained. The latest full `uv run pytest` before this qualification cycle, recorded in `dataset/qualification/baseline/`, is **92 passed and 2 failed**. Those two failures are the known missing file `outputs/phase3/frames/000/stokes/render.exr` in `tests/test_phase4.py::test_dolp_on_a_known_pixel_matches_polarization_maps` and `test_feature_stack_widths_and_order`. They are not treated as passing. Any other failure is a new regression. Git HEAD at that record is `0609c61124005144060a8f9eb673d73510b57ff5`. The working tree was already dirty; the tracked diff is `dataset/qualification/baseline/diff.patch`.

## History: earlier suite

- An earlier full `uv run pytest` ended at 85 passed and the same 2 failed. Scene-generation tests, including the Blender export, passed. `mitsuba_scene.py`, `sfp_scene.py`, and `material_bsdf` were not edited.
- Mesh-hit normals on the prototype sphere: median 0.684° at 256². The gate is that median, at most 1°. The 95th percentile stays under the old 8° cap. One triangle can still reach about 165°, so the maximum is not a 1° result.
- Continuous-sphere comparison remains about 1.3° median on that 256² frame and about 4.4° on the coarser diagnostic frame. That number is reported and is not the gate.
- Depth: `|position - camera|` exceeds the depth AOV by the 0.01 m near clip, to within 0.4 mm. Raw depth is not the pinhole distance.
- 256×256, 64-sample render used `cuda_ad_spectral_polarized` on CUDA. `eta` is 1.5 after the `ext_ior` patch. Whole-frame `frac_dop_gt_1_05` is 0 and `n_stokes_inequality` is 0. Sphere-only mean DoLP is 0.2237. Details are in `docs/stokes_256.md`.
- The same `scene.xml` also rendered on `llvm_ad_spectral_polarized` (256×256, 4 samples, shape `(256, 256, 15)`, integrator `StokesIntegrator`).
- Z-up material profiles replaced the old table. The Phase 2 numbers are preserved in `configs/scenes/material_profiles_phase2.json`.
- Two spheres in one frame received different material ids. A hidden cloud is left unlabeled by `label_positions`.
- `scene-gen --split-mode` writes `train.txt`, `validation.txt`, and `test.txt`.

## Material classes on this sphere

Measured at 256², 64 samples, CUDA, sphere object mask, `S0 > 1e-4`. `keeps` means max DoLP above 0.2.

- dielectric: keeps, max DoLP 1.0. IOR 1.33, 1.40, 1.50, 1.60, and 1.70 all keep.
- pplastic at the Phase 2 preset: keeps, max DoLP 0.56.
- diffuse, plastic, roughdielectric: does_not, max DoLP 0. The roughness sweep at Beckmann alpha 0.02, 0.10, 0.20, 0.35, and 0.50 is also 0.
- conductor Cu: keeps, max DoLP 0.238. Ag 0.040 and Al 0.147 do not. Au 0.199 is just under 0.2.
- roughconductor at alpha 0.1: Cu 0.246 and Au 0.214 keep. Al and Ag do not.

Random sampling of conductor metals is limited to the metals that kept polarization. A dielectric IOR that was not in the sweep is still labeled `untested` on the object. `roughdielectric` stays out of the primary mix.

## Pilot

`dataset/pilot/pilot_report.json`. Six one-object scenes completed (seeds 20–25) and one two-object scene (seed 42), at 64×64 and 4 samples. Seeds 40, 41, and 43 failed placement: the floor inset cannot hold the third object in 80 attempts. That is a packing limit, not a render failure. Splits of the seven completed scenes are in `dataset/pilot/splits/`.

## Not done, and not claimed

- Dataset v1 generation and SfP training. The 16-scene pilot is not a generalization result.
- A 1° result against the continuous analytic sphere.
- Every pixel of the mesh under 1°. Pixels above 8° on the reference sphere are partial coverage, described below.
- `roughdielectric` as a polarized material at any alpha in the sampled range, on this lighting.

## Dataset v1 qualification

`--num-scenes` counts samples that pass file, EXR, mask, and finite checks. Failures go to `failures/` and are absent from `run_manifest.json` and from the split files. A changed config hash refuses to resume.

The 12 sphere pixels above 8° sit on outward faces. Their geometric normal disagrees with the `prim_index` face by about 102°, so they are partial coverage and stay out of the angle score. Closed meshes with a reversed face, a degenerate face, or a non-manifold edge are rejected. `dragon_local` is not required to be watertight.

Identity scenes at 64×64 are in `dataset/qualification/identity/`. Interior pixels, those whose eight neighbors share the object id, match an independent ray-triangle test. The hidden sphere is in the metadata and has 0 pixels. The 5 mm gap scene has both objects. The partial-occlusion scene has both objects. Object ids are in `dataset/qualification/identity_ids.png`.

Primary draws are the five dielectric IORs, the measured `pplastic` preset, copper, and rough copper and gold at Beckmann alpha 0.1. DoLP median, 90th, and 95th percentiles, and the fraction above 0.2, use illuminated object pixels (`S0 > 1e-4`) as the denominator. Rough gold's maximum is 0.253 but its 95th percentile is 0.177 and only 0.75% of those pixels exceed 0.2, so the maximum alone does not describe it. Rough dielectric alpha 0.1 again has DoLP 0 on this Mitsuba 3.9.1 render. The distributions are in `configs/scenes/material_profiles.json` under `primary_distribution_spp16` and in `dataset/qualification/primary_dolp.png`.

The random pilot is 8 one-object and 8 tabletop samples at 128×128 and 16 spp. Thirteen tabletop attempts failed placement or export and were not counted. The eight successes include object counts 2, 3, 4, and 5. On one 128² scene, 16 spp versus 256 spp has mean absolute S0 0.025, mean absolute DoLP 0.0084, and median AoLP error 1.20°. AoLP error is the smallest difference on a 180° period, on pixels with DoLP at least 0.05. That residual is small enough for this fixed-camera broadband set. It is one scene.

The Stokes EXR on that pilot has 15 float32 channels: RGB plus RGB for S0–S3. There is no alpha. `Bitmap.split` returns five images. The loader averages S0–S3 and ignores the leading RGB.

## Final gates, 16 Nov 2026

Evidence is `dataset/qualification/final/evidence.json`. Dataset v1 was not generated.

- The five original empty-id seeds pass after identical BSDFs are kept from being merged. Mitsuba 3.9.1 merges shapes whose BSDF parameter lists match and gives the merged mesh an empty id. The second copy of a rough BSDF is written with `alpha_u` and `alpha_v` equal to `alpha`. Seed `269837089` ids are `mesh-object_00` through `mesh-object_03`, `mesh-ground`, and `emit-light`.
- An object named `sphere` with IOR 1.33 keeps 1.33. The prototype eta check runs only when `mode` is `prototype`.
- `shape_index` / `prim_index` is the same for two instances of one shapegroup. `SurfaceInteraction.instance` is a different pointer and is null when the hit is not an instance. Production exports distinct shapes.
- World vertices of `rerun_269837089` match the source OBJ to a maximum of about `1e-8` m.
- At 256², separated objects have 682 and 552 interior pixels, partial occlusion has 952 and 746, and the hidden object has 0 pixels. Interior ray checks agreed on the 12 pixels sampled per object.
- 16 spp fails the predeclared caps on the multi-object scene: S0 relative p95 is 0.336 (cap 0.15) and AoLP p95 is 11.5° (cap 10°). 64 spp still has S0 p95 0.216. Two 256 spp renders, seeds 0 and 1, have S0 p95 0.152, so the 0.15 cap is unusable and was not loosened. spp is not frozen. See `configs/scenes/dataset_v1.yaml`.
- Rough gold at 256 spp has median DoLP 0.079, p95 0.174, and 0.34% of 886 illuminated pixels above 0.2. It stays in the weak stratum.
- The loader returns RGB `128x128x3` and the five training views. A path render of the same XML differs from the leading RGB by mean absolute 0.0072 at 16 spp. A second Stokes render with the same seed differs by S0 p95 about `5e-8`.
- The 128² pilot visibility rule, written after the histogram, is 80 object pixels for training eligibility. Objects below that stay in the sample.

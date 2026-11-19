# 256 by 256 Stokes render

Command:

```bash
uv run scene-gen --config configs/scenes/sphere.yaml --resolution 256 256 --spp 64 --output dataset/samples
```

The published sample is `dataset/samples/scene_000000`.

- Variant: `cuda_ad_spectral_polarized` on CUDA. This is not an LLVM fallback.
- Mitsuba 3.9.1.
- Resolution 256 by 256, 64 samples, box filter.
- BSDF patch `bsdf.mat-obj_sphere.ext_ior` is present. The mesh-hit normal check passed: median 0.684°, maximum 1.277°. The continuous-sphere median in the same file is 1.292°, so the 1° claim is only for the mesh hit.

Stokes statistics from `metadata.json`, on pixels with `S0 > 1e-4` over the whole frame (23476 pixels). Walls and the ground are in that mask, so the mean is pulled down.

- `max_dolp` 0.9999997786865131
- `mean_dolp` 0.012874690917336475
- `max_dop` 0.9999997786865131
- `frac_dop_gt_1_05` 0
- `n_stokes_inequality` 0
- `n_dolp_above_excess` 0
- `mean_abs_s3_over_s0` 8.68e-7

On the sphere object id only, 1337 pixels with `S0 > 1e-4`:

- mean DoLP 0.22370418901336533
- max DoLP 0.9999997786865131
- mean DoP 0.22370426381306768

`polarization.exr` is the linear Mitsuba bitmap. Nothing was clamped. The `spp >= 64` check did not raise. Spectral copies are still averaged as broadband RGB, as in `convention.py`.

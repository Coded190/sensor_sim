# sensor_sim

Phase 0 replicates Mitsuba 3's quickstart, the polarized Cornell box, and the dielectric and conductor sphere setups from the polarization docs.

## Setup

The project uses [uv](https://docs.astral.sh/uv/). Mitsuba and Dr.Jit are already pinned in `uv.lock`.

```bash
uv sync
uv run pytest
uv run phase0
```

`uv run phase0` renders four jobs at 256×256 and 64 samples per pixel:

| Scene | Variant | What it checks |
| --- | --- | --- |
| `cornell_box` | `scalar_rgb` | Mitsuba hello world |
| `cbox_pol` | `llvm_ad_spectral_polarized` | Tutorial scene with glass and gold spheres |
| `dielectric_sphere` | `llvm_ad_spectral_polarized` | One dielectric sphere, interior IOR 1.5 |
| `conductor_spheres` | `llvm_ad_spectral_polarized` | Two conductors, `eta = 0.052`, `k = 3.905` |

`scalar_spectral_polarized` is the debug variant. The saved figures use `llvm_ad_spectral_polarized`, which is the vectorized CPU variant shipped in the Mitsuba wheel. The `_ad` suffix is automatic differentiation. Phase 0 does not use derivatives.

Override one job with:

```bash
uv run phase0 --scene dielectric_sphere --variant scalar_spectral_polarized --resolution 64 --spp 8
```

## Outputs

Each job writes `outputs/phase0/<scene>_<variant>/`.

`render.exr` is a linear OpenEXR file. Polarized renders have 15 channels: RGB, then `S0`, `S1`, `S2`, and `S3`, each as three spectral copies. Values are 32-bit floats and can be greater than 1. A normal image viewer shows only the first three channels.

The PNG files are the pictures to open. RGB and `S0` use `clip(linear, 0, 1) ** (1/2.2)`. `S1`, `S2`, and `S3` use a diverging scale whose limit is the 99th percentile of the absolute value. That limit is printed on each panel. `contact_sheet.png` puts the panels on one figure.

Inspect the EXR channel list with:

```bash
uv run python - <<'EOF'
import mitsuba as mi
mi.set_variant("scalar_rgb")
print(mi.Bitmap("outputs/phase0/dielectric_sphere_llvm_ad_spectral_polarized/render.exr"))
EOF
```

## Definitions

Stokes components are averaged over their three spectral copies. On these scenes the copies match because the materials use constant spectra. Summary statistics use pixels with `S0 > 1e-4`.

- `S0` is intensity.
- `S1` is horizontal versus vertical linear polarization.
- `S2` is +45° versus −45° linear polarization.
- `S3` is right versus left circular polarization.
- `DoP = sqrt(S1^2 + S2^2 + S3^2) / S0`. Noise-free light has `DoP <= 1`. `frac_dop_gt_1_05` is the fraction of valid pixels with `DoP > 1.05`, which is a Monte Carlo noise check.
- `DoLP = sqrt(S1^2 + S2^2) / S0`. This is the linear part divided by total intensity. `mean_dolp` averages that over the whole frame, so unpolarized walls pull it down. `max_dolp` is the brightest linear-polarization pixel.
- `AoLP = 0.5 * arctan2(S2, S1)` in radians, range `(-pi/2, pi/2]`. The PNG shows degrees. Pixels with `DoLP < 0.05` are masked.
- `mean_abs_s3` is the mean of `|S3|` in radiance units. `mean_abs_s3_over_s0` divides by `S0` first, so it does not depend on exposure.

A single specular bounce of unpolarized light does not produce circular polarization. Nonzero `S3` in the conductor scene is the two-bounce signal described in the Mitsuba polarization docs. The dielectric and conductor images are reconstructions of those documented materials. The original ART comparison scenes were not published, so these renders are not a pixel match to those figures.

Phase 0 does not write a hyperspectral cube. Spectral mode still integrates wavelengths to RGB unless a spectral film is requested. The extra EXR channels here are Stokes components.

## Phase 1

`uv run phase1` writes `outputs/phase1/`. Stokes images come first. The 8-band cube is a second render of the same scene. The two are not combined, so Stokes channels stay out of the wavelength file.

The Cornell box uses the Phase 0 scene. Polarization is `cuda_ad_spectral_polarized`, 256×256, 64 samples. The cube is `cuda_ad_spectral` with `specfilm`, 256 samples. Bands are 37.5 nm wide from 400 to 700 nm, named `band_01_400` through `band_08_662`. Each band is a `regular` spectrum equal to 1 on that window. `specfilm` does not accept the `uniform` spectrum plugin. `outputs/phase1/cornell_cuda/spectral_profile.png` plots the mean linear radiance of each band for the whole frame, the light, the red wall, the green wall, and a box face. Those regions are fractions of the image, recorded in `spectral_profile.json`.

The digital twin is a ball inside a smooth dielectric shell in the same gray room. Outer radius 0.45, inner radius 0.40, so the film is 0.05 thick (11% of the outer radius). That is a visible coating, not a micron film. The shell IOR is 1.5 and does not disperse. The inner ball is diffuse, with reflectance 0.15, 0.25, 0.55, and 0.85 at 400, 500, 600, and 700 nm. `cuda_ad_rgb` cannot evaluate that spectrum, so the ball's RGB picture is `cuda_ad_spectral` integrated to RGB by an HDR film. Its Stokes render is `cuda_ad_spectral_polarized`. Its cube uses the same eight bands.

The Cornell box is diffuse, so its Stokes image is unpolarized: DoLP stays at 0. The ball's shell is the part that polarizes. On the saved CUDA render, `max_dolp` is 0.927 and `mean_dolp` is 0.010. The mean is low because the walls are unpolarized.

The spectral profile is mean linear radiance in each band, not the surface reflectance by itself. The lamp is an RGB color, and Mitsuba turns that into a non-flat spectrum, so the ball's radiance curve is not a copy of 0.15, 0.25, 0.55, 0.85. A test renders a directly lit square with that reflectance under a flat emitter and checks that the 662.5–700 nm band is brighter than the 400–437.5 nm band.

Isaac Sim is not used here. The later scene, after the training phase, is Simple Room: `/Isaac/Environments/Simple_Room/simple_room.usd`. The same ball would sit on its table. Isaac does not compute polarization.

## Phase 2

`uv run phase2` compares six BSDFs on one sphere in the gray Cornell room. The sphere radius is 0.45, the same pose as the Phase 0 dielectric sphere. Walls stay gray. The integrator is `stokes` around `path`. Each job is `llvm_ad_spectral_polarized`, 256×256, 64 samples, seed 0. These jobs are not part of `uv run phase0`.

Shared parameters, where the plugin has them: interior IOR 1.5, exterior IOR 1.0, Beckmann `alpha` 0.1. `alpha` 0.1 is the plugin default. Diffuse reflectance is 0.5 for `pplastic`, `plastic`, and `diffuse`.

| BSDF | Parameters | max DoLP | mean \|S3\| / S0 | Keeps polarization |
| --- | --- | ---: | ---: | --- |
| `dielectric` | int IOR 1.5, ext IOR 1.0 | 0.876 | 0 | yes |
| `thindielectric` | int IOR 1.5, ext IOR 1.0 | 0 | 0 | no |
| `roughdielectric` | int IOR 1.5, ext IOR 1.0, Beckmann alpha 0.1 | 0 | 0 | no |
| `pplastic` | int IOR 1.5, ext IOR 1.0, reflectance 0.5, alpha 0.1 | 0.704 | 0 | yes |
| `plastic` | int IOR 1.5, reflectance 0.5 | 0 | 0 | no |
| `diffuse` | reflectance 0.5 | 0 | 0 | no |

`max_dolp` is the maximum of `sqrt(S1^2 + S2^2) / S0` over pixels with `S0 > 1e-4`. `mean_abs_s3_over_s0` is the mean of `|S3| / S0` on those pixels. A BSDF keeps polarization when `max_dolp > 0.2` and `max_dolp` is at least five times the diffuse control. The diffuse control is 0, so the binding test is the 0.2 threshold. DoLP images use a 0–1 scale. `mean_dolp` stays near 0.006 for the two polarized materials because the walls are unpolarized.

None of the six plugins raised an unsupported-variant error, so each row is a finished render. The runner still records `status=unsupported` and continues if a polarized BSDF reports that it is not implemented. `roughdielectric` looks like glass in RGB and has `max_dolp` 0, so roughness is not a substitute for a polarized BSDF. `mean |S3| / S0` is 0 for every material: one specular bounce of unpolarized lamp light does not make circular polarization.

`measured_polarized` is not in this comparison. It loads Baek et al. 2020 isotropic Mueller pBRDFs from `.pbsdf` files, and only in `*_spectral_polarized` variants. It is not a general measured-BRDF importer. Some non-`_ad` variants have crashed on it ([mitsuba3#959](https://github.com/mitsuba-renderer/mitsuba3/issues/959)). Those files were not downloaded.

Recommendation from this table: keep the closed dielectric shell for the transparent film. Use `pplastic` if the object is later treated as opaque coated plastic. `thindielectric` does not polarize, so a single-sheet thin film is not the polarized path. The numbers match the decision rule, so the choice is not left open.

The chart is `outputs/phase2/comparison.png`. Contact sheets are under `outputs/phase2/<material>_llvm_ad_spectral_polarized/`.

## Phase 3

`uv run phase3` writes a 32-frame proof set to `outputs/phase3/`. It is not part of `uv run phase0`, and it is not the Phase 4 training set. The camera stays fixed. Seed 0 draws the ball's `x` and `z` from `[-0.35, 0.35]`, a ceiling-light shift from `[-0.25, 0.25]`, and a radiance scale from `[0.5, 1.5]` around 18. Frames 0–23 are the dielectric shell (thickness 0.05, IOR 1.5). Frames 24–31 are one `pplastic` sphere and no shell. The same seed rebuilds the same records.

Each frame is 256×256 on `cuda_ad_spectral_polarized` (Stokes, 64 samples) and `cuda_ad_spectral` (`specfilm`, 256 samples). The Stokes EXR holds linear RGB plus `S0`–`S3`. The cube uses the same eight bands as Phase 1, 37.5 nm wide from 400 to 700 nm, named `band_01_400` through `band_08_662`. `specfilm` is not combined with the Stokes integrator.

The film mask is 1 where the first-hit `shape_index` is the outer sphere named `film-shell`. That AOV uses a box filter and 1 sample, so the indices stay whole numbers. A `pplastic` frame has no `film-shell`, so its mask is all zeros. `mask.png` is black for 0 and white for 1. `n_film_pixels` in `params.json` is the count of 1s. The mask render uses `cuda_ad_spectral`. If that AOV fails, the frame is rendered again with `llvm_ad_spectral` and the log says `status=fallback`. A failed Stokes or cube render stops the job. It is not replaced with a CPU image.

`outputs/phase3/manifest.json` lists every frame. `outputs/phase3/index.png` is the 32 display-RGB images. `frames/000/stokes/contact_sheet.png` and `frames/024/stokes/contact_sheet.png` are the full Stokes sheets for one film frame and the first `pplastic` frame. `max_dolp` is still the maximum of `sqrt(S1^2 + S2^2) / S0` over pixels with `S0 > 1e-4`.

Phase 4 does not start here. Isaac Sim Simple Room stays later.

## Phase 4

`uv run phase4` reads `outputs/phase3` and writes `outputs/phase4`. It does not render new frames. The task is the film mask: a pixel is film where the first hit is `film-shell`. `pplastic` masks are all zeros. There are no real captures, so this is not a sim-to-real result.

The split is by frame. Train is `000`–`019` and `024`–`027`. Test is `020`–`023` and `028`–`031`. The color threshold and the logistic models are fit on the training frames only.

IoU is computed per frame as intersection / union of the prediction and `mask.png`. If both are empty, IoU is 1. The headline is the mean of the eight test frames. Film and `pplastic` means are reported separately because an empty `pplastic` frame scores 0 as soon as any pixel is called film. Pixel accuracy is not used.

The color baseline is `R / (R+G+B)` where the sum is above `1e-4`, else 0. The prediction is feature `>=` threshold. The threshold is the smallest value in `{0, 0.01, ..., 1}` that maximizes mean training-frame IoU. On this set that value is 0.44. The film ball is brown and the `pplastic` ball is gray, so this cue can reject the gray ball without using polarization.

The other three methods are logistic regression on the linear EXR values, with `class_weight="balanced"` and `random_state=0`. Features are standardized by the training-pixel mean and NumPy standard deviation (divide by N). A pixel is film when the predicted probability is above 0.5. DoLP is `sqrt(S1^2 + S2^2) / S0` where `S0 > 1e-4`, else 0, after averaging the three spectral copies. The cube channels are the linear `band_01_400` through `band_08_662` radiances. AoLP and S3 are not features. Training film pixels are 109642 of 1572864, a fraction of 0.0697, which is why the balanced weights are on.

Test mean frame IoU:

| Method | All 8 | 4 film | 4 pplastic |
| --- | ---: | ---: | ---: |
| color threshold 0.44 | 0.927612 | 0.855224 | 1 |
| RGB | 0.222653 | 0.445305 | 0 |
| RGB + DoLP | 0.453277 | 0.906553 | 0 |
| RGB + DoLP + cube | 0.445926 | 0.891851 | 0 |

These are the means in `metrics.json`, not rounded further. Training means are close to these test means, so the gap is not a split accident. DoLP raises film-frame IoU from 0.445305 to 0.906553. The eight-band cube does not improve on DoLP. The color threshold still wins the eight-frame mean because it leaves every test `pplastic` frame empty. The balanced logistic models call part of the gray ball film, so those frames score 0. `outputs/phase4/overlays.png` shows that on frame `020` and frame `028`. `outputs/phase4/comparison.png` is the bar chart. The full table, including each frame, is `outputs/phase4/metrics.json`.

Isaac Sim Simple Room stays later.

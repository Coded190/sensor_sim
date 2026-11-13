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

## Partial polyethylene film

`uv run film-sample`, `uv run film-dataset`, `uv run film-train`, and `uv run film-sfp` are separate from `uv run phase0`. They compare two U-Nets on a partial dielectric cap. Model A reads linear RGB. Model B reads that RGB plus four ideal linear analyzers. The geometry mask is the label. DoLP, AoLP, and S3 are saved and are not inputs.

Blender is not in apt without a password on this machine. The binary in use is the official Blender 4.5.9 tarball at `tools/blender/blender-4.5.9-linux-x64/blender`, which is gitignored. `assets/blender/create_film_scene.py` runs with `--background` and writes `assets/blender/film_ball.blend`. Mitsuba loads the exported OBJ meshes. The Mitsuba Blender add-on is not used.

The current scene is an open-front box, Z-up, in meters. The camera is at `(0, -0.55, 0.32)`, looking toward +Y, with a 55° field of view. From that view the left wall (`-X`) is diffuse red `(0.63, 0.06, 0.04)`, the right wall is diffuse green `(0.15, 0.45, 0.09)`, and the back wall is diffuse blue `(0.15, 0.25, 0.65)`. The floor and ceiling are gray diffuse reflectance 0.8. The front wall is omitted. A 0.12 m by 0.12 m pedestal, 0.04 m tall and diffuse yellow `(0.76, 0.64, 0.08)`, lifts the ball. The YAML radius 0.05 m is still the outer film surface. The plastic sphere stops 20 µm inside that, the cap covers 0.45 of the sphere, and the cap axis points toward the camera. The shell is smooth dielectric, interior IOR 1.50, exterior IOR 1.0, roughness 0. The ball is diffuse reflectance 0.5. These material numbers are provisional. Beside the pedestal, on the green side, is the Stanford `dragon_vrip_res4` reconstruction from the Stanford 3D Scanning Repository (Graphics Laboratory, August 1996), scaled to 0.12 m tall and shaded diffuse red `(0.70, 0.05, 0.04)`. The scan is stored in `assets/dragon/`. `lighting.intensity` is Mitsuba area-emitter radiance in W/sr/m². The lamp is a 0.15 m ceiling rectangle at radiance 10. Blender stores the matching power, π · width · height · intensity watts. OBJ files include one averaged normal per vertex. The blend assigns a Principled transmission shader to the film. That shader is a viewport stand-in. Blender does not run Mitsuba's polarized dielectric. The viewport uses AgX at exposure +2. The PNG uses `mapped = (4 * linear) / (1 + 4 * linear)` and then `clip(mapped, 0, 1) ** (1/2.2)`. The EXR stays linear. The display frame is `outputs/film/sample_000001/rgb.png`. On that frame the geometry mask has 5330 film pixels and mean film S0 is 0.00922608.

The earlier lamp at `(1, 1, 2)`, 0.2 m by 0.2 m, left mean film S0 at 0.00318666. With the lamp beside the camera, mean film S0 is 0.0602642. A bare diffuse ball in that same light has median linear radiance about 0.51, and the walls about 0.73. The filmed ball's median is about 0.041. Normal-incidence Fresnel reflectance at IOR 1.5 is `((1.5-1)/(1.5+1))^2 = 0.04`, so the cap is mostly a dark reflection of the room plus a small lamp image. Raising the radiance scales the ball and the walls together, and the walls reach 1 first. The 12-image dataset and the U-Net numbers below were rendered with the earlier lamp and have not been regenerated. `elapsed_s` is the wall time of `mi.render` through the NumPy copy. The 512×512 Stokes frame, 64 samples, took 0.918 s on `llvm_ad_spectral_polarized`. No CUDA variant is called.

The film mesh is 160 rings by 512 segments around the axis. A coarser cap let the ball poke through the triangle centers, because the sagitta was larger than the 20 µm thickness, and the mask came out speckled. At this density the sagitta is about 1 µm. On the earlier close-up camera that mask had 23951 film pixels and 640 edge pixels, one contour. The Cornell frame above is the same mesh seen from farther away, so the pixel count is 5330.

Stokes channels are read by name and the three spectral copies are averaged. The metadata records that this is a spectral render with constant IOR, not one wavelength. The same frame on `llvm_ad_mono_polarized` has mean absolute DoLP difference 0.00031533794018152845 on pixels where both S0 values clear 1e-4. Analyzer images use

- I0 = (S0 + S1) / 2
- I45 = (S0 + S2) / 2
- I90 = (S0 − S1) / 2
- I135 = (S0 − S2) / 2

with the Stokes vector in Mitsuba's camera frame. Reconstructing S0, S1, and S2 from those four images matches the source to 5.551115123125783e-17. The four analyzers do not measure S3. S3 is stored and its mean absolute value on valid pixels is 0 on this single-bounce dielectric. DoLP on the film reaches 0.999682437531784. The mean DoLP over the 262144 pixels with S0 > 1e-4 is 0.056554749001624674. The same film pixels on an unfilmed control reach max DoLP 0. DoLP is `sqrt(S1^2 + S2^2) / S0` where S0 > 1e-4, else 0. AoLP is `0.5 * arctan2(S2, S1)`, stored as 0 where DoLP < 0.05 or S0 ≤ 1e-4.

The mask is the first-hit `shape_index` of `film-shell`, box filter, 1 sample, same camera as the EXRs. `film_mask.png` stores uint8 values 0 and 1, not 0 and 255. The first sample's overlay is `outputs/film/sample_000001/overlay.png`.

The dataset is 12 images in `outputs/film/dataset`. Train has 7, validation 2, test 3. Views of one scene id stay in one split. Sphere coverages 0.15 and 0.75 are test-only. The ellipsoid, scale `(1, 0.75, 1.25)`, is test-only. Validation checks shapes, finite values, mask values, and that split rule.

Both networks are `FilmUNet`: four 2×2 pools, base width 32, channels 32, 64, 128, 256, bottleneck 512, two 3×3 convolutions with batch norm and ReLU in each block, skip concatenation, and a 1×1 logit. A parameter is one trained scalar, including batch-norm scale and shift. Model A has 7765985 parameters. Model B has 7767137. The difference is the 1152 weights in the first convolution, `(7 − 3) × 32 × 9`.

Training is CPU-only, PyTorch 2.15.0+cpu, `CUDA_VISIBLE_DEVICES` empty. Adam, learning rate 0.001, betas 0.9 and 0.999, weight decay 0, batch size 2, 40 epochs, seed 0, no augmentation. The loss is half BCE-with-logits and half Dice, with Dice smooth constant 1. Inputs are standardized by the training-pixel mean and NumPy standard deviation (divide by N). The film fraction on the training pixels is 148299 / 1835008 = 0.080817, above 0.05, so BCE has no positive-class weight. A prediction is film when the sigmoid probability is above 0.5. The test checkpoint is the epoch with the highest validation mean IoU: epoch 3 for Model A (validation IoU 0.897428) and epoch 10 for Model B (validation IoU 0.924666). Later epochs keep lowering the training loss and the validation IoU falls toward 0, so the last epoch is not the reported model. One epoch takes about 5 s.

IoU is TP / (TP + FP + FN) per image. Precision is TP / (TP + FP). Recall is TP / (TP + FN). An image with no positive label and no positive prediction scores 1. A zero denominator otherwise scores 0. The headline is the mean of the 3 test images. The standard deviation is NumPy's, divide by N. With 3 images that std is a description of this set, not a sampling error from a large test.

| Model | Input | IoU mean | IoU std | Precision mean | Recall mean |
| --- | --- | ---: | ---: | ---: | ---: |
| A | RGB | 0.9230452178044396 | 0.02631436004657484 | 0.9735802181652985 | 0.9468904242255695 |
| B | RGB + I0, I45, I90, I135 | 0.9010834984500997 | 0.056237333366255485 | 0.9061262449672948 | 0.9943124833787476 |

Per test image, IoU / precision / recall, copied from `metrics.json`:

| Sample | Scene | A | B |
| --- | --- | --- | --- |
| sample_000010 | sphere coverage 0.15 | 0.9452346041055718 / 0.9632424355621965 / 0.9806054152722847 | 0.8236547014972123 / 0.8236547014972123 / 1.0 |
| sample_000011 | sphere coverage 0.75 | 0.8860779200560464 / 0.9679421768707483 / 0.9128673964473315 | 0.9240644119965467 / 0.9353343465045593 / 0.9871285937687959 |
| sample_000012 | ellipsoid coverage 0.45 | 0.9378231292517006 / 0.9895560420629509 / 0.9471984609570923 | 0.9555313818565401 / 0.9593896869001125 / 0.9958088563674465 |

On these three images, adding the analyzers does not raise mean IoU. It raises recall and lowers precision. The low-coverage sphere is the image that drops: Model B's recall is 1 and its precision equals its IoU, so the error is extra film around the cap. DoLP on that frame is bright at the silhouette of the whole ball, which is a larger region than the 0.15 cap. The high-coverage sphere and the ellipsoid both gain IoU. `outputs/film/eval/comparison.png` is the bar chart. `outputs/film/eval/sample_000010_sphere_cov15.png` and `sample_000012_ellipsoid_cov45.png` show RGB, the four analyzers, DoLP, AoLP, the mask, both predictions, and the error maps. The full numbers are `outputs/film/eval/metrics.json`.

This is a 12-image CPU probe. It is not a sim-to-real result. The film is a smooth constant-IOR shell, the analyzers are noiseless, and S3 is not an input.

## Shape from polarization

`uv run film-sfp` renders two views of two scenes with `cuda_ad_spectral_polarized` and fits camera-frame shading normals. Mean angular error is the mean, over pixels in one region with S0 > 1e-4 and a unit normal, of `arccos(clip(dot(prediction, truth), -1, 1))` in degrees. The truth is the Mitsuba shading normal `nn`, rotated into the camera and pointed back at the camera. View A is the only training view. View B is a sideways camera of the same objects. The loss is the mean of `1 - dot(prediction, truth)` on object pixels. Wall pixels are left out of the loss. Training is 20 epochs, 4 steps per epoch, batch 4 crops of 256², Adam, learning rate 0.001, seed 0, CPU. The three inputs are Stokes plus DoLP plus `cos(2 AoLP)` and `sin(2 AoLP)`, the four ideal analyzers, and linear RGB. Each also gets the per-pixel view direction. AoLP is `0.5 * arctan2(S2, S1)`, stored as 0 where DoLP < 0.05.

The standard scene is one opaque `pplastic` ball, interior IOR 1.5, exterior IOR 1.0, diffuse reflectance 0.5, Beckmann alpha 0.1, in a gray room of reflectance 0.6. The ball covers 111091 / 262144 = 0.4238 of view A and 86317 / 262144 = 0.3293 of view B. Max DoLP on the ball is 0.819473 on view A. The Fresnel baseline uses the diffuse branch, so the azimuth equals AoLP, and it takes the smaller zenith when the curve has two matches. The oracle also tries azimuth + π and keeps the candidate closer to the truth. That oracle is not the baseline.

| Model | View A ball | View B ball |
| --- | ---: | ---: |
| Stokes | 4.205 | 7.245 |
| Analyzers | 3.911 | 8.776 |
| RGB | 3.589 | 8.149 |
| Fresnel diffuse | 61.698 | 60.354 |
| Fresnel oracle | 39.658 | 38.723 |

On this ball, RGB matches the polarization networks on the held-out camera. The fit is real, and it does not show that polarization is what carried the normal. The closed-form Fresnel error stays near 60°.

The Cornell scene keeps the filmed ball and replaces the dragon with smooth copper (`material: Cu`). Dragon max DoLP on view A is 0.405396, above the 0.2 gate. The Fresnel baseline uses the specular branch on the film only, azimuth = AoLP + π/2, and is not applied to the copper. The visible diffuse ball is the sliver past the cap: 239 pixels on view A and 354 on view B.

| Model | View | Film | Dragon | Ball | Objects |
| --- | --- | ---: | ---: | ---: | ---: |
| Stokes | A | 6.555 | 20.558 | 10.244 | 15.482 |
| Stokes | B | 7.212 | 24.922 | 13.308 | 19.424 |
| Analyzers | A | 11.299 | 29.064 | 15.022 | 22.607 |
| Analyzers | B | 8.307 | 33.101 | 17.751 | 25.429 |
| RGB | A | 17.482 | 32.710 | 24.419 | 27.244 |
| RGB | B | 21.371 | 36.904 | 27.803 | 32.111 |
| Fresnel specular | A | 65.404 |  |  |  |
| Fresnel specular | B | 64.232 |  |  |  |
| Fresnel oracle | A | 39.053 |  |  |  |
| Fresnel oracle | B | 38.765 |  |  |  |

On the held-out Cornell camera, Stokes stays ahead of RGB on the film (7.212° against 21.371°) and on the copper dragon (24.922° against 36.904°). These are two views of one scene, not the 21 held-out images in Ba et al. (ECCV 2020), whose learned model reported 18.5° and whose physics baselines sat between 41.4° and 49.0°. The numbers are in `outputs/film/sfp/summary.json`. Display frames and error images are under `outputs/film/sfp/standard/` and `outputs/film/sfp/cornell/`.

## Held-out filmed shapes

`uv run film-sfp` runs the two experiments below. `uv run film-sfp 01` and `uv run film-sfp 02` run one of them. The numeric tables are written by that command into `outputs/film/sfp/01_cornell_film/summary.json` and `outputs/film/sfp/02_heldout_shapes/summary.json`. This checkout does not contain those tables. The renders and the training are meant to run on a machine whose GPU is free. The previous `outputs/film/sfp/summary.json` is left as it is.

Both experiments use `cuda_ad_spectral_polarized`, 512², 64 samples, seed 0, and a maximum path depth of 8. The lamp is the near ceiling rectangle and it is unpolarized. There is no polarizer and no quarter-wave plate. Experiment 01 keeps the Cornell lamp from `configs/film/scene.yaml` (radiance 10). Experiment 02 uses the gray-room rectangle at radiance 18 and aims it at a fixed point, so the lamp does not turn toward each object. A polarized source is a later experiment.

The loss, the error PNGs, and every reported angle use the outer film only. A pixel counts when its shape index is the film shell, S0 is above 1e-4, and the shading normal has unit length. Walls, the inner diffuse solid, and the copper mesh are omitted. The copper mesh stays in the Cornell picture. The summary has no key for it, and the error PNGs leave those pixels blank.

Mean angular error is the mean of `arccos(clip(dot(prediction, truth), -1, 1))` in degrees on those pixels. The median is the median of the same angles. The three fractions are the share of those angles strictly below 11.25°, 22.5°, and 30°. The comparison is `<`, the same cutoff SfPUEL's `normal_criterion.py` uses. Experiment 02 reports each test shape and the unweighted mean of the ten shape scores. A larger film does not count more than a smaller one. The Fresnel baseline is the specular branch on the film, azimuth = AoLP + π/2, and it keeps the smaller zenith. The oracle, which also tries azimuth + π, is labeled separately and is not the baseline. S3 is saved in `measurement.npz` and drawn as `s3.png`. The summary reports the mean of `|S3| / S0` on the same film pixels. S3 is not a network input.

Training starts after that experiment's renders. If `torch.cuda.is_available()` and `torch.cuda.mem_get_info` reports at least 8 GiB free, training uses that GPU. Otherwise it uses the CPU and the log says why. Nothing else on the machine is killed. The mask trainer is still CPU-only: `src/sensor_sim/film/train.py` clears `CUDA_VISIBLE_DEVICES` before it imports Torch, and the shape job does not import that module.

`uv.lock` installs the CPU build of Torch 2.15. On the GPU machine, after `uv sync` and before `uv run film-sfp`, install the CUDA 13.2 wheel that matches this Torch release:

```bash
uv pip install --reinstall torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
```

Do not run `uv sync` again before the job, or it will put the CPU wheel back. The driver has to support CUDA 13.2. Torch 2.15 does not publish a cu128 wheel. The log line `sfp stage=device selected=` records which device the run actually used.

The schedule is 20 epochs, Adam, learning rate 0.001, betas 0.9 and 0.999, weight decay 0, seed 0. Experiment 02 draws one 256² crop from each of the 50 training images per epoch and groups them in batches of 4. The last batch has the remaining 2 images. Experiment 01 has only view A, so each epoch draws 4 crops of that one frame and the batch stays 4. A single crop would make BatchNorm see a batch of 1. Every crop is placed so that it contains a film pixel. Inputs are standardized by the mean and NumPy standard deviation (divide by N) of the training film pixels. Evaluation runs the network in eval mode on the full 512² frame, so BatchNorm uses its running statistics.

### 01_cornell_film

`outputs/film/sfp/01_cornell_film/view_a/` and `view_b/`. This is the stress scene: colored walls, the filmed ball, and the copper mesh in the picture. Training uses view A's film pixels. View B is not an input to the optimizer. The meshes are built in Python from `configs/film/scene.yaml` and `assets/dragon/dragon_vrip_res4.ply`, then written under the output directory. Blender is not required, and the exported OBJs are not part of the git checkout.

### 02_heldout_shapes

`outputs/film/sfp/02_heldout_shapes/train/001_sphere/` through `050_bumpy/`, and `test/001_ellipsoid/` through `010_cone/`. The room is gray. One object is in each frame, and the camera looks at that object. Seeds are disjoint. The test split has no spheres. A render raises if the film covers less than 0.15 of that 512² frame. The Cornell views are not mixed into this experiment.

The 50 training shapes are 4 spheres, 8 ellipsoids, 8 rounded boxes, 6 capsules, 6 tori, 6 cylinders, 6 cones, and 6 bumpy ellipsoids. The 10 test shapes are 2 ellipsoids, 2 rounded boxes, 2 bumpy ellipsoids, and one each of capsule, torus, cylinder, and cone. Each instance has its own seed, a bounding radius from 3 cm to 8 cm, a yaw about Z, and a floor position in the middle of the room. The film is a closed dielectric shell, IOR 1.5, roughness 0, offset 20 µm along the mesh normals. The solid inside is diffuse and sits 10 µm inside the base surface. Cylinder and cone rims are fillets, and the cone tip is round, so the offset shell stays closed. The rounded-box exponent stays below 3 and the bumpy wave goes to zero at the poles, so the chord sagitta stays under the 20 µm thickness. The torus tube is thick enough that a side view can cover 0.15 of the frame. `test_contact.png` in the experiment directory is the ten test RGBs beside their Stokes error maps.

Every render folder gets `rgb.png`, `normal_gt.png`, `dolp.png`, `aolp.png`, and `s3.png`. After training it also gets `normal_stokes.png`, `error_stokes.png`, and the same pair for the analyzer network, the RGB network, and the specular Fresnel baseline.

SfPUEL's archives were not downloaded. The training zip is about 20,000 Mitsuba 2 environment-map renders of bare objects, including metal, with no polyethylene film. The test archive stores four polarizer PNGs as relative 8-bit or 16-bit intensities, with no linear Stokes vector and no S3, and it scores an object mask. Those images are not a filmed-gripper test. The angle columns use their cutoffs so a later table can be read beside theirs. The pixels and the lighting remain ours.


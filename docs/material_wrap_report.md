# Material randomization and conformal plastic wrap

This is a qualification of two capabilities: a canonical material specification, and a separate plastic shell. It is not a dataset. Nothing here is marked `visually_reviewed`, `admitted`, or `registered`.

The gallery is `dataset/qualification/material_wrap_review/review/index.html`. Mitsuba RGB images are the evidence. The scientific scenes are built from `MaterialSpec` by the direct mapper in `src/sensor_sim/scene_gen/material_spec.py`. They are not renders of the Blender-exported materials. The add-on export is a separate transfer test.

## 1. Repository state

- Branch `main`
- HEAD `e5ef2a3403e87d9977eef70ed98ae0db80075df5`
- Uncommitted files are the modules, Blender scripts, tests, and this report listed in section 14. `dataset/` is gitignored.

## 2. Pinned exporter

Blender 4.5.9 LTS at `tools/blender-4.5.9-linux-x64/blender`. Mitsuba 3.9.1. Dr.Jit 1.5.0. Variant `cuda_ad_spectral_polarized`. Add-on commit `8848ded5e5df575f2cd0a9c10061d3f6b7137905`, vendored at `third_party/mitsuba_blender` and enabled as `bl_ext.user_default.mitsuba_blender`.

The inspected mapping is in `convert/export/materials/principled.py` and `basic.py`. Principled sockets become a Mitsuba `principled` BSDF. Glass roughness 0 becomes `dielectric`. Glass roughness above 0 becomes `roughdielectric` with Mitsuba `alpha` equal to the square of the Blender roughness. IOR 1 and roughness 0 becomes `thindielectric`. Glossy becomes `conductor` or `roughconductor`. `convert/export/mesh.py` writes one shape per material slot. For mesh objects it reads `b_object.data`, so Shrinkwrap and Solidify are applied before export.

## 3. Transfer results

The probe is `assets/blender/material_transfer_probe.py`. The XML is `dataset/qualification/material_wrap_review/transfer/transfer.xml`. Comparison tolerance is `1e-4`. Each case has three flags. A finished XML file is only `export_supported`.

All 11 cases exported. All 11 had `parameter_transfer_verified` true for the sockets that were set:

- Principled base color, roughness, metallic, anisotropy, and transmission weight matched. IOR matched when transmission was above 0.
- Diffuse red reflectance matched.
- Glossy smooth and rough exports produced `conductor` and `roughconductor`. The Glossy color is not the Mitsuba preset name. The scientific conductor name `Al` is set only by `MaterialSpec`.
- Glass IOR 1.45 matched `int_ior`. Rough glass exported `alpha` as roughness squared. Exterior IOR is omitted by the Glass node. The scientific mapper sets `ext_ior` to 1.
- The image texture and the normal-map adapter are present in the XML.
- A cube with two slots exported both diffuse colors, red on the top and blue on the other faces.

`polarized_renderer_compatible` was measured by instantiating the direct mapper on `cuda_ad_spectral_polarized` and rendering a 96 by 72 probe at 4 samples. It is not inferred from the XML.

| Family | Direct construction | Polarized instantiation and RGB probe |
| --- | --- | --- |
| diffuse | yes | yes |
| pplastic | yes | yes |
| dielectric | yes | yes |
| roughdielectric | yes | yes |
| thindielectric | yes | yes |
| conductor `Al` | yes | yes |
| roughconductor `Al` | yes | yes |
| principled | yes | yes, on the plastic preset |

`thindielectric` loaded and rendered. It is a single interface. It is not a finite-thickness film and not an interference coating. No Stokes-sign or Fresnel check was run. Instantiation does not establish polarization accuracy.

## 4. Canonical specification

`src/sensor_sim/scene_gen/material_spec.py`, schema `material_spec_v1`.

Fields: schema version, material id, family, optional preset, `color_space=linear_rgb`, family parameters, texture hashes, seed, and the sampled values. Linear RGB is not a measured spectrum.

Roughness convention: scientific rough families store Mitsuba Beckmann alpha directly. Blender Glass and Glossy export the square of the 0–1 roughness into that alpha. Principled roughness is exported unchanged.

IOR is stored as interior and exterior. Exterior IOR for the scientific families is 1. Conductor presets are `Al`, `Cu`, and `Au`.

Validation is per family. A dielectric specification that contains `metallic` raises `MaterialSpecError`. The mapper raises on an unsupported family and does not substitute another BSDF.

Sampling ranges, all uniform, live in `SAMPLING` in that module. Diffuse reflectance is drawn from 0.15 to 0.80. Plastic alpha is drawn from 0.03 to 0.35. Dielectric IOR is drawn from 1.33 to 1.70. Those ranges avoid 0 and 1 reflectance.

Assignment and the four seeds are in `src/sensor_sim/scene_gen/material_assign.py`. The streams are geometry, placement, material, and wrap.

## 5. Families and ranges

The families above are the initial vocabulary. Presets used in the gallery include `matte_gray` reflectance `(0.50, 0.50, 0.50)`, `matte_red` `(0.70, 0.12, 0.10)`, `matte_blue` `(0.12, 0.22, 0.65)`, plastic alpha `0.04` and `0.28` with the same diffuse color, dielectric IOR `1.45` and `1.60`, and aluminum.

## 6. Polarized compatibility

Every probed family in the table in section 3 instantiated on `cuda_ad_spectral_polarized` and wrote an RGB PNG under `dataset/qualification/material_wrap_review/polarized/`. No family was replaced after a failure. No LLVM fallback was used.

## 7. Shrinkwrap

`assets/blender/build_plastic_wrap.py` builds the shell in Blender 4.5.9.

- Partial: a grid over the target, `SHRINKWRAP` with `NEAREST_SURFACEPOINT` and a positive offset, then `SOLIDIFY`.
- Full: a copy of the target, inflated, `SHRINKWRAP` to the nearest surface with the offset, optional `DISPLACE` wrinkles, then `SOLIDIFY`.

Modifiers are applied before the OBJ and the Mitsuba export. The target file hash was unchanged.

The full-shell PLY matches the applied vertices after the measured axis map `(x, y, z) -> (x, -z, y)`, with position coverage 1.0. Face indices are not identical: the exporter triangulates and splits vertices by normal. Passing `axis_forward=Y` and `axis_up=Z` did not leave the partial-sheet coordinates equal to the applied OBJ, so partial PLY coverage is 0 and those exports are not called a match.

## 8. Wrap results

Geometry checks use minimum shell-to-target clearance and the median distance from outer-face centroids to the inner faces. The requested offset is 0.004 m and the requested thickness is 0.002 m. Clearance must stay within 0.0015 m of the offset. Thickness must stay within 0.001 m of the request.

| Target | Partial | Full smooth | Full wrinkle 0.0015 m |
| --- | --- | --- | --- |
| box | unsupported: median thickness 0.048 m | pass, clearance 0.00346 m, thickness 0.00115 m, PLY coverage 1 | pass |
| sphere | unsupported: clearance 0.00157 m | pass, clearance 0.00578 m, thickness 0.00194 m, PLY coverage 1 | pass |
| apple | geometry pass, clearance 0.00607 m, thickness 0.00199 m; PLY coverage 0 | pass, clearance 0.00499 m, thickness 0.00194 m, PLY coverage 1 | pass |

Partial success is not claimed for the box or the sphere. The apple partial shell is a separate mesh. Its add-on PLY did not match the applied OBJ, so the gallery image for that sheet is the applied OBJ loaded by the direct Mitsuba path, not the exported PLY.

## 9. Film specification

`src/sensor_sim/scene_gen/plastic_wrap.py`, schema `wrap_spec_v1`.

The shell is a finite-thickness mesh. It is not a glossy coat and not an interference coating.

- Outer interface: air to plastic. `int_ior=1.5`, `ext_ior=1`. The normal points out of the film into the air. Mitsuba's convention is that `int_ior` is the medium opposite the geometric normal.
- Inner interface: intended to be plastic to substrate. The substrates in this suite are diffuse, which is a surface BSDF, not a volumetric IOR. The measured clearance is a few millimeters, so the medium outside the inner face is air. The inner interface is therefore film-to-air. `optical_film_to_substrate` is `not_validated` for every shell that passed. This is not a validated finite-thickness film-to-substrate model.
- Sidewalls of a partial sheet are assigned the same air-to-plastic dielectric. That assignment is explicit in `interfaces.sidewall`.

`thindielectric` remains the separate single-interface comparison.

Face groups are split by the normal's alignment with the direction away from the target center. Outer, inner, and sidewall faces are written as separate OBJ shapes and given separate ids. The substrate keeps its own mesh and its own diffuse material.

## 10. Gallery

`dataset/qualification/material_wrap_review/review/index.html`

Material comparisons use one sphere, one camera, one light, 640 by 480, 64 samples, seed 17521, and `cuda_ad_spectral_polarized`. Exposure is the sRGB gamma in `write_rgb_png`. Only the specification changes.

Visible differences in those PNGs:

- `family_diffuse.png` is a matte gray sphere. `reflectance_high.png` is the same sphere in red.
- `rough_low.png` is plastic with Beckmann alpha 0.04 and shows a sharp rectangular highlight. `rough_high.png` uses alpha 0.28 and the highlight is gone.
- `family_dielectric.png` shows the refractive caustic of IOR 1.45 rather than a diffuse body.
- `two_components.png` is a blue diffuse body with a separate aluminum lid.

Wrap rows show the unwrapped red target, the apple partial sheet, and the smooth and wrinkled full shells for the box, sphere, and apple. The sphere shell shows a dark rim and a sharp highlight over the red substrate, on a gray ground. Close views are the `*_close.png` files.

## 11. Mitsuba evidence

Direct material renders and their JSON specifications are under `dataset/qualification/material_wrap_review/materials/`. Wrap renders are under `gallery_src/`. The transfer XML is `transfer/transfer.xml`. Polarized probes are under `polarized/`. Resolved BSDF dictionaries are stored next to the material PNGs.

## 12. Replay and hashes

`replay_spec` returns the saved parameters without drawing again. The unit tests check that a new material seed changes the sampled plastic parameters and does not change a target mesh hash, that a new wrap seed does not change the target id or the offset, and that a failed `store_result` does not replace a valid sample.

Passing shells record `target_unchanged` true. The source hash is the hash of the target OBJ before and after Blender. The shell hash is the hash of the applied vertex and triangle bytes.

## 13. Tests

- `tests/test_material_wrap.py` and the scene-generation modules listed in the run: 112 passed in 135.88 s.
- Full `pytest`: 238 passed, 2 failed, 192.58 s.

The two failures are the existing pair `tests/test_phase4.py::test_dolp_on_a_known_pixel_matches_polarization_maps` and `test_feature_stack_widths_and_order`. Both raise `FileStream` at `src/sensor_sim/phase4.py:128` because `outputs/phase3/frames/000/stokes/render.exr` is missing. That file was not created. The previous full run was 231 passed and the same 2 failures. The 7 new tests account for 238.

## 14. Files created

- `src/sensor_sim/scene_gen/material_spec.py`
- `src/sensor_sim/scene_gen/material_assign.py`
- `src/sensor_sim/scene_gen/plastic_wrap.py`
- `src/sensor_sim/scene_gen/material_review.py`
- `assets/blender/material_transfer_probe.py`
- `assets/blender/build_plastic_wrap.py`
- `tests/test_material_wrap.py`
- `docs/material_wrap_report.md`

Qualification output is under `dataset/qualification/material_wrap_review/`. No protected module was edited. No dependency was added. Pins were not changed.

## 15. Existing outputs

Dataset v1 remains 1533 files and 457,265,977 bytes. `assets/candidates/index.json` remains `081bf07291a4cc3f8f2af1fe27b8f8beb0164153100244aefa25df1ba209c157`. The locked room, quality, and tabletop bundle hashes checked by `tests/test_semantic_scenes.py` were not rewritten. `fixture_sphere` and `prop_apple` were read and not modified.

## 16. Limitations and next step

Independent gates:

1. Blender material construction: pass. The probe scene built the requested nodes.
2. Blender-to-Mitsuba export support: pass. All 11 cases produced a BSDF in the XML.
3. Parameter-transfer verification: pass for the requested sockets, with the documented omissions. Exterior IOR is not a Glass socket. Glossy color is not a conductor preset.
4. Direct Mitsuba BSDF construction: pass for the eight families.
5. `cuda_ad_spectral_polarized` instantiation: pass for every probed family. This is not a polarization-accuracy result.
6. Small RGB render completion: pass for those probes and for the gallery images.
7. Visible material-parameter variation: pass. Red versus gray, and alpha 0.04 versus 0.28, are visible at the fixed camera and light.
8. Partial-wrap geometry: pass for the apple only. The box sheet thickness was 0.048 m. The sphere sheet clearance was 0.00157 m. Those two classes are unsupported for partial coverage.
9. Full-wrap geometry: pass for the box, the sphere, and the apple, including the 0.0015 m wrinkle. The box shell thickness measured 0.00115 m against a 0.002 m request, inside the 0.001 m tolerance.
10. Outer and inner interface assignment: the face groups and BSDF ids are separate. The optical film-to-substrate interface is `not_validated` because the substrate is diffuse and the shell clearance is air.
11. Replay and target-mesh immutability: pass in the unit tests and in the wrap runs that completed.

The useful next step is a partial-sheet construction whose applied thickness stays at the requested value for a box, and a film-to-substrate interface that does not depend on an air gap. That is a different optical model from the one measured here. It should not be claimed until that measurement exists.

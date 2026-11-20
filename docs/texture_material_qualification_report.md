# Texture-aware materials and wrap qualification

The gallery is `dataset/qualification/material_wrap_review/textured/gallery/index.html`. Every caption names the renderer. Mitsuba RGB is the scientific evidence. Blender Cycles is the preview. The new packages are unregistered. Nothing was committed, and nothing was marked visually reviewed or admitted.

Renderer pin: Mitsuba variant `cuda_ad_spectral_polarized`, 192×144, seed 17531. Panel, apple, can, and widget images use 16 samples. Shell images use 64 samples. Blender Cycles uses 32 samples on CUDA. Stokes-sign, Fresnel-curve, and polarization-accuracy audits were not run. A finite RGB image is not a polarization result.

## 1. Repository state

Branch `main`, HEAD `e5ef2a3403e87d9977eef70ed98ae0db80075df5`. The protected module hashes in `tests/test_semantic_scenes.py` still match, including `room_generator.py`, `room_schema.py`, `task_replay.py`, and `tabletop_assets.py`. `assets/candidates/index.json` still matches its locked hash. Dataset v1 still has 1533 files and 457265977 bytes. `prop_apple/1.0.0/geometry/render.obj` and `prop_can/1.0.0/geometry/render.obj` were read and not rewritten. `indoor_mitsuba.py` was not edited. The textured packages live only under the qualification directory and are loaded by this harness.

## 2. Material schema and roughness

`MaterialSpec` remains `material_spec_v1`. A missing texture list still means the old scalar preset. Texture records have a role (`base_color`, `roughness_blender`, `alpha_mitsuba`, `normal_opengl`), a package-relative path, a color space (`srgb` or `non_color`), a resolution, a SHA-256, a UV transform, and an input class. An absolute path, a `local_review/` path, a missing file, or a hash mismatch raises. The three package manifests were checked and contain relative texture paths only.

Roughness has one stored meaning. Blender Principled roughness stays in the range 0–1. For the Beckmann plastic, glossy, and conductor preview mapping, Mitsuba alpha is the square of that roughness, and the inverse preview is the square root of alpha. The qualification source image stores Blender roughness: 51/255 = 0.2 on the red half and 204/255 = 0.8 on the blue half. Those values were squared once while writing `alpha_mitsuba.exr`, which stores 0.04 and 0.64. `to_mitsuba` binds that EXR and does not square it again. A roughness map without a resolved alpha map raises. Non-finite or out-of-range values raise and are not clamped. The unit test checks inputs 0, 0.5, and 1, and the image pixels 0, 0.5, 1, and 0.2.

Scalar IOR and the BSDF family are labeled assumed or preset parameters. The authored maps are labeled `authored_appearance`. They are not measured reflectance and not measured IOR. Spectral upsampling of an sRGB bitmap is the renderer’s color representation. Conductors reject a base-color bitmap. A pure dielectric rejects a diffuse base-color bitmap. `thindielectric` is not used as the film result.

## 3. Mesh UV

`read_obj` and `write_obj` still return and write geometry only. `read_mesh` and `write_mesh` keep face-corner position, UV, and normal indices, including a seam where one vertex has two texture coordinates, and including negative OBJ indices. Collision meshes omit UVs and the package provenance says so.

Overlap rejection uses pixel centers that are strictly inside a UV triangle. A shared edge is not an overlap. The unit test covers a seam round trip, a two-triangle quad that shares a diagonal, a real overlap, and island padding.

Texel density is the mean, over the three edges of every UV triangle, of texture-edge length in pixels divided by world-edge length in metres.

The panel is an explicit 1 m quad facing +Y. Its density on the 256×256 map is 256 px/m. Mitsuba loaded that OBJ through `tabletop_render._scene` with `vertex_texcoords=true` and `vertex_normals=true` (4 vertices, 2 faces).

The apple unwrap is a dominant-axis projection into six padded UV cells. It uses the packaged 478 vertices and the 504 source polygons, fan-triangulated to 952 triangles. The same positions come back from the OBJ round trip. A requested island padding of 0.02 failed because the measured gap was 0.00398 UV units. Validation then passed at padding 0, with no overlapping islands. Density is 772.65 px/m, which is 7.73 px/cm on this 256 map. This is a first-pass projection. It is a material and texture change on the packaged vertices, not a new sculpt and not mesh detail stored only in the normal map.

## 4. Mitsuba bitmaps and filters

`tabletop_render.mitsuba_bsdf` sends a `material_spec_v1` dictionary through `to_mitsuba`. Color bitmaps are not raw. Data bitmaps are raw. The plugin dictionary contains `filename`, `raw`, `filter_type`, and `wrap_mode`. Filter notes stay outside that dictionary. A non-identity UV transform is applied as `to_uv`.

On this pin, for both the sRGB base-color PNG (`raw=false`) and the alpha EXR (`raw=true`):

- `bilinear` loaded
- `nearest` loaded
- `trilinear` was rejected
- `ewa` was rejected

The rejection text is: `Invalid filter type "trilinear", must be one of: "nearest", or "bilinear"!` The same wording is returned for `ewa`. This pin does not silently fall back. The renders request `bilinear`. Trilinear and anisotropic filtering are not claimed.

The full panel BSDF, loaded from `_scene`, exposes `diffuse_reflectance.data`, `alpha.data`, and `normalmap.data`. The normal-off BSDF has no `normalmap` key, so that render uses the geometric normals stored on the mesh. `pplastic.alpha` was bound to the EXR and rendered. It was not skipped. The normal map is an OpenGL tangent-space shading perturbation. It is not geometric surface detail and it is not shape-from-polarization ground truth.

## 5. On/off Mitsuba renders

The panel pair keeps the mesh, camera, light, exposure, variant, seed, and sample count fixed. `write_rgb_png` applies sRGB gamma to the RGB channels of the Stokes film. The difference below is the mean absolute difference of those 8-bit PNG values over every pixel and every RGB channel. It is not a reflectance error and not a Stokes error.

The camera looks toward -Y. Image left is the blue, high-U side. That side’s source roughness is 0.8 and its alpha is 0.64. Image right is the red, low-U side, source roughness 0.2, alpha 0.04. A green marker sits in the low-U, high-V corner of the base-color map. The normal map tilts that same corner in the green channel and tilts the opposite corner in the red channel.

| Pair | Mean absolute 8-bit difference | What the image shows |
| --- | ---: | --- |
| Base color on versus off | 68.16 | On is blue on the left and red on the right. Off is one salmon color, the scalar diffuse reflectance. |
| Roughness map versus constant alpha 0.04 | 7.62 | The smooth red half matches (right-half mean RGB 214.08, 59.78, 55.48 versus 214.09, 59.71, 55.47). The rough blue half changes (left-half mean RGB 57.03, 69.83, 198.75 versus 33.80, 53.86, 195.86). |
| Normal map on versus off | 5.04 | The off render has no normal-map plugin. The left-half mean moves from 57.03, 69.83, 198.75 to 60.14, 73.41, 206.91. |

Blender Cycles renders the same panel OBJ, the same three images, and a tangent-space Normal Map node. The preview shows the same blue/red split and the green corner. Pixel equality with Mitsuba was not required and was not measured.

## 6. Blender preview and the pinned add-on

`preview_nodes` builds the Blender description from the same specification: sRGB base color into Base Color, Non-Color roughness into Roughness, and a Non-Color OpenGL normal through a tangent Normal Map into Normal. The Cycles script uses those sockets. Metallic is 0 and IOR is 1.45 for the plastic preview.

The pinned add-on probe was run again. Eleven export cases were verified. The node report records the image texture as sRGB on a Diffuse BSDF Color socket, and the normal texture as Non-Color through a Normal Map node into Principled Normal. Principled export does not match the direct polarized `pplastic` BSDF. The image and normal exports are recorded separately from that direct mapper. The scientific path remains `to_mitsuba`.

## 7. Component assignment

The widget has two components and two saved records. Geometry seed 11, placement seed 12, and wrap seed 13 stay fixed while the material seed changes.

| Material seed | Body | Lid | Body UV transform |
| ---: | --- | --- | --- |
| 41 | `pplastic` with the panel texture set | `roughconductor` | identity |
| 42 | `pplastic` with the same texture set | `conductor` | scale 2 |

Replay restores the texture list and the parameters. Changing the stored geometry seed to 99 and the wrap seed to 77, while keeping material seed 41, leaves the specification unchanged. The two OBJ files are written once. The Mitsuba images of seed 41 and seed 42 differ by a mean absolute 8-bit value of 19.22. The body keeps its mesh. The lid is a separate mesh translated by 0.21 m in Z. The two families are not layered on one surface.

## 8. Apple, can, and the Poly Haven reference

`prop_apple/1.0.0` is unchanged: 478 vertices, 504 polygons, bounds about 76.5 × 82.2 × 81.2 mm, scalar diffuse reflectance `[0.62, 0.14, 0.12]`, no textures. Its Mitsuba image is the flat red apple.

The unregistered textured apple reuses those 478 vertices and the 952 fan triangles. The Mitsuba and Blender Cycles images show the authored color, roughness, and normal maps on that silhouette. That comparison is a material and texture change.

Poly Haven Food Apple 01 is CC0, 7012 triangles, source dimensions 97.6 × 95.9 × 85.4 mm. The 1k hashes are in `summary.json`. The normal convention is OpenGL. Blender scaled it by 0.95110 so its height matches the packaged apple height of 0.081214 m, and used the same camera origin, target, and horizontal field of view. The Cycles image is an authored look benchmark. It is not measured reflectance and not measured IOR.

The existing can is conductor `Al`, rendered from its packaged mesh with no base-color bitmap. A constant fill of RGB `(0.55, 0.62, 0.7)` makes the metal silhouette visible because a smooth conductor otherwise reflects the black background. That fill is a lighting choice, not a measured illumination.

## 9. Plastic wrap

Wrap geometry, the separate film mesh, and the interface record are the existing ones. The substrate material seed is not an input to the shell mesh. The wrap seed changes the wrap request’s seed and does not change the substrate texture records. Outer faces are assigned as air-to-plastic dielectric with exterior IOR 1. Inner faces of an opaque substrate remain `not_validated_opaque_substrate`.

| Target | Mode | Geometry remeasure | Clearance | Thickness | Mitsuba RGB | Film optics |
| --- | --- | --- | ---: | ---: | --- | --- |
| Box | Full | Passed | 0.003464 m | 0.001155 m | Rendered | Not validated |
| Sphere | Full | Passed | 0.005776 m | 0.001938 m | Rendered | Not validated |
| Box | Partial | Failed the thickness check (measured 0.0481 m against the 0.002 m solid-shell test) |  |  | Rendered | Not validated |
| Sphere | Partial | Failed the clearance check (0.001572 m) |  |  | Rendered | Not validated |

The earlier `wraps/results.json` also records the box and sphere partial shells as geometry failures, and the two full shells as geometry successes. The four results for the textured box plus its shell are separate: the full-shell geometry check passed; the outer and inner assignments are the statuses above; Mitsuba constructed the dielectric and the textured substrate and wrote an RGB image; the film optics were not validated. The unwrapped textured box, from the same camera, shows the red and blue faces. The shell image shows the glass envelope and the area-light reflection. The gray rectangle under the shells is a probe backdrop so the dielectric has a background. These wrap figures use the exemption `controlled_material_probe`.

## 10. Tests

`pytest` from the repository root: 265 passed, 2 failed, in 207 seconds.

The two failures are pre-existing and are outside this change:

- `tests/test_phase4.py::test_dolp_on_a_known_pixel_matches_polarization_maps`
- `tests/test_phase4.py::test_feature_stack_widths_and_order`

Both stop in `phase4.py` because `outputs/phase3/frames/000/stokes/render.exr` is not on disk. `phase4.py` was not edited.

The texture tests cover legacy and seamed OBJ round trips, a shared-edge quad, overlap, island padding, missing and mismatched textures, the roughness pixels, portable package loading, component replay, seed independence, `pplastic.alpha` and `normalmap` consumption through `_scene`, and rejection of `trilinear` on this variant. The qualification script also copied a package to another directory, checked the manifest, and rendered it.

## 11. What this does not establish

The packages are not visually reviewed, admitted, or registered. Dataset generation was not run. Film optics are not physically validated. Polarization accuracy was not measured. Blender and Mitsuba were not required to match pixels. The normal map is shading only. The Poly Haven apple and the generated maps are authored appearance. The dominant-axis apple unwrap is a first pass, and its 0.02 UV padding request failed at a measured gap of 0.00398.

# Report: realistic object appearance for sensor_sim

This report is for an agent that cannot open the repository. It describes the code, the objects, the scene pipeline, a downloaded realistic reference apple, and the specific work required to make the project's objects look realistic. Paths are absolute on the machine `/home/cmoore/sensor_sim`.

## Goal

The project builds synthetic indoor scenes for shape-from-polarization research. Objects currently look flat because each mesh has one BSDF and one uniform color. The goal is realistic shading and surface detail on the objects that already exist, without inventing a second material system and without mixing two BSDF families on one object.

Two renderers must stay distinct in every image and note:

- Blender 4.5.9 Cycles or EEVEE is the file a person opens and orbits. Label those images `Blender Cycles` or `Blender EEVEE`.
- Mitsuba 3.9.1, variant `cuda_ad_spectral_polarized`, is the scientific renderer. Label those images `Mitsuba`. A Blender preview is not evidence that the Mitsuba scene has the same maps.

## Reference asset that was downloaded

A free realistic apple was downloaded from Poly Haven. It is public domain under CC0. No account was required. The author recorded by the API is Oliver Harries. The asset page is `https://polyhaven.com/a/food_apple_01`. The API info file is saved with the download.

Local directory, which is gitignored:

`/home/cmoore/sensor_sim/local_review/reference_food_apple_01/`

Published facts from `info.json`:

- Name: Food Apple 01
- Category: Food & Kitchen/Produce/Fruit
- Triangles: 7012
- Maximum texture resolution offered: 8192 by 8192
- Dimensions: 97.561 by 95.902 by 85.390 mm
- Texel density in the API: 38537.58. That is about 385 pixels per centimetre at the 8k maps. The files actually downloaded are the 1k set, so the density of these copies is about one eighth of that.
- Description from the publisher: realistic ripe red apple, skin blemishes, glossy surface, organic stem, natural color variation

Files downloaded at 1k:

| File | Size | Role |
| --- | --- | --- |
| `food_apple_01_1k.gltf` | 2705 bytes | glTF 2.0 material and mesh description |
| `food_apple_01.bin` | 170680 bytes | Positions, normals, UV coordinates, and triangle indices |
| `food_apple_01_1k.blend` | 445545 bytes | Blender file. Textures are external, not packed |
| `textures/food_apple_01_diff_1k.jpg` | 1024 by 1024 RGB | Base color. Blender color space in the blend file is sRGB |
| `textures/food_apple_01_rough_1k.jpg` | 1024 by 1024 RGB | Roughness. Blender color space is Non-Color |
| `textures/food_apple_01_nor_gl_1k.jpg` | 1024 by 1024 RGB | OpenGL tangent-space normal map. Blender color space is Non-Color |
| `textures/food_apple_01_ao_1k.jpg` | 1024 by 1024 RGB | Ambient occlusion. Present on disk. Not connected in the blend material |
| `textures/food_apple_01_arm_1k.jpg` | 1024 by 1024 RGB | Packed map. Present on disk. Not connected in the blend material |

The glTF was exported by Khronos glTF Blender I/O v1.6.16. One mesh, one material, double sided. The mesh primitive has `POSITION`, `NORMAL`, and `TEXCOORD_0`. Accessor counts: 4019 vertices, 21036 indices, which is 7012 triangles. Bounds from the position accessor, in the glTF file's coordinates: minimum about `[-0.0484, 0, -0.0458]`, maximum about `[0.0492, 0.0854, 0.0501]` metres.

The glTF material `food_apple_01` wires three textures and one scalar:

- `pbrMetallicRoughness.baseColorTexture` points at `textures/food_apple_01_diff_1k.jpg`
- `pbrMetallicRoughness.metallicRoughnessTexture` points at `textures/food_apple_01_rough_1k.jpg`
- `metallicFactor` is 0
- `normalTexture` points at `textures/food_apple_01_nor_gl_1k.jpg`
- There is no separate roughness scalar in the glTF. In glTF, roughness is the green channel of the metallic-roughness image. This JPEG is a grayscale roughness image, so red, green, and blue match.

The Blender file was opened in Blender 4.5.9. The mesh object is `food_apple_01`: 3504 vertices, 7012 polygons, one UV layer named `UVMap`, one material named `food_apple_01`. Node links that were actually connected:

- Image Texture of `textures/food_apple_01_diff_1k.jpg`, color space sRGB, into Principled BSDF Base Color
- Image Texture of `textures/food_apple_01_rough_1k.jpg`, color space Non-Color, into Principled BSDF Roughness
- Image Texture of the OpenGL normal, color space Non-Color, into a Normal Map node, then into Principled BSDF Normal
- Texture Coordinate UV output into a Mapping node, then into all three image Vector inputs

Scalar values read from that Principled BSDF:

- Metallic = 0
- IOR = 1.45
- Roughness socket is linked, so the 0.5 default is unused
- Subsurface Weight = 0
- Subsurface Radius = `[1.0, 0.2, 0.1]` and Subsurface Scale = 0.05, but the weight is zero, so subsurface is off
- Coat is not linked. Its weight was not driving the material

So this realistic apple does not rely on subsurface, a coat, an ambient-occlusion connection, or a metallic map. The visible realism comes from the mesh detail, the UV map, the base-color image, the roughness image, the normal image, metallic 0, and IOR 1.45.

Poly Haven also publishes DirectX normals (`nor_dx`) and 2k, 4k, and 8k copies. The blend and the glTF use the OpenGL normal (`nor_gl`). Blender's tangent-space normal is the OpenGL orientation, green channel up. Do not connect `nor_dx` without flipping the green channel.

## What the project's apple has instead

Package directory:

`/home/cmoore/sensor_sim/assets/candidates/prop_apple/1.0.0/`

`asset.json` says:

- `asset_id`: `prop_apple`
- `version`: `1.0.0`
- Units metres, up axis Z, right handed
- Bounds about x `-0.0306` to `0.0459`, y `-0.0411` to `0.0411`, z `0` to `0.0812`
- 504 polygons, 952 triangles, profile `closed_solid`
- One material: `type` `diffuse`, `reflectance` `[0.62, 0.14, 0.12]`, note `mitsuba_diffuse_depolarizes`
- No `textures` entry
- Geometry files are `geometry/render.obj` and `geometry/collision.obj`
- The OBJ writer emits only `v` and `f` lines. There are no `vt` texture coordinates and no `vn` normals

That single reflectance is one linear RGB color. The same three numbers are the color at every surface point. Linear means doubling the light doubles the stored number. It is not a photograph and it is not an sRGB file.

A Blender Cycles comparison of this mesh, before and after a procedural UV texture experiment, is in `/home/cmoore/sensor_sim/local_review/apple_blender_material/`. Those PNGs are Blender Cycles, not Mitsuba. The textured experiment added a generated base-color image, roughness image, and normal image, plus subsurface and a wax coat. The 952-triangle silhouette did not change. That experiment is a viewing test. It is not wired into Mitsuba and it did not modify the asset package.

## Software pins

- Repository root: `/home/cmoore/sensor_sim`
- Python environment: `/home/cmoore/sensor_sim/.venv`
- Blender: `/home/cmoore/sensor_sim/tools/blender-4.5.9-linux-x64/blender`, version 4.5.9 LTS
- Mitsuba 3.9.1 and Dr.Jit 1.5.0
- Scientific variant: `cuda_ad_spectral_polarized`
- GPU used for Mitsuba: RTX 5080
- Coordinate system for indoor scenes: Z up, metres, right handed
- Room interiors use the southwest finished-floor corner as the origin and finished floor at z = 0
- `.gitignore` ignores `.venv/`, `outputs/`, `tools/`, `dataset/`, and `local_review/`

## Package and material code

Asset packages are defined in `src/sensor_sim/scene_gen/asset_package.py`. A package is a directory `asset_id/version/` with `asset.json` and the files named by that manifest. The manifest may include a `textures` key. Current generated objects leave it unused. The package hash is the SHA-256 of a canonical JSON document. Adding a texture file changes the hash. Lifecycle flags `visually_reviewed`, `admitted`, and `registered` must stay false unless a person sets them.

Mesh IO is `src/sensor_sim/scene_gen/mesh_io.py`. `write_obj` writes positions and face indices only. `read_obj` ignores texture and normal indices. Any UV work has to extend both functions, or the coordinates will be dropped the next time a mesh is saved.

Material specifications are `src/sensor_sim/scene_gen/material_spec.py`, schema `material_spec_v1`. `make_spec` accepts a `textures` list and stores it. `to_mitsuba` never reads that list. It builds one BSDF from scalar parameters. Families and the fields that belong to each family:

| Family | Parameters | Meaning |
| --- | --- | --- |
| `diffuse` | `reflectance`, one linear RGB | Lambertian. Mitsuba diffuse has no roughness. It depolarizes |
| `pplastic` | `diffuse_reflectance`, `int_ior`, `ext_ior`, `alpha`, `distribution` | Rough plastic. Diffuse base under a dielectric coat. `alpha` is Beckmann RMS slope, not Blender's 0–1 roughness. `ext_ior` must be 1. Sampling ranges: reflectance 0.20–0.70, alpha 0.03–0.35, IOR 1.40–1.60 |
| `dielectric` | `int_ior`, `ext_ior` | Smooth glass or clear film |
| `roughdielectric` | those plus `alpha` and `distribution` | Frosted glass |
| `thindielectric` | `int_ior`, `ext_ior` | Thin sheet, used for plastic wrap |
| `conductor` | `material` | Smooth metal. Allowed names: `Al`, `Cu`, `Au` |
| `roughconductor` | `material`, `alpha`, `distribution` | Rough metal. Alpha sampling range 0.03–0.30 |
| `principled` | `base_color`, `roughness`, `metallic`, `anisotropic`, `spec_trans`, `ior` | Blender-like shader. Not the family to use for the scientific apple |

`to_mitsuba` raises if the Mitsuba type does not match the requested family. Do not substitute a different family when a parameter is missing.

`src/sensor_sim/scene_gen/material_assign.py` samples those scalar ranges from an independent material seed. Seeds are split into streams `geometry`, `placement`, `material`, and `wrap`. A texture implementation must keep those streams independent. Changing the material seed must not move geometry.

`src/sensor_sim/scene_gen/tabletop_render.py` function `mitsuba_bsdf` is the other converter, used for packaged furniture materials. It accepts `diffuse`, `dielectric`, `pplastic`, `conductor`, and `roughconductor`. Unknown types raise. It also has no bitmap path.

Blender transfer is a separate path. `assets/blender/material_transfer_probe.py` and `blender_recipe` in `material_spec.py` describe node sockets. The pinned Mitsuba Blender add-on commit is `8848ded5e5df575f2cd0a9c10061d3f6b7137905`. Do not describe a Blender export as the scientific material.

## Objects that exist

Catalog root for the small props: `/home/cmoore/sensor_sim/assets/candidates/`. Each item below is version `1.0.0` unless noted. They are procedural meshes with the same OBJ limits and, unless a later edit adds them, no UV maps and no texture images.

Props: `prop_apple`, `prop_bowl`, `prop_bowl_glossy`, `prop_bottle`, `prop_can`, `prop_can_rough`, `prop_chair`, `prop_crate`, `prop_crate_open`, `prop_cup`, `prop_marker`, `prop_phone`, `prop_plate`, `prop_small_box`, `prop_spoon`, `prop_tape_roll`, `prop_book`.

Fixtures: `fixture_sphere`, `fixture_cone`.

Structural pieces also packaged as candidates: `table_rect`, `shelf_two`, `room_floor`, `room_ceiling`, `room_wall_south`, `room_wall_north`, `room_wall_east`, `room_wall_west`.

Infinigen imports, not the procedural prop generator: `infinigen2_bowl`, `infinigen2_table`, `infinigen2_room_shape`, `infinigen_stable_desk`, `infinigen_stable_floor`, `infinigen_stable_shelf`.

Furniture for the six furnished rooms is built by `src/sensor_sim/scene_gen/interior_assets.py` and packaged under `dataset/qualification/diverse_interior_review/catalog/`. Those meshes are also OBJ files of boxes and simple parts with one diffuse or plastic color. Room types and the furniture they require are in `src/sensor_sim/scene_gen/interior_scenes.py`: kitchen, dining, living, bedroom, office, workshop. A lab scene reuses the workshop shell. There is no separate lab mesh.

## How a scene is assembled

There are two different generators. Do not mix them.

Dataset v1, `src/sensor_sim/scene_gen/generate.py` function `random_config`, places one to four analytic primitives on a ground plane. That file is hash-locked by tests. It is the old prototype. Do not add room shells or texture maps there.

The indoor generator is the one that needs realistic objects:

1. Build or load an asset package.
2. Choose a room kind. `src/sensor_sim/scene_gen/scene_context.py` draws without replacement from room, kitchen, dining, living, bedroom, office, workshop, and lab. `room` uses the dining shell. `lab` uses the workshop shell. A scene must have a floor, a ceiling, and south, north, east, and west walls, plus at least one context object that is not the test subject. A ground plane alone is rejected.
3. Place furniture and props. Support contact is measured by `src/sensor_sim/scene_gen/indoor_contact.py`. The allowed gap is 1 mm and the allowed penetration of the support plane is 0.1 mm.
4. `src/sensor_sim/scene_gen/placement_separation.py` function `require_outside` rejects a new object whose triangle-to-triangle distance to another object is below 4 mm, or whose box sits inside another mesh and whose centroid is inside that mesh. The room shell is not a blocker. The flat top of the support is not a blocker. Raised parts of the support mesh, such as pillows, are blockers. Tabletop placement (`tabletop_layout.py`), scene-quality placement (`scene_quality.py`), semantic placement (`semantic_composition.py`), and interior validation (`interior_scenes.py` `validate_layout`) call this check.
5. Materials are attached as one BSDF per mesh.
6. Plastic wrap, `src/sensor_sim/scene_gen/plastic_wrap.py`, builds a separate thin shell mesh around a target. It is geometry plus a dielectric material. It is not a texture.
7. Mitsuba renders through `src/sensor_sim/scene_gen/indoor_mitsuba.py`. That file is hash-locked. Call it. Do not edit it.
8. Review bundles are written under `dataset/qualification/`. Local handoff zips and blend files go in `local_review/`.

Scene JSON instances store `instance_id`, `asset_id`, `version`, `package_sha256`, a 4 by 4 row-major matrix, and sometimes a role. The matrix moves the mesh. Materials are read back from the package. They are not stored as textures on the instance.

## What is missing

Compared with the Poly Haven apple, each project object is missing:

1. A UV map on the render mesh.
2. A base-color image for dielectric and plastic objects.
3. A roughness image, or a documented reason that one scalar roughness is enough.
4. A tangent-space OpenGL normal image when the triangles do not already carry the small surface detail.
5. Enough triangles for the silhouette. The project apple has 952 triangles. The reference apple has 7012, plus a stem. Normal maps do not change the outline.
6. A Mitsuba path that can consume those images. `to_mitsuba` and `mitsuba_bsdf` drop them.
7. A Blender preview path that connects Image Texture nodes with the color spaces above. The current quality-check blend builder paints Principled sockets with constants.
8. UVs that survive `write_obj` and `read_obj`.

Subsurface and coat are available on Blender's Principled BSDF and were tried on the project apple. They are not what the downloaded realistic apple uses. Do not make them mandatory. Metal objects must not receive a base-color paint. Their color comes from `Al`, `Cu`, or `Au`. Diffuse objects that should show a highlight need a family change to `pplastic` or `roughdielectric`. Adding a roughness image to `diffuse` does nothing, because that BSDF has no roughness parameter.

Ambient occlusion is shipped by Poly Haven and is not connected in their Blender material. Do not add it to the scientific renderer. The path tracer computes occlusion.

## Files that must not be edited

Tests compare SHA-256 hashes. Editing these fails the suite:

- `src/sensor_sim/scene_gen/generate.py`
- `src/sensor_sim/scene_gen/placement.py`
- `src/sensor_sim/scene_gen/indoor_mitsuba.py`
- `src/sensor_sim/scene_gen/structural.py`
- `src/sensor_sim/scene_gen/room_generator.py`
- `src/sensor_sim/scene_gen/room_schema.py`
- `src/sensor_sim/scene_gen/tabletop_assets.py`
- `src/sensor_sim/scene_gen/task_replay.py`
- `src/sensor_sim/scene_gen/cli.py`
- `src/sensor_sim/scene_gen/meshcheck.py`
- `src/sensor_sim/scene_gen/xml_patch.py`
- `src/sensor_sim/film/sfp_scene.py`
- `src/sensor_sim/film/mitsuba_scene.py`
- `src/sensor_sim/scenes.py`
- `assets/candidates/index.json`, current hash `081bf07291a4cc3f8f2af1fe27b8f8beb0164153100244aefa25df1ba209c157`
- `dataset/dataset_v1/`, 1533 files, 457265977 bytes

Also do not change Mitsuba, Dr.Jit, Blender, or CUDA pins. Do not mark objects `visually_reviewed`, `admitted`, or `registered`.

`mesh_io.py`, `material_spec.py`, `material_assign.py`, `tabletop_render.py` `mitsuba_bsdf`, `interior_assets.py`, `semantic_assets.py`, and the Blender preview scripts are the editable surface.

## Steps to fix

1. Extend `write_obj` and `read_obj` so texture coordinates are optional and preserved. A mesh without UVs must still load. Add a test that a round trip keeps `vt` indices.
2. Generate a UV map for each render mesh that should carry images. Smart-project UVs are acceptable for a first procedural object. Record the UV method and the island margin in the asset provenance. Do not silently drop UVs when the collision mesh is written. The collision mesh does not need UVs.
3. For each material class, produce the maps the reference apple actually wires, at a stated pixel size:
   - Plastic and fruit: base color as sRGB, roughness as Non-Color, OpenGL normal as Non-Color, one IOR.
   - Metal: roughness and OpenGL normal only, plus the existing metal name.
   - Diffuse paper or paint: base color only.
   - Clear wrap: no color map. A normal map only if the wrinkle is not already in the shell mesh.
4. Store the images in the package and list them under the existing `textures` key with role, color space, resolution, and sha256. Update the package hash. Keep the old package version if any saved scene points at `1.0.0`, and write a new version instead of rewriting it.
5. Teach `to_mitsuba` to emit a Mitsuba `bitmap` for reflectance or `diffuse_reflectance`, and to wrap the BSDF in `normalmap` when a normal image exists. Set `raw` true on the normal bitmap so Mitsuba does not apply sRGB. Before using a roughness bitmap on `pplastic`, instantiate it on `cuda_ad_spectral_polarized` with this Mitsuba 3.9.1 pin and record whether a textured `alpha` renders. Official current Mitsuba documents textured alpha on `roughplastic`. Textured roughness on the polarized `pplastic` plugin was added in a later Mitsuba change. If this pin rejects it, keep alpha as one scalar and still ship the roughness image for Blender.
6. Teach the Blender scene builder to create a UV layer and three Image Texture nodes with the color spaces above. Connect UV through a Mapping node the way the Poly Haven file does.
7. Render two labeled images of the same framed object. One is Blender Cycles. One is Mitsuba. State resolution, samples per pixel, and the view transform. Do not compare an overexposed render with a dark one and call that a material difference.
8. Re-run placement checks after any mesh or scale change. The 4 mm separation and the 1 mm support gap are in metres. A larger apple can intersect objects that the 8 cm apple cleared.
9. Add tests for the schema, the Mitsuba node types, the UV round trip, and the rule that a missing texture is an error rather than a silent return to one RGB color.

## Component list

These are the controls a person can look for on one object. The first group is the project standard. The second group exists on Blender's Principled BSDF or in the Poly Haven download and is not required for the first realistic version.

Required for a realistic single-family object:

| Component | Where it lives | What it does |
| --- | --- | --- |
| Mesh positions and faces | `geometry/render.obj` | The silhouette and the real slopes |
| UV map | `vt` coordinates, also glTF `TEXCOORD_0` | Places every image on the surface. Image textures use this |
| Vertex normals | `vn` or glTF `NORMAL` | Shading before a normal map |
| BSDF family | `asset.json` materials and `MaterialSpec.family` | The reflection law: diffuse, plastic, metal, or glass |
| Base color | One linear RGB, or an sRGB image | The color of a diffuse or plastic surface. Not used to paint metal |
| Roughness | Blender 0–1 image, Non-Color. Mitsuba Beckmann alpha | Sharp or broad highlights. Strongly affects polarization |
| IOR | One number | Strength of the dielectric reflection and refraction. Reference apple uses 1.45 |
| Normal map | Non-Color OpenGL image through a Normal Map node, or Mitsuba `normalmap` with `raw` true | Small bumps that are not in the mesh. Does not move the outline |

Present on the downloaded apple or in Blender, but not required to copy:

| Component | What it does | Decision |
| --- | --- | --- |
| Ambient occlusion image | Darkens creases with a baked shadow | Leave it out of Mitsuba. Poly Haven's own Blender node tree does not connect it |
| ARM packed image | One file whose channels are usually occlusion, roughness, and metal | Do not use the packed file. Use the separate roughness image |
| DirectX normal | Same as the OpenGL normal with the green channel flipped | Do not connect it in Blender or glTF |
| Metallic | Blends plastic and metal | Keep at 0 for this apple. A metal object is a conductor family, not a painted metallic value |
| Subsurface weight, radius, and scale | Light scattering inside fruit, skin, milk, or wax | Off on the reference apple. Optional later for fruit |
| Coat weight, coat roughness, coat IOR | Extra clear layer | Not connected on the reference apple. Optional for wax |
| Specular IOR level and specular tint | Non-physical specular brightness and color | Leave at the defaults. Mitsuba says not to tint the specular lobe for physical realism |
| Emission, alpha, transmission, sheen, anisotropy, thin film | Other Principled layers | Out of scope for one ordinary opaque object |

## Definition reminder for any number in a later report

- Triangle count is the number of triangles after fan triangulation, not the OBJ polygon count.
- Roughness in Blender is the Principled socket from 0 to 1. Mitsuba `alpha` is the Beckmann root-mean-square slope. They are not the same number. The project currently converts with a square root when sending alpha to Blender roughness.
- Clearance is the minimum triangle-to-triangle distance in metres. The placement failure threshold is 0.004 m.
- A linear RGB value is the shading color. An sRGB JPEG is a display encoding of a color image. A Non-Color image is data and must not be run through that sRGB conversion.

# Tabletop asset diversity

Branch `main`, HEAD `0609c61124005144060a8f9eb673d73510b57ff5`. No commit was created. New assets were marked `technically_validated` and then `renderer_validated` after their Stokes EXRs read back. `visually_reviewed`, `admitted`, and `registered` were not set. Existing candidate statuses were not changed. `assets/library/` was not populated.

This is not general text-to-3D and not general-purpose room generation. The scenes use pinned asset ids, one existing room shell, and three explicit furniture arrangements.

## Initial state

Protected hashes at the start of the milestone matched `docs/task_to_render_validation_report.md`:

- `sfp_scene.py` `3287fc531e50069d3fb912292f3a3bc907fa1e80a39d1c1a30398a8940ea3ea0`
- `mitsuba_scene.py` `6bdd40ac8ed12a243b0693da78c87219f3c5d51d66466b8772cd42c89d9b0653`
- `scenes.py` `752e7c5c7aa067eaaa77559b307066a8c57d72a481bd8853c1a0194d80aa0022`
- `xml_patch.py` `6b8ced6842897aa462eb1d855d4f7fe8cf8702e9f796af5bebe497d4b3942e32`
- `generate.py` `78ee05ab9afe551436b0f35836880e8c6b1680e8c559f642083269ff3931ddf8`
- `cli.py` `29a5ddc431bc4fa13887afed2fab774f485331d7aa5e97d0732b7d42a806763f`
- `meshcheck.py` `051676b437f4ae4394d201b8470cf179ddeebbbb3d25f3b96479f3fbeb684198`
- `placement.py` `650e9b36e9fb4422877b4cb392b9d7253c258cc1385329be502661a9d6ca466d`

`indoor_mitsuba.py` and `structural.py` were not edited. Their hashes stayed `dfbd5e9019be0d512f787ce2ec1ebffd78cf60cf52fff427d00a8a1e6c26d2b3` and `6e839ad9debb3bcbc4b16c1016f1eac77b6af58e8f1dc93b49c5302e007db1a7`. `mesh_io.py` was not edited. Pytest before the new packages: 172 passed, 2 failed. Both failures are the historical Phase 3 tests missing `outputs/phase3/frames/000/stokes/render.exr`.

Mitsuba variants include `cuda_ad_spectral_polarized`. Loaded plugins: `diffuse`, `pplastic` (reported as `SpecularDiffusePolarized`, Beckmann, `alpha` 0.1, `eta` 1.5), `conductor` for `Al` and `Cu`, `roughconductor` for `Al`, and `dielectric`. A cylinder OBJ with no `vn` lines still received distinct per-vertex normals from Mitsuba 3.9.1, so curved meshes were not given a second normal writer.

`task_layout.resolve` was not changed. New placement is in `tabletop_layout.py` and calls `measure_support` and `mesh_clearance`. Replay still uses `task_replay.replay`.

## New assets

All packages are version `1.0.0`, generator `tabletop`, license `permitted`, Z-up meters, contact at local z=0. Profile is `closed_solid` except the chair, which is `multi_component`. The render mesh is the collision mesh. Every triangle count is under 2500. Status is `renderer_validated` only after the inspection EXR read back. Human review is still pending.

| Asset | BSDF | Package SHA-256 | Geometry SHA-256 | Triangles |
| --- | --- | --- | --- | --- |
| `prop_bowl` | diffuse, depolarizing reference | `a83b3f4802ae1e67779d0b88df9a9b32ba3ba5289bc3aa5eae4466d89fe6757e` | `8aa45f5661d43c5cd5234df3925959340e4d02889fdf3c6ea8be9d4d50c05a85` | 448 |
| `prop_bowl_glossy` | pplastic | `4b79e821c3794bc5b84feefc23b3f1a050efd6d2115ff1a077dc5af6c7be647c` | same as `prop_bowl` | 448 |
| `prop_plate` | diffuse, depolarizing reference | `ae570fcda54faff68502e30efd18bcfe104586ba21158685d49ad62c6467aa18` | `ef56e67ae275cf34cbe41bd41d5d6c187f19fae75e8b27c88455e58079a46c09` | 504 |
| `prop_bottle` | pplastic | `048f51d4dd8de6c394b93836bde0cec02d25e0ffc8781b889dc589a10f2b6525` | `34c9a5a076e12bb583039ea8fa2c1eb5df81ceb68e362c022e56f7ff296a08ab` | 448 |
| `prop_can` | conductor `Al` | `2e2b526a41260f3da87402150d8055b9f8f2157adb945c6855f5cd34836811c1` | `84f2c74b43525f14401ac2353ab871f7095d694eed0d2e636830d28d1bb7fe07` | 384 |
| `prop_can_rough` | roughconductor `Al`, Beckmann `alpha` 0.1 | `c1f1cf7796ed22a63e24f6e42252f3a9ee0fffadd618945b49ef1bbaff8da120` | same as `prop_can` | 384 |
| `prop_spoon` | conductor `Cu` | `4b74f2db42f3d26ce16f76cd1623de7737fb5b2b53e9ab1d76bd2ab2ade8e62b` | `682969a728341c16b1a14b448b0492d43415b8588ca2aefa86c6fe381dfde007` | 1236 |
| `prop_marker` | pplastic | `44c02566156b1b681b74c08c387927d222c22c678055457780d19c2521b3a98e` | `e08f31b8458008aac7b0adf39356112f88973b5c30bbeaa68457e41197a221a9` | 200 |
| `prop_phone` | pplastic | `0879874aa0f1d0531ee57e7bb489a6be675d710afb012fa58627dc4568935d94` | `f29d2a36f3de953bc642232c247fa03f8bf1184e43079d736bc61a501755ec5e` | 156 |
| `prop_tape_roll` | pplastic | `510abcecb7850708f93a697745a3a3bffb1a312dee10a984ad2eb53e8136a0d5` | `fa77d1c910e16c8dfde3677db7a3127ff1db70e36c073292b3c19a9cdef830ff` | 784 |
| `prop_apple` | diffuse, depolarizing reference | `0e487b40fb707ed5cd36ce6b4087e88ad73d552bf121ac6c82aaee05b46adbd5` | `dace11254d1664787cd0ceceafe707177057da996b00065d5de490e9270c4627` | 952 |
| `prop_small_box` | diffuse, depolarizing reference | `c5a963f347896a5a3f65b2e18ad947a1971e67020e38a9a4b26e4bc199422d4d` | `43c23d42e15214e60cfff999d776efa663552c936e3e451277b56dcc56b8c0cb` | 12 |
| `prop_chair` | diffuse, depolarizing reference | `333c26c0adbdcc7485788eb7c07df301b78aa8ebe870c4f1102c3fd42e5c1350` | `937d992c583a53a73d4fb06c78c000132931f2027f38015e04a91479c5033bd4` | 72 |

`prop_bowl_glossy` and `prop_can_rough` contain the exact OBJ bytes of their base packages. `alpha` 0.1 is Mitsuba Beckmann RMS slope, the existing `PHASE2_ALPHA`, not an artistic roughness in `[0, 1]`. `pplastic` is an approximate polarization-aware plastic. Mitsuba diffuse is a depolarizer, not a full matte-dielectric model. A valid Stokes EXR is not a Fresnel or S3-sign measurement.

Measured extents, meters: bowl 0.110 x 0.110 x 0.045, plate 0.140 x 0.140 x 0.012, bottle 0.056 x 0.056 x 0.160, can 0.068 x 0.068 x 0.115, spoon 0.136 x 0.030 x 0.009 (handle ends at x=0.140; the head starts at x=0.004), marker 0.130 x 0.016 x 0.016, phone 0.145 x 0.070 x 0.009, tape 0.076 x 0.076 x 0.023, apple about 0.077 x 0.082 x 0.081, box 0.070 x 0.045 x 0.040, chair 0.420 x 0.400 x 0.880 with the seat top at 0.46 m. The spoon passed `closed_solid`, so it is one watertight mesh. The apple is not a surface of revolution: its x extents are -0.031 m to 0.046 m.

The bowl, plate, bottle, and can are thickened surfaces of revolution. The tape roll is a torus, so the center hole is real. The phone screen is an inset in the same mesh and the same BSDF. No new transmitting dielectric was added. `thindielectric` was not used. `prop_cup`, `prop_book`, `prop_crate`, and `prop_crate_open` were not regenerated and are not in these recipes.

Retrieval text does not contain `book`, `cup`, `mug`, `table`, `shelf`, `crate`, `container`, or `sphere`. After the new assets were `renderer_validated`, query `book` still returned `prop_book` first and query `mug` still returned `prop_cup` first.

## Room

The shell is the existing `room_floor`, `room_ceiling`, and four wall packages at identity. The south wall mesh already contains the door opening and the north wall mesh already contains the window opening. No painted opening was added. No daylight emitter was added. The opening stays dark because there is no exterior source. That is a geometric opening, not a calibrated window.

Furniture sits on the floor support polygon. The world tabletop height is the table matrix applied to the package support surface. For these arrangements that height is 0.790 m, not a hardcoded substitute for the local 0.74 m top. Floor contact gap is 0 m for the table, shelf, and chair. Furniture does not intersect the shell or the other furniture. The room size was not changed. This milestone does not generate arbitrary room dimensions.

Arrangements, translations in meters:

- `south_open`: table `[1.50, 1.35, 0.05]`, shelf `[2.90, 2.50, 0.05]`, chair `[0.45, 1.45, 0.05]`
- `east_work`: table `[0.70, 1.60, 0.05]`, shelf `[2.80, 0.50, 0.05]`, chair `[2.10, 1.70, 0.05]`
- `west_shelf`: table `[1.90, 1.10, 0.05]`, shelf `[0.40, 2.40, 0.05]`, chair `[3.30, 1.20, 0.05]`

The chair is room furniture. It is not counted as a tabletop object.

## Tabletop recipes

Selection is `explicit_asset_v1` with score 1.0. TF-IDF did not choose these assets. Scale is `[1, 1, 1]`. Yaw is about Z only. Objects are placed largest-footprint first, inside the table support polygon inset by 0.015 m, with `measure_support` and a surface clearance above the configured separation. A failure is kept. Nothing was scaled to force a fit. All nine primary seeds succeeded, so reserved seeds `17104`–`17106` were not used.

Occupied fraction is the 4 mm raster area of the XY-projected triangles divided by the table support area, 0.840 m². Footprint sum is the sum of XY convex-hull areas. Nearest clearance is the minimum surface clearance among placed tabletop objects, in meters. Instance count is not density.

| Recipe | Seeds | Instances | Geometry types | Separation | Occupied fraction | Nearest clearance |
| --- | --- | --- | --- | --- | --- | --- |
| `tabletop_polarization_controlled` | 17101, 17102, 17103 | 8 | 6 shapes, including the bowl and can material pairs | 0.030 m | 0.0667–0.0669 | 0.030–0.059 m |
| `tabletop_mixed_objects` | 17101, 17102, 17103 | 10 | 10 | 0.015 m | 0.0666–0.0668 | 0.034–0.096 m |
| `tabletop_dense_clutter` | 17101, 17102, 17103 | 16 | 10 | 0.008 m | 0.0959–0.0961 | 0.011–0.035 m |

Sixteen instances cover about 10 percent of the table. That arrangement is more crowded than the eight-object scene and it is not a full table. The crate was not included.

Controlled objects: matte bowl, glossy bowl, plate, bottle, polished can, rough can, spoon, phone. Mixed objects: matte bowl, plate, bottle, polished can, spoon, marker, phone, tape, apple, box. Dense objects: two matte bowls, one plate, one bottle, two polished cans, one rough can, two spoons, two markers, one tape, two apples, one box, one phone.

## Comparisons

Six extra bundles, seed `17121`, on the `south_open` room. Each pair shares the room, furniture, cameras, ceiling light, and the tabletop object's support point. The object yaw is 0.4 rad.

- Bowl: `prop_bowl` versus `prop_bowl_glossy`. Same geometry hash and transform. The BSDF changes from diffuse to `pplastic` with the same base RGB.
- Can: `prop_can` versus `prop_can_rough`. Same geometry hash and transform. The BSDF changes from smooth `Al` to Beckmann `alpha` 0.1.
- Geometry: `prop_plate` versus `prop_apple`. For this pair only, both renders use diffuse reflectance `[0.82, 0.82, 0.78]`. The apple package itself stays its own red diffuse material in the mixed and dense scenes. The changed variable is the mesh.

These pairs are not mirrored objects in one image. Differences in the Stokes images are not, by themselves, evidence of a particular physical effect.

## Rendering

Variant `cuda_ad_spectral_polarized`. Integrator `stokes` around `path`, max depth 5. Render seed `17111`. Samples per pixel 16 for scenes and 8 for asset inspection.

The room light is one rectangle area emitter, center `[2.12, 1.72, 2.40]` m, size 0.80 m by 0.50 m, rotated 180 degrees about X so the normal points down. Radiance RGB is 40. A probe at 4 samples per pixel gave RGB mean / 99th percentile of 0.045 / 0.16 at radiance 8, 0.225 / 0.79 at radiance 40, and 0.675 / 2.38 at radiance 120. Radiance 40 was frozen for every scene. The emitter is unpolarized. This is not calibrated radiometry.

Room overview: origin `[0.55, 0.48, 1.72]`, target `[2.35, 2.15, 0.75]`, field of view 58 degrees, 512 by 384. Tabletop view: inside the room, south of the table, field of view about 76 degrees for the controlled arrangement, 1024 by 768. The exact pose is stored in each bundle. Asset inspection is 256 by 192, three views, radiance 4, with the light above the asset rather than in front of the camera.

All 13 assets and all 15 scene bundles rendered. Each reopened EXR is float32, finite, and named with the existing 15 `CHANNEL_NAMES` in order. Scene shapes are `[384, 512, 15]` and `[768, 1024, 15]`. No asset or scene failed.

Visibility uses the real OBJ meshes and the same camera rays. A nearer hit replaces the id only when it is closer by more than 1e-6 m. Equal distance keeps the smaller id. Background is 0. Low support means the visible count is below 1 percent of the frame, or below half of that object's unoccluded projection when the occlusion is partial. One percent of the 1024 by 768 frame is 7864 pixels. On the controlled tabletop view the counts were: plate 1130, bowl 843, glossy bowl 946, phone 1247, polished can 1784, rough can 1258, spoon 151, bottle 926. All eight are below the 1 percent rule, so none is marked polarization-ready. The spoon is the weakest. The frame includes the room around the table, which is why a clearly visible object can still be a small fraction of the image.

## Tests

Final pytest: 191 passed, 2 failed. The failures are the same Phase 3 pair and the same missing file:

- `tests/test_phase4.py::test_dolp_on_a_known_pixel_matches_polarization_maps`
- `tests/test_phase4.py::test_feature_stack_widths_and_order`

Both raise `FileStream` for `outputs/phase3/frames/000/stokes/render.exr`. The new tabletop tests passed, including package hashes, shared geometry, retrieval winners, room support, seed reproducibility, rejected overlap at a 2 m separation, the 15-channel EXR, real-mesh occlusion, comparison equality, gallery links, and the unchanged Scene A–C and six task `scene.json` hashes.

## What was demonstrated

Pinned assets can be packed on the existing table inside the existing room, replayed without retrieval, and rendered through the existing Stokes path with diffuse, `pplastic`, conductor, and roughconductor materials. Material pairs share geometry. One pair holds the diffuse reflectance fixed and changes the mesh.

What was not demonstrated: natural-language room generation, text-to-3D, Fresnel correctness, a polarized light source, dynamic stability, robot reach, or a tabletop that fills the table. A static room that looks coherent is not a physics simulation.

## Gallery

`dataset/qualification/tabletop_scenes/review/index.html`

```text
xdg-open dataset/qualification/tabletop_scenes/review/index.html
```

Visual review, admission, and registration are pending.

## Protected hashes and git status

The eight protected hashes above are unchanged. `indoor_mitsuba.py` remains `dfbd5e9019be0d512f787ce2ec1ebffd78cf60cf52fff427d00a8a1e6c26d2b3`. `structural.py` remains `6e839ad9debb3bcbc4b16c1016f1eac77b6af58e8f1dc93b49c5302e007db1a7`. No commit was made. Dataset v1 was not modified. Final `git status --short`:

```text
 M .gitignore
 M README.md
 M configs/film/rendering.yaml
 M configs/film/train.yaml
 M pyproject.toml
 M src/sensor_sim/film/dataset.py
 M src/sensor_sim/film/geometry.py
 M src/sensor_sim/film/heldout.py
 M src/sensor_sim/film/sample.py
 M src/sensor_sim/film/sfp.py
 M src/sensor_sim/film/sfp_math.py
 M src/sensor_sim/film/shape_mesh.py
 M src/sensor_sim/film/train.py
 M tests/test_heldout.py
 M uv.lock
?? assets/blender/build_review_blend.py
?? assets/blender/build_task_review_blend.py
?? assets/blender/create_dataset_scene.py
?? assets/blender/create_indoor_scene.py
?? assets/candidates/
?? configs/scenes/
?? docs/
?? portable_review/
?? scripts/
?? src/sensor_sim/policy/
?? src/sensor_sim/scene_gen/
?? tests/test_asset_package.py
?? tests/test_film_shape_sheet.py
?? tests/test_indoor_audit.py
?? tests/test_indoor_contact.py
?? tests/test_indoor_geometry.py
?? tests/test_indoor_identity.py
?? tests/test_indoor_scenes.py
?? tests/test_indoor_transform.py
?? tests/test_metric_audit.py
?? tests/test_policy.py
?? tests/test_qualification_gates.py
?? tests/test_reference_clearance.py
?? tests/test_scene_gen.py
?? tests/test_scene_normals.py
?? tests/test_stokes_convention_audit.py
?? tests/test_tabletop_assets.py
?? tests/test_tabletop_layout.py
?? tests/test_tabletop_render.py
?? tests/test_task_render.py
?? tests/test_task_scenes.py
?? third_party/
```

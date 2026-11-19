# Indoor asset pipeline implementation report

Branch `main`, HEAD `0609c61124005144060a8f9eb673d73510b57ff5`. Nothing was committed. The protected and read-only files listed below have the same SHA-256 before and after this work.

Baseline pytest, recorded before the new modules existed: 138 passed, 2 failed. Final pytest: 161 passed, 2 failed. The two failures are the same pre-existing tests, both raised by the missing file `outputs/phase3/frames/000/stokes/render.exr`:

- `tests/test_phase4.py::test_dolp_on_a_known_pixel_matches_polarization_maps`
- `tests/test_phase4.py::test_feature_stack_widths_and_order`

No new regression was introduced. The 23 new passes are the indoor package, geometry, transform, identity, contact, and scene tests.

## Milestone status

These states are separate. A package is not admitted because it rendered.

| Milestone | State | What that means |
| --- | --- | --- |
| M1 package library | validated | Hash, path, approval, lock, and recovery tests passed |
| M2 structural generator | validated | Floor, walls, ceiling, door, and window passed profile checks |
| M3 candidate packaging | validated | 13 procedural packages plus 6 Infinigen extracts; `room_walls` rejected |
| M4 renderer qualification | validated for the procedural scene assets and the synthetic identity/Stokes gates | Infinigen extracts were not given `renderer_validated` |
| M5 assembler | validated | Scenes repeat, contacts pass, support graphs are acyclic |
| M6 three scenes | validated technically | Human visual review and admission are pending |
| Production library | empty | No human approval record was created |

`visually_reviewed` and `admitted` were not set. `assets/library/` has no registered versions.

## Modules

- `src/sensor_sim/scene_gen/asset_package.py`
- `src/sensor_sim/scene_gen/asset_validate.py`
- `src/sensor_sim/scene_gen/asset_cli.py`
- `src/sensor_sim/scene_gen/mesh_io.py`
- `src/sensor_sim/scene_gen/structural.py`
- `src/sensor_sim/scene_gen/indoor_transform.py`
- `src/sensor_sim/scene_gen/infinigen_extract.py`
- `src/sensor_sim/scene_gen/indoor_mitsuba.py`
- `src/sensor_sim/scene_gen/indoor_contact.py`
- `src/sensor_sim/scene_gen/indoor_cli.py`
- `assets/blender/create_indoor_scene.py`

Dataset v1 files were not edited. Indoor placement does not call `place_objects`. The indoor writer does not call `_disambiguate_duplicate_bsdfs`.

## CLI

Package catalog:

```text
python -m sensor_sim.scene_gen.asset_cli package --manifest manifest.json --source dir --destination out --role geometry/render.obj=render
python -m sensor_sim.scene_gen.asset_cli inspect --package out
python -m sensor_sim.scene_gen.asset_cli validate --package out
python -m sensor_sim.scene_gen.asset_cli verify --package out
python -m sensor_sim.scene_gen.asset_cli list --candidates assets/candidates --library assets/library
python -m sensor_sim.scene_gen.asset_cli mark --catalog assets/candidates --asset-id room_floor --version 1.0.0 --status technically_validated
python -m sensor_sim.scene_gen.asset_cli register --library assets/library --package out --approval approval.json
python -m sensor_sim.scene_gen.asset_cli reconcile --library assets/library
```

`register` requires an external approval record bound to the asset id, version, and exact `package_sha256`. Automated tests use temporary fixtures and do not write a production approval. `mark` accepts only `technically_validated` and `renderer_validated`.

Scenes:

```text
python -m sensor_sim.scene_gen.indoor_cli --catalog assets/candidates --output dataset/qualification/indoor_scenes --spp 16
```

`--no-render` still checks support, collision, and per-asset Mitsuba loads.

## Package contract

Exclusion policy version 1 omits only `admission_report.json`, `*.blend1`, `*.log`, `*.tmp`, and regular files under `logs/`, `cache/`, `caches/`, `tmp/`, or `temp/`. Every other regular file is payload. Symlinks, sockets, FIFOs, and device files are rejected, including inside an excluded directory. Paths must be NFC relative POSIX paths. Backslashes, colons, control characters, trailing spaces or dots, Windows reserved names, and case-folded collisions are rejected. Reads use `O_NOFOLLOW`, then `fstat`, then the same descriptor.

`asset.json` is stored as canonical JSON. New manifests cannot contain `status`. The package digest still drops a legacy `status` field and `package_sha256` before hashing the manifest body. The package preimage is the canonical UTF-8 document `{"files": sorted {path, sha256}}` plus a newline. The digest is lowercase hex.

The candidate index is `assets/candidates/index.json`. The production index, when used, is `assets/library/index.json`. Version bytes live at `assets/library/<asset_id>/<version>/` and are not rewritten to change status. Publication holds an exclusive `fcntl` lock across the existence check and the rename. An index failure leaves the version directory and records an orphan. Reconciliation lists orphans and does not delete or register them.

## Geometry

Degenerate triangles are those with area below `1e-14 m²`. An edge with one triangle is a boundary, two is an interior edge, and any other count is non-manifold. A vertex is non-manifold when its link is not one path or one cycle. Interior edges must be traversed once in each direction. Closed solids must have positive signed volume. Face normals, after normalization, must match the geometric normal with dot product at least `0.9999`. A unit normal that points backward fails. `structural_solid` also requires thickness of at least 5 mm along the recorded unit axis and extent of at most 20 m. Thickness is the span of vertex projections on that axis. Floors and ceilings use `+Z`. North and south walls use `+Y`. East and west walls use `+X`.

An 8 m floor passes the indoor structural profile. `mesh_issues` still reports `bounds` for that mesh. That Dataset v1 check was not changed.

Procedural OBJ bytes repeat for the same generator version, seed, and parameters. The writer fixes vertex order, face order, nine decimal places, and LF newlines. Infinigen extraction does not make that claim.

## Transform

The serialized matrix is 16 row-major floats applied to column vectors, `M_world = M_parent @ M_local`, meters, Z up, right-handed. On this Mitsuba 3.9.1 install, `ScalarTransform4f` uses that same matrix. The basis conversion is the identity, measured by loading a translated, rotated, and scaled triangle and comparing world positions. Normals use the inverse transpose multiplied by the sign of the determinant, because `cross(Au, Av) = det(A) inv(A)^T cross(u, v)`. Negative scale is accepted and reported as `mirrored`. Shear, singular matrices, and condition numbers above `1e8` are rejected. A parent non-uniform scale followed by a 45 degree local rotation is shear. The same scale followed by a 90 degree rotation stays orthogonal and matches Mitsuba. Blender `matrix_world` for translation, rotation, non-uniform scale, and negative scale matched the loaded Mitsuba positions. The parent-scale plus child-rotation case was rejected as shear.

## Assets

Procedural packages are `sensor_sim` geometry, license status `permitted`. They are candidates, not registered assets.

Renderer-validated, because each was loaded alone and included in a Stokes scene render:

- `room_floor` `7310820bbdeae89fc84bc9434d9d66f67ce250dd40161dca16af2b48d0dcb699`
- `room_ceiling` `4293c89790629947908ba926f124c1b7ba07ab7b016c7a0a912a5dd120112722`
- `room_wall_west` `f77feeeef415630acbc22d0e7a59129a195dae6a89b9c0f2cacca39132fb46e5`
- `room_wall_east` `edeca4c7b887b2f685a8f51a18e1be185620a3ff8cff3b9aa16f886c7bacfdec`
- `room_wall_south` `07bff88068eacd6c24a58ac808f9049179e3c7e627c8fe23409e9c749f5b2543` (36 triangles, door)
- `room_wall_north` `154c4c090506e4834c5072d7415eb30e70dad625b3874db669a1a170f3dabab8` (48 triangles, window and sill)
- `table_rect` `af388a7ce55f371fd572e9e82226c017a8f46502e6e787161fe253e4db17e135`
- `shelf_two` `91efd1e8f6079ba2c4b709cf808435390a9636f712a2211bd876e413a7f9344b`
- `prop_crate` `a4373a42cf42c4def1006e825e495e88cd070b9d800e41e928efba2e9a317997`
- `prop_cup` `f29e35c44c6bd973a13bb4ffa44518de7df3420a090edd89276cd8c16d584844`
- `prop_book` `05d4fd32e18335fe5be336ffef3ace52e8695cba9a2299d33f4a62d8e940e72e`

Technically validated, not placed in the three scenes, so not marked renderer-validated:

- `fixture_sphere`
- `fixture_cone`

Infinigen candidates stay license-unresolved. This report does not decide whether their meshes may be redistributed. Stable extracts record that `infinigen_gpl` at `10c1d76f` is GPL-3.0 and is imported while the indoors code loads. Infinigen 2 source is BSD-3-Clause at `53033dd`; extracted-mesh redistribution was not decided. None are in the production library.

| Candidate | Status | Triangles |
| --- | --- | --- |
| `infinigen_stable_floor` | technically_validated | 59 |
| `infinigen_stable_desk` | technically_validated | 2044 |
| `infinigen_stable_shelf` | technically_validated | 10996 |
| `infinigen2_table` | technically_validated | 524 |
| `infinigen2_bowl` | technically_validated | 27308 |
| `infinigen2_room_shape` | candidate, `unapproved_room_shell` | 96 |
| `infinigen2_room_walls` | rejected, empty | 0 |

The bowl and the stable shelf are above the 2,500-triangle `reference_clearance` limit. Their render meshes do not claim independent clearance, and they were not decimated. Procedural collision meshes are separate files. Every procedural scene mesh is under 2,500 triangles, so its collision resource is qualified.

## Identity

Mitsuba 3.9.1 was measured again. Ten cube primitives with equal diffuse parameters are 10 entries in the scene dict and 1 loaded shape, id `""`, 240 vertices. Sixty shapes also load as 1 shape. Shared BSDF XML references and separately named BSDF blocks with the same parameters do the same. The merge happens at Mitsuba load, after the scene dict or XML still lists every shape.

The optical scene keeps those BSDFs. Identity loads each shape by itself, so the id is the shape name and the vertex buffer is that shape's buffer. A nearest-hit buffer stores integer ids. A closer hit replaces the id. An equal distance keeps the smaller id. Miss pixels stay 0. The buffer is not filtered and is not the optical image. For both the 10-shape and 60-shape grids, every identity from 1 through N was present and no other id appeared.

Two optical renders of the same 16 by 16, 4 spp, seed 19, `independent` sampler, `cuda_ad_spectral_polarized` scene differed by maximum absolute error 0, mean absolute error 0, and relative mean absolute error 0. A second build of that optical scene, with no identity change, matched the control inside that repeatability. The tolerance was not widened.

Different material slots on one logical object stayed two loaded shapes. That case uses two intended reflectances, 0.2 and 0.8. It is not a recoloring of identical materials.

In the demonstration scenes the four walls share one diffuse reflectance. The optical load therefore merges them. Scene A reports 6 identity loads and 3 optical shapes. Scene B reports 8 and 5. Scene C reports 11 and 8. The identity count is the per-asset load. The smaller optical count is the identical-BSDF merge, including the area lights being excluded from the object count.

## Stokes

Variant `cuda_ad_spectral_polarized` is required at runtime. The film is `hdrfilm` with `pixel_format` `rgb`, so the base image is three channels. `integrator.aov_names()` on this install returns, in order:

`S0.R`, `S0.G`, `S0.B`, `S1.R`, `S1.G`, `S1.B`, `S2.R`, `S2.G`, `S2.B`, `S3.R`, `S3.G`, `S3.B`.

`render_sample._names` was called and not edited. With 15 channels and those 12 AOV names, it returns `CHANNEL_NAMES`. The EXR is written with those names and reopened. The struct fields match. A dielectric sphere, 32 by 32, 16 spp, seed 7, produced finite float32 data. Mean channel values included `S0.R` 0.015509, `S1.R` 0.002541, and `S2.R` -0.009917. `S3` was 0 on this dielectric scene. The `S3` sign in `STOKES_FRAME` was not independently remeasured. The RGB Stokes triplets are RGB-converted components, not wavelengths. This is not a Fresnel forward model.

Scene EXRs use the same writer. Their RGB review PNGs are the `R`, `G`, and `B` channels of that Stokes render, converted with the sRGB gamma for viewing. The sample count is 16. The sampler is `independent`. The seed is 17011.

## Placement

Gap limit 1 mm. Penetration limit 0.1 mm. Measurement tolerance of the planar test is `1e-6 m`. The method is signed distance to the support polygon plane plus even-odd containment. A support polygon that lies on a larger object, such as a wall top under the ceiling, is accepted when its vertices fall in the object's projected footprint and the plane gap is inside the limits. AABB overlap is not contact. The tests cover a resting box, a 0.4 mm gap, a 2 mm gap, a 0.05 mm penetration, a 0.2 mm penetration, a point outside the polygon, four table legs, a sphere pole, a cone base, a cyclic graph, a bad collision hash, and a mesh above 2,500 triangles without a proxy.

Scene A has 8 passing contacts. Scene B has 14. Scene C has 17. All three support graphs are acyclic. Checked collision pairs had no unintended intersection. Every recorded gap in the generated scenes was 0 within the measurement rounding.

## Scenes

All three are qualification scenes. Their manifests say they are not visually reviewed, admitted, or registered, and they list every package hash.

- `dataset/qualification/indoor_scenes/scene_a/`
- `dataset/qualification/indoor_scenes/scene_b/`
- `dataset/qualification/indoor_scenes/scene_c/`

Each directory contains `scene.json`, `validation.json`, Stokes EXRs, RGB PNGs, and per-view Stokes JSON. Interior size is 4.0 by 3.2 by 2.5 m. The south wall has a door. The north wall has a window and a sill. Scene B adds the table and shelf on the floor. Scene C adds the crate on the floor, the dielectric cup on the table, and the book on the lower shelf. The cup is a procedural cylinder, not the Infinigen bowl. The bowl remains above 2,500 triangles and license-unresolved.

Review images:

- `dataset/qualification/indoor_scenes/scene_a/overview.png`
- `dataset/qualification/indoor_scenes/scene_a/door.png`
- `dataset/qualification/indoor_scenes/scene_a/window.png`
- `dataset/qualification/indoor_scenes/scene_b/overview.png`
- `dataset/qualification/indoor_scenes/scene_b/table_contact.png`
- `dataset/qualification/indoor_scenes/scene_c/overview.png`
- `dataset/qualification/indoor_scenes/scene_c/prop_crate_contact.png`
- `dataset/qualification/indoor_scenes/scene_c/prop_cup_contact.png`
- `dataset/qualification/indoor_scenes/scene_c/prop_book_contact.png`
- `dataset/qualification/indoor_scenes/scene_c/table_contact.png`
- `dataset/qualification/indoor_scenes/scene_c/shelf_contact.png`

The openings look black because there is no exterior light or shell outside the room. The lit edges are the jambs and wall faces. Rebuilding a scene directory from the same catalog and seed rewrites the same `scene.json`.

## Protected hashes

Unchanged:

- `sfp_scene.py` `3287fc531e50069d3fb912292f3a3bc907fa1e80a39d1c1a30398a8940ea3ea0`
- `mitsuba_scene.py` `6bdd40ac8ed12a243b0693da78c87219f3c5d51d66466b8772cd42c89d9b0653`
- `scenes.py` `752e7c5c7aa067eaaa77559b307066a8c57d72a481bd8853c1a0194d80aa0022`
- `xml_patch.py` `6b8ced6842897aa462eb1d855d4f7fe8cf8702e9f796af5bebe497d4b3942e32`
- `generate.py` `78ee05ab9afe551436b0f35836880e8c6b1680e8c559f642083269ff3931ddf8`
- `cli.py` `29a5ddc431bc4fa13887afed2fab774f485331d7aa5e97d0732b7d42a806763f`
- `meshcheck.py` `051676b437f4ae4394d201b8470cf179ddeebbbb3d25f3b96479f3fbeb684198`
- `placement.py` `650e9b36e9fb4422877b4cb392b9d7253c258cc1385329be502661a9d6ca466d`
- `render_sample.py` `44eb92b92b8d3cd92b9cc6f073ee99172880bd69f2b0c13b4ca9128e8d4da0f7`
- `metric_audit.py` `39a2b97d87d931292db24bf6b160d77ee9de9555e22d8c0bc59817ff85ee30dd`
- `convention.py` `e1e796886351315f861f28b0fd0d7c314dba4a42c55534368cfccf0b19ac1e8c`

Blender 4.5.9, Mitsuba 3.9.1, and `cuda_ad_spectral_polarized` were the versions used. `pyproject.toml` and `uv.lock` were already dirty and were not changed for this work.

## Pending decisions

- No person has visually reviewed or admitted a package.
- Infinigen extracted-mesh redistribution, including the GPL import path on the stable checkout, is unresolved.
- `S3` handedness is still the recorded project convention, not a new measurement.
- Identical optical BSDFs still merge in the combined Mitsuba scene. Identity is the separate per-shape load and integer buffer.
- Static contact is not dynamic stability.

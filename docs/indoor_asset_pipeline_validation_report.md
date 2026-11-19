# Indoor asset pipeline validation

Inspection is from the working tree on `main` at `0609c61124005144060a8f9eb673d73510b57ff5`. Protected-file hashes match the implementation report. Pytest before the task-layer tests were added was 161 passed and 2 failed. The failures are only:

- `tests/test_phase4.py::test_dolp_on_a_known_pixel_matches_polarization_maps`
- `tests/test_phase4.py::test_feature_stack_widths_and_order`

Both raise `FileStream` on the missing file `outputs/phase3/frames/000/stokes/render.exr`. They are pre-existing. Dataset v1 files were not edited.

## Catalog

`assets/candidates/index.json` has top-level keys `schema_version` and `assets`. Every package was reopened with `verify_package` during retrieval and during the scene replay. The production library has no approval records and remains empty.

Renderer-validated project assets used by Scenes A–C: `room_floor`, `room_ceiling`, `room_wall_south`, `room_wall_north`, `room_wall_west`, `room_wall_east`, `table_rect`, `shelf_two`, `prop_crate`, `prop_cup`, `prop_book`.

`fixture_sphere` and `fixture_cone` are only `technically_validated`. Infinigen extracts are `technically_validated` or `candidate` and their license status is `unresolved`. `infinigen2_room_walls` is rejected because the mesh is empty. `infinigen2_room_shape` remains an unapproved candidate. Nothing is visually reviewed, admitted, or registered.

`prop_crate` is a solid closed box. It has no interior metadata, so it is not a container. `prop_crate_open` is a new package, hash `5c870892f1ac0e7b55e2d34473ab4b344614373b8090be487c0434af15738a36`. Its cavity is 0.37 by 0.27 by 0.185 m, with a top opening, an interior bottom, and a 5 mm margin. It was loaded as one Mitsuba shape and written to a 15-channel Stokes EXR, then marked `renderer_validated`. The solid crate package was not rewritten.

## Identity

`identity_buffer` casts one ray per pixel from a pinhole above the shapes and keeps the smallest finite ray distance. The stored value is an `int32`. Background is 0. A strictly nearer hit replaces the id. An equal distance within `1e-6` m keeps the smaller id. There is no interpolation.

A new test places a wide slab in front of a smaller cube and a third cube beside the slab. The hidden cube has no pixels. The front slab and the side cube do. Two coincident cubes produce only identity 1. The 10-shape grid still shows identities 1 through 10. The dielectric cup's identity, when rendered as geometry, is the first surface along the ray. It does not list every object that contributes to refraction.

The optical control at 16 by 16, 4 samples per pixel, seed 19, still has maximum, mean, and relative error 0. Optical BSDFs were not changed to create ids. In Scenes A–C the four walls share one diffuse reflectance, so the combined optical load merges them. The per-asset identity loads remain one shape per instance: 6, 8, and 11.

## Stokes

Scene C `overview.exr` reopened as float32 with shape 160 by 256 by 15. The names are `R`, `G`, `B`, then `S0.R` through `S3.B`. The variant is `cuda_ad_spectral_polarized`. The film is RGB. Non-finite count is 0 and negative `S0` count is 0.

`mi.render` of this film returns Stokes components after spectral-to-RGB conversion. A pre-conversion spectral Stokes image is not what that call returns, so the cone inequality on the RGB channels is a diagnostic. It is not a hard pass or fail. On the Scene C overview, with tolerance `1e-4` and an `S0` floor of `1e-4`, the diagnostic violation count is 0. A diffuse constant-light control has a smaller mean absolute `S1.R` than the dielectric control. This is not a Fresnel validation. The `S3` sign was not remeasured. The 15-channel contract is independent of the diagnostic.

## Contacts and replay

Every support record in Scenes A–C uses meters. Every record passed. Gap is 0 and penetration is 0, inside the 1 mm gap limit and the 0.1 mm penetration limit. Scene A has 8 structural contacts: four wall bottoms on the floor interior and the ceiling on four wall tops. Scene B adds 6 object contacts: four table legs and two shelf sides on the floor. Scene C adds the crate, the cup, and the book, for 17 contacts. The package hashes in those manifests match the candidate files. Replaying the assembler with `render=False` into a separate directory reproduced `scene.json` byte for byte and did not change Scene C. The indoor CLI still overwrites a directory when that directory is passed as `--output`. The default path is the original scene directory, so replay must not use it.

Pixel equality is not required. The earlier same-seed optical control differed by zero on a small render. The qualification EXRs were not required to be byte-identical.

## Review page

`dataset/qualification/indoor_scenes/review/index.html` is offline. It links to the original Scene A, B, and C images and lists each instance package hash. Visual review and admission remain pending.

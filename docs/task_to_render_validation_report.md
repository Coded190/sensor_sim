# Task-to-render validation

Branch `main`, HEAD `0609c61124005144060a8f9eb673d73510b57ff5`. No commit was created. Catalog statuses were not changed. `visually_reviewed`, `admitted`, and `registered` were not set. `assets/library/` was not populated.

## Audit

Protected-file hashes match `docs/indoor_asset_generation_implementation_report.md`. No mismatch.

Pytest at the start of this milestone: 169 passed, 2 failed. Pytest at the end: 172 passed, 2 failed. The two failures are the same tests and the same cause as the historical record:

- `tests/test_phase4.py::test_dolp_on_a_known_pixel_matches_polarization_maps`
- `tests/test_phase4.py::test_feature_stack_widths_and_order`

Both raise `FileStream` because `outputs/phase3/frames/000/stokes/render.exr` is missing. The three new passes are `tests/test_task_render.py`. No other failure appeared.

`cuda_ad_spectral_polarized` is installed. Every candidate package reopened with `verify_package`. The six seed-17011 bundles still match the catalog hashes. `replay` reproduced each saved `scene.json` byte for byte, and the source `bundle.json` and `scene.json` hashes were unchanged after rendering.

Scene A–C files and the six source bundle directories were not overwritten.

## What you can open

Gallery:

`dataset/qualification/task_scene_renders/review/index.html`

```text
xdg-open dataset/qualification/task_scene_renders/review/index.html
```

The gallery links the original Scene A–C images, the six task renders, and the asset views. It states that visual review, admission, and registration are pending.

## Asset inspection

Production renders are in `dataset/qualification/asset_review/<asset_id>/inspection_v2/`. Each directory has `front.png`, `side.png`, `elevated.png`, matching EXRs, and `summary.json` with the verified package hash, meter bounds, vertex, polygon, and triangle counts, materials, and image hashes.

`prop_crate` is still the solid box. `prop_crate_open` shows the open cavity. `fixture_sphere` and `fixture_cone` are in the gallery section **Technical Fixtures — Not Production Assets**. Their catalog status is still `technically_validated`.

An earlier darker pass remains at `inspection_v1` and was not overwritten. Use `inspection_v2`.

## Six task renders

All six saved bundles rendered. Outputs are `dataset/qualification/task_scene_renders/<task_id>/render_17011_lit/`. Each has `overview.png`, `detail.png`, `overview.exr`, `detail.exr`, and `validation.json`.

| Task | Source run | Rendered | Optical shapes | Identity loads | EXR |
| --- | --- | --- | --- | --- | --- |
| book_on_table | `e54a5e18636df8e7` | yes | 2 | 2 | 15 float32 names, finite |
| cup_on_shelf | `8a0a6b98b048ef83` | yes | 2 | 2 | same contract |
| book_inside_crate | `e547b5c803391dbe` | yes | 2 | 2 | same contract |
| crate_and_book_on_table | `44ac50f76b66afcb` | yes | 3 | 3 | same contract |
| cup_near_book | `95d8b3f496b78836` | yes | 3 | 3 | same contract |
| objects_in_workspace | `3cf07be40ab68252` | yes | 2 | 2 | same contract |

The channel order is the existing `CHANNEL_NAMES`: `R`, `G`, `B`, then `S0.R` through `S3.B`. These EXRs are not a Fresnel validation. The `S3` sign was not remeasured.

Materials are a fixed map. Assets that already appeared in Scenes A–C use that qualified render color or, for the cup, the dielectric BSDF. `prop_crate_open` and the fixtures use the reflectance stored in the package. Object matrices are the saved matrices. The camera is a front-elevated view framed from the combined bounds, field of view 45 degrees, margin 1.25. The light is one area light beside the camera, radiance 40. Resolution is 256 by 192. Samples per pixel are 16. Seed is 17011. Variant is `cuda_ad_spectral_polarized`.

Contact checks were recomputed with `measure_support`. The book-on-table gap is 0 m. None of the six renders failed.

A darker first task pass exists only as `render_17011` where it was completed before the lighting change. The gallery uses `render_17011_lit`.

## Capabilities

Demonstrated on this run:

1. A task name selects a catalog asset through the existing local TF-IDF retriever. That is not Sentence-BERT and not general text-to-3D.
2. The solid `prop_crate` is retrieved by the word crate and then rejected for containment because it has no interior. `prop_crate_open` is used instead.
3. Instance scale is recorded from the package bounds. These six scenes use scale 1.
4. Saved poses satisfy `supported_on`, and the book sits in the open-crate interior.
5. The saved bundle is the source of the render. Retrieval was not run again.
6. `replay` matches the saved `scene.json` without a model call.
7. The saved bundle renders in Mitsuba.
8. The EXR read-back matches the 15-channel contract.

Not supported: articulated doors or windows, robot reachability, dynamic stability, and treating a Stokes EXR as proof of Fresnel correctness.

## Protected hashes

Unchanged from the implementation report:

- `sfp_scene.py` `3287fc531e50069d3fb912292f3a3bc907fa1e80a39d1c1a30398a8940ea3ea0`
- `mitsuba_scene.py` `6bdd40ac8ed12a243b0693da78c87219f3c5d51d66466b8772cd42c89d9b0653`
- `scenes.py` `752e7c5c7aa067eaaa77559b307066a8c57d72a481bd8853c1a0194d80aa0022`
- `xml_patch.py` `6b8ced6842897aa462eb1d855d4f7fe8cf8702e9f796af5bebe497d4b3942e32`
- `generate.py` `78ee05ab9afe551436b0f35836880e8c6b1680e8c559f642083269ff3931ddf8`
- `cli.py` `29a5ddc431bc4fa13887afed2fab774f485331d7aa5e97d0732b7d42a806763f`
- `meshcheck.py` `051676b437f4ae4394d201b8470cf179ddeebbbb3d25f3b96479f3fbeb684198`
- `placement.py` `650e9b36e9fb4422877b4cb392b9d7253c258cc1385329be502661a9d6ca466d`

## What to look at

- Is the book clearly on the table, and the cup clearly on the shelf?
- Is the book inside the open crate, or only sitting in the opening?
- Are the crate and book separated on the table?
- Are the cup and book near each other without overlapping?
- Do the wall, floor, and furniture inspection views show the shape you expect?
- Is the open crate obviously hollow and the solid crate obviously closed?

Visual review has not been performed.

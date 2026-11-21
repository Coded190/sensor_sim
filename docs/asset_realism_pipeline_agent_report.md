# Agent report: asset generation, placement, and Mitsuba export

This report is for the next agent. Read it before changing indoor assets or scenes. The qualification run under `dataset/qualification/asset_realism/` did not fix the two problems that block the next milestone: assets still look low quality, and scene placement is still a sparse arrangement of a few objects. Do not start a new milestone until both are fixed.

HEAD when this qualification was built: `8f16edb8b75f33589deee318a84dd08ac8baf3aa`. Gallery: `dataset/qualification/asset_realism/gallery/index.html`. Shorter result note: `docs/asset_realism_scene_composition_report.md`.

Pytest at that run: 269 passed, 2 failed. Both failures are pre-existing:

- `tests/test_phase4.py::test_dolp_on_a_known_pixel_matches_polarization_maps`
- `tests/test_phase4.py::test_feature_stack_widths_and_order`

Both die in `src/sensor_sim/phase4.py:128` because `outputs/phase3/frames/000/stokes/render.exr` is absent. Do not create that file to hide the failure. Do not edit phase 4 for this asset task.

Protected files that must stay unchanged: `indoor_mitsuba.py`, `room_schema.py`, `room_generator.py`, `task_replay.py`, `tabletop_assets.py`, `interior_assets.py`, `interior_scenes.py`, `semantic_composition.py`, Dataset v1, `assets/candidates/index.json`, and the historical behavior of `infinigen_extract.py`.

## Blocking defects

### 1. Asset appearance never left Infinigen

Infinigen should be the source of the render mesh, UVs, geometric normals, material slots, and baked textures (albedo, roughness, normal, metallic). The running pipeline takes an OBJ of positions and faces, or a PLY of positions, normals, and UVs, and then paints a flat color in Mitsuba.

The break is in the bakeoff extractor, and the new ingest repeats it.

`dataset/qualification/indoor_generator_bakeoff/scripts/extract_local_obj.py` function `_write_obj` writes only `v` and `f`. `main` reads `mesh.vertices` and `mesh.polygons` from the unevaluated mesh datablock. It does not evaluate the dependency graph, does not realize geometry nodes, does not write `vt` or `vn`, does not write `usemtl`, and does not bake or copy images. It does store `material_slots` as names in `asset.json`. A material name is not a material.

`src/sensor_sim/scene_gen/infinigen_extract.py` function `package_extracts` then copies that OBJ byte-for-byte into `geometry/render.obj`. The manifest `materials` list is `{"id": slot_N, "name": <slot name>}`. There is no `type`, no reflectance, no texture path, and no BSDF. `publish_candidate` and `set_candidate_status(..., "technically_validated")` run on that incomplete package. License status stays `unresolved`. Stable commit `b11700ebac77695d8f53538705cfdc0cd890c54d` also records that `infinigen_gpl` at `10c1d76f` is GPL-3.0. Infinigen 2 preview commit `53033dd55c7191cbc436be6ffbb63c026935a509` is BSD-3-Clause for the source, and extracted-mesh redistribution is still unresolved. Do not mark these project-owned.

`src/sensor_sim/scene_gen/asset_realism.py` function `_ingest_existing` reads those same OBJs again (`qual_desk`, `qual_shelf`, `qual_table`, `qual_bowl`). `read_mesh` therefore reports `uv=False`. The Infinigen 2 preview Mitsuba PLY files `sofa.ply` and `lamp-material_52.ply` do have normals and `s`/`t`. Those become `qual_sofa` and `qual_lamp`. They still have no albedo, roughness, normal, or metallic images. `face_materials` is set to the string `default` for every face, so the per-material PLY split (`sofa_002-material_25.ply` and similar) is discarded. Only one sofa part is packaged.

`scripts/infinigen/export_qualification.py` does not export anything. If `INFINIGEN_ROOT` is missing it returns exit code 2, `reason=infinigen_checkout_missing`. If the checkout matches the pin it still returns exit code 4, `reason=exporter_interface_must_be_read_from_this_commit`. No Infinigen checkout was on this machine. The 83 MB preview blend and the 628 MB stable blend are still at:

- `dataset/qualification/indoor_generator_bakeoff/candidates/infinigen_2_preview/generated/run1/scene.blend`
- `dataset/qualification/indoor_generator_bakeoff/candidates/infinigen_indoors_stable/generated/run1/scene.blend`

Those blends are the place the materials still exist. They were not re-opened for this qualification.

The room renders in `asset_realism.render_scenes` ignore package materials entirely. Every object gets `{"type": "diffuse", "reflectance": [0.55, 0.32, 0.22]}`. The sofa on/off pair in the gallery used a hand-made 64×64 PNG that is not in the sealed package. It only proves that a bitmap on `qual_sofa` UVs changes the Mitsuba image. It is not the Infinigen shader.

`gap_assets.py` fills missing categories with axis-aligned boxes (legs, seat, door, screen). `chair` and `monitor` have two seeds. That is a coverage fallback, not a high-detail asset. Do not use these as the finished furniture if an Infinigen object of that category exists in the blend.

### 2. Placement is a three-object layout, and the critic can pass it

`composition_rules.py` hard-codes three instances per room. Living room is sofa, television, table. Office is desk, monitor, chair. Positions are fractions of width and depth. Yaw is written by hand. The solver in `composition_solver.solve` only moves an object when a hard rule already fails. The twelve saved scenes start valid, so the solver trace stops at step 0 with `hard_constraints_satisfied`. It does not design the room.

`render_scenes` builds a Mitsuba scene with the floor, the ceiling, and those three objects. It does not emit walls, doors, windows, or the other furniture the room type needs. The camera sits inside the floor plan. The earlier furnished interiors in `interior_scenes.py` do have walls and more objects, but those objects are the box builders.

`semantic_for` in `infinigen_ingest.py` assigns `forward` from the category string. The docstring says the frame comes from the mesh. The function does not read face normals. An imported sofa whose seat faces a different local axis still gets `forward = +Y`. Facing tests then pass or fail against that assigned vector, not against the upholstery.

`evaluate` in `composition_solver.py` has these holes:

- `between_pair` looks up the instance id `television`. A room without that id throws `KeyError` or, if the branch is skipped, does not test the table.
- `on_surface` compares `source.z` with `target.z + target.size[2]`. `size` is a hand-written box, not the packaged mesh bounds. A real desk height will not match.
- `door_clearance` uses a fixed door `x0=0.35`, `x1=1.25`, `depth_m=0.9` from `build_scenes`, not the opening in the room spec if those numbers drift.
- `front_clearance` and `faces_open_room` both dot the forward vector with world `-Y`. They do not look at the room interior.
- Soft scores in `scene_critic.scene_report` are the minimum center distance and the object count. They do not measure clearance, doorway access, or whether a sofa faces a television in the image.

A scene with three non-overlapping boxes and a correct dot product is `valid=True`. That is why `living_1` through `workshop_2` all passed. The reversed-sofa file `scenes/rejected_sofa_reversed/scene.json` fails with `failure_reason=facing`. That test is real. It does not make the accepted rooms well composed.

## Pipeline, stage by stage

Each stage lists what the code should do, what it does, and the check that fails when the link is broken.

### Stage A. Generate with Infinigen

Should: run commit `b11700eb` or `53033dd` in an external checkout, keep the `.blend`, record the command and config hash, and produce more than one geometry variant per major category.

Does: `scripts/infinigen/run_indoor_qualification.sh` and `export_qualification.py` only check that a checkout exists and that HEAD matches. They do not call Infinigen. The blends already on disk are from the earlier bakeoff. `comparison/recommendation.md` says a second stable run changed mesh buffers, so blend bytes are not a replay key. Frozen exported files are the replay key.

Check: `INFINIGEN_ROOT` unset must exit 2. A wrong HEAD must exit 3. After a real run, `source_runs/` must contain the blend, the command, and the commit. Today `source_runs/` does not contain a new generation.

### Stage B. Export geometry and materials from Blender

Should: evaluate the dependency graph, realize instances and modifiers, bake displacement into the mesh when it changes the shape, write the high-detail render mesh with per-corner UVs, geometric normals, and per-face material ids, and bake albedo, roughness, normal, and metallic maps.

Does: `extract_local_obj.py` writes positions and polygon indices from `obj.data` and a JSON list of slot names. The Mitsuba add-on PLY export under `compatibility/mitsuba_export/infinigen_2_preview/meshes/` kept normals and UVs and split some objects by material slot (`sofa_002-material_25.ply`). It did not write the shader images. The bakeoff recommendation says the original Mitsuba XML failed on texture type `mix`.

Check: exported OBJ or the packaged `render.obj` must contain `vt`, `vn`, and `usemtl`, and a sibling texture directory must contain the baked maps with hashes. `qual_desk` fails this check (`uv=False` in the ingest log). `qual_sofa` has `vt` and `vn` and still has no texture files in the sealed package.

### Stage C. Package

Should: one qualification package per asset, render mesh separate from collision proxy, relative texture paths, semantic frame measured on the mesh, license text copied from the pin, no call to `publish_candidate`.

Does: `infinigen_ingest.ingest_mesh` calls `create_package` only. It writes `geometry/render.obj` and `geometry/collision.obj`. Proxy method is `stride_keep_support` in `collision_proxy.py`: keep upward faces and every Nth triangle. `surface_deviation` samples vertices against a stride of proxy triangles. The tolerance is `max(0.001, 0.005 * extent)` metres. Support distance is allowed an extra 2 cm. Mitsuba is supposed to load `render.obj` only. The room renderer does load `render.obj`. Good.

The manifest `materials` entries are still `{id, name, type: diffuse, reflectance: [0.6, 0.6, 0.6]}` unless the caller passes textures. `_ingest_existing` passes no textures. `frame.semantic` is the category default from `semantic_for`, not a measured normal.

Check: `verify_package` on `dataset/qualification/asset_realism/packages/<id>/1.0.0`. Then open `asset.json` and require `textures` length greater than 0 for any asset whose blend has a shader. Today that check fails for every sealed qualification package.

### Stage D. Bind materials in Mitsuba

Should: `material_spec.to_mitsuba` bind sRGB base color, Blender roughness squared once into Beckmann alpha, and OpenGL normals. Record `bound`, `unsupported`, `preview_only`, `missing`, or `failed` for every map. Do not bind a metallic blend unless a probe on `cuda_ad_spectral_polarized` changes the image. Do not put a base-color bitmap on a conductor.

Does: `texture_binding` implements those statuses. `UNSUPPORTED_TEXTURE_ROLES` is `metallic`, `ambient_occlusion`, `height`, `opacity`, `orm_packed`. `to_mitsuba` drops those roles before `_apply_textures` and logs them. Base color, alpha, and normal still bind, which the earlier texture qualification demonstrated on a panel and which the sofa probe demonstrated for base color only. `tabletop_render.mitsuba_bsdf` calls `to_mitsuba` when `schema_version` is `material_spec_v1`. A package material shaped `{type: diffuse, reflectance: ...}` uses the scalar branch and never looks at textures.

`asset_realism.render_scenes` does not call `mitsuba_bsdf` on the package. It substitutes one diffuse color. `interior_scenes.render_scene` does use `loaded["manifest"]["materials"][0]`, which for `furn_*` is a scalar BSDF and for Infinigen candidates is a name-only slot that `mitsuba_bsdf` cannot turn into a textured BSDF.

Check: the scene dict passed to `mi.load_dict` must contain `type: bitmap` for base color, roughness alpha, and normal on an imported asset, and the on/off PNGs must differ. The room gallery fails this check. The sofa probe PNGs pass it for base color only, with a fake map.

### Stage E. Semantic frame

Should: unit orthogonal right-handed axes, `right = forward × up`, and a measured agreement between `screen_normal` and the screen faces, or between sofa forward and the backrest-to-seat vector.

Does: `semantic_frame.require_frame` checks the algebra. `screen_agrees` and `sofa_forward_from_regions` exist and are unit-tested. `semantic_for` does not call them. The validation string on an imported sofa says the forward points from the backrest even when nobody computed those centroids.

Check: for `qual_sofa`, compute the mean normal of the faces that point horizontally and compare it to `frame.semantic.forward`. If the dot product is below 0.85, the frame is wrong. Do this before trusting `faces_toward`.

### Stage F. Place the scene

Should: load a real room shell (floor, ceiling, four walls, openings), place Infinigen assets with measured frames, enforce support, clearance, facing, and doorway rules, and reject a hard failure.

Does, in two stacks:

1. Production interiors, `interior_scenes.py`. `_compose_living` and the other `_compose_*` functions pick explicit centers and yaws. `_chairs` aims a chair's local `-Y` at a table with `atan2`. There is no sofa-to-television dot-product check. Furniture meshes come from `interior_assets._box` assemblies via `write_obj` (positions and faces only). `render_scene` sends the first manifest material through `_scene_for` and `render_stokes`. This stack was not edited. It is still the furnished scenes a user sees in `dataset/qualification/diverse_interior_review` if those bundles exist.

2. Qualification composer, `asset_realism.build_scenes`. Calls `room_schema.make_specification` for dimensions and openings, then `composition_rules.BUILDERS[kind]`, then `composition_solver.solve`, then writes `scene.json`. `_room_shell` writes only `floor.obj` and `ceiling.obj`. No wall meshes. Three instances. Solver usually does nothing because the hand placement already passes.

`placement_separation.py` and `indoor_contact.py` are not called by `build_scenes`. Triangle clearance is not measured. Support is the `on_floor` rule `abs(z) <= 0.01` or the `on_surface` size guess.

Check: a living-room Mitsuba image must show a sofa facing a television, with walls and a door clearance. `gallery/mitsuba_living_1.png` shows a sofa, a table, and a dark screen-like object on a floor, with no walls. Treat that image as a failed composition even though `critic.valid` is true.

### Stage G. Mitsuba render and replay

Should: `cuda_ad_spectral_polarized`, Stokes EXR, RGB PNG via `write_rgb_png`, package hash and world matrix saved, replay loads those bytes and does not regenerate or re-solve.

Does: `render_scenes` calls `require_variant`, `tabletop_render._scene`, `render_stokes` at 160×120 and 4 samples, seed 17531, then `write_rgb_png`. The matrix is yaw about Z plus the position from the solver. Replay is "read `scene.json` again". There is no function that rebuilds the Mitsuba dict from the file and checks it against a stored hash of the BSDF. `interior_scenes.replay_interior` does hash the older bundles. The new scenes do not use that function.

Check: delete the PNG, run a replay entry point, and get the same package hashes and the same material dict. That entry point does not exist yet for `asset_realism`.

## What the tests actually lock

`tests/test_asset_realism.py` locks:

- `usemtl` round trip in `mesh_io.write_mesh` / `read_mesh`.
- A right-handed frame and a rejected flipped `right` axis.
- `sofa_forward_from_regions` on two hand-picked points.
- `office(4.0, 3.4)` solves as valid, and a monitor with `screen_normal = +Y` and `screen_up = -Z` fails `screen_faces` or `screen_up`.
- `chair(1)` and `chair(2)` differ in max X. They do not differ in silhouette quality.

The tests do not open a blend, do not require a texture file on `qual_sofa`, do not render a room, and do not call `placement_separation`. A green test run is compatible with the two blocking defects.

## Required fix order

Do these before any new research milestone.

1. Point `INFINIGEN_ROOT` at a checkout of `53033dd55c7191cbc436be6ffbb63c026935a509` for the preview assets and, separately, `b11700ebac77695d8f53538705cfdc0cd890c54d` for the stable desk and shelf. Read the exporter at that commit. Do not guess the current documentation. Record the command in `dataset/qualification/asset_realism/source_runs/`.

2. Replace the use of `extract_local_obj.py` for new packages. Export evaluated meshes plus baked albedo, roughness, normal, and metallic maps. Keep per-face material ids. Do not decimate the render mesh.

3. Extend `infinigen_ingest.ingest_mesh` so a package with zero texture files fails when the export directory contains images. Stop setting every face to `default` when the PLY name or the OBJ `usemtl` has a slot. Measure `semantic_for` from face normals and named components. Reject the package when the declared screen or sofa forward disagrees with the mesh.

4. Change `render_scenes` so each instance uses `mitsuba_bsdf` on a `material_spec_v1` built from that package's textures. Add the four walls. Call `require_scene_context` and `placement_separation` on the placed meshes, using the real vertex bounds, not `size` in the rule file.

5. Rebuild the twelve scenes from those packages. A room that still has only three objects, or that still uses `gap_*` while the blend contains that category, is not done. Keep `rejected_sofa_reversed` and add the other planted failures the same way: upside-down monitor, chair turned away, blocked appliance front, intersection, missing support.

6. Re-render primary, secondary, and close-up Mitsuba views. Compare each imported asset with a Blender Cycles view of the same object from the source blend. Pixel equality is not the test. Missing maps, missing UVs, and a flat brown diffuse in the room render are failures.

## File map

| File | Role in the chain | Current failure |
| --- | --- | --- |
| `scripts/infinigen/export_qualification.py` | Start an external Infinigen export | Returns before any export |
| `dataset/qualification/indoor_generator_bakeoff/scripts/extract_local_obj.py` | Historical mesh grab | `v` and `f` only |
| `src/sensor_sim/scene_gen/infinigen_extract.py` | Packages that grab into `assets/candidates` | Slot names, no textures, calls `publish_candidate` |
| `src/sensor_sim/scene_gen/infinigen_ingest.py` | New qualification packager | Re-packs the same incomplete meshes; frame is by category |
| `src/sensor_sim/scene_gen/mesh_io.py` | `write_mesh` / `read_mesh` keep UVs, normals, `usemtl` | Works; unused by the old extractor |
| `src/sensor_sim/scene_gen/material_spec.py` | Bind or mark each texture | Works when a spec is passed; room render never passes one |
| `src/sensor_sim/scene_gen/tabletop_render.py` `mitsuba_bsdf` / `_scene` | Scientific BSDF and OBJ load | Used, but with a flat diffuse dict |
| `src/sensor_sim/scene_gen/gap_assets.py` | Fallback shapes | Box components for missing categories |
| `src/sensor_sim/scene_gen/composition_rules.py` | Intended room relations | Three hand-placed objects |
| `src/sensor_sim/scene_gen/composition_solver.py` | Yaw and translation search | No-op when the hand placement already passes |
| `src/sensor_sim/scene_gen/scene_critic.py` | Hard and soft report | Hard set is too weak; soft set is center distance |
| `src/sensor_sim/scene_gen/asset_realism.py` | Audit, ingest, scenes, gallery | Wires the broken stages together |
| `src/sensor_sim/scene_gen/interior_assets.py` | Box furniture for the older interiors | Still the meshes in the six furnished room types |
| `src/sensor_sim/scene_gen/interior_scenes.py` | Older composer and `render_scene` | Scalar material from the manifest; do not edit this file to patch it |
| `src/sensor_sim/scene_gen/indoor_mitsuba.py` | Stokes render and `write_rgb_png` | Call it; do not edit it |

## Log lines that matter

Search these after the next run:

- `infinigen_ingest event=package` must show `uv=True` and a non-zero texture count for sofa, desk, table, chair, and monitor.
- `material_spec event=mitsuba` must show `textures=` greater than 0 on those room objects.
- `asset_realism event=scene` must not be the only evidence of placement. Add a log for clearance in metres and facing dot product per relation.
- `reason=infinigen_checkout_missing` means Stage A did not run.

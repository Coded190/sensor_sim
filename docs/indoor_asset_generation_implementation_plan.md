# Indoor asset generation implementation plan

Do not start these milestones until the approval gates below are accepted. Dataset v1, `random_config`, `scene-gen`, `xml_patch.py`, `sfp_scene.py`, `mitsuba_scene.py`, and `material_bsdf` stay unchanged. Do not train, commit, or implement the Fresnel model in this sequence.

Infinigen is an offline proposal source. The library and the assembler must run without an Infinigen installation. The first coding task is Milestone 1 together with the structural generator in Milestone 2. It is not a bulk import of an Infinigen scene, and it is not a Mitsuba export of the 8,987,842-polygon stable blend.

## Sources and commits

- Infinigen 2.0 preview, commit `53033dd55c7191cbc436be6ffbb63c026935a509`, for scene-composition examples and for the table and bowl candidates. Any call to `room_livingroom_rand` stays inside a version-specific adapter after that pinned function is read. The CLI does not forward room dimensions.
- Infinigen indoors-stable, commit `b11700ebac77695d8f53538705cfdc0cd890c54d`, only for the extracted floor, desk, and shelf. The generator is not deterministic. The empty `terrain/__init__.py` on that tag must be repaired inside its own environment before `generate_indoors` imports. `infinigen_gpl` at `10c1d76f` is GPL-3.0 and is on that import path. `OcMesher` at `bb895a01` is BSD-3-Clause.
- ProcTHOR produced no meshes. Do not write a Unity extractor.
- BlenderProc commit `cdb0b160ac1ce826152bfd6448daaec84e605571` is not a dependency.
- `mesh_for_family` and `write_obj` remain the sphere and cone regression path.
- `mesh_issues` defaults stay as they are for Dataset v1.
- Export stays `bpy.ops.export_scene.mitsuba` with add-on `8848ded5e5df575f2cd0a9c10061d3f6b7137905` and Blender 4.5.9. The indoor material and identity writer is new. It does not call `_disambiguate_duplicate_bsdfs`.

## Modules to add

These files do not exist yet.

- `src/sensor_sim/scene_gen/asset_package.py` and `python -m sensor_sim.scene_gen.asset_cli` for candidates, hashes, and registration.
- `src/sensor_sim/scene_gen/asset_validate.py` for the indoor profiles.
- `src/sensor_sim/scene_gen/structural.py` for floor, wall segments, ceiling, doorway, and window components.
- `src/sensor_sim/scene_gen/infinigen_extract.py`, with a 2.0 adapter and a stable adapter. Neither is imported by the scene assembler.
- `src/sensor_sim/scene_gen/indoor_cli.py` and a small indoor package for templates, support placement, and scene manifests. It must not call `cli.main` or `random_config`.
- `assets/blender/create_indoor_scene.py` for component meshes, datablock copies, and the pinned export. It does not replace `create_dataset_scene.py`.
- An indoor Mitsuba writer next to the indoor package, not inside `xml_patch.py`.

Qualification output stays under `dataset/qualification/`. It never writes `dataset/dataset_v1/`.

## Milestone 1 — Package and candidate framework

Implement the multi-file manifest, canonical package hash, immutable registration, separate technical, visual, and registration statuses, atomic index update, and rejection tests.

Hash rules: SHA-256 per file; relative POSIX paths sorted; package hash is SHA-256 of the UTF-8 JSON list of `{relative_path, sha256}` with sorted keys. Include geometry, material metadata, textures, collision, previews, and retained source files. Exclude `admission_report.json`, logs, caches, and `.blend1` backups. Hash `asset.json` with `package_sha256` empty and `status` omitted. No absolute paths. Reload checks the hash.

Registration writes a temp directory on the same filesystem, verifies hashes, renames the version once, then replaces `index.json` under a lock. A second registration of an existing version fails. If the index write fails, record an orphan and do not delete a version another process finished. A named reconciliation command lists orphans. A status edit does not change the package hash.

Tests: a tampered byte fails; a rejected candidate is not copied into `assets/library/`; an index failure leaves a recorded orphan; two processes do not delete each other's completed version.

## Milestone 2 — Structural generator and topology profiles

Generate candidate packages for a floor, wall segments, a ceiling, a doorway, and a window. Openings are separate wall, jamb, header, and sill meshes. Width, length, height, thickness, and opening positions are arguments.

Profiles: `closed_solid`, `multi_component`, `structural_solid`, and `open_surface`. `open_surface` allows boundary edges and rejects an edge with more than two faces, bad interior winding, and non-manifold vertices. It does not call `mesh_issues(..., closed=False)` alone.

Tests: a 4 m floor and an 8 m floor pass the indoor extent profile and still fail unchanged `mesh_issues`. An open quad passes `open_surface`. A non-manifold sheet fails it. A zero-vertex mesh, the `room_walls` case, is rejected. Dataset v1 tests that call `mesh_issues` without the new arguments still use the 2 m bound.

Each component records vertex count, polygon count, triangle count from `calc_loop_triangles()`, connected components, bounds, boundary edges, non-manifold edges, degenerate faces, and normal status. The floor exposes a support surface on its top face. That surface is a polygon in the asset frame, not the bounding-box center.

## Milestone 3 — Candidate extraction and initial admission

Extract only these measured objects into candidates, using the extraction contract in the architecture document:

- Infinigen 2.0 `table_dining` and `bowl.001`.
- Stable `living-room_0/0.floor`, the simple-desk spawn, and the large-shelf spawn.

Reject empty evaluated meshes. Do not treat `room_shape.001` as an admitted floor or wall. It has 63 vertices, 48 polygons, and zero thickness. It may be tried later under `open_surface` and stays unapproved until that profile passes. Do not register a failure.

Preserve commit, configuration, seed, Blender version, `matrix_world`, and local vertices. Do not bake the world transform twice. Run static geometry reports. Record license and provenance, including the GPL-3.0 `infinigen_gpl` import on the stable path. Do not mark a package cleared for distribution.

The bowl's triangle count is measured here. If it exceeds 2,500 triangles, keep the render mesh, do not claim `reference_clearance`, and add a collision proxy only as a new hashed resource. Do not decimate the render mesh in place.

## Milestone 4 — Blender and Mitsuba admission

Admit representative structural, furniture, and object candidates on Blender 4.5.9, the pinned add-on, and Mitsuba 3.9.1.

The identity experiment, before the writer is called complete, covers:

1. Several shapes referencing one shared BSDF.
2. Separately named BSDF blocks with identical physical parameters.
3. Separate object and mesh datablocks with identical parameters.
4. A logical object with several geometric components.
5. Several material slots on one component.
6. Two instances of one asset.
7. Ten renderable shapes.
8. Sixty renderable shapes.

For each case, read the XML, load Mitsuba 3.9.1, require a nonempty unique shape id, map the id to component, slot, instance, and logical object, render, and compare the image with a control that uses the same optical parameters. Do not change roughness, IOR, reflectance, or conductor parameters to create ids. Do not edit `xml_patch.py`. If identical optical materials cannot keep unique ids, the gate fails and a separate identity pass may be proposed. That pass keeps the same geometry and transforms and is not the optical render.

Source materials, textures, and the assigned Mitsuba BSDF are stored separately. `mix` and `color_ramp` are not required to load.

The Stokes test uses `cuda_ad_spectral_polarized` and the 15 names in `CHANNEL_NAMES`: `R`, `G`, `B`, `S0.R`, `S0.G`, `S0.B`, `S1.R`, `S1.G`, `S1.B`, `S2.R`, `S2.G`, `S2.B`, `S3.R`, `S3.G`, `S3.B`. Reopen the EXR and check names, order, float32, and finite values. A controlled render must show that those Stokes channels are the renderer output. `S0` through `S3` RGB triplets are RGB-converted values, not wavelengths. Record the `STOKES_FRAME` claim, including the unremeasured S3 sign. Unnamed `ch00`–`ch14` fails. Correct names on an arbitrary buffer also fail. This test does not start the Fresnel model.

## Milestone 5 — Supported scene assembler

Implement templates, loading of approved versions only, instance transforms, and support surfaces on the floor, the table or desk, and the shelf. The only placement mode is `supported`. Write `scene.json` and a preview.

The demonstrator is one rectangular room with floor, walls, ceiling, doorway, window, one table or desk, one shelf, props on the floor, on the table, and on the shelf. Every object has a support record: supporting instance, support surface, contact frame, transform, measured gap or penetration, and tolerance. A bowl uses a contact frame from its mesh. Table legs may have several floor contacts. No floating object and no physics.

A candidate may be loaded only in a run labeled as a qualification test. That run is not an approved scene.

## Milestone 6 — Scene validation and renderer qualification

Check room structure, containment, unintended intersection, intended contact, margin, identity, export, Mitsuba load, shape ids, and the Stokes EXR from Milestone 4. Clearance for a mesh above 2,500 triangles is reported as not independently measured unless a qualified proxy was used. The result is a qualification scene, not a Dataset v1 sample.

## Milestone 7 — Follow-on sensor plan

After Milestones 1–6 pass, write the next plan for RGB, depth, normals, polarization, camera and light variation, explicit BSDF studies, a new indoor dataset contract, and a later forward model. Do not implement the Fresnel model in that writing task. Do not modify Dataset v1.

## Evidence already measured

- Infinigen 2.0 semantic signatures match at seed 17011. The blend hashes do not: `c95fe369f8ad504f79dd66c1934d21ef0f650c129b9508807ebb61c086903e5e` and `09d0e9ff0e33f0bf4a65b9b494e03f3d5241456a2fd0c32582c2ee2736842871`.
- Infinigen stable signatures do not match: same 186 names and matrices, 8 geometry-buffer hashes and 14 material lists differ. Counts of those eight meshes match, so the difference is in the stored arrays, not in a missing object.
- Ten identical BSDFs load as one empty id. Distinct diffuse colors load as separate ids. That color change is not the solution.
- ProcTHOR object poses match and four wall materials do not. The JSON is not a mesh regression.
- Local OBJ extracts were not loaded in Mitsuba. Their `asset_mitsuba_status` is `not_tested`.

## Deferred

- Multi-room houses and language-directed layout.
- Floating objects, `free` placement, `suspended` placement, and physics.
- 3D-FRONT, 3D-FUTURE, and Holodeck.
- A Unity extractor.
- Importing BlenderProc or `infinigen_gpl` into this repository.
- The full stable blend through Mitsuba.
- Loading library assets from `random_config`.
- The Fresnel model, the long-run CLI shutdown hang, and any change to the Dataset v1 four-object limit or the 2,500-triangle clearance limit.

## Approval gates

Required before milestone code:

1. Infinigen 2.0, commit `53033dd`, is the leading scene-composition source.
2. Stable Infinigen, commit `b11700eb`, is only a supplemental mesh source.
3. The structural generator lives in `sensor_sim`.
4. The multi-file immutable package and hash rules above are the library contract.
5. Placement is `supported` only.
6. The indoor Mitsuba writer is separate from `xml_patch.py`, and identical optical parameters must keep unique shape ids.
7. A license decision covers generated Infinigen meshes, including the GPL-3.0 `infinigen_gpl` import on the stable path. Internal research use and redistribution are not the same decision.
8. Indoor scenes use a new output path. Dataset v1 stays frozen.

## Checklist

None of these are complete. Do not check one because an earlier row passed.

- [ ] Candidate extracted.
- [ ] Candidate technically validated.
- [ ] Candidate renderer validated.
- [ ] Candidate visually reviewed.
- [ ] Candidate admitted.
- [ ] Candidate registered as an immutable library version.
- [ ] Asset loaded into a scene from an approved version.
- [ ] Supported placement validated.
- [ ] Scene exported to Mitsuba.
- [ ] Shape identity verified for identical optical materials at 10 and at 60 shapes.
- [ ] Stokes EXR verified with the 15 named channels and a controlled read-back.

The first admission set, after the gates above, is one floor, one wall, one ceiling, one table or desk, one shelf, one bowl or other prop, and the existing sphere and cone as regression assets only. A failure stays in `assets/candidates/` with its report.

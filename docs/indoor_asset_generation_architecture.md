# Indoor asset generation architecture

This document records what the repository does today, what the indoor-generator bakeoff measured, and the architecture proposed for the next implementation. Proposed behavior is labeled proposed. The library, the structural generator, and the scene assembler are not implemented. Dataset v1 was not changed.

Evidence is in `dataset/qualification/indoor_generator_bakeoff/`.

## Roles

Infinigen 2.0 preview, commit `53033dd55c7191cbc436be6ffbb63c026935a509`, is the leading offline source for furnished scene composition and for candidate furniture and props. The interiors from that commit are the ones selected for scene composition. It is not a runtime dependency of `sensor_sim`, and it is not yet a qualified sole asset source. `room_livingroom_rand` is a function in that pinned tree. It is not a stable production API. A later adapter may call it only after that exact implementation is checked, and the call stays inside the adapter.

Infinigen indoors-stable, commit `b11700ebac77695d8f53538705cfdc0cd890c54d`, is a supplemental source of extracted geometry: the measured floor, desk, and shelf. It is not the default room generator. Its semantic signature is not deterministic. A frozen candidate can still be loaded later if the package stores the exact mesh bytes, that commit, the generator configuration, and the package hash. Do not describe the stable generator as deterministic.

`sensor_sim` owns the structural generator, the asset package, admission, versioning, and the scene assembler. The structural generator creates floor, wall segments, ceiling, doorway, and window components. An opening is those pieces, not a missing mesh and not a Boolean this repository does not have. Room dimensions are template parameters. This generator is not `place_objects` and not the Dataset v1 ground plane at `FLOOR_Z = 0.001`.

ProcTHOR is not an asset source. Its house JSON is a layout example only. BlenderProc 2.8.0, commit `cdb0b160ac1ce826152bfd6448daaec84e605571`, is an external reference. Its GPL-3.0 code is not imported. 3D-FRONT and 3D-FUTURE were not downloaded.

Blender 4.5.9 and add-on commit `8848ded5e5df575f2cd0a9c10061d3f6b7137905` are the export path. Mitsuba 3.9.1 is the renderer.

## Three outputs

An asset is reusable geometry and metadata. A table is an asset. A wall segment is an asset.

A template records parameters, asset choices, opening positions, and support rules. It does not store one realized pose.

A generated scene is one seed. It records the template id and version, exact asset versions and package hashes, instance ids, logical object ids, component ids, local-to-world transforms, material-slot assignments, support relationships, lighting, camera, and validation results. A bowl on a table is an instance, not a new asset.

`asset_type` is `structural`, `furniture`, or `object`. A composite environment package, if one is added later, is a versioned bundle of those assets. A generated room is a scene, not an environment asset. A support surface is metadata on an asset or component. It is not its own asset type unless a later design needs a separate reusable support mesh.

The only initial placement mode is `supported`. There is no `free` mode, no `suspended` mode, no floating staircase, no hanging constraint, and no physics. An object raised because it rests on a shelf is supported.

## What the current code does

Verified in this tree:

- `src/sensor_sim/scene_gen/assets.py` `build_asset` builds one local mesh. The eight names in `FAMILIES` use `mesh_for_family` and `require_clean(..., closed=True)`. `asset_id` there is the family name. That path is the regression object generator. It is not a room generator.
- `src/sensor_sim/film/geometry.py` `write_obj` writes one vertex array and one face array.
- `src/sensor_sim/scene_gen/meshcheck.py` `mesh_issues` rejects a bounding box longer than 2 m. With `closed=True` it requires every edge to have two faces. With `closed=False` it skips the non-manifold and winding checks. Dataset v1 keeps those defaults.
- `assets/blender/create_dataset_scene.py` imports an OBJ only when `kind == "obj"`, and only when that file contains exactly one mesh. It then copies the mesh datablock. The dataset path does not import glTF, USD, or `.blend`.
- `src/sensor_sim/scene_gen/blender_export.py` runs that script with Blender 4.5.9 and the pinned add-on. The operator is `bpy.ops.export_scene.mitsuba`.
- `src/sensor_sim/scene_gen/xml_patch.py` `_disambiguate_duplicate_bsdfs` is the production workaround for identical BSDFs. The bakeoff did not call it. The ten-cube export shows why an indoor scene cannot depend on it.
- `src/sensor_sim/scene_gen/placement.py` `place_objects` puts every object on `FLOOR_Z = 0.001`. That is the dataset ground plane, not a room floor.
- `scene-gen` remains the Dataset v1 entry point. `random_config` was not changed.
- `src/sensor_sim/scene_gen/metric_audit.py` `CHANNEL_NAMES` is the 15-name Stokes EXR contract cited below. `src/sensor_sim/scene_gen/convention.py` `STOKES_FRAME` is the project claim for S1, S2, and S3 signs. The bakeoff did not remeasure the S3 sign.

## Identity

These names are different:

- `library_asset_id`: the reusable asset. `table` and `sphere` are categories, not ids.
- `asset_instance_id`: one placement of that asset.
- `logical_object_id`: the object used for supervision.
- `component_id`: one named geometric piece. It is not a material island.
- `material_slot_id`: one material region on a component.
- Mitsuba shape id: one exported renderable shape.

One geometric component can hold several material slots and can export as several shapes. Several components can belong to one instance and one logical object. The manifest maps component to asset, slot to component, instance to logical object, shape to component, and shape to slot.

The stable subset made the split concrete. Seven Blender objects, including a datablock copy of the desk, became 12 PLY shapes because the desk, shelf, wall, and staircase each had more than one material slot. Those 12 shapes loaded with nonempty unique ids only after each received its own diffuse reflectance. A shared diffuse did not. Changing reflectance to force an id is not an acceptable writer. That result is a failure mode, not the design.

Exported Blender and Mitsuba names are sanitized and checked for collisions. Blender's automatic `.001` suffix is not a permanent id. A source object name is a locator, not proof that the mesh is valid or unique.

## Extraction contract

Proposed. The adapter does not edit the source `.blend`.

For each source object it obtains the evaluated object and mesh, records whether modifiers or Geometry Nodes ran, and records whether displacement is in the mesh or only in a shader. It writes the evaluated mesh in a defined asset-local frame and stores the source `matrix_world` separately. Units and axes are converted once into meters and Z-up. The same world transform is not baked into the vertices and also stored as that transform. Displayed dimensions are not treated as a substitute for the matrix.

The adapter records non-unit scale, rotation, negative scale, parent transforms, nested collections, shared datablocks, multiple objects that form one asset, multiple material slots, empty evaluated meshes, and modifiers that change the mesh. After extraction it computes bounds and the geometry statistics in the admission section. Normals, winding, and topology are checked against the declared profile.

Provenance stored with the candidate includes the source scene id, object name and hierarchy path, generator commit, configuration and seed when the source has them, Blender version, evaluated-mesh status, local-to-source transform, the unit and axis conversion, and the adapter version.

The bakeoff extracts were local vertex dumps plus a stored `matrix_world`. They are candidates, not admissions. `room_walls` evaluated to 0 vertices and is rejected. `room_shape.001` has 63 vertices, 48 polygons, and zero thickness. It is not approved as a floor, wall, or support surface.

## Package and hash

Proposed layout:

```text
assets/candidates/<candidate_id>/
assets/library/index.json
assets/library/<library_asset_id>/<version>/
```

Each package can contain `asset.json`, `geometry/`, `materials/`, `textures/`, `collision/`, `previews/`, `source/`, and, after admission, `admission_report.json`. Several geometry files are allowed.

Each included file has a SHA-256. Paths in the hash list are normalized relative POSIX paths, sorted by path. The package hash is SHA-256 of the UTF-8 serialization of that list. The canonical encoding is UTF-8 JSON with sorted keys and no absolute machine paths. Include geometry, material metadata, textures, collision data, previews, and retained source files. Exclude `admission_report.json`, logs, caches, and Blender backup files. Hash `asset.json` with `package_sha256` set to an empty string and with `status` omitted from the hashed representation. A status change does not change the package hash. Loading an approved version recomputes the hash.

Registration validates the candidate, writes a temporary directory on the same filesystem as the library, checks every file hash and the package hash, renames the version into a new path, and then replaces `index.json` under a lock. An existing version is never overwritten. If the index write fails, the new directory is recorded as an orphan. A reconciliation procedure inspects orphans. It does not delete a version another process may have finished.

Technical admission, visual review, and library registration are separate statuses. A failed candidate stays under `assets/candidates/` with its report.

## Topology and triangle counts

`closed_solid` calls today's `mesh_issues(..., closed=True)`.

`structural_solid` uses the same closed checks with an indoor extent, proposed at 20 m, so an 8 m floor is not rejected by the Dataset v1 2 m bound. The stable living-room floor object is 8.0 m by 8.38 m by 0.015 m and would fail `mesh_issues` today.

`multi_component` applies a profile to each named component. One failure rejects the package.

`open_surface` allows boundary edges and still rejects an edge used by more than two faces, inconsistent interior winding, and non-manifold vertices where the check can be made. It is not `mesh_issues(..., closed=False)` alone, because that mode skips the non-manifold and winding checks. The first structural assets can still be thickened closed boxes. `room_shape.001` may be examined under `open_surface` later. That examination is not an admission.

Admission also records vertex count, polygon count, triangle count from `calc_loop_triangles()` or an equivalent explicit triangulation, connected-component count, bounds, boundary-edge count, non-manifold-edge count, degenerate-face count, and whether normals are finite and consistently oriented. Polygon count is not triangle count. The bakeoff did not triangulate the extracts, so those triangle counts are unknown.

`reference_clearance` stays limited to meshes of at most 2,500 triangles. A heavier mesh may remain render geometry. Its independent clearance is not claimed. If clearance is required, a separate collision proxy is stored, hashed on its own, and checked for bounds and alignment against the render mesh. Extraction does not decimate. A simplified mesh is a new candidate version. The Infinigen 2.0 bowl, 13,652 vertices and 13,466 polygons, is the first prop that needs this triangle check before placement qualification. The stable blend also contains a trinket mesh of 2,568,384 polygons. That mesh was not extracted and was not passed to `reference_clearance`.

## Support

Proposed. The only mode is `supported`.

A support surface records `support_surface_id`, the owning component, a local frame, a polygon or mesh region, the outward normal, the usable region, an optional edge margin, and optional allowed categories. The first surfaces are the floor top, a table or desk top, and shelf levels. The surface is not inferred from a bounding-box center.

The scene records the placed instance, the supporting instance, the support surface, the contact frame, the target transform, the measured gap or penetration, and the tolerance. Intended contact inside the region and within tolerance is allowed. Penetration beyond tolerance is rejected. Intersection elsewhere is rejected. The footprint must stay inside the usable region and the margin. A table may have several floor-contact regions, one per leg. A bowl uses a contact frame taken from its mesh. A flat disk is not invented if the mesh has none.

The stable desk and shelf origins sit near z = 0.123 m, next to that floor object's top. That is a bakeoff measurement of origins and bounds. It is not a stored support polygon, and it is not an admission of those meshes.

The assembler stores the transforms from the seed. It does not run physics.

## Scene assembly

Proposed. `python -m sensor_sim.scene_gen.indoor_cli` loads approved versions and a template. It does not call `random_config` or `scene-gen`, and it does not require Infinigen at runtime.

A qualification run may load a candidate only when the output is labeled a test and is not an approved or production scene.

The first demonstrator is one rectangular room with floor, walls, ceiling, one doorway, one window, one table or desk, one shelf, and props resting on the floor, the table, and the shelf. Every object has a support relationship. There is no floating object.

## Materials

Three records stay separate: the original Infinigen material, any preserved textures and UV maps, and the Mitsuba BSDF assigned for rendering. Principled graphs, `mix`, `color_ramp`, and shader-only displacement are not assumed to load. The package records source material names, slots, texture files, whether detail is real mesh, and provenance. The scene assigns an explicit Mitsuba BSDF and its parameters to each slot. A slot name is not a claim that the source shader was reproduced. BSDF parameters are not changed to force unique shape ids.

## Blender and Mitsuba

Five measured results:

1. Blender 4.5.9 opened the Infinigen 2 blend (150 objects) and the stable blend (186 objects). The copied add-on reported `mitsuba_blender registered (with mitsuba v3.9.1)`.
2. `bpy.ops.export_scene.mitsuba` returned `FINISHED` for the Infinigen 2 scene (217 XML shapes, 87 of them PLY) and for a seven-object stable subset.
3. Loading those original XML files in Mitsuba 3.9.1 failed. Infinigen 2 stopped on texture type `mix`. The stable subset stopped on texture type `color_ramp`. That failure belongs to the source-scene export. It was not a load of each extracted OBJ. Those OBJs are `not_tested`.
4. Ten cubes with one shared Principled material, separate objects and separate datablocks, loaded as one shape with an empty id, 240 vertices and 120 faces. Writing the XML `name` into `id` still produced one empty id. After each PLY was given its own diffuse reflectance, the Infinigen 2 set loaded as 60 shapes with nonempty unique ids and 144,587 faces, and the stable subset loaded as 12 shapes, 0 empty ids, and 34,134 faces. Distinct colors are not an acceptable way to keep ids.
5. A wrapper that did not use the protected film modules rendered one PLY with integrator `stokes` on `cuda_ad_spectral_polarized`. The EXR is 64 by 64 by 15, float32, all finite, minimum 0, maximum about 1.80. The channels are unnamed `ch00` through `ch14`. That file does not meet the Stokes contract below.

The indoor writer is separate from `_disambiguate_duplicate_bsdfs`. Before it is accepted it must test a shared BSDF reference, separately named BSDF blocks with identical parameters, separate datablocks with identical parameters, multi-component objects, several slots on one component, two instances of one asset, ten shapes, and sixty shapes. For each case the test inspects the XML, loads Mitsuba 3.9.1, requires a nonempty unique id on every renderable shape, maps each id back to component, slot, instance, and logical object, and compares a render with a control that uses the same optical parameters. Roughness, IOR, reflectance, and conductor parameters are not perturbed to create uniqueness. If identical optical materials cannot keep unique ids, the gate fails. An identity or segmentation pass may then be proposed. It must keep the same geometry and transforms, and it is not the optical render. Identity and optical behavior are both required.

## Stokes contract

The indoor EXR must use these 15 float32 names, in this order, from `CHANNEL_NAMES` in `src/sensor_sim/scene_gen/metric_audit.py`:

`R`, `G`, `B`, `S0.R`, `S0.G`, `S0.B`, `S1.R`, `S1.G`, `S1.B`, `S2.R`, `S2.G`, `S2.B`, `S3.R`, `S3.G`, `S3.B`.

`S0` through `S3` are RGB-converted Stokes triplicates, not wavelength-resolved spectra. `convention.py` states the project claim: the reference axis is the camera horizontal, S1 is horizontal versus vertical, S2 is +45 versus -45, and S3 is claimed right versus left circular. The bakeoff did not independently measure the S3 sign. The indoor test records that claim. It does not treat the claim as a new measurement, and it does not start a Fresnel comparison.

The test loads `cuda_ad_spectral_polarized`, renders on that variant, writes the EXR, reopens it, and checks names, order, count, dtype, and finiteness. A controlled render must show that the Stokes channels are the renderer output, not an arbitrary 15-channel buffer with the right names. Unnamed `ch00`–`ch14` fails. Geometry load and this check are separate results. The protected film modules are not edited for the test.

## Validation split

Asset admission asks whether the package is a reusable asset: hashes, geometry, topology, units, frame, origin, components, slots, textures, collision proxy when present, support metadata, Blender import, Mitsuba load, and shape ids. Technical admission and visual review are separate.

Scene validation asks whether one arrangement is valid: room structure, containment, unintended intersection, intended contact, overhang and margin, identity, export, shape ids, renderability, and camera-dependent visibility. The Dataset v1 80-pixel rule is not an admission rule. There is no floating-object scene check.

## Risks

- Treating an Infinigen preview as proof that Mitsuba loaded the scene. The original XML did not load.
- Treating the stable seed as a deterministic mesh contract. Eight mesh buffers and 14 material lists changed between two runs.
- Calling `room_walls` or `room_shape.001` a structural asset.
- Using polygon counts as the 2,500-triangle clearance limit.
- Changing a BSDF parameter to keep shape ids, or extending the production XML patch to do it.
- Importing GPL-3.0 BlenderProc or `infinigen_gpl` into `sensor_sim`. The stable indoors import path does import `infinigen_gpl`.
- Accepting `ch00`–`ch14` as Stokes.

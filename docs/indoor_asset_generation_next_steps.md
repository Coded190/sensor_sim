# Indoor asset generation: source-verified next steps

This roadmap is a plan. Writing it does not implement, test, or validate any milestone. Proposed modules are labeled proposed. They are not in the checkout.

Inspection date is the working tree on branch `main` at `0609c61124005144060a8f9eb673d73510b57ff5`. The tree was already dirty. See `dataset/qualification/indoor_generator_bakeoff/status_evidence.md` for the status list and protected hashes. Those hashes were recorded before this file was added.

## Direction

Infinigen 2.0 preview, commit `53033dd55c7191cbc436be6ffbb63c026935a509`, is the offline source for furnished scene composition and for candidate furniture and props. It is not a runtime dependency and not a qualified sole asset source.

Infinigen indoors-stable, commit `b11700ebac77695d8f53538705cfdc0cd890c54d`, is a supplemental source for the extracted floor, desk, and shelf only. It is not the default room generator. Repeating its seed is not byte-level determinism.

`sensor_sim` owns the structural generator, the package library, validation, and the scene assembler. Indoor output uses `dataset/qualification/indoor_scenes/` and, later, a separate dataset path. It does not use `dataset/dataset_v1/`.

Placement is `supported` only. There is no `free` mode, no `suspended` mode, no floating object, and no physics.

## Status matrix

| Milestone | Status | What exists | What that does not prove |
| --- | --- | --- | --- |
| M1 Package framework | `not_started` | No `asset_package.py`, `asset_cli`, or `assets/library/` | A written hash rule is not an implementation |
| M2 Structural generator | `not_started` | `mesh_issues` still rejects extent above 2 m | Dataset v1 checks are not an indoor room generator |
| M3 Extraction and admission | `partially_complete` | Six local OBJs plus manifests under the bakeoff tree | Extraction is not admission, triangulation, or an OBJ Mitsuba load |
| M4 Blender/Mitsuba admission | `partially_complete` | Source scenes exported. Original XML loads failed. Ten identical BSDFs collapsed | Export is not a load. A load is not optical equivalence. `ch00`–`ch14` is not the Stokes contract |
| M5 Scene assembler | `not_started` | `place_objects` is the Dataset v1 floor path | No indoor template or `indoor_cli` |
| M6 Scene qualification | `not_started` | No `dataset/qualification/indoor_scenes/` | No validated indoor scene |
| M7 Follow-on plan | `not_started` | An outline exists in `docs/indoor_asset_generation_implementation_plan.md` | That outline, and this roadmap, are not Milestone 7 |

## Evidence already on disk

Completed as bakeoff measurement, not as milestone acceptance:

- Six meshes were written as local OBJ plus `asset.json`. Counts are in `comparison/asset_inventory.json`. All six are `not_admitted` and `asset_mitsuba_status: not_tested`.
- `room_walls` evaluated to 0 vertices and is a rejected diagnostic, not an asset.
- `room_shape.001` has 63 vertices, 48 polygons, and zero thickness. It is not an approved floor, wall, or support surface.
- Blender 4.5.9 opened both Infinigen blends. `bpy.ops.export_scene.mitsuba` returned `FINISHED` for the Infinigen 2 scene and for a seven-object stable subset.
- Mitsuba 3.9.1 on `cuda_ad_spectral_polarized` failed the Infinigen 2 source XML on texture type `mix` and the stable subset XML on `color_ramp`. Those failures do not decide the extracted OBJs.
- Ten cubes with one shared material loaded as one shape, id `""`, 240 vertices and 120 faces. Copying the XML name into `id` did not split it. Distinct diffuse colors later produced separate ids. Changing color is not an accepted fix.
- A Stokes integrator wrote a finite 64 by 64 by 15 float32 EXR whose channels are `ch00`–`ch14`.

Nothing above is shape-identity success, optical equivalence, a named Stokes EXR, or scene validation.

## Dependency order

```text
Gate A roles and license
        |
        v
M1 package  -->  M2 structure
        |              |
        +------+-------+
               v
        early M4 feasibility
        synthetic meshes only
               |
               v
              M3 candidate admission
               |
               v
        formal M4 identity and Stokes
               |
               v
              M5 assembler
               |
               v
         M6 three scenes
               |
               v
         Gate F visual inspection
               |
               v
              M7 plan
```

A failed gate stops the next box. The early feasibility check is part of M4's estimate. It runs on synthetic geometry and the proposed indoor writer before a large admission effort, because a library of meshes that cannot keep shape ids is not useful. Formal M4 still runs again on admitted candidates. Do not weaken a criterion to pass a gate.

## Proposed modules

These paths are proposed. They are absent from the checkout.

- `src/sensor_sim/scene_gen/asset_package.py`
- `python -m sensor_sim.scene_gen.asset_cli`
- `src/sensor_sim/scene_gen/asset_validate.py`
- `src/sensor_sim/scene_gen/structural.py`
- `src/sensor_sim/scene_gen/infinigen_extract.py`
- `src/sensor_sim/scene_gen/indoor_cli.py`
- `assets/blender/create_indoor_scene.py`
- An indoor Mitsuba writer beside the indoor package. It does not call `_disambiguate_duplicate_bsdfs` and does not edit `xml_patch.py`.

`indoor_cli` does not call `cli.main` or `random_config`. Infinigen is not imported for scene assembly.

## Package contract

Lifecycle states:

- `candidate`: files exist under `assets/candidates/` and validation has not passed.
- `technically_validated`: schema, file hashes, geometry, topology, and metadata pass.
- `renderer_validated`: Blender export, Mitsuba load, shape identity, optical equivalence, and the applicable render checks pass.
- `visually_reviewed`: a person approved the images.
- `admitted`: a person approved the asset for the library.
- `registered`: that admitted version is on disk and indexed.

Automated checks may set `technically_validated` and `renderer_validated` only. They must not set `visually_reviewed`, `admitted`, or `registered`. Status for a published version lives in `assets/library/index.json` or a sibling record, not by rewriting the version directory. A status change does not change package bytes or `package_sha256`. Failures remain under `assets/candidates/` with a report.

Hash procedure. Two implementations must hash the same bytes.

1. Reject absolute paths, a path containing `..`, a repeated normalized path, and any symlink. No alternative is approved in this roadmap.
2. Normalize each kept path to a relative POSIX path with `/` separators and no leading `./`.
3. SHA-256 the raw file bytes of every payload file except `asset.json`.
4. For `asset.json`, parse the object, delete the keys `package_sha256` and `status` if present, and hash the UTF-8 bytes of `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"`.
5. Build a list of `{"path": <posix>, "sha256": <lowercase hex>}` sorted by `path` ascending, Unicode code point order.
6. The package-hash preimage is the UTF-8 bytes of `json.dumps({"files": that_list}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"`.
7. `package_sha256` is the lowercase hex SHA-256 of that preimage.

Payload includes geometry, material metadata, textures, collision files, previews, and retained source files. Excluded, and only these classes: logs, caches, temporary files, Blender `*.blend1` backups, and `admission_report.json`. No absolute path appears in the preimage.

Registration writes a temporary directory on the same filesystem, verifies every file hash and the package hash, renames the complete directory into `assets/library/<id>/<version>/` once, then replaces `index.json` under an exclusive lock. An existing version path is never replaced. If the index update fails, record an orphan. Reconciliation lists orphans and can attach one to the index after a person checks it. It does not delete a directory only because the index lacks it. A second reconciliation of an already recorded orphan changes nothing.

Tests: interrupted rename, index-write failure, two concurrent registrations, registering the same version twice, a flipped byte, a missing file, `..`, an absolute path, a symlink, and a repeated reconciliation.

## Coordinates and profiles

Indoor frames are Z-up, meters, and right-handed. Local `+Z` is up for a structural asset whose support normal points up. A transform is 16 floats, row-major, applied to column vectors in the Blender sense: world = M local. Parent then local is `M_world = M_parent @ M_local`. Serialization is that 16-float list in `asset.json` or `scene.json`. Proposed contact tolerance is 1 mm of allowed gap and rejection of penetration deeper than 0.1 mm. Those numbers are indoor defaults. They are not `FLOOR_Z` and not the Dataset v1 5 mm clearance rule.

`closed_solid` requires every edge to have two faces, consistent winding, no degenerate face below `1e-14` square meters, finite vertices, and finite unit normals when normals are supplied. Boundary edges fail.

`multi_component` is valid only when every disconnected piece is a named component that passes its own profile. An unnamed extra piece fails.

`structural_solid` is `closed_solid` plus thickness at least 5 mm and maximum extent at most 20 m. Zero thickness fails. The Dataset v1 2 m check is a separate function and stays unchanged.

`open_surface` allows boundary edges. It rejects an edge with more than two faces, inconsistent interior winding, degenerate faces, non-finite vertices, and non-manifold vertices. It is not `mesh_issues(..., closed=False)` alone, because that call skips non-manifold and winding checks. Zero thickness is allowed only under this profile.

Every generated or extracted component records vertices, polygons, triangulated face count from `calc_loop_triangles()` or an equivalent triangulation, connected components, bounds, boundary edges, non-manifold edges, degenerate faces, and normal status.

The structural generator, proposed in `structural.py`, emits floor, wall segments, ceiling, and doorway and window pieces. An opening is wall, jamb, header, and sill meshes with tested dimensions, no unintended gap, and no overlap. A missing mesh is not an opening. Dimensions are arguments. The generator does not call `place_objects`.

## Extraction and the first admission set

The adapter reads the evaluated mesh, records modifiers and whether displacement is mesh or shader-only, writes local vertices, and stores `matrix_world` separately. It converts units and axes once. It does not bake that matrix into the vertices and apply it again on load. It records non-unit scale, rotation, negative scale, parents, collections, and shared datablocks. It does not save the source `.blend`. It does not decimate. A simplified mesh is a new candidate version.

First admission candidates:

- Infinigen 2.0 `table_dining` (266 vertices, 278 polygons) and `bowl.001` (13,652 vertices, 13,466 polygons).
- Stable floor (69 / 59), simple desk (1,032 / 2,044), and large shelf (5,592 / 10,996).
- Sphere and cone from `mesh_for_family`, as regression fixtures, not as realistic furniture.

`room_shape.001` may be tried under `open_surface` and stays unapproved as a floor, wall, or support until that profile passes. `room_walls` stays rejected.

The bowl's polygon count is not its triangle count. Measure triangles before applying the 2,500-triangle `reference_clearance` limit. Above that limit, the render mesh gets no independent clearance claim unless a separate hashed collision proxy is qualified.

Each candidate gets a technical report and a preview. A person must review the preview before `visually_reviewed` or `admitted`. The stable path's GPL-3.0 `infinigen_gpl` import is recorded and is not treated as a clearance to redistribute the meshes.

## Early feasibility check, then formal M4

Before a long admission pass, run the identity and Stokes check on synthetic meshes with the new writer. Count this work inside the M4 estimate of 4–8 person-days. Do not add those days again.

Identity cases:

1. Several shapes referencing one shared BSDF.
2. Separately named BSDF blocks with identical optical parameters.
3. Separate objects and mesh datablocks with identical parameters.
4. One logical object with several geometric components.
5. Several material slots on one component.
6. Two instances of one asset.
7. Ten renderable shapes.
8. Sixty renderable shapes.

For each case, read the XML, load Mitsuba 3.9.1, require one nonempty unique id per expected renderable shape, and map each id to logical object, component, slot, and instance. Compare the loaded shape count with the expected count. Render and compare with a control that uses the same geometry, transforms, and optical parameters. Do not change IOR, roughness, reflectance, or conductor parameters to create ids. Names are sanitized and checked for collisions. A Blender `.001` suffix is not a permanent id.

If identical optical materials cannot keep unique ids, stop. A separate identity or segmentation pass may be proposed. It must keep the same geometry and transforms and must not be called the optical render. Identity and appearance are both required.

Stokes, on the same synthetic scene and again at formal M4: variant `cuda_ad_spectral_polarized`, exactly these 15 float32 names in order from `CHANNEL_NAMES`:

`R`, `G`, `B`, `S0.R`, `S0.G`, `S0.B`, `S1.R`, `S1.G`, `S1.B`, `S2.R`, `S2.G`, `S2.B`, `S3.R`, `S3.G`, `S3.B`.

Reopen the EXR from disk. Values must be finite. The names must be the channels the renderer produced, not a relabel of an arbitrary 15-wide array. `S0` through `S3` RGB triplets are RGB-converted Stokes values, not wavelengths. Record `STOKES_FRAME` in `src/sensor_sim/scene_gen/convention.py`, including the claim that S3 is right versus left circular. That sign was not remeasured. This gate does not validate a Fresnel model. `render_sample._names` may be reused only when the integrator's AOV names are already those 15 names. It must not be used to rename `ch00`–`ch14`.

If the early check fails, formal asset renderer qualification and Milestones 5 and 6 stay blocked until the design decision is made.

## Placement and assembly

`supported` only. Each record names the placed asset and instance, the supporting asset and instance, the contact surface, relative and world transforms, the contact frame, measured gap and penetration, tolerances in meters, and the geometry or proxy hashes used. Contact is tested on the surface mesh, not by bounding-box overlap. Support links form an acyclic graph. Intended contact within 1 mm gap and 0.1 mm penetration is allowed under the proposed defaults. Intersection elsewhere is rejected. A table may have several leg contacts. A bowl uses a contact frame from its mesh. Static support is not a claim that the object is dynamically stable.

A repeat of assembly with the same template version, seed, package hashes, placement parameters, materials, lights, camera, and exporter versions must reproduce the manifest, identities, placements, and transforms. Mitsuba pixels and any new Infinigen generation may differ. The manifest records those versions and hashes.

Normal scenes load registered versions only and verify `package_sha256`. A qualification test may load a candidate only when the output is labeled a test and is not an approved scene.

## Three scenes

Output root: `dataset/qualification/indoor_scenes/`. Not Dataset v1. Images are renders of the scene, not drawings.

- Scene A, `scene_a/`: floor, walls, ceiling, doorway, and window. Checks geometry, dimensions, transforms, export, load, shape ids, and Stokes.
- Scene B, `scene_b/`: Scene A plus an admitted table or desk and an admitted shelf, both supported by the floor.
- Scene C, `scene_c/`: at least one separate admitted object on the floor, one on the table, and one on the shelf. No floating object. If the bowl is not admitted, use another admitted prop or mark Scene C blocked. Do not weaken the bowl tests.

Each directory contains `scene.json`, package hashes, transforms, support records, collision results and resource hashes, expected versus loaded shape ids, export and load reports, a Stokes EXR meeting the channel contract, a full-room overview, and close-ups of openings and contacts. The ten- and sixty-shape tests stay in their own directory. Tests do not set `visually_reviewed` or `admitted`. Gate F is the technical pass plus a chance for a person to look at the three scenes before the library grows.

## Milestone completion

M1. Inputs: the hash and lifecycle rules above. Outputs: the package modules and tests. Evidence: tamper, symlink, orphan, and concurrency logs. Human decision: none for the mechanism. Blocks M2's registration of structural candidates, and blocks M3 registration. Does not prove a mesh is a good asset.

M2. Inputs: M1. Outputs: structural candidates and profile tests. Evidence: an 8 m floor passes the indoor profile and fails unchanged `mesh_issues`. Human decision: none. Blocks M5's use of structural assets. Does not prove Mitsuba identity.

M3. Inputs: M1, M2 profiles, and the early M4 check having a viable identity design. Outputs: candidate packages and technical reports for the admission set, with measured triangle counts. Human decision: visual review before admission. Blocks formal M4 on real assets and blocks M5. Does not prove renderer qualification. A failed candidate is not registered.

M4. Inputs: the early check plus technically validated candidates. Outputs: identity and Stokes reports. Human decision: accept a separate identity pass only if the identical-parameter tests fail. Blocks M5 and M6. Passing M4 does not admit an asset visually and does not validate S3 as a Fresnel sign.

M5. Inputs: registered versions from M3 and M4. Outputs: assembler, template, and one manifest. Human decision: none for the mechanism. Blocks M6. Does not prove the three scenes.

M6. Inputs: M5. Outputs: scenes A, B, and C with the artifact list above. Human decision: Gate F inspection. Blocks library growth and M7. A viewport screenshot is not Mitsuba validation.

M7. Inputs: Gate F. Output: a separate follow-on document for RGB, depth, normals, polarization, cameras, lights, BSDF studies, a new dataset contract, and a later forward model. This roadmap is not that document. M7 does not implement the Fresnel model.

## Schedule

Estimates are person-days, not calendar guarantees. M4 is the highest risk.

| Span | Person-days |
| --- | --- |
| M1 | 2–4 |
| M2 | 3–5 |
| M3 | 3–5 |
| M4, including the early check | 4–8 |
| M5 | 4–6 |
| M6 | 3–5 |
| M7 | 1–2 |
| Through M3 | 8–14 |
| Through M4 | 12–22 |
| Through M5 | 16–28 |
| Through M6 | 19–33 |
| Through M7 | 20–35 |

The package exists after M1. The structural generator exists after M2. A renderer-qualified library exists only after formal M4 and human admission. The assembler exists after M5. The three scenes exist after M6 and are not accepted until Gate F. A license decision, a failed identity design, hardware, and review time can make calendar time longer than these sums.

## Gates

- Gate A. Roles, supported-only placement, separate indoor output, and a written license choice for `infinigen_gpl`. The license note must say whether it blocks internal use, redistribution, or both. This roadmap does not make that legal choice.
- Gate B. M1 and M2 tests pass. Dataset v1 `mesh_issues` behavior is unchanged.
- Gate C. At least one floor, one wall, and one ceiling pass technical admission. At least one furniture candidate and one prop pass, or the missing prop is named as a blocker. Failures stay out of the library.
- Gate D. Ten- and sixty-shape identity tests pass with identical optical parameters. Mappings and optical controls pass. The 15-name Stokes EXR passes read-back.
- Gate E. Approved versions load by hash. Supported contact checks pass. Manifests repeat.
- Gate F. Scenes A–C pass technical validation and Mitsuba rendering, and a person can inspect them.
- Gate G. M7 is written only after Gate F.

## Checklist

None of these are done.

- [ ] Implemented.
- [ ] Tested.
- [ ] Renderer validated.
- [ ] Visually reviewed by a person.
- [ ] Admitted by a person.
- [ ] Registered as an immutable version.

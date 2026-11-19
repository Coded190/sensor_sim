# Asset library architecture

This is an investigation. No library was implemented, no mesh was generated, and Dataset v1 was not modified. The detailed notes are in `dataset/qualification/asset_library_architecture/`.

## Current flow

`cli.main` expands `strata` into slots and retries a failed slot in place. `random_config` builds each mesh in Python, places it, and writes an OBJ. Blender only imports that OBJ. The add-on exports a binary PLY. `patch_xml` assigns the Mitsuba BSDF. Mitsuba renders the 15-channel EXR. `paint_ids` and schema 3 write the training ids. `publish` moves the staging directory only after the publication audit.

Geometry is not created inside Blender. `build_asset` calls `sample_shape_params` and `mesh_for_family` for the eight families, then `require_clean`. There is no mesh cache. Dragon is a separate ASCII PLY path and is rejected by `random_config`.

`asset_id` is the family name. `asset_instance_id` hashes the family plus the shape parameters. The logical object is `object_00` with `object_index` 1. The Mitsuba id is `mesh-object_00`. `visible_pixels` is the ownership count. `object_id_pixel_count` is the image count.

## What a library would change

Only the mesh source. `random_config` would load a validated local OBJ when the stratum names `library_asset_id`, then call the same `place_objects`, datablock copy, XML patch, and publisher. The asset file has no material. The preset stays on the stratum.

The library format is OBJ plus `asset.json`, because `bpy.ops.wm.obj_import` is the only mesh import the dataset script calls. The add-on already writes PLY for Mitsuba. glTF and USD are not referenced by the add-on or the dataset script, so they are unverified. A `.blend` link would still need the datablock copy.

Admission is once per version: hash, `mesh_issues` under the declared closed or open policy, and one import-export load. Every scene still runs placement, 5 mm clearance, the floor, the 80-pixel rule, and publication. Admission cannot see a future pose.

`library_asset_id`, version, and geometry hash go into the stratum, the instance-id hash, and the sample metadata. Resume keeps refusing a changed fingerprint. Dataset v1 hashes stay as they are.

## Risks

Identical shape parameters already share an `asset_instance_id` across splits. A shared Blender datablock drops objects, so the copy stays. More than four identical BSDFs is outside the tested opacity patch. A cone at 0.036 m can fail the 80-pixel rule even after admission if the pose is poor. Production pairs above 2,500 triangles were not measured by the independent clearance checker.

## Later experiment

A separate dataset id should admit a sphere and a 0.05 m cone, reject a degenerate mesh, and run a few CLI scenes. That experiment is specified and was not run.

The forward model and any new generator are also out of scope.

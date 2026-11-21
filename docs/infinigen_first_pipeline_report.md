# Infinigen-first pipeline

LivingRoom seed 21 attempt 3 is the first full scene. The blend was not regenerated. The Indoors pin is `b11700ebac77695d8f53538705cfdc0cd890c54d`. The local patch file hashes to `2f98aa479d074cbfa508e51f92f7d725de5b2d8e0ee97780feb8a635381297e9`. Nothing was committed, pushed, or registered. Dataset v1 and `assets/candidates/index.json` were left unchanged.

## Coverage flags

A flag is true only for the counts in its row. Definitions are in the evidence list below. The machine-readable copy is `dataset/qualification/asset_realism/reports/livingroom_21_acceptance.json`.

| Flag | Value | Counts |
| --- | --- | --- |
| generation_valid | true | attempt 3, 26 lights, 2 cameras |
| geometry_valid | true | 186 source meshes, corner error 1.36e-6 m |
| support_valid | false | 17 of 124 contacts passed |
| semantic_valid | false | sofa-to-television dot 0.221 |
| materials_export_valid | true | 523 regions, 873 texture hashes |
| materials_render_valid | true | 460 pplastic, 26 roughdielectric, 37 diffuse |
| mitsuba_subset_renderable | true | 45-instance preview |
| mitsuba_full_scene_renderable | true | 187 instances, 2 cameras, Stokes EXR |
| replay_preview_valid | false | the `/tmp` preview was not kept |
| replay_full_scene_valid | true | copied package, 187 instances, 873 hashes |
| polarization_qualified | false | separate from the Stokes film |

## Seventeen evidence items

1. **Generation.** Attempt 3 ran `generate_indoors` with `singleroom.gin` and no `fast_solve.gin`. Coarse finished in 0:31:18. `spawn_timeout` count is 0. The blend is `source_runs/LivingRoom_full_21_retry/scene.blend`, SHA-256 `6d38b354627858e43fda07127b05a1fe4386cd134d42a9ec52b42ae550249a99`. Cycles wrote `frames/Image/camera_0/Image_0_0_0001_0.png` in 85.56 s.

2. **Attempts 1 and 2 stayed on disk.** Attempt 1 hung in `compute_stable_poses` with no blend. Attempt 2 died in `room_walls` when `Concrete.generate()` received `vertical`. The checkout filters kwargs by signature. The patch file was not edited again for this qualification.

3. **Export axis.** Official OBJ is Y-up. The package maps `(x, y, z)` to Blender `(x, -z, y)` and stores the instance translation on the matrix. Surface tolerance is `max(1 mm, 0.1% of the source diagonal)`. That tolerance is export integrity. It is separate from shape-from-polarization ground truth.

4. **Surface samples.** 186 exported source instances were compared. Twenty-four deterministic evaluated world vertices were matched to every exported vertex. All 186 passed. The maximum corner error on the later eight-triangle check is 1.36e-6 m. The calibration marker is the 187th instance and is an added helper.

5. **Normals, winding, and faces.** Eight source triangles per instance were matched by nearest exported vertex. Winding agreement is 1.0 on every instance. Exported face lines are never fewer than source polygons, so the lost-face count is 0. The maximum normal angle is 22.15 degrees, and every such angle above 1 degree comes from a triangle smaller than 1 mm², where a micrometre of OBJ rounding moves the normal. Triangles of at least 1 mm² stay within 1 degree.

6. **Instance records.** Each of the 187 instances has an id, asset id, 4×4 matrix, parent key, and material regions. The calibration marker's parent is null because the package adds it. Shared geometry is allowed. The checked instance matrices for the sofa, television, monitor, desk, and door are pure translations; Infinigen baked rotation into the vertices.

7. **Support.** `measure_support` uses the lowest 400 exported vertices against the supporter's top polygon. The gap limit is 1 mm and the penetration limit is 0.1 mm. The living-room floor polygon is the floor AABB at Z = 0.1124 m. The sofa gap is 0.023 mm and passes. The desk, cabinet, shelves, bookcase, television stand, both rugs, and the desk lamp also pass. The plant gap is 13.0 mm and fails. Skirting penetration is 0.22 mm and fails. 76 of 78 shelf trinkets have no supporter within 5 cm of a furniture AABB top; that test does not see an interior shelf. Seed 11's sofa gap of 0.516 m remains a rejected measurement in `tests/test_infinigen_validation.py`.

8. **Semantics.** A facing relation passes when the horizontal forward and the direction to the target have dot product at least 0.5. The sofa forward is `[-0.962, -0.272, 0]`, measured from backrest to seat. Its dot with the direction to the television is 0.221, so `semantic_valid` is false. The television's dominant front faces the sofa (dot 0.925). The monitor faces the sofa (dot 0.626). The room has no chair, which is recorded as `not_present` and passes. Door facing and door access do not all pass. The walking-path segment from the first door to the sofa/display midpoint is clear of the tested furniture boxes.

9. **Material ledger.** 523 rows close the count: 509 paired to a source slot, 13 package slots whose source index has no material, and 1 calibration marker added by the package. The earlier 500 pplastic + 22 dielectric sum is 522 because it omitted the calibration diffuse. 217 blend meshes have zero material slots, are not hidden, and are outside the 523. 219 regions have a displacement link on the material output. 7 have a bump link. A Cycles `BUMP` displacement method is a shader setting. `displacement_mode` on the package is `NONE`.

10. **BSDF choice.** The audited families rendered in the full scene are 460 `pplastic`, 26 `roughdielectric`, and 37 `diffuse`. `pplastic` is used for coated surfaces and the report records that Mitsuba's model does not conserve energy. Rough glass stays `roughdielectric`, which is depolarizing on this pin. Metalness without optical constants becomes `diffuse`, with `metallic_optical_constants` listed as lost. The 13 source-less slots are `diffuse` for the same reason: a missing node graph is not turned into `pplastic`. `measured_polarized` is unused because no measured pBRDF was supplied. `quantitative_polarization` is false wherever the choice is an approximation.

11. **Normal-map direction.** The probe in `tests/test_infinigen_probes.py` renders an OpenGL tangent map. A tilt toward -X is brighter under the light at x = -0.6 than a tilt toward +X. A green-high tilt, which is +Y when `POS_Y` is the bitangent, is brighter under a +Y light than a green-low tilt. Green was left unflipped. The bake settings in the blend are tangent space, `normal_g = POS_Y`.

12. **Swatches.** `gallery/LivingRoom/21/qualified/` has one plane each for wood (`pplastic`, `shader_wood`), plaster (`pplastic`), fabric (`pplastic`, `shader_rug`), plastic (`pplastic`, `shader_black`), metal (`diffuse`, `shader_metal`), and glass (`roughdielectric`, `shader_glass`). Wood, plaster, fabric, plastic, and metal show a shaded plane. The glass plane stayed dark in front of a red backdrop, so that swatch does not demonstrate transmission.

13. **Lights, from the evaluated depsgraph.** 22 area lights are square, one-sided, spread π radians, and their evaluated translation is the origin even when the parent is a window. They are omitted with `reason=light_transform_at_origin`. The one-sided Lambertian formula `radiance = power / (π × area)` was not applied to a light at the origin. Four ceiling lamps have `shadow_soft_size` 0.025056 m and power 55.610649 W. They are Lambertian spheres: area `4πr²` = 0.007889 m², radiance `power / (area × π)` = 2243.67 W/sr/m². The isotropic point value `power / (4π)` = 4.425 W/sr was not used. The Blender center is at Z ≈ 2.907 m, inside the fixture. The render center is Z = 2.824 m, which is fixture bottom minus radius minus 2 mm, so the sphere hangs below the shade.

14. **Full room images.** `full_primary.png` and `full_secondary.png` are the two blend cameras, `camera_0_0` and `camera_0_1`, at 640×360 and 16 samples. Both EXRs are float32 with shape `(360, 640, 15)`. `render_stokes` writes that buffer only when the channel names match the Stokes film. The PNG is uint8 sRGB from `write_rgb_png` at exposure 1. Linear mean RGB on the primary EXR is about `[0.276, 0.261, 0.266]`. The pictures show the doorway, window, television, stand, rug, desk, and shelf. Environment radiance is 1.2 and is labeled as a replacement for the unconverted world shader.

15. **Why 62 meshes were welded.** A direct load of the archived scatter asked CUDA for a 4 GB block and failed. Four corals alone are about 14.6 million faces, with median edges near 0.2 mm. Textures are 225 MB and are not the bottleneck. The render copies grid-weld vertices. The cell diagonal equals that mesh's export tolerance, because a vertex can move by almost the full diagonal when the other vertices in the cell sit at the opposite corner. 62 meshes went from 36,728,435 faces to 13,134,704. The largest measured move is 1.47 mm, on the sofa, whose tolerance is 2.64 mm. Every welded mesh is within its own tolerance. The archived OBJ paths in `scene_manifest.json` are unchanged. `geometry_valid` uses those archived meshes.

16. **Replay.** The package directory was copied to `dataset/qualification/asset_realism/replay/LivingRoom_21_package`. That directory has no blend. `replay_infinigen_scene` checked 187 instances, 26 light records, and 873 texture hashes, rendered, deleted the PNG and EXR, and rendered again. `replay_matches` is true. The variant is `cuda_ad_spectral_polarized`.

17. **Tests.** Targeted coverage is in `tests/test_infinigen_validation.py`, `tests/test_infinigen_probes.py`, and `tests/test_material_choice.py`: instance parent, support gap, facing, BSDF choice, the normal direction, light radius and two-sided area, and the cluster bound. Full pytest is 314 passed and 2 failed, in 220 seconds. The log is `reports/pytest_full_livingroom.log`. Both failures are `tests/test_phase4.py` opening the missing `outputs/phase3/frames/000/stokes/render.exr`. That file was not created.

## What the pictures are

Diagnostic layers, after the lamp spheres were lowered:

- `qualified/layer_architecture.png`: shell, doorway, window, ceiling lamp.
- `qualified/layer_furniture.png`: sofa, desk, stand, and rug under the environment.
- `qualified/layer_small_objects.png`: the small-object set.
- `qualified/layer_scatter.png`: the shelf trinkets on a white environment field.
- `qualified/full_primary.png` and `qualified/full_secondary.png`: the paired room.

`gallery/LivingRoom/21/mitsuba_primary.png` is the older 45-instance preview. It is the subset flag, and it is a different image from the full room.

## Seed 11

Seed 11 is the fast scene and stays rejected. Its sofa gap is 0.516 m in the evaluated blend, and `measure_support` fails it. The blend hash is `b54a7cb6f37d516e21419d936fc1e0c8e8de7009ac28acd0b78760792b732c0c`.

## Expansion

The LivingRoom gate used for expansion is generation, geometry, the 523-row audit, the paired full-room renders, and the copied replay. `polarization_qualified` stays false. `support_valid` and `semantic_valid` also stay false, and the numbers above are the reason.

The expansion queue is `scripts/infinigen/run_full_expansion.py`. It calls the same `coarse_command` as `indoor_generate` with `fast=False`, and the output folder is an absolute path. The rooms are Kitchen 31, Bedroom 41, DiningRoom 51, Bathroom 61, Office 71, then two more full seeds for each of the six categories: LivingRoom 22 and 23, Kitchen 32 and 33, Bedroom 42 and 43, DiningRoom 52 and 53, Bathroom 62 and 63, Office 72 and 73. LivingRoom 21 is not in that list. Workshop is not a tag. The log is `reports/expansion_queue.log`. Fast seeds do not count.

The first Kitchen 31 run reached `room_floors` after populate and died because `utility_floor` named the modules `ceramic.plaster` and `ceramic.tile`. Those entries are now the classes `Plaster` and `Tile`. The failed directory is `source_runs/Kitchen_full_31_floor_fail`. It has no blend. The current checkout diff is `reports/indoors_local_current.patch`, SHA-256 `2730888215eddd934318b4298a1effd00a7e3ebe4ea4660fd30718d632e0fe31`. The earlier patch file `indoors_local.patch` is unchanged at `2f98aa479d074cbfa508e51f92f7d725de5b2d8e0ee97780feb8a635381297e9`.

The retried Kitchen 31 finished. Coarse took 1521 s and the Cycles render returned 0. The blend is `source_runs/Kitchen_full_31/scene.blend`, 2.2 GB, SHA-256 `a790f39a079d71c66016704e6515dcb61e92f2f698da1770747fa0c16c435d99`. The coarse log does not mention `fast_solve`. The Cycles frame shows a stove, counter, and shelving, which is a different room from LivingRoom 21. Bedroom 41 was started after that. `reports/room_diversity.json` lists which of the 18 planned blends exist. A seed without a blend is not counted, and two different seed numbers are not treated as diversity until their dimensions, counts, and hashes are both present.

## Sources

Indoors is `/home/cmoore/infinigen_src/indoors-stable` at `b11700eb`. Infinigen 2.0.0a5 is installed and did not produce a sofa, lamp, or material ball. Articulated objects were inventoried and not placed. `infinigen_gpl` at `10c1d76f` is GPL-3.0. The license of the generated meshes stays unresolved. The a5 wheel hash ends in `cb82f`. The sdist matches `0a57d2d80fc85168b3702f7e4a16f3483e9258d9232105c14ec48d525c06a77b`.

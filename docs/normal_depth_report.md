# Normal and depth on the prototype sphere

Measured on the Blender-exported UV sphere, 192 segments by 96 rings, radius 0.05 m, center `(0, 0, 0.051)`. The camera is `(0, -0.55, 0.32)` looking at `(0, 0, 0.18)`. The diagnostic pass is `llvm_ad_spectral_polarized`, 64×64, box filter, 1 sample. AOVs were `sh_normal`, `geo_normal`, `shape_index`, `prim_index`, `position`, and `depth`. Mitsuba 3.9.1 accepted that string in one integrator.

## What the 4 degree number was

The earlier 3.5–4.5° median compared the AOV normal with the radius to the **continuous** sphere along the camera ray. That intersection sits outside the triangle. It is not the point Mitsuba shaded.

On pixels whose position AOV is within 3 mm of the sphere and whose geometric normal faces the camera (`dot(normal, -ray) > 0.5`, 66 pixels):

| Comparison | Median | Max |
| --- | ---: | ---: |
| `geo_normal` vs `sh_normal` | 0.00° | 0.02° |
| `geo_normal` vs radius through the position hit | 0.60° | 1.26° |
| Share of those pixels under 1° | 0.91 |  |
| `geo_normal` vs radius to the continuous sphere | 4.43° |  |

The 1° criterion holds as a median on the mesh hit. It does not hold for the continuous sphere. The automated gate is the median, at most 1°, plus the 95th percentile under the old 8° cap. `analytic_sphere_median_deg` in the sample metadata is the continuous-sphere number and is not the gate. The 256², 64-sample CUDA sample measured mesh-hit median 0.684° and maximum 1.277°.

## Pixels above 8 degrees

A second pass on that same 256² scene, with `prim_index` enabled, found 12 of 1032 interior pixels above 8°. They sit on primitive 0 or 1. Every face of the exported sphere, all 36480 of them, has a positive outward dot with the vector from the center through the face centroid. Those two primitives point outward as well. The geometric normal on the 12 pixels is about 102° from the primitive's own face normal, so the pixel is not a sample of that triangle. The coverage rule is: a pixel whose geometric normal disagrees with its `prim_index` face by more than 20° is partial coverage and is left out of the angle score. The face stays in the mesh. A fixture with one reversed triangle, one degenerate triangle, or a non-manifold edge is rejected before export. Open assets, including `dragon_local`, are not required to be watertight.

## What the AOV normal is

On the eight most camera-facing sphere pixels, `geo_normal` matched the face normal of the `prim_index` triangle to 0.012° or better. `sh_normal` matched `geo_normal`. The PLY `nx,ny,nz` values match `shape.vertex_normal` for the same vertex index. A corner of the hit triangle can differ from the face normal by about 2°, which is the split between a vertex radius and the flat triangle. The AOV is the face normal, not a re-smoothed radius and not a camera-space vector.

The primary SfP target stays the camera-space shading normal. On this mesh it is the same direction as the geometric normal, then rotated by `world_normals_to_camera` so it points toward the camera.

## Depth

For the same sphere pixels, `|position - camera_origin| - depth` has median 10.23 mm. The near clip is 10 mm. `|position - camera_origin| - (depth + 0.01)` has median 0.23 mm and max 0.39 mm. Raw depth is not the pinhole distance. Adding `near_clip` recovers it to a fraction of a millimetre. The leftover is the tilt of the near plane: the ray meets that plane at `near_clip / dot(ray, camera_forward)`, which equals `near_clip` only on the optical axis.

Mitsuba's depth AOV documentation describes distance from the pinhole. This render does not match that sentence. The metadata keeps the measured convention.

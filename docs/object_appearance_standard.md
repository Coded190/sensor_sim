# Object appearance standard

This is the viewing standard for one material on one object. Blender 4.5 shows it with a Principled BSDF. Mitsuba keeps the same job split across its own BSDF plugins. A texture image does not replace the BSDF family.

## What every object has

1. A mesh with a UV map. U and V are the 2D coordinates that place an image on the surface. Image textures use that map. A mesh with only vertex positions cannot carry an image.
2. One BSDF family. Diffuse, rough plastic, rough metal, or dielectric. Do not mix families on one object.
3. Base color. One linear RGB value, or an sRGB image connected to Base Color. Linear means twice the light is twice the stored number. A display PNG is sRGB and is converted to linear before shading.
4. Roughness. One value, or a Non-Color image. In Blender, 0 is a sharp reflection and 1 is diffuse. Mitsuba stores Beckmann alpha, the root-mean-square microfacet slope, not that 0–1 slider.
5. IOR. One number for plastic, glass, and the wax coat. About 1.45–1.5. Metal uses the measured metal name, not this number.
6. Normal. The triangle normal, plus a Non-Color tangent-space normal image through a Normal Map node when the mesh does not already contain the small surface detail.

## What a class adds

| Class | Family | Extra inputs |
| --- | --- | --- |
| Paper, paint, cloth | Diffuse | Base-color image only |
| Plastic, fruit skin | Rough plastic | Base-color image, roughness image, normal image, IOR 1.45 |
| Fruit and wax | Rough plastic plus subsurface and coat | Subsurface weight, radius, and scale. Coat weight, coat roughness, and coat IOR |
| Metal | Rough conductor | Roughness image and normal image. No painted base color |
| Clear glass or wrap | Dielectric | IOR only, unless the surface is frosted or wrinkled |

## What is not part of this standard

Metallic maps, ambient occlusion, emission, alpha, transmission, sheen, anisotropy, and thin film. A metallic map blends two families. Ambient occlusion is a baked shadow the path tracer already computes.

## Apple demonstration

The generated mesh is `assets/candidates/prop_apple/1.0.0`, 952 triangles, one diffuse reflectance `[0.62, 0.14, 0.12]`, and no UVs. The Blender Cycles comparison is in `local_review/apple_blender_material/`. Those PNGs are Blender Cycles renders, not Mitsuba renders. The textured apple adds a UV map, a base-color image, a roughness image, a tangent normal image, subsurface scale 4 mm, and a wax coat of weight 0.25. The mesh silhouette is unchanged.

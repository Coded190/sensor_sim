# Blender to Mitsuba export

Measured on this machine. The project Mitsuba pin stays 3.9.1.

## Versions

| Piece | Version |
| --- | --- |
| Blender | 4.5.9 LTS, hash `8bf95cbd38d1`, `tools/blender-4.5.9-linux-x64/blender` |
| Blender Python | 3.11.11 |
| Project Python | 3.12.3 |
| Project Mitsuba | 3.9.1 |
| Project Dr.Jit | 1.5.0 |
| Mitsuba inside Blender's Python | 3.9.1, Dr.Jit 1.5.0 |

`cuda_ad_spectral_polarized` imports in the project environment and a 16×16 Stokes render completed on it. The same variant list is present in Blender's Python after installing the 3.9.1 wheel there. That install does not change the project environment.

PyPI's newest Mitsuba release is 3.9.1. `mitsuba==3.10.0` is not installable. The project was not upgraded.

## Add-on

The current `mitsuba-blender` master (`d3df446`, 2026-09-23) does not register on Mitsuba 3.9.1. `plugins/postprocess/filmic.py` subclasses `mi.PostProcess`, and 3.9.1 has no such type.

Commit `71302ac` registers far enough to import plugins, then fails because `mi.RayFlags.Default` does not exist. The 3.9.1 enum has `Empty` and `Minimal`, not `Default`.

Commit `ff23ac4` registers, then mesh export calls `mi.Mesh.from_corners`. That method is not on the 3.9.1 `Mesh`.

Commit `8848ded5e5df575f2cd0a9c10061d3f6b7137905` (2026-09-06, "fix the HUE converter node type name") registers with Mitsuba 3.9.1. Its mesh writer uses `from_corners` only when the method exists, and otherwise builds the mesh with `vertex_positions`. A headless export of a sphere, ground plane, camera, and area light finished with `Info: Scene exported successfully!` and did not call `mi.render`. The project interpreter then loaded that XML with `mi.load_file`.

The vendored copy is `third_party/mitsuba_blender`, with the license in `third_party/mitsuba_blender_LICENSE`. Blender loads it as the extension `bl_ext.user_default.mitsuba_blender`.

## Export call

The operator id, from `mitsuba_blender/io/__init__.py` in that commit, is `export_scene.mitsuba`.

```python
bpy.ops.preferences.addon_enable(module="bl_ext.user_default.mitsuba_blender")
bpy.ops.export_scene.mitsuba(
    filepath="scene_exported.xml",
    export_ids=True,
    ignore_background=True,
    axis_forward="Y",
    axis_up="Z",
    check_existing=False,
)
```

`axis_forward="Y"` and `axis_up="Z"` keep a Z-up scene. A sphere placed at `(0.1, -0.2, 0.051)` stayed there. The default axes are `-Z` forward and `Y` up, which rotate the scene out of this project's Z-up frame.

Meshes are written as PLY files next to the XML, with paths relative to the XML. The Glass BSDF becomes a Mitsuba `dielectric` whose `int_ior` is the Glass IOR. The exterior IOR is omitted. In Mitsuba 3.9.1 that omission loads as `eta` 1.4995845556259155 rather than 1.5. The rendered XML adds `<float name="ext_ior" value="1"/>` from `scene_config.json`, which loads as `eta` 1.5. The untouched add-on file is kept as `scene_exported.xml`.

The camera is a Blender perspective camera aimed with `to_track_quat("-Z", "Y")`. After this export, its matrix matches `mi.ScalarTransform4f.look_at` for the configured origin, target, and up, to `2.6e-7`. Field of view, clip planes, and resolution are the Blender camera and render settings. The add-on reads the sample count from Cycles, so the builder sets `scene.cycles.samples` and `scene.cycles.max_bounces` before export. The rendered copy sets the film reconstruction filter to `box` so the Stokes image and the AOV pass share a filter.

Importing the XML back into Blender is not part of the dataset path.

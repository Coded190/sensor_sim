"""Frame and render the downloaded Poly Haven apple in Blender Cycles.

The source blend and textures stay in local_review. This does not call Mitsuba.
"""

from pathlib import Path

import bpy
import mathutils

SOURCE = Path("/home/cmoore/sensor_sim/local_review/reference_food_apple_01/food_apple_01_1k.blend")
OUT_DIR = SOURCE.parent
PNG = OUT_DIR / "blender_cycles_food_apple_01.png"
BLEND = OUT_DIR / "blender_cycles_food_apple_01.blend"


def main() -> None:
    bpy.ops.wm.open_mainfile(filepath=str(SOURCE))
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    try:
        scene.cycles.device = "GPU"
    except TypeError:
        scene.cycles.device = "CPU"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1200
    scene.render.film_transparent = False

    missing = []
    for image in bpy.data.images:
        if image.name == "Render Result":
            continue
        image.reload()
        width, height = image.size
        print(f"BLENDER_IMAGE name={image.name} size={width}x{height} colorspace={image.colorspace_settings.name} file={image.filepath}")
        if width == 0 or height == 0:
            missing.append(image.name)
    if missing:
        raise RuntimeError(f"reason=missing_texture images={','.join(missing)}")

    apple = bpy.data.objects["food_apple_01"]
    points = [apple.matrix_world @ vertex.co for vertex in apple.data.vertices]
    low = mathutils.Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    high = mathutils.Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    center = (low + high) * 0.5
    radius = max(high.x - low.x, high.y - low.y, high.z - low.z)

    world = bpy.data.worlds.new("studio_world") if scene.world is None else scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.62, 0.64, 0.66, 1.0)
        background.inputs["Strength"].default_value = 0.25
    scene.world = world

    camera_data = bpy.data.cameras.new("examine_camera")
    camera_data.lens = 85.0
    camera = bpy.data.objects.new("examine_camera", camera_data)
    camera.location = center + mathutils.Vector((0.0, -radius * 2.4, radius * 0.35))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(camera)
    scene.camera = camera

    for name, offset, energy, size in (
        ("key", mathutils.Vector((0.55, -0.85, 0.75)), 18.0, 0.45),
        ("fill", mathutils.Vector((-0.7, -0.35, 0.25)), 6.0, 0.6),
        ("rim", mathutils.Vector((0.15, 0.8, 0.55)), 10.0, 0.3),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.size = size * radius
        light = bpy.data.objects.new(name, light_data)
        light.location = center + offset * radius
        aim = center - light.location
        light.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(light)

    scene.render.filepath = str(PNG)
    bpy.ops.render.render(write_still=True)
    print(f"BLENDER_CYCLES_RENDER path={PNG}")

    for image in bpy.data.images:
        if image.packed_file is None and image.filepath:
            image.pack()
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND))
    print(f"BLENDER_FILE path={BLEND}")


if __name__ == "__main__":
    main()

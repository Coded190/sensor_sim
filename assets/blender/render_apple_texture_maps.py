"""Blender Cycles apple: the generated mesh, then UV maps on Principled BSDF.

This script does not call Mitsuba. Every PNG name starts with blender_cycles_.
"""

from __future__ import annotations

from pathlib import Path

import bpy

ROOT = Path("/home/cmoore/sensor_sim")
APPLE = ROOT / "assets/candidates/prop_apple/1.0.0/geometry/render.obj"
OUT = ROOT / "local_review" / "apple_blender_material"
SIZE = 1024
# The packaged apple material is one linear RGB reflectance. Blender stores the
# same numbers in the Principled Base Color socket.
FLAT_LINEAR_RGB = (0.62, 0.14, 0.12, 1.0)


def _reset() -> bpy.types.Object:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    try:
        scene.cycles.device = "GPU"
    except TypeError:
        scene.cycles.device = "CPU"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 960
    scene.render.film_transparent = False
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.55, 0.58, 0.62, 1.0)
    background.inputs["Strength"].default_value = 0.12
    scene.world = world
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(APPLE))
    created = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if len(created) != 1:
        raise RuntimeError(f"reason=import_count count={len(created)}")
    obj = created[0]
    obj.name = "prop_apple"
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    return obj


def _studio(obj: bpy.types.Object) -> None:
    import mathutils

    total = mathutils.Vector((0.0, 0.0, 0.0))
    for vertex in obj.data.vertices:
        total += obj.matrix_world @ vertex.co
    center = total / max(len(obj.data.vertices), 1)
    camera_data = bpy.data.cameras.new("camera")
    camera_data.lens = 85.0
    camera = bpy.data.objects.new("camera", camera_data)
    camera.location = center + mathutils.Vector((0.0, -0.22, 0.05))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    for name, location, energy, size in (
        ("key", center + mathutils.Vector((0.12, -0.16, 0.16)), 8.0, 0.12),
        ("fill", center + mathutils.Vector((-0.16, -0.08, 0.06)), 2.5, 0.18),
        ("rim", center + mathutils.Vector((0.02, 0.14, 0.12)), 5.0, 0.08),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        light.location = location
        aim = center - location
        light.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.collection.objects.link(light)


def _unwrap(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.151917, island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    if not obj.data.uv_layers:
        raise RuntimeError("reason=no_uv_layer")


def _image(name: str, color):
    image = bpy.data.images.new(name, SIZE, SIZE, alpha=False, float_buffer=False)
    image.colorspace_settings.name = color
    image.filepath_raw = str(OUT / f"{name}.png")
    image.file_format = "PNG"
    return image


def _principled(name: str):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (400, 0)
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (700, 0)
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material, shader


def _assign(obj, material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


def _render(path: Path) -> None:
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print(f"BLENDER_CYCLES_RENDER path={path}")


def _flat(obj) -> None:
    material, shader = _principled("apple_flat_linear_rgb")
    shader.inputs["Base Color"].default_value = FLAT_LINEAR_RGB
    shader.inputs["Roughness"].default_value = 1.0
    shader.inputs["Metallic"].default_value = 0.0
    _assign(obj, material)
    _render(OUT / "blender_cycles_apple_flat.png")


def _procedural(obj):
    material = bpy.data.materials.new("apple_procedural_source")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 8.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.55
    fine = nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 45.0
    fine.inputs["Detail"].default_value = 4.0
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.35
    color_ramp.color_ramp.elements[0].color = (0.28, 0.03, 0.02, 1.0)
    color_ramp.color_ramp.elements[1].position = 0.72
    color_ramp.color_ramp.elements[1].color = (0.55, 0.07, 0.04, 1.0)
    mid = color_ramp.color_ramp.elements.new(0.55)
    color_ramp.name = "apple_basecolor_ramp"
    rough_ramp = nodes.new("ShaderNodeValToRGB")
    rough_ramp.color_ramp.elements[0].position = 0.30
    rough_ramp.color_ramp.elements[0].color = (0.18, 0.18, 0.18, 1.0)
    rough_ramp.color_ramp.elements[1].position = 0.80
    rough_ramp.name = "apple_roughness_ramp"
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.55
    bump.inputs["Distance"].default_value = 0.0015
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(mapping.outputs["Vector"], fine.inputs["Vector"])
    links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(fine.outputs["Fac"], rough_ramp.inputs["Fac"])
    links.new(fine.outputs["Fac"], bump.inputs["Height"])
    links.new(color_ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(rough_ramp.outputs["Color"], shader.inputs["Roughness"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["IOR"].default_value = 1.45
    if "Subsurface Weight" in shader.inputs:
        shader.inputs["Subsurface Weight"].default_value = 0.65
    if "Subsurface Scale" in shader.inputs:
        shader.inputs["Subsurface Scale"].default_value = 0.004
    if "Subsurface Radius" in shader.inputs:
        shader.inputs["Subsurface Radius"].default_value = (1.0, 0.25, 0.12)
    if "Coat Weight" in shader.inputs:
        shader.inputs["Coat Weight"].default_value = 0.25
    if "Coat Roughness" in shader.inputs:
        shader.inputs["Coat Roughness"].default_value = 0.12
    if "Coat IOR" in shader.inputs:
        shader.inputs["Coat IOR"].default_value = 1.45
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    _assign(obj, material)
    print("BLENDER_PRINCIPLED_SOCKETS " + ",".join(socket.name for socket in shader.inputs))
    return material, color_ramp, rough_ramp, bump


def _bake_emit(obj, source_socket, image, name: str) -> None:
    material = obj.data.materials[0]
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    emission = nodes.new("ShaderNodeEmission")
    output = next(node for node in nodes if node.type == "OUTPUT_MATERIAL")
    links.new(source_socket, emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.select = True
    nodes.active = tex
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.bake(type="EMIT")
    image.save()
    print(f"BLENDER_BAKE map={name} path={image.filepath_raw}")
    links.new(next(node for node in nodes if node.type == "BSDF_PRINCIPLED").outputs["BSDF"], output.inputs["Surface"])
    nodes.remove(emission)
    nodes.remove(tex)


def _bake_maps(obj, color_ramp, rough_ramp, bump) -> dict:
    base = _image("blender_cycles_apple_basecolor", "sRGB")
    rough = _image("blender_cycles_apple_roughness", "Non-Color")
    normal = _image("blender_cycles_apple_normal", "Non-Color")
    _bake_emit(obj, color_ramp.outputs["Color"], base, "basecolor")
    _bake_emit(obj, rough_ramp.outputs["Color"], rough, "roughness")
    material = obj.data.materials[0]
    nodes = material.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = normal
    tex.select = True
    nodes.active = tex
    bpy.context.scene.render.bake.normal_space = "TANGENT"
    bpy.ops.object.bake(type="NORMAL")
    normal.save()
    print(f"BLENDER_BAKE map=normal path={normal.filepath_raw}")
    nodes.remove(tex)
    _ = bump
    return {"basecolor": base, "roughness": rough, "normal": normal}


def _textured(obj, images) -> None:
    material, shader = _principled("apple_textured")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    texcoord = nodes.new("ShaderNodeTexCoord")
    base = nodes.new("ShaderNodeTexImage")
    base.image = images["basecolor"]
    base.image.colorspace_settings.name = "sRGB"
    rough = nodes.new("ShaderNodeTexImage")
    rough.image = images["roughness"]
    rough.image.colorspace_settings.name = "Non-Color"
    normal_image = nodes.new("ShaderNodeTexImage")
    normal_image.image = images["normal"]
    normal_image.image.colorspace_settings.name = "Non-Color"
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.space = "TANGENT"
    normal_map.inputs["Strength"].default_value = 0.6
    links.new(texcoord.outputs["UV"], base.inputs["Vector"])
    links.new(texcoord.outputs["UV"], rough.inputs["Vector"])
    links.new(texcoord.outputs["UV"], normal_image.inputs["Vector"])
    links.new(base.outputs["Color"], shader.inputs["Base Color"])
    links.new(rough.outputs["Color"], shader.inputs["Roughness"])
    links.new(normal_image.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["IOR"].default_value = 1.45
    if "Subsurface Weight" in shader.inputs:
        shader.inputs["Subsurface Weight"].default_value = 0.65
    if "Subsurface Scale" in shader.inputs:
        shader.inputs["Subsurface Scale"].default_value = 0.004
    if "Subsurface Radius" in shader.inputs:
        shader.inputs["Subsurface Radius"].default_value = (1.0, 0.25, 0.12)
    if "Coat Weight" in shader.inputs:
        shader.inputs["Coat Weight"].default_value = 0.25
    if "Coat Roughness" in shader.inputs:
        shader.inputs["Coat Roughness"].default_value = 0.12
    if "Coat IOR" in shader.inputs:
        shader.inputs["Coat IOR"].default_value = 1.45
    _assign(obj, material)
    _render(OUT / "blender_cycles_apple_textured.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obj = _reset()
    _studio(obj)
    _unwrap(obj)
    _flat(obj)
    _procedural(obj)
    images = _bake_maps(obj, *_procedural_outputs(obj))
    _textured(obj, images)
    blend = OUT / "blender_cycles_apple_textured.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    print(f"BLENDER_FILE path={blend} uv_layers={len(obj.data.uv_layers)}")


def _procedural_outputs(obj):
    nodes = obj.data.materials[0].node_tree.nodes
    ramps = [node for node in nodes if node.type == "VALTORGB"]
    bump = next(node for node in nodes if node.type == "BUMP")
    color_ramp = min(ramps, key=lambda node: node.location.x)
    # Both ramps are at default location. Identify by the first element's color.
    color_ramp = nodes["apple_basecolor_ramp"]
    rough_ramp = nodes["apple_roughness_ramp"]
    return color_ramp, rough_ramp, bump


if __name__ == "__main__":
    main()

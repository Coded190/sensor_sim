"""Write two Blender files for a local quality check.

Each file places the subject meshes inside a furnished room. The room kind
comes from the scene-context distribution. A bare ground plane is rejected
before the file is saved.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import bpy
import mathutils

ROOT = Path("/home/cmoore/sensor_sim")
sys.path.insert(0, str(ROOT / "src"))

from sensor_sim.scene_gen.scene_context import compose_scene_records, require_scene_context, sample_room_kinds
from sensor_sim.scene_gen.placement_separation import blockers_for, place_clear, support_frame
from sensor_sim.scene_gen.asset_validate import triangulate
from sensor_sim.scene_gen.mesh_io import read_obj

REVIEW = ROOT / "local_review"
SPHERE = ROOT / "assets/candidates/fixture_sphere/1.0.0/geometry/render.obj"
BOX = ROOT / "dataset/qualification/material_wrap_review/targets/box.obj"
APPLE = ROOT / "assets/candidates/prop_apple/1.0.0/geometry/render.obj"
SHELL = ROOT / "dataset/qualification/material_wrap_review/wraps/sphere_full/smooth/shell_applied.obj"
OUT_RANDOM = REVIEW / "material_randomization.blend"
OUT_WRAP = REVIEW / "material_wrap.blend"
SUBJECTS = {"sphere": SPHERE, "box": BOX, "apple": APPLE}


def _reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.55, 0.55, 0.55, 1.0)
        background.inputs["Strength"].default_value = 0.15
    scene.world = world


def _collection(name: str):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def _material(name: str, color, roughness: float, metallic: float = 0.0, transmission: float = 0.0, ior: float = 1.5):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    shader.inputs["Roughness"].default_value = float(roughness)
    shader.inputs["Metallic"].default_value = float(metallic)
    if "Transmission Weight" in shader.inputs:
        shader.inputs["Transmission Weight"].default_value = float(transmission)
    shader.inputs["IOR"].default_value = float(ior)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def _assign(obj, material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


def _link(obj, collection):
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def _import(path: Path, name: str, collection):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(path))
    created = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if len(created) != 1:
        raise RuntimeError(f"reason=import_count path={path} count={len(created)}")
    obj = created[0]
    obj.name = name
    return _link(obj, collection)


def _matrix(values, offset) -> mathutils.Matrix:
    flat = [float(value) for row in values for value in row] if isinstance(values[0], (list, tuple)) else [float(value) for value in values]
    rows = [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]
    matrix = mathutils.Matrix(rows)
    matrix[0][3] += float(offset[0])
    matrix[1][3] += float(offset[1])
    matrix[2][3] += float(offset[2])
    return matrix


def _package_material(record: dict):
    manifest = json.loads((Path(record["catalog"]) / record["asset_id"] / record["version"] / "asset.json").read_text(encoding="utf-8"))
    spec = manifest["materials"][0]
    color = spec.get("reflectance") or spec.get("diffuse_reflectance") or [0.7, 0.7, 0.7]
    metallic = 1.0 if spec.get("type") in {"conductor", "roughconductor"} else 0.0
    roughness = min(1.0, float(spec["alpha"]) ** 0.5) if "alpha" in spec else 0.6
    if spec.get("type") == "dielectric":
        return _material(record["instance_id"], [0.95, 0.97, 0.95], 0.04, transmission=1.0, ior=float(spec.get("int_ior", 1.5)))
    return _material(record["instance_id"], color, roughness, metallic=metallic)


def _place_room(loaded: dict, collection, offset) -> None:
    for record in loaded["bundle"]["instances"]:
        mesh = Path(record["catalog"]) / record["asset_id"] / record["version"] / "geometry" / "render.obj"
        obj = _import(mesh, f"{loaded['room_kind']}_{record['instance_id']}", collection)
        obj.matrix_world = _matrix(record["matrix"], offset)
        _assign(obj, _package_material(record))


def _rest(obj, xy, support_z: float, offset) -> None:
    obj.location = (float(xy[0]) + float(offset[0]), float(xy[1]) + float(offset[1]), 0.0)
    bpy.context.view_layer.update()
    minimum = min((obj.matrix_world @ mathutils.Vector(corner)).z for corner in obj.bound_box)
    obj.location.z += float(support_z) + float(offset[2]) - minimum


def _label(text: str, location, collection) -> None:
    curve = bpy.data.curves.new(text, "FONT")
    curve.body = text
    curve.size = 0.08
    obj = bpy.data.objects.new(text, curve)
    obj.location = location
    collection.objects.link(obj)


def _camera(loaded: dict, offset, view: str = "primary") -> None:
    camera_data = bpy.data.cameras.new(f"camera_{loaded['room_kind']}")
    camera_data.lens = 28.0
    camera = bpy.data.objects.new(f"camera_{loaded['room_kind']}", camera_data)
    source = loaded["bundle"]["cameras"][view]
    origin = mathutils.Vector(source["origin"]) + mathutils.Vector(offset)
    target = mathutils.Vector(source["target"]) + mathutils.Vector(offset)
    direction = target - origin
    camera.matrix_world = mathutils.Matrix.LocRotScale(origin, direction.to_track_quat("-Z", "Y"), mathutils.Vector((1.0, 1.0, 1.0)))
    camera.data.angle = float(source["fov_deg"]) * 3.14159265 / 180.0
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    light_data = bpy.data.lights.new(f"area_{loaded['room_kind']}", "AREA")
    light_data.energy = 40.0
    light_data.size = 1.8
    light = bpy.data.objects.new(f"area_{loaded['room_kind']}", light_data)
    light.location = (target.x, target.y, max(origin.z, target.z) + 1.4)
    light.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.collection.objects.link(light)


def _subject_material(name: str, spec: dict):
    color = spec["color"]
    return _material(name, color, spec["roughness"], metallic=spec.get("metallic", 0.0), transmission=spec.get("transmission", 0.0), ior=spec.get("ior", 1.5))


OUT_COMBINED = REVIEW / "furnished_material_scenes.blend"
FIXED = {
    "sphere": {"color": (0.70, 0.12, 0.10), "roughness": 1.0},
    "box": {"color": (0.45, 0.45, 0.42), "roughness": 0.20, "ior": 1.5},
    "apple": {"color": (0.91, 0.91, 0.93), "roughness": 0.12, "metallic": 1.0},
}
DRAWS = [
    FIXED,
    {
        "sphere": {"color": (0.12, 0.22, 0.65), "roughness": 1.0},
        "box": {"color": (0.86, 0.42, 0.22), "roughness": 0.18, "metallic": 1.0},
        "apple": {"color": (0.15, 0.55, 0.18), "roughness": 0.47, "ior": 1.5},
    },
    {
        "sphere": {"color": (0.91, 0.91, 0.93), "roughness": 0.45, "metallic": 1.0},
        "box": {"color": (0.82, 0.62, 0.12), "roughness": 1.0},
        "apple": {"color": (0.50, 0.50, 0.50), "roughness": 1.0},
    },
]
DRAW_NOTES = [
    "Draw 1: red diffuse, smooth plastic, aluminum",
    "Draw 2: blue diffuse, copper, green rough plastic",
    "Draw 3: rough aluminum, yellow diffuse, gray diffuse",
]
LAYOUTS = {
    "row": {"sphere": (-0.16, 0.0), "box": (0.0, 0.0), "apple": (0.16, 0.0)},
    "triangle": {"sphere": (-0.12, -0.08), "box": (0.12, -0.08), "apple": (0.0, 0.10)},
    "column": {"sphere": (-0.08, -0.12), "box": (0.10, 0.0), "apple": (-0.02, 0.12)},
}


def _section(name: str):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def _child(parent, name: str):
    collection = bpy.data.collections.new(name)
    parent.children.link(collection)
    return collection


def _move(obj, translation, offset) -> None:
    obj.location = (
        float(translation[0]) + float(offset[0]),
        float(translation[1]) + float(offset[1]),
        float(translation[2]) + float(offset[2]),
    )


def _local_subject(name: str, path: Path) -> dict:
    vertices, polygons = read_obj(path)
    return {"name": name, "vertices": vertices, "faces": triangulate(polygons)}


def _subject_group(parent, name: str, room_kind: str, materials: dict, layout: dict, note: str, slot: int) -> None:
    loaded = compose_scene_records(room_kind, ["sphere", "box", "apple"])
    offset = (0.0, slot * 12.0, 0.0)
    collection = _child(parent, name)
    _place_room(loaded, collection, offset)
    frame = support_frame(loaded)
    subjects = [_local_subject(subject, SUBJECTS[subject]) for subject in layout]
    placed = place_clear(subjects, blockers_for(loaded, frame), frame["polygon"], offsets=layout)
    for subject, translation in zip(subjects, placed["translations"], strict=True):
        obj = _import(SUBJECTS[subject["name"]], f"{name}_{subject['name']}", collection)
        _assign(obj, _subject_material(f"{name}_{subject['name']}", materials[subject["name"]]))
        _move(obj, translation, offset)
    support = frame["z"]
    _label(
        f"{note}. room={room_kind} geometry={loaded['geometry_kind']} context_objects={loaded['report']['context_objects']} pattern_kept={placed['pattern_kept']}",
        (0.2 + offset[0], frame["polygon"][:, 1].mean() + offset[1] + 0.8, support + 0.45),
        collection,
    )
    _camera(loaded, offset, "primary")
    require_scene_context(loaded["records"], room_kind=room_kind)


def _wrap_group(parent, slot: int) -> None:
    loaded = compose_scene_records("lab", ["sphere", "plastic_shell"])
    offset = (0.0, slot * 12.0, 0.0)
    collection = _child(parent, "lab_workshop_shell")
    _place_room(loaded, collection, offset)
    frame = support_frame(loaded)
    bare_mesh = _local_subject("sphere_unwrapped", SPHERE)
    shell_mesh = _local_subject("plastic_shell", SHELL)
    placed = place_clear(
        [bare_mesh, shell_mesh],
        blockers_for(loaded, frame),
        frame["polygon"],
        offsets={"sphere_unwrapped": (-0.18, 0.0), "plastic_shell": (0.18, 0.0)},
    )
    red = _material("sphere_red_diffuse", (0.70, 0.12, 0.10), 1.0)
    film = _material("plastic_film_dielectric", (0.95, 0.97, 0.95), 0.02, transmission=1.0, ior=1.5)
    bare = _import(SPHERE, "sphere_unwrapped", collection)
    _assign(bare, red)
    _move(bare, placed["translations"][0], offset)
    wrapped = _import(SPHERE, "sphere_inside_shell", collection)
    _assign(wrapped, red)
    _move(wrapped, placed["translations"][1], offset)
    shell = _import(SHELL, "plastic_shell", collection)
    _assign(shell, film)
    _move(shell, placed["translations"][1], offset)
    _label(
        f"Shrink wrap. Lab uses the {loaded['geometry_kind']} shell. Left sphere is bare. Right sphere has a separate plastic shell. context_objects={loaded['report']['context_objects']} pattern_kept={placed['pattern_kept']}",
        (frame["polygon"][:, 0].mean() + offset[0] - 0.8, frame["polygon"][:, 1].mean() + offset[1] + 0.7, frame["z"] + 0.4),
        collection,
    )
    _camera(loaded, offset, "task")
    require_scene_context(loaded["records"], room_kind="lab")


def build_combined() -> None:
    _reset()
    kinds = sample_room_kinds(random.Random(17521), 4)
    layout_kinds = kinds[:3]
    draw_kind = kinds[3]
    randomization = _section("01_material_randomization")
    slot = 0
    for layout, room_kind in zip(("row", "triangle", "column"), layout_kinds, strict=True):
        _subject_group(
            randomization,
            f"fixed_{layout}_{room_kind}",
            room_kind,
            FIXED,
            LAYOUTS[layout],
            f"Fixed materials, {layout} placement",
            slot,
        )
        slot += 1
    for index, (draw, note) in enumerate(zip(DRAWS, DRAW_NOTES, strict=True), start=1):
        _subject_group(
            randomization,
            f"draw{index}_{draw_kind}",
            draw_kind,
            draw,
            LAYOUTS["row"],
            note,
            slot,
        )
        slot += 1
    wrap = _section("02_shrink_wrap")
    _wrap_group(wrap, slot)
    REVIEW.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_COMBINED))
    print(f"BLEND path={OUT_COMBINED} layout_kinds={','.join(layout_kinds)} draw_kind={draw_kind} wrap=lab")


if __name__ == "__main__":
    build_combined()

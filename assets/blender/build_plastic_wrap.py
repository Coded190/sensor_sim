"""Build one Shrinkwrap shell, apply the modifiers, and export it.

The target mesh file is only read. The shell is a new object.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import bmesh
import mathutils

ADDON = "bl_ext.user_default.mitsuba_blender"


def _params() -> dict:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("missing -- params.json")
    return json.loads(Path(argv[argv.index("--") + 1]).read_text(encoding="utf-8"))


def _clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    enabled = bpy.ops.preferences.addon_enable(module=ADDON)
    if enabled != {"FINISHED"}:
        raise RuntimeError(f"could not enable {ADDON}: {enabled}")


def _import_obj(path: Path):
    bpy.ops.wm.obj_import(filepath=str(path))
    obj = bpy.context.selected_objects[0]
    obj.name = "target"
    return obj


def _apply(obj) -> None:
    bpy.context.view_layer.objects.active = obj
    for name in [modifier.name for modifier in obj.modifiers]:
        bpy.ops.object.modifier_apply(modifier=name)


def _normals_away_from(obj, target) -> None:
    """Point face normals out of the film, away from the target center, before Solidify."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    mesh = bmesh.from_edit_mesh(obj.data)
    center = mathutils.Vector((0.0, 0.0, 0.0))
    for vertex in target.data.vertices:
        center += vertex.co
    center = target.matrix_world @ (center / max(len(target.data.vertices), 1))
    average = mathutils.Vector((0.0, 0.0, 0.0))
    origin = mathutils.Vector((0.0, 0.0, 0.0))
    for face in mesh.faces:
        average += face.normal
    for vertex in mesh.verts:
        origin += vertex.co
    origin = obj.matrix_world @ (origin / max(len(mesh.verts), 1))
    if average.dot(origin - center) < 0.0:
        for face in mesh.faces:
            face.normal_flip()
        mesh.normal_update()
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")


def _partial(target, spec: dict):
    points = [vertex.co for vertex in target.data.vertices]
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    zs = [point.z for point in points]
    span = 0.55 * min(max(xs) - min(xs), max(ys) - min(ys))
    span = max(float(span), 0.04)
    center_x = 0.5 * (min(xs) + max(xs))
    center_y = 0.5 * (min(ys) + max(ys))
    # Keep the object at the origin so vertex coordinates stay in the target file's frame.
    bpy.ops.mesh.primitive_grid_add(size=span, x_subdivisions=int(spec["subdivision"]) * 8, y_subdivisions=int(spec["subdivision"]) * 8, location=(0.0, 0.0, 0.0))
    sheet = bpy.context.active_object
    height = max(zs) + float(spec["offset_m"]) + float(spec["thickness_m"]) + 0.001
    for vertex in sheet.data.vertices:
        vertex.co.x += center_x
        vertex.co.y += center_y
        vertex.co.z = height
    sheet.name = "shell"
    wrap = sheet.modifiers.new("shrinkwrap", "SHRINKWRAP")
    wrap.target = target
    wrap.wrap_method = "NEAREST_SURFACEPOINT"
    wrap.offset = float(spec["offset_m"])
    _apply(sheet)
    _normals_away_from(sheet, target)
    if spec["wrinkle_amplitude_m"] > 0.0:
        texture = bpy.data.textures.new("wrinkle", "CLOUDS")
        texture.noise_scale = 1.0 / max(float(spec["wrinkle_frequency_per_m"]), 1.0)
        displace = sheet.modifiers.new("wrinkle", "DISPLACE")
        displace.texture = texture
        displace.strength = float(spec["wrinkle_amplitude_m"])
        displace.direction = "NORMAL"
        _apply(sheet)
        _normals_away_from(sheet, target)
    solid = sheet.modifiers.new("solidify", "SOLIDIFY")
    solid.thickness = float(spec["thickness_m"])
    solid.offset = 1.0
    solid.use_rim = True
    _apply(sheet)
    return sheet


def _full(target, spec: dict):
    shell = target.copy()
    shell.data = target.data.copy()
    shell.name = "shell"
    bpy.context.collection.objects.link(shell)
    bpy.context.view_layer.objects.active = shell
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.transform.shrink_fatten(value=float(spec["offset_m"]) + float(spec["thickness_m"]))
    bpy.ops.object.mode_set(mode="OBJECT")
    wrap = shell.modifiers.new("shrinkwrap", "SHRINKWRAP")
    wrap.target = target
    wrap.wrap_method = "NEAREST_SURFACEPOINT"
    wrap.offset = float(spec["offset_m"])
    _apply(shell)
    _normals_away_from(shell, target)
    if spec["wrinkle_amplitude_m"] > 0.0:
        texture = bpy.data.textures.new("wrinkle", "CLOUDS")
        texture.noise_scale = 1.0 / max(float(spec["wrinkle_frequency_per_m"]), 1.0)
        displace = shell.modifiers.new("wrinkle", "DISPLACE")
        displace.texture = texture
        displace.strength = float(spec["wrinkle_amplitude_m"])
        displace.direction = "NORMAL"
        _apply(shell)
        _normals_away_from(shell, target)
    solid = shell.modifiers.new("solidify", "SOLIDIFY")
    solid.thickness = float(spec["thickness_m"])
    solid.offset = 1.0
    solid.use_rim = True
    _apply(shell)
    return shell


def _write_obj(obj, path: Path, world: bool) -> None:
    mesh = obj.data
    lines = [f"o {obj.name}"]
    for vertex in mesh.vertices:
        co = obj.matrix_world @ vertex.co if world else vertex.co
        lines.append(f"v {co.x:.8f} {co.y:.8f} {co.z:.8f}")
    for polygon in mesh.polygons:
        indices = " ".join(str(index + 1) for index in polygon.vertices)
        lines.append(f"f {indices}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _boundary_count(obj) -> int:
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    mesh.edges.ensure_lookup_table()
    count = sum(1 for edge in mesh.edges if edge.is_boundary)
    mesh.free()
    return count


def main() -> None:
    params = _params()
    spec = params["spec"]
    output = Path(params["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    _clear()
    target = _import_obj(Path(params["target_obj"]))
    before = Path(params["target_obj"]).read_bytes()
    shell = _partial(target, spec) if spec["mode"] == "partial" else _full(target, spec)
    applied = output / "shell_applied.obj"
    _write_obj(shell, applied, world=False)
    xml = output / "shell.xml"
    for obj in list(bpy.context.scene.objects):
        obj.hide_render = obj.name != "shell"
    bpy.ops.export_scene.mitsuba(filepath=str(xml), export_ids=True, ignore_background=True, axis_forward="Y", axis_up="Z", check_existing=False)
    after = Path(params["target_obj"]).read_bytes()
    meta = {
        "boundary_edges": _boundary_count(shell),
        "vertex_count": len(shell.data.vertices),
        "polygon_count": len(shell.data.polygons),
        "target_unchanged": before == after,
        "applied_obj": str(applied),
        "xml": str(xml),
    }
    (output / "blender_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"WRAP_EXPORT mode={spec['mode']} vertices={meta['vertex_count']} boundary={meta['boundary_edges']}")


if __name__ == "__main__":
    main()

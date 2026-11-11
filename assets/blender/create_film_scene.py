"""Build the ball, partial film, room, light, and camera. Run by Blender.

Blender is started with ``--background --python create_film_scene.py -- params.json``.
Mesh coordinates are written from this process so the OBJ axes match the
blend file. The materials below are Blender Principled BSDFs with the
same IOR and reflectance as the Mitsuba scene. Blender does not evaluate
Mitsuba's polarized dielectric, so a Blender render has no Stokes image.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
import mathutils

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sensor_sim.film.geometry import (
    CORNELL_HIGH,
    CORNELL_LOW,
    DRAGON_CENTER_XY,
    DRAGON_HEIGHT,
    DRAGON_REFLECTANCE,
    PLATFORM_REFLECTANCE,
    WALL_COLORS,
    ball_center_offset,
    cap_shell,
    cornell_shell,
    platform_mesh,
    reference_points,
    uv_sphere,
    write_obj,
)
from sensor_sim.film.lighting import area_power_watts


def _args() -> dict:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("missing -- params.json")
    payload = Path(argv[argv.index("--") + 1])
    return json.loads(payload.read_text())


def _mesh_object(name: str, vertices, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _look_at(obj, target, track: str, up: str) -> None:
    direction = mathutils.Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat(track, up).to_euler()


def _set_socket(node, name: str, value) -> None:
    socket = node.inputs.get(name)
    if socket is None:
        names = ", ".join(item.name for item in node.inputs)
        raise RuntimeError(f"Principled BSDF has no {name}. Sockets: {names}")
    socket.default_value = value


def _principled(name: str, color, *, transmission: float, ior: float, roughness: float, specular: float):
    """Viewport stand-in. Transmission 1 is glass; specular 0 is Lambert."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.use_backface_culling = False
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    _set_socket(bsdf, "Base Color", color)
    _set_socket(bsdf, "Metallic", 0.0)
    _set_socket(bsdf, "Roughness", roughness)
    _set_socket(bsdf, "IOR", ior)
    _set_socket(bsdf, "Transmission Weight", transmission)
    _set_socket(bsdf, "Specular IOR Level", specular)
    return material


def _assign(obj, material) -> None:
    obj.data.materials.append(material)


def _smooth(obj) -> None:
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.data.update()


def _export(obj, path, smooth: bool = True) -> None:
    write_obj(path, _world_vertices(obj), _triangles(obj.data), smooth=smooth)


def _place_dragon(path: Path, floor_z: float):
    """Stand the Stanford scan on the floor. The PLY is Y-up."""
    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=str(path))
    created = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if len(created) != 1:
        names = [obj.name for obj in created]
        raise RuntimeError(f"expected one dragon mesh from {path}, got {names}")
    dragon = created[0]
    dragon.name = "dragon"
    # +90° about X sends the scan's +Y up-axis to +Z. The opposite sign
    # puts the back on the floor and the feet in the air.
    dragon.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    bpy.context.view_layer.update()
    height = _world_span(dragon)[2]
    if height <= 0.0:
        raise RuntimeError(f"dragon from {path} has no height")
    factor = DRAGON_HEIGHT / height
    dragon.scale = (factor, factor, factor)
    bpy.context.view_layer.update()
    low, high = _world_bounds(dragon)
    dragon.location.x += DRAGON_CENTER_XY[0] - 0.5 * (low.x + high.x)
    dragon.location.y += DRAGON_CENTER_XY[1] - 0.5 * (low.y + high.y)
    dragon.location.z += float(floor_z) - low.z
    bpy.context.view_layer.update()
    return dragon


def _world_bounds(obj):
    corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    low = mathutils.Vector((min(corner.x for corner in corners), min(corner.y for corner in corners), min(corner.z for corner in corners)))
    high = mathutils.Vector((max(corner.x for corner in corners), max(corner.y for corner in corners), max(corner.z for corner in corners)))
    return low, high


def _world_span(obj):
    low, high = _world_bounds(obj)
    return high - low


def main() -> None:
    params = _args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    radius = float(params["radius"])
    scale = tuple(float(v) for v in params["scale"])
    thickness = float(params["thickness"])
    coverage = float(params["coverage"])
    axis = tuple(float(v) for v in params["axis"])
    out_dir = Path(params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # The YAML radius is the outer surface of the wrap. The plastic stops at
    # the inner film surface so the two meshes do not occupy the same points.
    # Triangle sagitta must stay under the 20 µm shell or the ball shows
    # through the middle of each face. 512 by 160 keeps the sagitta near 1 µm.
    reflectance = float(params["reflectance"])
    room_reflectance = float(params["room_reflectance"])
    ior = float(params["ior"])
    gap = float(params["floor_gap"])
    floor = float(CORNELL_LOW[2])
    offset = ball_center_offset(radius, scale[2], gap, floor_z=floor)
    ball_mesh = uv_sphere(radius - thickness, scale=scale, n_rings=128, n_phi=256)
    ball = _mesh_object("ball", ball_mesh["vertices"] + offset, ball_mesh["faces"])
    _smooth(ball)
    _assign(
        ball,
        _principled(
            "ball",
            (reflectance, reflectance, reflectance, 1.0),
            transmission=0.0,
            ior=1.5,
            roughness=1.0,
            specular=0.0,
        ),
    )

    shell = cap_shell(radius, thickness, coverage, axis, scale=scale, n_rings=160, n_phi=512)
    film = _mesh_object("film-shell", shell["vertices"] + offset, shell["faces"])
    _smooth(film)
    _assign(film, _principled("film-shell", (1.0, 1.0, 1.0, 1.0), transmission=1.0, ior=ior, roughness=0.0, specular=0.5))

    pedestal = platform_mesh(floor)
    platform = _mesh_object("platform", pedestal["vertices"], pedestal["faces"])
    # Leave the pedestal flat. Averaged corner normals point through the floor.
    _assign(
        platform,
        _principled(
            "platform",
            (*PLATFORM_REFLECTANCE, 1.0),
            transmission=0.0,
            ior=1.5,
            roughness=1.0,
            specular=0.0,
        ),
    )

    wall_objects = {}
    for name, mesh in cornell_shell().items():
        color = WALL_COLORS.get(name, (room_reflectance, room_reflectance, room_reflectance))
        obj = _mesh_object(name, mesh["vertices"], mesh["faces"])
        _smooth(obj)
        _assign(obj, _principled(name, (*color, 1.0), transmission=0.0, ior=1.5, roughness=1.0, specular=0.0))
        wall_objects[name] = obj

    dragon_path = ROOT / "assets" / "dragon" / "dragon_vrip_res4.ply"
    dragon = _place_dragon(dragon_path, floor)
    _smooth(dragon)
    _assign(
        dragon,
        _principled(
            "dragon",
            (*DRAGON_REFLECTANCE, 1.0),
            transmission=0.0,
            ior=1.5,
            roughness=0.6,
            specular=0.2,
        ),
    )

    radiance = float(params["light"]["intensity"])
    size_x = float(params["light"]["size"][0])
    size_y = float(params["light"]["size"][1])
    power_watts = area_power_watts(size_x, size_y, radiance)
    light_data = bpy.data.lights.new("light", type="AREA")
    light_data.shape = "RECTANGLE"
    light_data.size = size_x
    light_data.size_y = size_y
    light_data.color = (1.0, 1.0, 1.0)
    light_data.energy = power_watts
    light_data.exposure = 0.0
    light_data.normalize = True
    light = bpy.data.objects.new("light", light_data)
    light.location = params["light"]["position"]
    bpy.context.collection.objects.link(light)
    light_up = "Y" if abs(float(params["light"].get("up", [0, 0, 1])[1])) > 0.5 else "Z"
    _look_at(light, params["camera"]["target"], "-Z", light_up)

    camera_data = bpy.data.cameras.new("camera")
    camera_data.angle = float(params["camera"]["fov_deg"]) * 3.141592653589793 / 180.0
    camera = bpy.data.objects.new("camera", camera_data)
    camera.location = params["camera"]["position"]
    bpy.context.collection.objects.link(camera)
    _look_at(camera, params["camera"]["target"], "-Z", "Y")
    view_layer = bpy.context.scene
    view_layer.camera = camera
    view_layer.render.engine = "CYCLES"
    view_layer.cycles.device = "CPU"
    # AgX has a shoulder, and +2 stops is a 4x exposure. That is the viewport
    # counterpart of the PNG map. The two curves are not the same pixels.
    view_layer.view_settings.view_transform = "AgX"
    view_layer.view_settings.look = "None"
    view_layer.view_settings.exposure = 2.0
    view_layer.view_settings.gamma = 1.0
    world = bpy.data.worlds.new("world")
    view_layer.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0

    blend_path = Path(params["blend"])
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    _export(ball, out_dir / "ball.obj")
    _export(film, out_dir / "film_cap.obj")
    _export(platform, out_dir / "platform.obj", smooth=False)
    _export(dragon, out_dir / "dragon.obj")
    for name, obj in wall_objects.items():
        _export(obj, out_dir / f"{name}.obj")

    refs = reference_points(radius, coverage, axis, scale=scale, floor_z=floor)
    refs["film"] = refs["film"] + offset
    refs["uncovered"] = refs["uncovered"] + offset
    # Upper left of the blue back wall, clear of the ball and the dragon.
    refs["floor"] = __import__("numpy").array([-0.2, float(CORNELL_HIGH[1]), 0.4], dtype=float)
    object_names = ["ball", "film-shell", "platform", "dragon", *wall_objects, "light", "camera"]
    poses = {
        "camera": params["camera"],
        "light": {
            **params["light"],
            "radiance": radiance,
            "power_watts": power_watts,
            "exposure": 0.0,
            "normalize": True,
        },
        "materials": {
            "ball": "Principled diffuse, reflectance from YAML, specular 0",
            "film-shell": "Principled transmission, Mitsuba dielectric IOR, roughness 0. Not a polarized BSDF.",
            "platform": "Principled diffuse pedestal",
            "dragon": "Stanford dragon_vrip_res4, red diffuse (0.70, 0.05, 0.04). Y-up scan, +90 deg about X so +Y becomes +Z.",
            "floor": "gray diffuse",
            "ceiling": "gray diffuse",
            **{name: f"diffuse RGB {color}" for name, color in WALL_COLORS.items()},
        },
        "room_low": [float(v) for v in CORNELL_LOW],
        "room_high": [float(v) for v in CORNELL_HIGH],
        "floor_z": floor,
        "ball_center": [float(v) for v in offset],
        "points": {
            "film": refs["film"].tolist(),
            "uncovered": refs["uncovered"].tolist(),
            "floor": refs["floor"].tolist(),
            "uncovered_visible": refs["uncovered_visible"],
        },
        "objects": object_names,
    }
    Path(params["poses"]).write_text(json.dumps(poses, indent=2) + "\n")
    print(
        f"film blender=ok blend={blend_path} objects={','.join(poses['objects'])} "
        f"floor_z={floor:.6f} radiance={radiance:.6g} power_watts={power_watts:.6g}"
    )


def _world_vertices(obj):
    import numpy as np

    matrix = obj.matrix_world
    coords = [matrix @ vertex.co for vertex in obj.data.vertices]
    return np.array([[co.x, co.y, co.z] for co in coords], dtype=np.float64)


def _triangles(mesh):
    import numpy as np

    return np.array([[vertex for vertex in polygon.vertices] for polygon in mesh.polygons], dtype=np.int64)


if __name__ == "__main__":
    main()

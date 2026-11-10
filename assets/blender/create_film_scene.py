"""Build the ball, partial film, room, light, and camera. Run by Blender.

Blender is started with ``--background --python create_film_scene.py -- params.json``.
Mesh coordinates are written from this process so the OBJ axes match the
blend file. The materials below are Blender Principled BSDFs with the
same IOR and reflectance as the Mitsuba scene. Blender does not evaluate
Mitsuba's polarized dielectric, so a Blender render has no Stokes image.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import mathutils

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sensor_sim.film.geometry import cap_shell, floor_z, reference_points, room_box, uv_sphere, write_obj
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
    ball_mesh = uv_sphere(radius - thickness, scale=scale, n_rings=128, n_phi=256)
    ball = _mesh_object("ball", ball_mesh["vertices"], ball_mesh["faces"])
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
    film = _mesh_object("film-shell", shell["vertices"], shell["faces"])
    _assign(film, _principled("film-shell", (1.0, 1.0, 1.0, 1.0), transmission=1.0, ior=ior, roughness=0.0, specular=0.5))

    z_floor = floor_z(radius, scale[2], float(params["floor_gap"]))
    margin = float(params["margin"])
    points = [
        [-radius * scale[0], -radius * scale[1], z_floor],
        [radius * scale[0], radius * scale[1], radius * scale[2]],
        list(params["camera"]["position"]),
        list(params["light"]["position"]),
    ]
    array = __import__("numpy").asarray(points, dtype=float)
    low = array.min(axis=0) - margin
    high = array.max(axis=0) + margin
    low[2] = z_floor
    box = room_box(low, high)
    room = _mesh_object("room", box["vertices"], box["faces"])
    _assign(
        room,
        _principled(
            "room",
            (room_reflectance, room_reflectance, room_reflectance, 1.0),
            transmission=0.0,
            ior=1.5,
            roughness=1.0,
            specular=0.0,
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
    _look_at(light, params["camera"]["target"], "-Z", "Z")

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

    write_obj(out_dir / "ball.obj", _world_vertices(ball), _triangles(ball.data))
    write_obj(out_dir / "film_cap.obj", _world_vertices(film), _triangles(film.data))
    room = bpy.data.objects["room"]
    write_obj(out_dir / "room.obj", _world_vertices(room), _triangles(room.data))

    refs = reference_points(radius, coverage, axis, scale=scale, floor_z=z_floor)
    # A point on the far wall, offset in x so the camera ray misses the ball.
    refs["floor"] = __import__("numpy").array([0.22, float(high[1]), 0.3], dtype=float)
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
            "room": "Principled diffuse, room reflectance, specular 0",
        },
        "room_low": [float(v) for v in low],
        "room_high": [float(v) for v in high],
        "floor_z": z_floor,
        "points": {
            "film": refs["film"].tolist(),
            "uncovered": refs["uncovered"].tolist(),
            "floor": refs["floor"].tolist(),
            "uncovered_visible": refs["uncovered_visible"],
        },
        "objects": ["ball", "film-shell", "room", "light", "camera"],
    }
    Path(params["poses"]).write_text(json.dumps(poses, indent=2) + "\n")
    print(
        f"film blender=ok blend={blend_path} objects={','.join(poses['objects'])} "
        f"floor_z={z_floor:.6f} radiance={radiance:.6g} power_watts={power_watts:.6g}"
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

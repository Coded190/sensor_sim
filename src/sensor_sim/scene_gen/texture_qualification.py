"""Texture-aware qualification renders.

Mitsuba RGB from ``tabletop_render._scene`` is the scientific evidence.
Blender Cycles images are previews and are labeled separately. New packages
under this directory are not registered.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sensor_sim.scene_gen.asset_validate import triangulate
from sensor_sim.scene_gen.indoor_mitsuba import render_stokes, require_variant, write_rgb_png
from sensor_sim.scene_gen.material_assign import assign_components, replay_assignment
from sensor_sim.scene_gen.material_spec import FILTER_NOTE, make_spec, preview_nodes, roughness_image_to_alpha, to_mitsuba
from sensor_sim.scene_gen.mesh_io import read_mesh, read_obj, validate_uv, write_mesh, write_obj
from sensor_sim.scene_gen.plastic_wrap import classify_faces, make_wrap_spec, shell_report
from sensor_sim.scene_gen.scene_context import require_scene_context
from sensor_sim.scene_gen.structural import box_mesh as make_box_mesh
from sensor_sim.scene_gen.tabletop_render import _scene

logger = logging.getLogger("sensor_sim.texture_qualification")

ROOT = Path("dataset/qualification/material_wrap_review/textured")
RESOLUTION = (192, 144)
SPP = 16
SEED = 17531
BLENDER = Path("tools/blender-4.5.9-linux-x64/blender")
APPLE_OBJ = Path("assets/candidates/prop_apple/1.0.0/geometry/render.obj")
CAN_OBJ = Path("assets/candidates/prop_can/1.0.0/geometry/render.obj")
CAN_ASSET = Path("assets/candidates/prop_can/1.0.0/asset.json")
BOX_OBJ = Path("dataset/qualification/material_wrap_review/targets/box.obj")
WRAP_ROOT = Path("dataset/qualification/material_wrap_review/wraps")
MAP_SIZE = 256
IDENTITY_UV = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
SCALE_UV = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(package: Path, path: Path, role: str, color_space: str) -> dict:
    if path.suffix.lower() == ".exr":
        resolution = [MAP_SIZE, MAP_SIZE]
    else:
        image = Image.open(path)
        resolution = [int(image.size[0]), int(image.size[1])]
    return {
        "role": role,
        "path": path.relative_to(package).as_posix(),
        "color_space": color_space,
        "resolution": resolution,
        "sha256": _sha(path),
        "parameter_class": "authored_appearance",
        "channel": "rgb" if role == "base_color" else "r",
        "normal_orientation": "opengl_tangent",
        "normal_strength": 1.0,
        "uv_transform": list(IDENTITY_UV),
        "filter_type": "bilinear",
        "wrap_mode": "repeat",
        "raw": color_space == "non_color",
    }


def _write_maps(directory: Path) -> dict:
    """Author one asymmetric appearance set. Roughness is converted to alpha once."""
    directory.mkdir(parents=True, exist_ok=True)
    size = MAP_SIZE
    color = Image.new("RGB", (size, size), (210, 40, 35))
    rough = Image.new("L", (size, size), 51)
    normal = Image.new("RGB", (size, size), (128, 128, 255))
    alpha = np.full((size, size, 3), 0.04, dtype=np.float32)
    alpha[:, size // 2 :, :] = 0.64
    for x in range(size // 2, size):
        for y in range(size):
            color.putpixel((x, y), (35, 55, 205))
            rough.putpixel((x, y), 204)
    for x in range(size // 5):
        for y in range(4 * size // 5, size):
            color.putpixel((x, y), (40, 180, 55))
            normal.putpixel((x, y), (128, 230, 190))
    for x in range(4 * size // 5, size):
        for y in range(size // 5):
            normal.putpixel((x, y), (230, 90, 190))
    color.save(directory / "base_color.png")
    rough.save(directory / "roughness_blender.png")
    normal.save(directory / "normal_opengl.png")
    _write_exr(directory / "alpha_mitsuba.exr", alpha)
    source = np.asarray(Image.open(directory / "roughness_blender.png"), dtype=np.float64) / 255.0
    converted = roughness_image_to_alpha(source)
    if not np.allclose(converted, alpha[:, :, 0], atol=1e-6):
        raise RuntimeError("reason=alpha_conversion_mismatch")
    import mitsuba as mi

    reloaded = np.array(mi.Bitmap(str(directory / "alpha_mitsuba.exr")))
    if reloaded.shape[0] != MAP_SIZE or reloaded.shape[1] != MAP_SIZE or not np.allclose(reloaded[:, :, 0], alpha[:, :, 0], atol=1e-5):
        raise RuntimeError(f"reason=alpha_exr_roundtrip shape={reloaded.shape}")
    return {
        "roughness_left": 51 / 255.0,
        "roughness_right": 204 / 255.0,
        "alpha_left": 0.04,
        "alpha_right": 0.64,
        "conversion": "alpha = roughness^2 applied once while writing alpha_mitsuba.exr",
    }


def _write_exr(path: Path, image: np.ndarray) -> None:
    import mitsuba as mi

    require_variant()
    bitmap = mi.Bitmap(np.ascontiguousarray(image.astype(np.float32)))
    bitmap.write(str(path))


def _panel_mesh() -> dict:
    """A one-metre quad facing +Y, with UVs and a matching geometric normal."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    texcoords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    normals = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    corners = [[(0, 0, 0), (2, 2, 0), (1, 1, 0)], [(0, 0, 0), (3, 3, 0), (2, 2, 0)]]
    return {"vertices": vertices, "texcoords": texcoords, "normals": normals, "corners": corners}


def _project_mesh(vertices: np.ndarray, polygons: list[list[int]]) -> dict:
    """Project each triangle by its dominant axis into a padded UV cell.

    Vertex positions are copied. The method is a first-pass projection. The
    caller records its distortion and texel density.
    """
    triangles = triangulate(polygons)
    points = np.asarray(vertices, dtype=np.float64)
    minimum = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - minimum, 1e-8)
    cells = {
        (0, 1): np.array([0.02, 0.02]),
        (0, -1): np.array([0.36, 0.02]),
        (1, 1): np.array([0.70, 0.02]),
        (1, -1): np.array([0.02, 0.36]),
        (2, 1): np.array([0.36, 0.36]),
        (2, -1): np.array([0.70, 0.36]),
    }
    texcoords = []
    normals = []
    corners = []
    for triangle in triangles:
        tri = points[triangle]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = float(np.linalg.norm(normal))
        normal = np.array([0.0, 0.0, 1.0]) if length < 1e-12 else normal / length
        axis = int(np.argmax(np.abs(normal)))
        sign = 1 if normal[axis] >= 0.0 else -1
        origin = cells[(axis, sign)]
        axes = [index for index in range(3) if index != axis]
        normal_index = len(normals)
        normals.append(normal)
        face = []
        for index in triangle:
            uv = (points[int(index), axes] - minimum[axes]) / span[axes]
            texcoords.append(origin + 0.26 * uv)
            face.append((int(index), len(texcoords) - 1, normal_index))
        corners.append(face)
    return {"vertices": points, "texcoords": np.asarray(texcoords, dtype=np.float64), "normals": np.asarray(normals, dtype=np.float64), "corners": corners}


def _unique_islands(vertices: np.ndarray, polygons: list[list[int]]) -> dict:
    """One UV island per triangle, separated by padding. Positions stay put."""
    triangles = triangulate(polygons)
    points = np.asarray(vertices, dtype=np.float64)
    columns = int(np.ceil(np.sqrt(max(len(triangles), 1))))
    cell = 1.0 / columns
    texcoords = []
    normals = []
    corners = []
    for number, triangle in enumerate(triangles):
        tri = points[triangle]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = float(np.linalg.norm(normal))
        normal = np.array([0.0, 0.0, 1.0]) if length < 1e-12 else normal / length
        column, row = divmod(number, columns)
        origin = np.array([column * cell, row * cell])
        local = [np.array([0.2, 0.2]), np.array([0.8, 0.2]), np.array([0.5, 0.8])]
        normal_index = len(normals)
        normals.append(normal)
        face = []
        for index, uv in enumerate(local):
            texcoords.append(origin + uv * cell)
            face.append((int(triangle[index]), len(texcoords) - 1, normal_index))
        corners.append(face)
    return {"vertices": points, "texcoords": np.asarray(texcoords, dtype=np.float64), "normals": np.asarray(normals, dtype=np.float64), "corners": corners}


def _with_uv(vertices: np.ndarray, polygons: list[list[int]]) -> tuple[dict, dict]:
    projected = _project_mesh(vertices, polygons)
    try:
        report = validate_uv(projected, texture_resolution=(MAP_SIZE, MAP_SIZE), island_padding=0.0)
    except Exception as exc:
        packed = _unique_islands(vertices, polygons)
        report = validate_uv(packed, texture_resolution=(MAP_SIZE, MAP_SIZE), island_padding=0.0)
        report["uv_method"] = "per_triangle_islands"
        report["projection_rejected"] = str(exc)
        logger.info("texture_qualification event=uv_fallback reason=%s", exc)
        return packed, report
    report["uv_method"] = "dominant_axis_projection"
    try:
        validate_uv(projected, texture_resolution=(MAP_SIZE, MAP_SIZE), island_padding=0.02)
        report["padding_0_02_passed"] = True
    except Exception as exc:
        report["padding_0_02_passed"] = False
        report["padding_failure"] = str(exc)
    return projected, report


def _uv_image(mesh: dict, path: Path) -> None:
    image = Image.new("RGB", (256, 256), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    texcoords = np.asarray(mesh["texcoords"], dtype=np.float64)
    for face in mesh["corners"]:
        points = []
        for corner in face:
            uv = texcoords[int(corner[1])]
            points.append((float(uv[0]) * 255.0, (1.0 - float(uv[1])) * 255.0))
        draw.line(points + [points[0]], fill=(90, 210, 255), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _spec(material_id: str, package: Path, roles: set[str], alpha: float = 0.04) -> dict:
    textures = []
    names = {
        "base_color": ("textures/base_color.png", "srgb"),
        "roughness_blender": ("textures/roughness_blender.png", "non_color"),
        "alpha_mitsuba": ("textures/alpha_mitsuba.exr", "non_color"),
        "normal_opengl": ("textures/normal_opengl.png", "non_color"),
    }
    for role in ("base_color", "roughness_blender", "alpha_mitsuba", "normal_opengl"):
        if role in roles:
            relative, space = names[role]
            textures.append(_record(package, package / relative, role, space))
    spec = make_spec(
        material_id,
        "pplastic",
        {"diffuse_reflectance": [0.62, 0.14, 0.12], "int_ior": 1.45, "ext_ior": 1.0, "alpha": float(alpha), "distribution": "beckmann"},
        textures=textures,
        seed=SEED,
        sampled={
            "parameter_class": {
                "family": "assumed_preset",
                "int_ior": "assumed_preset",
                "scalar_alpha": "procedural_parameter",
                "textures": "authored_appearance",
            },
            "not_measured_reflectance": True,
            "not_measured_ior": True,
        },
    )
    spec["package_root"] = str(package)
    return spec


def _camera_light(vertices: np.ndarray, distance: float = 2.4) -> tuple[dict, dict]:
    import mitsuba as mi

    require_variant()
    points = np.asarray(vertices, dtype=np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = ((minimum + maximum) * 0.5).tolist()
    size = float(np.max(maximum - minimum))
    extent = max(size, 0.05)
    origin = (np.asarray(center) + np.array([0.15, -1.0, 0.38]) * extent * distance).tolist()
    camera = {"origin": origin, "target": center, "up": [0.0, 0.0, 1.0], "fov_deg": 32.0}
    transform = mi.ScalarTransform4f.look_at(origin=(np.asarray(center) + np.array([0.35, -0.55, 0.7]) * extent * distance).tolist(), target=center, up=[0.0, 0.0, 1.0])
    transform = transform @ mi.ScalarTransform4f.scale([1.6 * extent, 1.6 * extent, 1.0])
    light = {"type": "rectangle", "to_world": transform, "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 18.0}}}
    return camera, light


def _panel_view() -> tuple[dict, dict]:
    import mitsuba as mi

    require_variant()
    camera = {"origin": [0.5, 1.55, 0.62], "target": [0.5, 0.0, 0.5], "up": [0.0, 0.0, 1.0], "fov_deg": 38.0}
    transform = mi.ScalarTransform4f.look_at(origin=[0.15, 0.85, 1.25], target=[0.5, 0.0, 0.45], up=[0.0, 0.0, 1.0]) @ mi.ScalarTransform4f.scale([0.55, 0.4, 1.0])
    light = {"type": "rectangle", "to_world": transform, "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 14.0}}}
    return camera, light


def _render(instances: list[dict], camera: dict, light: dict, png: Path, spp: int = SPP, extras: dict | None = None) -> dict:
    require_scene_context(
        [{"id": item["id"], "asset_id": item["id"], "role": "subject"} for item in instances],
        room_kind="probe",
        allow_bare_plane=True,
        allow_without_context_objects=True,
        exemption="controlled_material_probe",
    )
    scene = _scene(instances, camera, RESOLUTION, light)
    if extras:
        scene.update(extras)
    png.parent.mkdir(parents=True, exist_ok=True)
    exr = png.with_suffix(".exr")
    stokes = render_stokes(scene, exr, spp, SEED)
    write_rgb_png(exr, png)
    observed = _observe(scene)
    payload = {
        "renderer": "Mitsuba",
        "variant": "cuda_ad_spectral_polarized",
        "spp": int(spp),
        "seed": SEED,
        "resolution": list(RESOLUTION),
        "png": str(png),
        "finite": stokes["finite"],
        "mean_rgb": _mean(png),
        "halves": _halves(png),
        "observed": observed,
        "filter_note": FILTER_NOTE,
    }
    logger.info("texture_qualification event=mitsuba_rgb path=%s mean=%s", png.name, [round(value, 2) for value in payload["mean_rgb"]])
    return payload


def _observe(scene: dict) -> dict:
    import mitsuba as mi

    loaded = mi.load_dict(scene)
    shapes = []
    for shape in loaded.shapes():
        if not hasattr(shape, "vertex_count"):
            continue
        shapes.append({
            "id": shape.id(),
            "vertices": int(shape.vertex_count()),
            "faces": int(shape.face_count()),
            "vertex_texcoords": bool(shape.has_vertex_texcoords()),
            "vertex_normals": bool(shape.has_vertex_normals()),
        })
    keys = [key for key in mi.traverse(loaded).keys() if any(token in key for token in ("alpha", "normalmap", "reflectance", "texcoord", "vertex_normal"))]
    return {"shapes": shapes, "parameter_keys": keys}


def _mean(png: Path) -> list[float]:
    image = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)
    return [float(value) for value in image.mean(axis=(0, 1))]


def _halves(png: Path) -> dict:
    image = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)
    width = image.shape[1]
    left = image[:, : width // 2].mean(axis=(0, 1))
    right = image[:, width // 2 :].mean(axis=(0, 1))
    return {
        "left_mean_rgb": [float(value) for value in left],
        "right_mean_rgb": [float(value) for value in right],
        "half_mean_abs": float(np.mean(np.abs(left - right))),
    }


def _difference(left: Path, right: Path) -> float:
    first = np.asarray(Image.open(left).convert("RGB"), dtype=np.float64)
    second = np.asarray(Image.open(right).convert("RGB"), dtype=np.float64)
    return float(np.mean(np.abs(first - second)))


def _write_package(package: Path, mesh: dict, uv_report: dict, extra: dict) -> None:
    if package.exists():
        shutil.rmtree(package)
    (package / "textures").mkdir(parents=True)
    (package / "geometry").mkdir()
    conversion = _write_maps(package / "textures")
    (package / "geometry" / "render.obj").write_bytes(write_mesh(mesh, name=package.parent.name))
    collision = {"vertices": mesh["vertices"], "corners": [[(int(corner[0]), None, None) for corner in face] for face in mesh["corners"]]}
    (package / "geometry" / "collision.obj").write_bytes(write_mesh(collision, name=package.parent.name + "_collision"))
    loaded = read_mesh(package / "geometry" / "render.obj")
    if not np.allclose(loaded["vertices"], np.asarray(mesh["vertices"]), atol=1e-8):
        raise RuntimeError("reason=vertex_roundtrip")
    textures = [
        _record(package, package / "textures" / name, role, space)
        for name, role, space in (
            ("base_color.png", "base_color", "srgb"),
            ("roughness_blender.png", "roughness_blender", "non_color"),
            ("alpha_mitsuba.exr", "alpha_mitsuba", "non_color"),
            ("normal_opengl.png", "normal_opengl", "non_color"),
        )
    ]
    manifest = {
        "asset_id": package.parent.name,
        "version": "1.0.0",
        "registered": False,
        "visually_reviewed": False,
        "admitted": False,
        "textures": textures,
        "uv": uv_report,
        "roughness_conversion": conversion,
        "geometry": {
            "vertex_count": int(len(mesh["vertices"])),
            "triangle_count": int(uv_report["triangle_count"]),
            "collision_uvs": "omitted_on_purpose",
        },
        "provenance": {
            "parameter_class": "authored_appearance",
            "not_measured_reflectance": True,
            "not_measured_ior": True,
            **extra,
        },
    }
    (package / "asset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _uv_image(mesh, package / "uv_layout.png")


def _copy_and_render(package: Path, destination: Path) -> dict:
    copied = destination / "copied_panel_package"
    if copied.exists():
        shutil.rmtree(copied)
    shutil.copytree(package, copied)
    text = (copied / "asset.json").read_text(encoding="utf-8")
    if "local_review/" in text or str(Path.home()) in text or '": "/' in text:
        raise RuntimeError("reason=manifest_path")
    spec = _spec("copied_panel", copied, {"base_color", "alpha_mitsuba"})
    png = destination / "mitsuba_copied_package.png"
    image = _render([{"id": "panel", "mesh": copied / "geometry" / "render.obj", "matrix": np.eye(4), "bsdf": spec}], *_panel_view(), png)
    return {"rendered": True, "finite": image["finite"], "mean_rgb": image["mean_rgb"], "manifest_paths_relative": True}


def _components(package: Path) -> dict:
    textures = json.loads((package / "asset.json").read_text(encoding="utf-8"))["textures"]
    body = {
        "instance_id": "widget",
        "component_id": "body",
        "families": ["pplastic"],
        "texture_sets": ["panel"],
        "uv_transforms": [IDENTITY_UV, SCALE_UV],
        "geometry_seed": 11,
        "placement_seed": 12,
        "wrap_seed": 13,
    }
    lid = {
        "instance_id": "widget",
        "component_id": "lid",
        "families": ["conductor", "roughconductor"],
        "geometry_seed": 11,
        "placement_seed": 12,
        "wrap_seed": 13,
    }
    sets = {"panel": textures}
    first = None
    second = None
    for seed in range(41, 90):
        records = assign_components([body, lid], seed, sets)
        if first is None:
            first = (seed, records)
            continue
        if records[0]["spec"]["textures"][0]["uv_transform"] != first[1][0]["spec"]["textures"][0]["uv_transform"]:
            second = (seed, records)
            break
    if second is None:
        raise RuntimeError("reason=uv_transform_not_selected")
    replay = [replay_assignment(record) for record in first[1]]
    moved = dict(body)
    moved["geometry_seed"] = 99
    moved["wrap_seed"] = 77
    same = assign_components([moved, lid], first[0], sets)
    return {
        "material_seed_a": first[0],
        "material_seed_b": second[0],
        "seed_a": [_public(record) for record in first[1]],
        "seed_b_families": [record["spec"]["family"] for record in second[1]],
        "seed_b_uv": second[1][0]["spec"]["textures"][0]["uv_transform"],
        "replay_matches": all(item["textures"] == record["spec"]["textures"] and item["parameters"] == record["spec"]["parameters"] for item, record in zip(replay, first[1], strict=True)),
        "geometry_and_wrap_seeds_do_not_change_material": same[0]["spec"]["textures"] == first[1][0]["spec"]["textures"] and same[0]["spec"]["parameters"] == first[1][0]["spec"]["parameters"],
        "stored_geometry_seed_changed": same[0]["seeds"]["geometry"] == 99 and first[1][0]["seeds"]["geometry"] == 11,
        "records_a": first[1],
    }


def _public(record: dict) -> dict:
    spec = dict(record["spec"])
    spec.pop("package_root", None)
    return {"instance_id": record["instance_id"], "component_id": record["component_id"], "material_id": spec["material_id"], "family": spec["family"], "seeds": record["seeds"], "spec": spec}


def _filter_probe(color_path: Path, data_path: Path) -> dict:
    import mitsuba as mi

    require_variant()
    rows = []
    for path, raw in ((color_path, False), (data_path, True)):
        for name in ("bilinear", "nearest", "trilinear", "ewa"):
            try:
                plugin = mi.load_dict({"type": "bitmap", "filename": str(path), "raw": raw, "filter_type": name, "wrap_mode": "repeat"})
                rows.append({"file": path.name, "raw": raw, "requested": name, "loaded": True, "plugin": str(plugin)})
            except Exception as exc:
                rows.append({"file": path.name, "raw": raw, "requested": name, "loaded": False, "error": f"{type(exc).__name__}: {exc}"})
    return {"requests": rows, "note": FILTER_NOTE}


def _reference() -> dict:
    info_path = Path("local_review/reference_food_apple_01/info.json")
    if not info_path.is_file():
        return {"present": False}
    info = json.loads(info_path.read_text(encoding="utf-8"))
    folder = info_path.parent / "textures"
    hashes = {}
    for name in ("food_apple_01_diff_1k.jpg", "food_apple_01_rough_1k.jpg", "food_apple_01_nor_gl_1k.jpg", "food_apple_01_nor_gl_1k.exr"):
        path = folder / name
        if path.is_file():
            hashes[name] = _sha(path)
    return {
        "present": True,
        "name": info.get("name", "food_apple_01"),
        "license": "CC0",
        "source": "https://polyhaven.com/a/food_apple_01",
        "author": info.get("author"),
        "triangles": info.get("polycount"),
        "dimensions": info.get("dimensions"),
        "texture_sha256": hashes,
        "normal_convention": "opengl_tangent",
        "parameter_class": "authored_appearance",
        "measured_reflectance": False,
        "measured_ior": False,
    }


def _protected() -> dict:
    from tests.test_material_wrap import INDEX_SHA256
    from tests.test_semantic_scenes import DATASET_V1, LOCKED

    files = {}
    for name, digest in LOCKED.items():
        if not name.endswith(".py"):
            continue
        path = Path("src/sensor_sim/scene_gen") / name
        files[name] = {"sha256": _sha(path), "matches_locked_hash": _sha(path) == digest}
    index = Path("assets/candidates/index.json")
    files["assets/candidates/index.json"] = {"sha256": _sha(index), "matches_locked_hash": _sha(index) == INDEX_SHA256}
    dataset = Path("dataset/dataset_v1")
    listing = [path for path in dataset.rglob("*") if path.is_file()]
    return {
        "files": files,
        "dataset_v1_files": len(listing),
        "dataset_v1_bytes": sum(path.stat().st_size for path in listing),
        "dataset_v1_matches": len(listing) == DATASET_V1["files"] and sum(path.stat().st_size for path in listing) == DATASET_V1["bytes"],
    }


def _shell_shapes(target_name: str, mode: str, vertices, faces, film: dict, gallery: Path) -> list[dict]:
    """Write outer and inner groups as separate meshes. One OBJ was a single dielectric volume."""
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append([int(face[0]), int(face[1]), int(face[2])])
        elif len(face) == 4:
            triangles.append([int(face[0]), int(face[1]), int(face[2])])
            triangles.append([int(face[0]), int(face[2]), int(face[3])])
    center = np.asarray(vertices, dtype=np.float64).mean(axis=0)
    groups = classify_faces(np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64), center)
    shapes = []
    for name, group in groups["groups"].items():
        if len(group) == 0:
            continue
        path = gallery / f"{target_name}_{mode}_{name}.obj"
        path.write_bytes(write_obj(vertices, [list(map(int, face)) for face in group], name=name))
        shapes.append({"id": f"film_{name}", "mesh": path, "matrix": np.eye(4), "bsdf": film})
    return shapes


def _wrap_view(points: np.ndarray) -> tuple[dict, dict, dict]:
    """Use the camera and area light that already showed the dielectric shell.

    The ground gives the glass a background. It is a controlled-probe backdrop,
    not a furnished room.
    """
    import mitsuba as mi

    require_variant()
    points = np.asarray(points, dtype=np.float64)
    center = ((points.min(axis=0) + points.max(axis=0)) * 0.5)
    camera = {"origin": (center + np.array([0.0, -1.15, 0.43])).tolist(), "target": center.tolist(), "up": [0.0, 0.0, 1.0], "fov_deg": 32.0}
    transform = mi.ScalarTransform4f.translate([0.15, -0.35, 1.05]) @ mi.ScalarTransform4f.rotate([1.0, 0.0, 0.0], 180.0) @ mi.ScalarTransform4f.scale([0.40, 0.40, 1.0])
    light = {"type": "rectangle", "to_world": transform, "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 6.0}}}
    ground = {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f.translate([0.0, 0.0, float(points[:, 2].min()) - 0.02]) @ mi.ScalarTransform4f.scale([1.5, 1.5, 1.0]),
        "bsdf": to_mitsuba(make_spec("wrap_ground", "diffuse", {"reflectance": [0.45, 0.45, 0.45]})),
    }
    return camera, light, {"ground": ground}


def _wrap_cases(box_package: Path, gallery: Path) -> list[dict]:
    rows = []
    textured = _spec("textured_box", box_package, {"base_color", "alpha_mitsuba"})
    for target_name, mode in (("box", "partial"), ("box", "full"), ("sphere", "partial"), ("sphere", "full")):
        shell_path = WRAP_ROOT / f"{target_name}_{mode}" / "smooth" / "shell_applied.obj"
        target_path = BOX_OBJ if target_name == "box" else Path(f"assets/candidates/fixture_sphere/1.0.0/geometry/render.obj")
        if target_name == "apple":
            target_path = APPLE_OBJ
        if not shell_path.is_file() or not target_path.is_file():
            rows.append({"target": target_name, "mode": mode, "rendered": False, "reason": "mesh_missing"})
            continue
        film = make_spec(f"{target_name}_{mode}_film", "dielectric", {"int_ior": 1.5, "ext_ior": 1.0})
        substrate_vertices, substrate_faces = read_obj(target_path)
        shell_vertices, shell_faces = read_obj(shell_path)
        spec = make_wrap_spec(f"{target_name}_{mode}", target_name, mode, offset_m=0.004, thickness_m=0.002, seed=90, substrate_family="pplastic" if target_name == "box" and mode == "full" else "diffuse")
        other = make_wrap_spec(f"{target_name}_{mode}", target_name, mode, offset_m=0.004, thickness_m=0.002, seed=91, substrate_family=spec["interfaces"]["inner"]["status"])
        geometry = {"passed_geometry": None}
        try:
            geometry = shell_report({"vertices": shell_vertices, "faces": shell_faces}, {"vertices": substrate_vertices, "faces": substrate_faces}, spec)
        except Exception as exc:
            geometry = {"passed_geometry": False, "remeasure_error": str(exc)}
        use_texture = target_name == "box" and mode == "full"
        substrate = textured if use_texture else make_spec(f"{target_name}_substrate", "diffuse", {"reflectance": [0.7, 0.12, 0.1]})
        mesh = box_package / "geometry" / "render.obj" if use_texture else target_path
        shapes = [{"id": "substrate", "mesh": mesh, "matrix": np.eye(4), "bsdf": substrate}]
        part_dir = shell_path.parent
        part_paths = [part_dir / f"{name}.obj" for name in ("outer", "inner", "sidewall") if (part_dir / f"{name}.obj").is_file()]
        if part_paths:
            shapes.extend({"id": f"film_{path.stem}", "mesh": path, "matrix": np.eye(4), "bsdf": film} for path in part_paths)
        else:
            shapes.extend(_shell_shapes(target_name, mode, shell_vertices, shell_faces, film, gallery))
        camera, light, extras = _wrap_view(np.vstack([substrate_vertices, shell_vertices]))
        png = gallery / f"mitsuba_{target_name}_{mode}_shell.png"
        image = _render(shapes, camera, light, png, spp=64, extras=extras)
        if use_texture:
            bare_png = gallery / "mitsuba_box_textured_unwrapped.png"
            bare = _render([shapes[0]], camera, light, bare_png, spp=32, extras=extras)
            rows.append({
                "target": "box",
                "mode": "unwrapped",
                "rendered": True,
                "finite": bare["finite"],
                "png": str(bare_png),
                "mean_rgb": bare["mean_rgb"],
                "outer_status": "no_shell",
                "inner_status": "no_shell",
                "optical_film_to_substrate": "not_applicable",
                "geometry": {"passed_geometry": None},
                "textured_substrate": True,
                "physically_validated_film_optics": False,
                "caption": "textured box with no shell, same camera as the full-shell figure",
            })
        rows.append({
            "target": target_name,
            "mode": mode,
            "rendered": True,
            "finite": image["finite"],
            "png": str(png),
            "mean_rgb": image["mean_rgb"],
            "outer_status": spec["interfaces"]["outer"]["status"],
            "inner_status": spec["interfaces"]["inner"]["status"],
            "optical_film_to_substrate": "not_validated",
            "geometry": {key: geometry.get(key) for key in ("passed_geometry", "minimum_clearance_m", "thickness_m", "counts", "remeasure_error")},
            "wrap_seed_changes_mesh_request": spec["seed"] != other["seed"] and spec["target_id"] == other["target_id"],
            "substrate_texture_independent_of_wrap_seed": True,
            "textured_substrate": use_texture,
            "validated_wrap_geometry": geometry.get("passed_geometry"),
            "verified_interface_assignment": spec["interfaces"]["outer"]["status"] == "assigned" and spec["interfaces"]["inner"]["status"] == "not_validated_opaque_substrate",
            "mitsuba_rgb": image["finite"] is True,
            "physically_validated_film_optics": False,
        })
    return rows


def _blender(jobs: list[dict], gallery: Path) -> dict:
    params = {"width": RESOLUTION[0], "height": RESOLUTION[1], "samples": 32, "jobs": jobs}
    path = gallery / "blender_jobs.json"
    path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    completed = subprocess.run([str(BLENDER), "--background", "--python", "assets/blender/render_texture_qualification.py", "--", str(path)], check=False, capture_output=True, text=True)
    (gallery / "blender_cycles.log").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    return {"returncode": completed.returncode, "images": [{"png": job["png"], "exists": Path(job["png"]).is_file(), "renderer": "Blender Cycles"} for job in jobs]}


def _save_bsdf(path: Path, spec: dict, package: Path) -> None:
    mapped = to_mitsuba(spec, package_root=package)
    path.write_text(json.dumps({"spec": _public({"instance_id": "", "component_id": "", "spec": spec, "seeds": {"geometry": 0, "placement": 0, "material": SEED, "wrap": 0}})["spec"], "bsdf": mapped, "filter_note": FILTER_NOTE}, indent=2, default=str) + "\n", encoding="utf-8")


def _gallery(gallery: Path, rows: list[dict]) -> None:
    body = [
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Texture qualification</title>",
        "<style>body{font-family:sans-serif;margin:2rem;background:#111;color:#eee} img{max-width:520px;background:#222} figure{margin:1rem 0}</style>",
        "</head><body><h1>Texture-aware qualification</h1>",
        "<p>Each caption names the renderer. Mitsuba RGB is the scientific evidence. Blender Cycles is the preview. These packages are not visually reviewed, admitted, or registered. A successful RGB render is not a polarization-accuracy result. Film optics are not physically validated.</p>",
    ]
    for row in rows:
        path = Path(row["png"])
        if not path.is_file():
            continue
        if path.parent != gallery:
            shutil.copy(path, gallery / path.name)
        body.append(f"<figure><img src=\"{path.name}\" alt=\"{row['renderer']}\"><figcaption><strong>{row['renderer']}</strong> {row['label']}</figcaption></figure>")
    body.append("</body></html>\n")
    (gallery / "index.html").write_text("\n".join(body), encoding="utf-8")


def _bounds(path: Path) -> dict:
    vertices, _polygons = read_obj(path)
    points = np.asarray(vertices, dtype=np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return {"minimum_m": minimum.tolist(), "maximum_m": maximum.tolist(), "size_m": (maximum - minimum).tolist(), "center_m": ((minimum + maximum) * 0.5).tolist(), "vertex_count": int(len(points))}


def build() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    require_variant()
    gallery = ROOT / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    apple_before = _sha(APPLE_OBJ)
    can_before = _sha(CAN_OBJ)
    panel = ROOT / "package" / "textured_panel" / "1.0.0"
    panel_mesh = _panel_mesh()
    panel_uv = validate_uv(panel_mesh, texture_resolution=(MAP_SIZE, MAP_SIZE), island_padding=0.0)
    panel_uv["uv_method"] = "explicit_quad_parameterization"
    _write_package(panel, panel_mesh, panel_uv, {"source": "qualification_quad"})
    apple_vertices, apple_polygons = read_obj(APPLE_OBJ)
    apple_mesh, apple_uv = _with_uv(apple_vertices, apple_polygons)
    apple_uv["source_vertex_count"] = int(len(apple_vertices))
    apple_uv["source_polygon_count"] = int(len(apple_polygons))
    apple_uv["comparison_label"] = "material_and_texture_on_the_packaged_vertices"
    apple = ROOT / "package" / "textured_apple" / "1.0.0"
    _write_package(apple, apple_mesh, apple_uv, {"source_mesh": "prop_apple/1.0.0", "geometry_changed": False, "registered": False})
    box_vertices, box_polygons = read_obj(BOX_OBJ)
    boxed, box_uv = _with_uv(box_vertices, box_polygons)
    box_uv["uv_method_note"] = box_uv["uv_method"]
    box = ROOT / "package" / "textured_box" / "1.0.0"
    _write_package(box, boxed, box_uv, {"source_mesh": "qualification box target", "positions_copied": True})
    if not np.allclose(boxed["vertices"], np.asarray(box_vertices), atol=1e-8):
        raise RuntimeError("reason=box_vertices_moved")
    camera, light = _panel_view()
    full = _spec("panel_full", panel, {"base_color", "roughness_blender", "alpha_mitsuba", "normal_opengl"})
    instances = [{"id": "panel", "mesh": panel / "geometry" / "render.obj", "matrix": np.eye(4), "bsdf": full}]
    images = {
        "base_rough_normal": _render(instances, camera, light, gallery / "mitsuba_base_rough_normal.png"),
        "base_off": _render([{"id": "panel", "mesh": panel / "geometry" / "render.obj", "matrix": np.eye(4), "bsdf": _spec("panel_base_off", panel, {"alpha_mitsuba", "normal_opengl"})}], camera, light, gallery / "mitsuba_base_off.png"),
        "rough_constant": _render([{"id": "panel", "mesh": panel / "geometry" / "render.obj", "matrix": np.eye(4), "bsdf": _spec("panel_rough_off", panel, {"base_color", "normal_opengl"})}], camera, light, gallery / "mitsuba_rough_constant.png"),
        "normal_off": _render([{"id": "panel", "mesh": panel / "geometry" / "render.obj", "matrix": np.eye(4), "bsdf": _spec("panel_normal_off", panel, {"base_color", "alpha_mitsuba"})}], camera, light, gallery / "mitsuba_normal_off.png"),
    }
    _save_bsdf(gallery / "panel_full_bsdf.json", full, panel)
    differences = {
        "base_color_mean_abs": _difference(gallery / "mitsuba_base_rough_normal.png", gallery / "mitsuba_base_off.png"),
        "roughness_mean_abs": _difference(gallery / "mitsuba_base_rough_normal.png", gallery / "mitsuba_rough_constant.png"),
        "normal_mean_abs": _difference(gallery / "mitsuba_base_rough_normal.png", gallery / "mitsuba_normal_off.png"),
    }
    apple_camera, apple_light = _camera_light(apple_vertices, 2.6)
    apple_full = _spec("apple_full", apple, {"base_color", "roughness_blender", "alpha_mitsuba", "normal_opengl"})
    apple_images = {
        "textured": _render([{"id": "apple", "mesh": apple / "geometry" / "render.obj", "matrix": np.eye(4), "bsdf": apple_full}], apple_camera, apple_light, gallery / "mitsuba_apple_textured.png"),
        "scalar": _render([{"id": "apple", "mesh": APPLE_OBJ, "matrix": np.eye(4), "bsdf": {"type": "diffuse", "reflectance": [0.62, 0.14, 0.12]}}], apple_camera, apple_light, gallery / "mitsuba_apple_scalar.png"),
        "normal_off": _render([{"id": "apple", "mesh": apple / "geometry" / "render.obj", "matrix": np.eye(4), "bsdf": _spec("apple_normal_off", apple, {"base_color", "alpha_mitsuba"})}], apple_camera, apple_light, gallery / "mitsuba_apple_normal_off.png"),
    }
    can_vertices, _can_faces = read_obj(CAN_OBJ)
    can_camera, can_light = _camera_light(can_vertices, 2.6)
    can_material = json.loads(CAN_ASSET.read_text(encoding="utf-8"))["materials"][0]
    can_fill = {"ambient": {"type": "constant", "radiance": {"type": "rgb", "value": [0.55, 0.62, 0.7]}}}
    can_image = _render(
        [{"id": "can", "mesh": CAN_OBJ, "matrix": np.eye(4), "bsdf": {"type": can_material["type"], "material": can_material["material"]}}],
        can_camera,
        can_light,
        gallery / "mitsuba_can_conductor.png",
        extras=can_fill,
    )
    body_vertices = box_vertices
    lid_vertices, lid_polygons = make_box_mesh([-0.10, -0.10, 0.0], [0.10, 0.10, 0.03])
    lid_path = gallery / "widget_lid.obj"
    lid_path.write_bytes(write_obj(lid_vertices, lid_polygons, name="lid"))
    assignment = _components(panel)
    seed_a = assignment["records_a"]
    widget_camera, widget_light = _camera_light(np.vstack([body_vertices, np.asarray(lid_vertices) + np.array([0.0, 0.0, 0.21])]), 2.8)
    def _widget(records, png):
        body_spec = dict(records[0]["spec"])
        body_spec["package_root"] = str(panel)
        lid_spec = records[1]["spec"]
        return _render(
            [
                {"id": "body", "mesh": box / "geometry" / "render.obj", "matrix": np.eye(4), "bsdf": body_spec},
                {"id": "lid", "mesh": lid_path, "matrix": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.21], [0, 0, 0, 1]], dtype=np.float64), "bsdf": lid_spec},
            ],
            widget_camera,
            widget_light,
            png,
            extras={"ambient": {"type": "constant", "radiance": {"type": "rgb", "value": 0.35}}},
        )
    widget_a = _widget(seed_a, gallery / "mitsuba_widget_seed_a.png")
    seed_b_records = assign_components(
        [
            {"instance_id": "widget", "component_id": "body", "families": ["pplastic"], "texture_sets": ["panel"], "uv_transforms": [IDENTITY_UV, SCALE_UV], "geometry_seed": 11, "placement_seed": 12, "wrap_seed": 13},
            {"instance_id": "widget", "component_id": "lid", "families": ["conductor", "roughconductor"], "geometry_seed": 11, "placement_seed": 12, "wrap_seed": 13},
        ],
        assignment["material_seed_b"],
        {"panel": json.loads((panel / "asset.json").read_text(encoding="utf-8"))["textures"]},
    )
    widget_b = _widget(seed_b_records, gallery / "mitsuba_widget_seed_b.png")
    (gallery / "widget_assignment.json").write_text(json.dumps({"seed_a": assignment["seed_a"], "seed_b": [_public(record) for record in seed_b_records]}, indent=2) + "\n", encoding="utf-8")
    wraps = _wrap_cases(box, gallery)
    copied = _copy_and_render(panel, gallery)
    filters = _filter_probe(panel / "textures" / "base_color.png", panel / "textures" / "alpha_mitsuba.exr")
    apple_bounds = _bounds(APPLE_OBJ)
    jobs = [
        {"kind": "textured", "obj": str((panel / "geometry" / "render.obj").resolve()), "png": str((gallery / "blender_cycles_panel.png").resolve()), "base_color": str((panel / "textures" / "base_color.png").resolve()), "roughness": str((panel / "textures" / "roughness_blender.png").resolve()), "normal": str((panel / "textures" / "normal_opengl.png").resolve()), "camera": camera, "ior": 1.45, "metallic": 0.0},
        {"kind": "textured", "obj": str((apple / "geometry" / "render.obj").resolve()), "png": str((gallery / "blender_cycles_apple_textured.png").resolve()), "base_color": str((apple / "textures" / "base_color.png").resolve()), "roughness": str((apple / "textures" / "roughness_blender.png").resolve()), "normal": str((apple / "textures" / "normal_opengl.png").resolve()), "camera": apple_camera, "ior": 1.45, "metallic": 0.0},
        {"kind": "flat", "obj": str(APPLE_OBJ.resolve()), "png": str((gallery / "blender_cycles_apple_flat.png").resolve()), "flat_color": [0.62, 0.14, 0.12], "roughness_value": 0.5, "camera": apple_camera, "ior": 1.45, "metallic": 0.0},
        {"kind": "polyhaven", "blend": str(Path("local_review/reference_food_apple_01/food_apple_01_1k.blend").resolve()), "png": str((gallery / "blender_cycles_polyhaven_apple.png").resolve()), "target_height_m": apple_bounds["size_m"][2], "target_center_m": apple_bounds["center_m"], "camera": apple_camera},
    ]
    from sensor_sim.scene_gen.material_review import run_transfer

    try:
        transfer = run_transfer(Path("dataset/qualification/material_wrap_review/transfer"))
    except Exception as exc:
        transfer = {"export_supported": False, "cases": [], "error": str(exc)}
        logger.info("texture_qualification event=transfer_fail error=%s", exc)
    blender = _blender(jobs, gallery)
    for name in ("base_color.png", "roughness_blender.png", "normal_opengl.png", "uv_layout.png"):
        source = (panel / "textures" / name) if name != "uv_layout.png" else (panel / name)
        if source.is_file():
            shutil.copy(source, gallery / f"thumbnail_{name}")
    shutil.copy(apple / "uv_layout.png", gallery / "thumbnail_apple_uv.png")
    captions = [
        {"renderer": "Mitsuba", "png": str(gallery / "mitsuba_base_rough_normal.png"), "label": "panel, base color, roughness map, and normal map on"},
        {"renderer": "Mitsuba", "png": str(gallery / "mitsuba_base_off.png"), "label": "panel, base color off, same roughness and normal"},
        {"renderer": "Mitsuba", "png": str(gallery / "mitsuba_rough_constant.png"), "label": "panel, constant alpha, base color and normal on"},
        {"renderer": "Mitsuba", "png": str(gallery / "mitsuba_normal_off.png"), "label": "panel, normal map off, geometric shading frame"},
        {"renderer": "Blender Cycles", "png": str(gallery / "blender_cycles_panel.png"), "label": "same panel maps, Blender Cycles preview"},
        {"renderer": "Mitsuba", "png": str(gallery / "mitsuba_apple_scalar.png"), "label": "prop_apple/1.0.0 scalar diffuse, unchanged mesh"},
        {"renderer": "Mitsuba", "png": str(gallery / "mitsuba_apple_textured.png"), "label": "same apple vertices with authored maps. This is a material and texture change, not a new mesh."},
        {"renderer": "Mitsuba", "png": str(gallery / "mitsuba_apple_normal_off.png"), "label": "textured apple with the normal map off"},
        {"renderer": "Blender Cycles", "png": str(gallery / "blender_cycles_apple_textured.png"), "label": "textured apple preview"},
        {"renderer": "Blender Cycles", "png": str(gallery / "blender_cycles_apple_flat.png"), "label": "packaged apple, flat color, no texture maps"},
        {"renderer": "Blender Cycles", "png": str(gallery / "blender_cycles_polyhaven_apple.png"), "label": "Poly Haven food_apple_01 scaled to the packaged apple height. Authored look benchmark, not measured reflectance or IOR."},
        {"renderer": "Mitsuba", "png": str(gallery / "mitsuba_can_conductor.png"), "label": "existing metal can, conductor Al, no base-color bitmap"},
        {"renderer": "Mitsuba", "png": str(gallery / "mitsuba_widget_seed_a.png"), "label": "two components, material seed A"},
        {"renderer": "Mitsuba", "png": str(gallery / "mitsuba_widget_seed_b.png"), "label": "same widget meshes, material seed B"},
    ]
    for row in wraps:
        if row.get("png"):
            label = row.get("caption") or f"{row['target']} {row['mode']} shell. Outer status {row['outer_status']}. Inner status {row['inner_status']}. Film optics not validated."
            captions.append({"renderer": "Mitsuba", "png": row["png"], "label": label})
    captions.append({"renderer": "Mitsuba", "png": str(gallery / "mitsuba_copied_package.png"), "label": "panel package copied to another directory and rendered from relative texture paths"})
    _gallery(gallery, captions)
    report = {
        "variant": "cuda_ad_spectral_polarized",
        "resolution": list(RESOLUTION),
        "spp": SPP,
        "seed": SEED,
        "panel_uv": panel_uv,
        "apple_uv": apple_uv,
        "apple_bounds_m": apple_bounds,
        "preview_nodes": preview_nodes(full),
        "differences": differences,
        "panel_halves": {key: value["halves"] for key, value in images.items()},
        "panel_shapes": images["base_rough_normal"]["observed"],
        "normal_off_has_no_normalmap": not any("normalmap" in key for key in images["normal_off"]["observed"]["parameter_keys"]),
        "full_has_normalmap": any("normalmap" in key for key in images["base_rough_normal"]["observed"]["parameter_keys"]),
        "full_has_alpha_bitmap": any(key.endswith("alpha.data") or ".alpha.data" in key for key in images["base_rough_normal"]["observed"]["parameter_keys"]),
        "apple": {key: {"mean_rgb": value["mean_rgb"], "texcoords": value["observed"]["shapes"]} for key, value in apple_images.items()},
        "can": {"mean_rgb": can_image["mean_rgb"], "material": can_material["material"], "family": can_material["type"]},
        "widget": {"seed_a_mean": widget_a["mean_rgb"], "seed_b_mean": widget_b["mean_rgb"], "mean_abs": _difference(gallery / "mitsuba_widget_seed_a.png", gallery / "mitsuba_widget_seed_b.png"), "assignment": {key: value for key, value in assignment.items() if key != "records_a"}},
        "wraps": wraps,
        "copied_package": copied,
        "filters": filters,
        "blender": blender,
        "transfer": {"export_supported": transfer.get("export_supported"), "cases": transfer.get("cases", []), "node_report": str(Path("dataset/qualification/material_wrap_review/transfer/node_report.json"))},
        "reference_apple": _reference(),
        "protected": _protected(),
        "source_hashes_unchanged": {"prop_apple_render.obj": _sha(APPLE_OBJ) == apple_before, "prop_can_render.obj": _sha(CAN_OBJ) == can_before},
        "polarization_accuracy": "not_audited",
        "film_optics": "not_validated",
    }
    (gallery / "summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("texture_qualification event=done differences=%s", differences)
    return report


def main() -> dict:
    return build()


if __name__ == "__main__":
    main()

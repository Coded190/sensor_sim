"""Build a Mitsuba package from an Infinigen export and the original blend graph.

The official individual OBJ export writes Y-up coordinates, bakes object rotation,
and drops translation. ``world_from_exported_vertex`` reverses that axis change.
The instance matrix then carries the translation from the untouched blend.
PLY export on this Indoors commit requires vertex colors, so the textured path is OBJ.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from sensor_sim.scene_gen.infinigen_ingest import measure_semantic_frame
from sensor_sim.scene_gen.light_units import classify_area_emitter, classify_point_emitter, polarization_role
from sensor_sim.scene_gen.material_choice import choose_material
from sensor_sim.scene_gen.material_spec import blender_roughness_to_alpha, make_spec, roughness_image_to_alpha

logger = logging.getLogger("sensor_sim.infinigen_package")

MESH_FORMAT = "obj"
MESH_FORMAT_REASON = "official_ply_export_requires_vertex_colors_and_drops_texture_slots"
AXIS_NOTE = "Official OBJ is Blender forward=-Z up=Y. Package meshes are Z-up metres. Translation stays on the instance."


class InfinigenPackageError(RuntimeError):
    """The export cannot become a Mitsuba package."""


def world_from_exported_vertex(vertex: list[float] | np.ndarray, translation: list[float] | np.ndarray) -> np.ndarray:
    """Map one official OBJ vertex back to Z-up world metres.

    The exporter stored ``(blender_x, blender_z, -blender_y)`` after applying
    the world rotation and before applying translation.
    """
    x_value, y_value, z_value = (float(item) for item in vertex)
    converted = np.array([x_value, -z_value, y_value], dtype=np.float64)
    return converted + np.asarray(translation, dtype=np.float64)


def vertical_fov_degrees(lens_mm: float, sensor_width_mm: float, width_px: int, height_px: int) -> float:
    """Vertical field of view for Blender AUTO sensor fit on a landscape frame."""
    half = math.atan((float(sensor_width_mm) * float(height_px)) / (2.0 * float(lens_mm) * float(width_px)))
    return math.degrees(2.0 * half)


def normal_convention(normal_space: str, normal_g: str) -> dict:
    """Record the bake convention that was read from the blend. Do not guess."""
    if normal_space != "TANGENT":
        raise InfinigenPackageError(f"reason=normal_space_untested value={normal_space}")
    if normal_g == "POS_Y":
        return {"tangent_space": True, "green_channel": "POS_Y", "flip_green": False, "orientation": "opengl_tangent", "source": "blend_render_bake"}
    if normal_g == "NEG_Y":
        return {"tangent_space": True, "green_channel": "NEG_Y", "flip_green": True, "orientation": "directx_tangent", "source": "blend_render_bake"}
    raise InfinigenPackageError(f"reason=normal_green_untested value={normal_g}")


def displacement_audit(materials: list[dict]) -> dict:
    """Separate bump from geometric displacement. Bump is not mesh detail."""
    methods = sorted({str(item.get("displacement_method")) for item in materials})
    geometric = [item for item in materials if item.get("displacement_method") in {"DISPLACEMENT", "BOTH"}]
    return {
        "source_methods": methods,
        "geometric_displacement_slots": len(geometric),
        "exported_mesh_displacement": False,
        "mitsuba_displacement_mode": "NONE",
        "claim": "no_exported_geometric_displacement" if not geometric else "unfit_for_geometric_normal_ground_truth",
    }


def sanitize_object_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_").replace(" ", "_")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_triplet(text: str) -> list[float]:
    return [float(item) for item in text.split()[:3]]


def parse_mtl(path: Path) -> dict[str, dict]:
    """Read the exporter MTL. Maps stay as file names next to that MTL."""
    materials: dict[str, dict] = {}
    current = None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("newmtl "):
            current = line.split(None, 1)[1]
            materials[current] = {"name": current, "maps": {}, "Kd": [0.6, 0.6, 0.6], "Pr": None, "Pm": 0.0, "Ni": 1.5, "d": 1.0, "bump_strength": 1.0}
            continue
        if current is None or not line or line.startswith("#"):
            continue
        key, _, rest = line.partition(" ")
        slot = materials[current]
        if key == "Kd":
            slot["Kd"] = _float_triplet(rest)
        elif key == "Pr":
            slot["Pr"] = float(rest.split()[0])
        elif key == "Pm":
            slot["Pm"] = float(rest.split()[0])
        elif key == "Ni":
            slot["Ni"] = float(rest.split()[0])
        elif key == "d":
            slot["d"] = float(rest.split()[0])
        elif key in {"map_Kd", "map_Pr", "map_Pm", "map_Bump", "map_Kn"}:
            parts = rest.split()
            if key == "map_Bump" and "-bm" in parts:
                slot["bump_strength"] = float(parts[parts.index("-bm") + 1])
                filename = parts[-1]
            else:
                filename = parts[-1]
            slot["maps"][key] = filename
    return materials


def _parse_face(token: str) -> tuple[int, int | None, int | None]:
    parts = token.split("/")
    vertex = int(parts[0])
    uv = int(parts[1]) if len(parts) > 1 and parts[1] else None
    normal = int(parts[2]) if len(parts) > 2 and parts[2] else None
    if vertex < 1 or (uv is not None and uv < 1) or (normal is not None and normal < 1):
        raise InfinigenPackageError("reason=negative_obj_index")
    return vertex - 1, None if uv is None else uv - 1, None if normal is None else normal - 1


def parse_obj(path: Path) -> dict:
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    normals: list[list[float]] = []
    groups: list[tuple[str, list]] = []
    material = "default"
    current: list = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("v "):
            vertices.append(_float_triplet(raw[2:]))
        elif raw.startswith("vt "):
            uvs.append([float(item) for item in raw[3:].split()[:2]])
        elif raw.startswith("vn "):
            normals.append(_float_triplet(raw[3:]))
        elif raw.startswith("usemtl "):
            if current:
                groups.append((material, current))
            material = raw.split(None, 1)[1].strip()
            current = []
        elif raw.startswith("f "):
            current.append(tuple(_parse_face(token) for token in raw.split()[1:]))
    if current:
        groups.append((material, current))
    if not groups and current == []:
        raise InfinigenPackageError(f"reason=empty_obj path={path.name}")
    return {"vertices": vertices, "uvs": uvs, "normals": normals, "groups": groups}


def _write_obj(path: Path, vertices: np.ndarray, uvs: list, normals: np.ndarray, faces: list, material: str) -> None:
    lines = [f"usemtl {material}"]
    for vertex in vertices:
        lines.append(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")
    for uv in uvs:
        lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
    if len(normals):
        for normal in normals:
            lines.append(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}")
    for face in faces:
        tokens = []
        for vertex, uv, normal in face:
            if uv is None and normal is None:
                tokens.append(str(vertex + 1))
            elif normal is None:
                tokens.append(f"{vertex + 1}/{uv + 1}")
            elif uv is None:
                tokens.append(f"{vertex + 1}//{normal + 1}")
            else:
                tokens.append(f"{vertex + 1}/{uv + 1}/{normal + 1}")
        lines.append("f " + " ".join(tokens))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compact(vertices: np.ndarray, uvs: list, normals: np.ndarray, faces: list) -> tuple[np.ndarray, list, np.ndarray, list]:
    used_v: list[int] = []
    used_t: list[int] = []
    used_n: list[int] = []
    map_v: dict[int, int] = {}
    map_t: dict[int, int] = {}
    map_n: dict[int, int] = {}
    remapped = []
    for face in faces:
        new_face = []
        for vertex, uv, normal in face:
            if vertex not in map_v:
                map_v[vertex] = len(used_v)
                used_v.append(vertex)
            uv_index = None
            if uv is not None:
                if uv not in map_t:
                    map_t[uv] = len(used_t)
                    used_t.append(uv)
                uv_index = map_t[uv]
            normal_index = None
            if normal is not None:
                if normal not in map_n:
                    map_n[normal] = len(used_n)
                    used_n.append(normal)
                normal_index = map_n[normal]
            new_face.append((map_v[vertex], uv_index, normal_index))
        remapped.append(tuple(new_face))
    new_uvs = [uvs[index] for index in used_t]
    new_normals = normals[used_n] if len(normals) and used_n else np.zeros((0, 3))
    return vertices[used_v], new_uvs, new_normals, remapped


def _square_roughness(source: Path, destination: Path) -> dict:
    raw = np.array(Image.open(source))
    scale = 65535.0 if raw.dtype == np.uint16 else 255.0
    alpha = roughness_image_to_alpha(raw.astype(np.float64) / scale)
    stored = np.rint(alpha * 65535.0).astype(np.uint16)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("I;16", (stored.shape[1], stored.shape[0]), np.ascontiguousarray(stored).tobytes()).save(destination)
    return {"path": destination, "min": float(alpha.min()), "max": float(alpha.max()), "mean": float(alpha.mean())}


def _audit_from_slot(slot: dict, audit: dict | None) -> dict:
    """Node-graph audit wins. MTL scalars fill only the keys the audit omitted."""
    merged = dict(audit or {})
    merged.setdefault("name", slot.get("name", ""))
    merged.setdefault("metallic", float(slot.get("Pm") or 0.0))
    merged.setdefault("roughness", float(slot["Pr"]) if slot.get("Pr") is not None else 0.5)
    transmission = float(merged.get("transmission") or 0.0)
    name = str(merged.get("name") or "")
    if "glass" in name.lower() or float(slot.get("d") or 1.0) < 0.99:
        transmission = max(transmission, 1.0)
    merged["transmission"] = transmission
    merged.setdefault("ior", float(slot.get("Ni") or 1.5))
    merged.setdefault("thin", "thin" in name.lower())
    maps = slot.get("maps") or {}
    merged.setdefault("base_color_linked", "map_Kd" in maps)
    merged.setdefault("roughness_linked", "map_Pr" in maps)
    merged.setdefault("normal_linked", "map_Kn" in maps or "map_Bump" in maps)
    merged.setdefault("base_color", slot.get("Kd"))
    return merged


def _spec_for_slot(slot: dict, *, material_id: str, texture_root: Path, package_root: Path, convention: dict, audit: dict | None = None) -> dict:
    """Choose the BSDF from the audited graph. ``pplastic`` is not the default."""
    maps = slot["maps"]
    choice = choose_material(_audit_from_slot(slot, audit))
    family = choice["family"]
    textures = []
    alpha_scalar = blender_roughness_to_alpha(float(slot["Pr"])) if slot.get("Pr") is not None else blender_roughness_to_alpha(float(choice["roughness"]))
    if "map_Pr" in maps:
        source = texture_root / maps["map_Pr"]
        if not source.is_file():
            raise InfinigenPackageError(f"reason=missing_required_texture path={source.name}")
        alpha_path = package_root / "textures" / "alpha" / f"{material_id}.png"
        stats = _square_roughness(source, alpha_path)
        alpha_scalar = float(stats["mean"])
        relative = alpha_path.relative_to(package_root).as_posix()
        textures.append(_texture_record("alpha_mitsuba", relative, alpha_path, "non_color", "procedural_parameter"))
        textures.append(_texture_record("roughness_blender", _relative_texture(package_root, source), source, "non_color", "procedural_parameter"))
    if "map_Kd" in maps:
        source = texture_root / maps["map_Kd"]
        if not source.is_file():
            raise InfinigenPackageError(f"reason=missing_required_texture path={source.name}")
        textures.append(_texture_record("base_color", _relative_texture(package_root, source), source, "srgb", "authored_appearance"))
    if "map_Bump" in maps or "map_Kn" in maps:
        filename = maps.get("map_Bump") or maps.get("map_Kn")
        source = texture_root / filename
        if not source.is_file():
            raise InfinigenPackageError(f"reason=missing_required_texture path={filename}")
        if abs(float(slot["bump_strength"]) - 1.0) > 1e-6:
            raise InfinigenPackageError(f"reason=normal_strength_not_applied value={slot['bump_strength']}")
        if convention["flip_green"]:
            raise InfinigenPackageError("reason=wrong_normal_convention")
        textures.append(_texture_record("normal_opengl", _relative_texture(package_root, source), source, "non_color", "authored_appearance", orientation=convention["orientation"]))
    kept = _textures_for_family(family, textures)
    kd = [min(max(float(channel), 0.0), 1.0) for channel in slot["Kd"]]
    ior = max(float(choice["ior"]), 1.0)
    if family == "diffuse":
        parameters = {"reflectance": kd}
    elif family == "pplastic":
        parameters = {"diffuse_reflectance": kd, "int_ior": 1.5, "ext_ior": 1.0, "alpha": float(alpha_scalar), "distribution": "beckmann"}
    elif family == "roughdielectric":
        parameters = {"int_ior": ior, "ext_ior": 1.0, "alpha": float(max(alpha_scalar, 1e-4)), "distribution": "beckmann"}
    elif family in {"dielectric", "thindielectric"}:
        parameters = {"int_ior": ior, "ext_ior": 1.0}
    else:
        raise InfinigenPackageError(f"reason=bsdf_not_built family={family}")
    spec = make_spec(material_id, family, parameters, textures=kept)
    spec["binding_status"] = "approximated" if choice["approximated"] else "bound"
    spec["polarization"] = polarization_role(family)
    spec["displacement_mode"] = "NONE"
    spec["metallic_status"] = "unsupported" if float(slot.get("Pm") or 0.0) > 0.5 else "constant_zero"
    spec["choice"] = {
        "family": choice["family"],
        "note": choice["note"],
        "preserved": choice["preserved"],
        "approximated": choice["approximated"],
        "lost": choice["lost"],
        "quantitative_polarization": choice["quantitative_polarization"],
        "qualitative_ok": choice["qualitative_ok"],
    }
    return spec


def _textures_for_family(family: str, textures: list[dict]) -> list[dict]:
    allowed = {
        "diffuse": {"base_color", "normal_opengl"},
        "pplastic": {"base_color", "alpha_mitsuba", "roughness_blender", "normal_opengl"},
        "roughdielectric": {"alpha_mitsuba", "roughness_blender", "normal_opengl"},
        "dielectric": {"normal_opengl"},
        "thindielectric": set(),
    }.get(family, set())
    return [item for item in textures if item["role"] in allowed]


def _relative_texture(package_root: Path, source: Path) -> str:
    try:
        return source.resolve().relative_to(package_root.resolve()).as_posix()
    except ValueError as exc:
        raise InfinigenPackageError(f"reason=texture_outside_package path={source}") from exc


def _texture_record(role: str, relative: str, source: Path, color_space: str, parameter_class: str, orientation: str = "opengl_tangent") -> dict:
    with Image.open(source) as image:
        resolution = [int(image.size[0]), int(image.size[1])]
    return {
        "role": role,
        "path": relative,
        "color_space": color_space,
        "channel": "r" if role in {"roughness_blender", "alpha_mitsuba"} else "rgb",
        "resolution": resolution,
        "sha256": _sha(source),
        "parameter_class": parameter_class,
        "uv_transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "normal_strength": 1.0,
        "normal_orientation": orientation,
        "filter_type": "bilinear",
        "wrap_mode": "repeat",
        "raw": color_space == "non_color",
    }


def _translation(matrix: list[list[float]]) -> list[float]:
    return [float(matrix[row][3]) for row in range(3)]


def _category(name: str) -> str:
    match = re.match(r"([A-Za-z]+)", name)
    text = match.group(1) if match else "object"
    return text[: -len("Factory")] if text.endswith("Factory") else text


def build_package(export_root: Path, graph_path: Path, package_root: Path, *, camera_npz: Path | None = None) -> dict:
    """Write ``scene_manifest.json`` and Z-up region meshes. The blend is not opened."""
    export_root = Path(export_root)
    package_root = Path(package_root)
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    convention = normal_convention(graph["normal_space"], graph["normal_g"])
    materials = [slot for obj in graph["objects"] if obj["type"] == "MESH" for slot in obj["materials"]]
    audit = displacement_audit(materials)
    if audit["geometric_displacement_slots"]:
        raise InfinigenPackageError("reason=displacement_not_in_mesh")
    by_name = {obj["name"]: obj for obj in graph["objects"]}
    obj_files = {path.parent.name: path for path in export_root.glob("*/*.obj")}
    assets = []
    instances = []
    seen: dict[str, str] = {}
    for obj in graph["objects"]:
        if obj["type"] != "MESH" or obj["hide_render"] or obj["vertices"] < 3:
            continue
        if "bbox_placeholder" in obj["name"] or obj["name"].startswith("hoof_parent"):
            continue
        folder = sanitize_object_name(obj["name"])
        source = obj_files.get(folder)
        if source is None:
            logger.info("infinigen_package event=missing_obj name=%s", obj["name"])
            continue
        parsed = parse_obj(source)
        translation = _translation(obj["matrix_world"])
        converted = np.vstack([world_from_exported_vertex(vertex, [0.0, 0.0, 0.0]) for vertex in parsed["vertices"]]) if parsed["vertices"] else np.zeros((0, 3))
        converted_normals = np.vstack([world_from_exported_vertex(vertex, [0.0, 0.0, 0.0]) for vertex in parsed["normals"]]) if parsed["normals"] else np.zeros((0, 3))
        lengths = np.linalg.norm(converted_normals, axis=1)
        converted_normals = np.divide(converted_normals, lengths[:, None], out=np.zeros_like(converted_normals), where=lengths[:, None] > 1e-12)
        mtl_path = source.with_suffix(".mtl")
        slots = parse_mtl(mtl_path) if mtl_path.is_file() else {}
        regions = []
        for index, (material_name, faces) in enumerate(parsed["groups"]):
            if not faces:
                continue
            region_vertices, region_uvs, region_normals, region_faces = _compact(converted, parsed["uvs"], converted_normals, faces)
            if len(region_uvs) == 0 and any("map_" in key for key in slots.get(material_name, {}).get("maps", {})):
                raise InfinigenPackageError(f"reason=missing_uvs material={material_name}")
            digest = hashlib.sha256()
            digest.update(np.ascontiguousarray(region_vertices).tobytes())
            digest.update(material_name.encode())
            digest.update(str(len(region_faces)).encode())
            key = digest.hexdigest()
            mesh_name = seen.get(key)
            if mesh_name is None:
                mesh_name = f"meshes/{folder}__{index}.obj"
                _write_obj(package_root / mesh_name, region_vertices, region_uvs, region_normals, region_faces, material_name)
                seen[key] = mesh_name
            slot = slots.get(material_name) or {"name": material_name, "maps": {}, "Kd": [0.6, 0.6, 0.6], "Pr": 0.5, "Pm": 0.0, "Ni": 1.5, "d": 1.0, "bump_strength": 1.0}
            spec = _spec_for_slot(slot, material_id=f"{folder}_{index}", texture_root=source.parent, package_root=package_root, convention=convention)
            regions.append({"slot": material_name, "mesh": mesh_name, "faces": len(region_faces), "spec": spec})
        if not regions:
            continue
        category = _category(obj["name"])
        frame = None
        if len(converted) > 3:
            corners = [[(int(corner[0]),) for corner in face] for _name, faces in parsed["groups"] for face in faces]
            frame = measure_semantic_frame({"vertices": converted, "corners": corners}, category)
        asset_id = folder
        assets.append({"id": asset_id, "mesh": regions[0]["mesh"], "regions": regions, "split_meshes": len(regions) > 1, "category": category})
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = translation
        instances.append({
            "id": folder,
            "asset_id": asset_id,
            "matrix": matrix.tolist(),
            "matrix_world_blend": obj["matrix_world"],
            "category": category,
            "frame": frame,
            "regions": regions,
            "parent": obj["parent"],
        })
        logger.info("infinigen_package event=instance name=%s regions=%d translation=%s", obj["name"], len(regions), translation)
    camera = _camera(graph, camera_npz)
    lights = _lights(graph)
    calibration = _calibration(package_root)
    marker = make_spec("calibration_marker", "diffuse", {"reflectance": [0.8, 0.05, 0.05]})
    assets.append({
        "id": "calibration_marker",
        "mesh": calibration["mesh"],
        "regions": [{"slot": "calibration", "mesh": calibration["mesh"], "faces": 18, "spec": marker}],
        "split_meshes": False,
        "category": "calibration",
    })
    instances.append({
        "id": "calibration_marker",
        "asset_id": "calibration_marker",
        "matrix": calibration["matrix"],
        "parent": None,
        "category": "calibration",
        "frame": None,
        "regions": [{"slot": "calibration", "mesh": calibration["mesh"], "spec": marker}],
    })
    manifest = {
        "schema": "infinigen_mitsuba_scene_v1",
        "variant": "cuda_ad_spectral_polarized",
        "units": "m",
        "world_up": [0.0, 0.0, 1.0],
        "handedness": "right",
        "axis_conversion": AXIS_NOTE,
        "mesh_format": MESH_FORMAT,
        "mesh_format_reason": MESH_FORMAT_REASON,
        "normal_convention": convention,
        "displacement": audit,
        "camera": camera,
        "lights": lights,
        "environment": {"type": "constant", "radiance": 0.35, "polarization": "unpolarized", "reason": "world_node_tree_not_converted"},
        "assets": assets,
        "instances": instances,
        "calibration": calibration,
    }
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "scene_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (package_root / "cameras" / "camera.json").parent.mkdir(parents=True, exist_ok=True)
    (package_root / "cameras" / "camera.json").write_text(json.dumps(camera, indent=2) + "\n", encoding="utf-8")
    (package_root / "lights" / "lights.json").parent.mkdir(parents=True, exist_ok=True)
    (package_root / "lights" / "lights.json").write_text(json.dumps(lights, indent=2) + "\n", encoding="utf-8")
    logger.info("infinigen_package event=manifest instances=%d assets=%d", len(instances), len(assets))
    return manifest


def _camera(graph: dict, camera_npz: Path | None) -> dict:
    cameras = [obj for obj in graph["objects"] if obj["type"] == "CAMERA" and obj["name"].endswith("_0")]
    if not cameras:
        raise InfinigenPackageError("reason=missing_camera")
    record = cameras[0]
    data = next(item for item in graph["cameras"] if item["name"] == record["name"])
    matrix = record["matrix_world"]
    origin = _translation(matrix)
    forward = np.array([-matrix[row][2] for row in range(3)], dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    width, height = graph["render_resolution"]
    camera = {
        "id": record["name"],
        "origin": origin,
        "target": (np.array(origin) + forward).tolist(),
        "up": [0.0, 0.0, 1.0],
        "fov_deg": vertical_fov_degrees(data["lens_mm"], data["sensor_width_mm"], width, height),
        "width": int(width),
        "height": int(height),
        "clip_start_m": data["clip_start_m"],
        "clip_end_m": data["clip_end_m"],
        "lens_mm": data["lens_mm"],
        "sensor_width_mm": data["sensor_width_mm"],
        "sensor_fit": data["sensor_fit"],
        "matrix_world": matrix,
    }
    if camera_npz is not None and Path(camera_npz).is_file():
        loaded = np.load(camera_npz)
        camera["cycles_K"] = loaded["K"].tolist()
        camera["cycles_T"] = loaded["T"].tolist()
        camera["cycles_hw"] = [int(item) for item in loaded["HW"]]
    return camera


def _lights(graph: dict) -> list[dict]:
    lights = []
    for obj in graph["objects"]:
        if obj["type"] != "LIGHT":
            continue
        data = next(item for item in graph["lights"] if item["name"] == obj["name"])
        position = _translation(obj["matrix_world"])
        record = {
            "id": sanitize_object_name(obj["name"]),
            "blender_type": data["light_type"],
            "position": position,
            "matrix": obj["matrix_world"],
            "intensity": float(data["energy"]),
            "color": data["color"],
            "size": data["size"],
            "polarization": "unpolarized",
            "parent": obj["parent"],
        }
        scale = (
            float(np.linalg.norm(np.asarray(obj["matrix_world"], dtype=np.float64)[0, :3])),
            float(np.linalg.norm(np.asarray(obj["matrix_world"], dtype=np.float64)[1, :3])),
            float(np.linalg.norm(np.asarray(obj["matrix_world"], dtype=np.float64)[2, :3])),
        )
        if data["light_type"] == "POINT":
            classified = classify_point_emitter(
                data["energy"],
                float(data.get("shadow_soft_size", data.get("radius_m", data.get("size", 0.0)))),
                position,
            )
        elif data["light_type"] == "AREA":
            classified = classify_area_emitter(
                data["energy"],
                shape=str(data.get("shape", "SQUARE")),
                size=float(data.get("size") or 0.0),
                size_y=float(data.get("size_y", data.get("size") or 0.0)),
                scale=scale,
                two_sided=bool(data.get("two_sided", False)),
                spread_rad=float(data.get("spread_rad", np.pi)),
                position=position,
            )
        else:
            classified = {"type": "omitted", "included": False, "reason": "light_type_not_converted", "conversion": "not_converted"}
        record.update(classified)
        lights.append(record)
    return lights


def _calibration(package_root: Path) -> dict:
    """An asymmetric marker: +X is longest, +Y is middle, +Z is shortest."""
    vertices = [
        [0, 0, 0], [0.40, 0, 0], [0.40, 0.02, 0], [0, 0.02, 0], [0, 0, 0.02], [0.40, 0, 0.02], [0.40, 0.02, 0.02], [0, 0.02, 0.02],
        [0, 0, 0], [0, 0.20, 0], [0.02, 0.20, 0], [0.02, 0, 0], [0, 0, 0.02], [0, 0.20, 0.02], [0.02, 0.20, 0.02], [0.02, 0, 0.02],
        [0, 0, 0], [0, 0, 0.10], [0.02, 0, 0.10], [0.02, 0, 0], [0, 0.02, 0], [0, 0.02, 0.10], [0.02, 0.02, 0.10], [0.02, 0.02, 0],
    ]
    faces = []
    for base in (0, 8, 16):
        quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
        for quad in quads:
            faces.append(tuple((base + index, None, None) for index in quad))
    path = package_root / "meshes" / "calibration_marker.obj"
    _write_obj(path, np.asarray(vertices, dtype=np.float64), [], np.zeros((0, 3)), faces, "calibration")
    translation = [8.5, 5.0, 0.2]
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return {
        "mesh": "meshes/calibration_marker.obj",
        "matrix": matrix.tolist(),
        "lengths_m": {"+X": 0.40, "+Y": 0.20, "+Z": 0.10},
        "expected_world_max": [translation[0] + 0.40, translation[1] + 0.20, translation[2] + 0.10],
    }

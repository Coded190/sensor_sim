"""Qualification driver for material transfer, BSDF compatibility, and plastic shells.

Scientific renders use MaterialSpec and the direct Mitsuba mapper. Blender
export is recorded separately and is not described as the scientific path.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import canonical_json_bytes
from sensor_sim.scene_gen.indoor_mitsuba import render_stokes, require_variant, write_rgb_png
from sensor_sim.scene_gen.material_assign import STREAMS, assign_fixed, replay_spec, sample_family
from sensor_sim.scene_gen.material_spec import FLOAT_TOLERANCE, SCHEMA_VERSION, content_hash, make_spec, preset, to_mitsuba, transfer_cases, validate_spec, values_close
from sensor_sim.scene_gen.mesh_io import read_obj, write_obj
from sensor_sim.scene_gen.plastic_wrap import file_hash, make_wrap_spec, mesh_hash, shell_report
from sensor_sim.scene_gen.scene_context import compose_scene_records, require_scene_context, sample_room_kinds
from sensor_sim.scene_gen.structural import box_mesh
from sensor_sim.scene_gen.tabletop_render import mitsuba_bsdf

logger = logging.getLogger("sensor_sim.scene_gen.material_review")

ROOT = Path("dataset/qualification/material_wrap_review")
BLENDER = Path("tools/blender-4.5.9-linux-x64/blender")
ADDON_COMMIT = "8848ded5e5df575f2cd0a9c10061d3f6b7137905"
RENDER_SEED = 17521
RESOLUTION = (640, 480)
SPP = 64
PROBE_SPP = 4


def _run_blender(script: Path, params: Path) -> subprocess.CompletedProcess:
    command = [str(BLENDER), "--background", "--python", str(script), "--", str(params)]
    logger.info("material_review event=blender script=%s", script.name)
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _texture(path: Path) -> None:
    import mitsuba as mi

    require_variant()
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[:4, :4] = (0.85, 0.10, 0.10)
    image[:4, 4:] = (0.10, 0.75, 0.15)
    image[4:, :4] = (0.10, 0.20, 0.80)
    image[4:, 4:] = (0.90, 0.80, 0.15)
    path.parent.mkdir(parents=True, exist_ok=True)
    mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=False).write(str(path))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_bsdfs(xml_path: Path) -> list[dict]:
    """Read BSDF type and numeric parameters from one exported scene."""
    root = ET.parse(xml_path).getroot()
    found = []

    def walk(node, parent_type=None):
        if _local(node.tag) != "bsdf":
            for child in node:
                walk(child, parent_type)
            return
        kind = node.attrib.get("type", "")
        record = {"id": node.attrib.get("id", ""), "type": kind, "parent": parent_type, "values": {}}
        for child in node:
            name = _local(child.tag)
            if name in {"float", "string", "boolean"}:
                record["values"][child.attrib.get("name", name)] = child.attrib.get("value")
            elif name in {"rgb", "spectrum"}:
                record["values"][child.attrib.get("name", name)] = child.attrib.get("value")
            elif name == "bsdf":
                walk(child, kind)
        found.append(record)

    walk(root)
    return found


def _numbers(value: str) -> list[float]:
    return [float(part) for part in value.replace(",", " ").split()]


def compare_transfer(xml_path: Path, cases: list[dict], log_text: str) -> list[dict]:
    bsdfs = parse_bsdfs(xml_path)
    text = xml_path.read_text(encoding="utf-8", errors="replace")
    warnings = [line for line in log_text.splitlines() if "WARN" in line or "Warning" in line]
    rows = []
    for case in cases:
        kind = case.get("kind", "recipe")
        export_supported = xml_path.is_file() and "<bsdf" in text
        verified = False
        notes = []
        if kind == "image_texture":
            verified = "bitmap" in text or "texture" in text
            if not verified:
                notes.append("image texture reference was not found in the XML")
        elif kind == "normal_map":
            verified = "normalmap" in text or "bumpmap" in text
            if not verified:
                notes.append("normal or bump adapter was not found in the XML")
        elif kind == "two_slots":
            red = any(item["values"].get("reflectance") and values_close(_numbers(item["values"]["reflectance"]), [0.8, 0.1, 0.1]) for item in bsdfs if item["type"] == "diffuse")
            blue = any(item["values"].get("reflectance") and values_close(_numbers(item["values"]["reflectance"]), [0.1, 0.2, 0.8]) for item in bsdfs if item["type"] == "diffuse")
            verified = red and blue
            if not verified:
                notes.append("the two face colors were not both present as diffuse reflectance")
        else:
            verified, notes = _recipe_transferred(case, bsdfs)
        family = case.get("expected_family")
        recipe = case.get("recipe") or {}
        matches_direct = bool(verified) and family not in {None, "principled"} and recipe.get("node") != "BSDF_PRINCIPLED"
        if recipe.get("node") == "BSDF_PRINCIPLED" or family == "principled":
            notes.append("Principled export is a preview path. It is not the scientific pplastic BSDF.")
            matches_direct = False
        if kind in {"image_texture", "normal_map"}:
            notes.append("The add-on XML is recorded separately from the direct polarized MaterialSpec mapper.")
            matches_direct = False
        rows.append({
            "case_id": case["case_id"],
            "export_supported": bool(export_supported),
            "parameter_transfer_verified": bool(verified),
            "polarized_renderer_compatible": None,
            "matches_direct_polarized_bsdf": matches_direct,
            "blender_node": recipe.get("node"),
            "blender_sockets": sorted((recipe.get("sockets") or {}).keys()),
            "xml_bsdf_types": sorted({item["type"] for item in bsdfs}),
            "xml_bitmaps": _xml_bitmaps(xml_path),
            "notes": notes,
            "warnings": warnings,
            "scientific_path": "direct_material_spec",
        })
    return rows


def _xml_bitmaps(xml_path: Path) -> list[dict]:
    """Image paths and plugin types written by the pinned add-on."""
    root = ET.parse(xml_path).getroot()
    found = []
    for texture in root.iter("texture"):
        if texture.attrib.get("type") != "bitmap":
            continue
        filename = ""
        raw = None
        for child in texture:
            if child.tag == "string" and child.attrib.get("name") == "filename":
                filename = child.attrib.get("value", "")
            if child.tag == "boolean" and child.attrib.get("name") == "raw":
                raw = child.attrib.get("value")
        found.append({
            "plugin": "bitmap",
            "parent": texture.attrib.get("name"),
            "filename": filename,
            "raw": raw,
            "color_space_in_xml": "raw_non_color" if raw == "true" else "exporter_default",
        })
    return found


def _recipe_transferred(case: dict, bsdfs: list[dict]) -> tuple[bool, list[str]]:
    recipe = case["recipe"]
    sockets = recipe["sockets"]
    notes = []
    node = recipe["node"]
    if node == "BSDF_DIFFUSE":
        match = _find_color(bsdfs, "diffuse", "reflectance", sockets["Color"])
        if not match:
            notes.append("diffuse reflectance did not match the requested color")
        return match, notes
    if node == "BSDF_GLASS":
        roughness = float(sockets["Roughness"])
        ior_name = "int_ior"
        ior_ok = _find_float(bsdfs, "dielectric" if roughness == 0.0 else "roughdielectric", ior_name, sockets["IOR"])
        if roughness == 0.0:
            rough_ok = True
        else:
            rough_ok = _find_float(bsdfs, "roughdielectric", "alpha", roughness * roughness)
        if not ior_ok:
            notes.append("glass IOR was missing or replaced")
        if not rough_ok:
            notes.append("glass roughness was not exported as Mitsuba alpha = roughness^2")
        if not any(item["type"] == "dielectric" and "ext_ior" in item["values"] for item in bsdfs):
            notes.append("exterior IOR is omitted by the Glass exporter; the scientific mapper sets it explicitly")
        return ior_ok and rough_ok, notes
    if node == "BSDF_GLOSSY":
        roughness = float(sockets["Roughness"])
        kind = "conductor" if roughness == 0.0 else "roughconductor"
        present = any(item["type"] == kind for item in bsdfs)
        if not present:
            notes.append(f"glossy export did not produce {kind}")
        if roughness > 0.0 and not _find_float(bsdfs, "roughconductor", "alpha", roughness * roughness):
            notes.append("glossy roughness was not exported as squared alpha")
            present = False
        notes.append("the Glossy color is not the Mitsuba conductor preset; the scientific conductor name is set only by MaterialSpec")
        return present, notes
    color_ok = _find_color(bsdfs, "principled", "base_color", sockets["Base Color"])
    rough_ok = _find_float(bsdfs, "principled", "roughness", sockets["Roughness"])
    metal_ok = _find_float(bsdfs, "principled", "metallic", sockets["Metallic"])
    trans_ok = _find_float(bsdfs, "principled", "spec_trans", sockets["Transmission Weight"])
    aniso_ok = _find_float(bsdfs, "principled", "anisotropic", sockets["Anisotropic"])
    ior_ok = True if float(sockets["Transmission Weight"]) == 0.0 else _find_float(bsdfs, "principled", "eta", sockets["IOR"])
    for name, ok in (("base_color", color_ok), ("roughness", rough_ok), ("metallic", metal_ok), ("spec_trans", trans_ok), ("anisotropic", aniso_ok), ("ior", ior_ok)):
        if not ok:
            notes.append(f"principled {name} did not match the requested non-default value")
    return all((color_ok, rough_ok, metal_ok, trans_ok, aniso_ok, ior_ok)), notes


def _find_float(bsdfs, kind, name, expected) -> bool:
    for item in bsdfs:
        if item["type"] != kind or name not in item["values"]:
            continue
        if values_close(float(item["values"][name]), float(expected), FLOAT_TOLERANCE):
            return True
    return False


def _find_color(bsdfs, kind, name, expected) -> bool:
    for item in bsdfs:
        if item["type"] != kind or name not in item["values"]:
            continue
        if values_close(_numbers(item["values"][name]), expected, FLOAT_TOLERANCE):
            return True
    return False


def _sensor(resolution=RESOLUTION, origin=None, target=None, fov: float = 32.0):
    import mitsuba as mi

    require_variant()
    width, height = resolution
    return {
        "type": "perspective",
        "fov": float(fov),
        "to_world": mi.ScalarTransform4f.look_at(origin=list(origin or [0.0, -1.15, 0.55]), target=list(target or [0.0, 0.0, 0.12]), up=[0.0, 0.0, 1.0]),
        "sampler": {"type": "independent"},
        "film": {"type": "hdrfilm", "width": int(width), "height": int(height), "pixel_format": "rgb", "rfilter": {"type": "box"}},
    }


def _light():
    import mitsuba as mi

    require_variant()
    transform = mi.ScalarTransform4f.translate([0.15, -0.35, 1.05]) @ mi.ScalarTransform4f.rotate([1.0, 0.0, 0.0], 180.0) @ mi.ScalarTransform4f.scale([0.40, 0.40, 1.0])
    return {"type": "rectangle", "to_world": transform, "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 6.0}}}


def _render(shapes: list[dict], png: Path, spp: int = SPP, resolution=RESOLUTION, origin=None, target=None, fov: float = 32.0, emitters: dict | None = None) -> dict:
    scene = {"type": "scene", "integrator": {"type": "stokes", "integrator": {"type": "path", "max_depth": 6}}, "sensor": _sensor(resolution, origin, target, fov)}
    if emitters:
        scene.update(emitters)
    else:
        scene["light"] = _light()
    for shape in shapes:
        scene[shape["id"]] = shape["mitsuba"]
    exr = png.with_suffix(".exr")
    png.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_stokes(scene, exr, spp, RENDER_SEED)
        write_rgb_png(exr, png)
    except Exception as exc:
        logger.info("material_review event=render_fail path=%s error=%s", png, exc)
        return {"rendered": False, "error": str(exc), "png": str(png), "spp": spp, "resolution": list(resolution), "variant": "cuda_ad_spectral_polarized"}
    return {"rendered": True, "png": str(png), "exr": str(exr), "spp": spp, "resolution": list(resolution), "variant": "cuda_ad_spectral_polarized", "exposure": "write_rgb_png sRGB gamma"}


def _room_emitters(loaded: dict) -> dict:
    """Ceiling, window, and task lamp stored on the furnished-room bundle."""
    from sensor_sim.scene_gen.interior_scenes import _lamp_emitter
    from sensor_sim.scene_gen.room_generator import _daylight_emitters
    from sensor_sim.scene_gen.semantic_composition import _ceiling_emitter

    lighting = loaded["bundle"]["lighting"]
    emitters = {"ceiling": _ceiling_emitter({"lighting": lighting})}
    for emitter in _daylight_emitters({"lighting": lighting}):
        emitters[emitter["id"]] = emitter["mitsuba"]
    emitters["task_lamp"] = _lamp_emitter(lighting["lamp"])
    logger.info(
        "material_review event=room_light room_kind=%s ceiling=%.4f lamp=%.4f windows=%d",
        loaded["room_kind"],
        float(lighting["ceiling"]["radiance"]),
        float(lighting["lamp"]["radiance"]),
        len(lighting["daylight"]["emitters"]),
    )
    return emitters


def _render_room(shapes, png, loaded, view: str, spp: int, resolution) -> dict:
    origin, target, fov = _room_camera(loaded, view)
    result = _render(shapes, png, spp=spp, resolution=resolution, origin=origin, target=target, fov=fov, emitters=_room_emitters(loaded))
    if not result["rendered"]:
        raise RuntimeError(f"reason=approval_render_failed path={png} error={result.get('error')}")
    return result


def probe_family(spec: dict, destination: Path) -> dict:
    """Instantiate and render one BSDF. Failure is recorded. The family is not replaced."""
    try:
        bsdf = to_mitsuba(spec)
    except Exception as exc:
        return {"family": spec["family"], "material_id": spec["material_id"], "constructed": False, "instantiated": False, "rendered": False, "error": str(exc), "polarized_renderer_compatible": False}
    result = _render([{"id": "probe", "mitsuba": {"type": "sphere", "center": [0.0, 0.0, 0.12], "radius": 0.12, "bsdf": bsdf}}], destination, spp=PROBE_SPP, resolution=(96, 72))
    result.update({"family": spec["family"], "material_id": spec["material_id"], "constructed": True, "instantiated": result["rendered"], "polarized_renderer_compatible": result["rendered"], "bsdf": bsdf})
    logger.info("material_review event=probe family=%s compatible=%s", spec["family"], result["polarized_renderer_compatible"])
    return result


def _shape_obj(instance_id: str, path: Path, bsdf: dict, matrix=None) -> dict:
    import mitsuba as mi

    require_variant()
    transform = mi.ScalarTransform4f() if matrix is None else mi.ScalarTransform4f(matrix)
    return {"id": instance_id, "mitsuba": {"type": "obj", "filename": str(path.resolve()), "to_world": transform, "bsdf": bsdf}}


def _write_box(path: Path) -> dict:
    vertices, polygons = box_mesh([-0.10, -0.10, 0.0], [0.10, 0.10, 0.20])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(write_obj(vertices, polygons, name="box"))
    return {"vertices": vertices, "faces": polygons, "path": path, "hash": mesh_hash(vertices, np.asarray(polygons))}


def run_transfer(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    texture = root / "checker.png"
    _texture(texture)
    cases = transfer_cases()
    params = {"xml": str(root / "transfer.xml"), "texture": str(texture.resolve()), "cases": cases, "node_report": str(root / "node_report.json")}
    params_path = root / "transfer_params.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    completed = _run_blender(Path("assets/blender/material_transfer_probe.py"), params_path)
    (root / "transfer_blender.log").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    xml = root / "transfer.xml"
    if completed.returncode != 0 or not xml.is_file():
        return {"export_supported": False, "returncode": completed.returncode, "cases": [], "log": str(root / "transfer_blender.log")}
    rows = compare_transfer(xml, cases, completed.stdout + completed.stderr)
    (root / "transfer_table.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    logger.info("material_review event=transfer cases=%d verified=%d", len(rows), sum(row["parameter_transfer_verified"] for row in rows))
    return {"export_supported": True, "addon_commit": ADDON_COMMIT, "cases": rows, "xml": str(xml)}


def run_polarized(root: Path) -> list[dict]:
    names = ["matte_gray", "matte_red", "plastic_smooth", "plastic_rough", "glass_145", "glass_160", "rough_glass", "thin_sheet", "aluminum", "aluminum_rough", "principled_plastic"]
    rows = []
    for name in names:
        spec = preset(name)
        rows.append(probe_family(spec, root / "polarized" / f"{name}.png"))
    (root / "polarized_compatibility.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def _supported_families(rows: list[dict]) -> set[str]:
    return {row["family"] for row in rows if row.get("polarized_renderer_compatible")}


def run_material_gallery(root: Path, supported: set[str]) -> list[dict]:
    """Same sphere, camera, and light. Only the MaterialSpec changes."""
    jobs = [
        ("family_diffuse", preset("matte_gray")),
        ("family_plastic", preset("plastic_rough")),
        ("family_dielectric", preset("glass_145")),
        ("family_conductor", preset("aluminum")),
        ("reflectance_low", assign_fixed("reflectance_low", "matte_blue")),
        ("reflectance_high", preset("matte_red")),
        ("rough_low", preset("plastic_smooth")),
        ("rough_high", preset("plastic_rough")),
        ("ior_low", preset("glass_145")),
        ("ior_high", preset("glass_160")),
    ]
    if "roughdielectric" in supported:
        jobs.append(("family_rough_dielectric", preset("rough_glass")))
    written = []
    for name, spec in jobs:
        if spec["family"] not in supported:
            written.append({"name": name, "rendered": False, "reason": "family_not_compatible", "spec": spec})
            continue
        png = root / "materials" / f"{name}.png"
        bsdf = to_mitsuba(spec)
        image = _render([{"id": "ball", "mitsuba": {"type": "sphere", "center": [0.0, 0.0, 0.12], "radius": 0.12, "bsdf": bsdf}}], png)
        (png.with_suffix(".json")).write_text(json.dumps({"spec": spec, "bsdf": bsdf, "render": image}, indent=2) + "\n", encoding="utf-8")
        written.append({"name": name, **image, "family": spec["family"], "material_id": spec["material_id"]})
    body = to_mitsuba(preset("matte_blue"))
    lid = to_mitsuba(preset("aluminum"))
    parts = _render([
        {"id": "body", "mitsuba": {"type": "cube", "to_world": _cube_transform([0.0, 0.0, 0.08], [0.16, 0.16, 0.12]), "bsdf": body}},
        {"id": "lid", "mitsuba": {"type": "cube", "to_world": _cube_transform([0.0, 0.0, 0.20], [0.16, 0.16, 0.04]), "bsdf": lid}},
    ], root / "materials" / "two_components.png")
    (root / "materials" / "two_components.json").write_text(json.dumps({"components": ["body", "lid"], "families": ["diffuse", "conductor"], "render": parts}, indent=2) + "\n", encoding="utf-8")
    written.append({"name": "two_components", **parts})
    (root / "materials" / "index.json").write_bytes(canonical_json_bytes({"images": written, "spp": SPP, "resolution": list(RESOLUTION), "seed": RENDER_SEED}))
    return written


def _cube_transform(center, scale):
    import mitsuba as mi

    require_variant()
    return mi.ScalarTransform4f.translate(center) @ mi.ScalarTransform4f.scale([value * 0.5 for value in scale])


def _load_mesh(path: Path) -> dict:
    vertices, polygons = read_obj(path)
    return {"vertices": vertices, "faces": polygons}


def run_wraps(root: Path) -> list[dict]:
    box = _write_box(root / "targets" / "box.obj")
    targets = [
        {"id": "box", "path": box["path"], "mesh": {"vertices": box["vertices"], "faces": box["faces"]}, "source_hash": file_hash(box["path"])},
        {"id": "sphere", "path": Path("assets/candidates/fixture_sphere/1.0.0/geometry/render.obj"), "source_hash": None},
        {"id": "apple", "path": Path("assets/candidates/prop_apple/1.0.0/geometry/render.obj"), "source_hash": None},
    ]
    for target in targets:
        target["source_hash"] = file_hash(target["path"])
        target["mesh"] = _load_mesh(target["path"])
    results = []
    for target in targets:
        full_ok = False
        for mode, wrinkle in (("partial", 0.0), ("full", 0.0), ("full", 0.0015)):
            if wrinkle > 0.0 and not full_ok:
                results.append({"target": target["id"], "mode": "full", "wrinkle_m": wrinkle, "geometry_passed": False, "reason": "smooth_full_wrap_did_not_pass"})
                continue
            spec = make_wrap_spec(f"{target['id']}_{mode}_{wrinkle}", target["id"], mode, offset_m=0.004, thickness_m=0.002, seed=17601 if wrinkle == 0.0 else 17611, wrinkle_amplitude_m=wrinkle, film_ior=1.5, substrate_family="diffuse")
            destination = root / "wraps" / f"{target['id']}_{mode}" / ("smooth" if wrinkle == 0.0 else "wrinkle")
            if destination.exists():
                shutil.rmtree(destination)
            params = {"spec": spec, "target_obj": str(target["path"].resolve()), "output_dir": str(destination.resolve())}
            params_path = root / "wraps" / f"{target['id']}_{mode}_{'smooth' if wrinkle == 0.0 else 'wrinkle'}_params.json"
            params_path.parent.mkdir(parents=True, exist_ok=True)
            params_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
            completed = _run_blender(Path("assets/blender/build_plastic_wrap.py"), params_path)
            (destination / "blender.log").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8") if destination.exists() else None
            row = {"target": target["id"], "mode": mode, "wrinkle_m": wrinkle, "spec": spec, "blender_returncode": completed.returncode, "source_hash_before": target["source_hash"]}
            applied = destination / "shell_applied.obj"
            meta_path = destination / "blender_meta.json"
            if completed.returncode != 0 or not applied.is_file():
                row.update({"geometry_passed": False, "reason": "blender_failed", "source_hash_after": file_hash(target["path"])})
                results.append(row)
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            shell = _load_mesh(applied)
            row["source_hash_after"] = file_hash(target["path"])
            row["target_unchanged"] = row["source_hash_before"] == row["source_hash_after"] and meta.get("target_unchanged") is True
            row["shell_hash"] = mesh_hash(shell["vertices"], _tri(shell["faces"]))
            row["boundary_edges"] = meta.get("boundary_edges")
            ply = _compare_ply(destination, shell)
            row["ply_match"] = ply
            try:
                report = shell_report(shell, target["mesh"], spec)
                row["geometry_passed"] = bool(row["target_unchanged"] and report["passed_geometry"])
                row["ply_matched"] = bool(ply["matched"])
                row["report"] = report
            except Exception as exc:
                row["geometry_passed"] = False
                row["reason"] = str(exc)
            if mode == "full" and wrinkle == 0.0:
                full_ok = bool(row.get("geometry_passed") and row.get("ply_matched"))
            if row.get("geometry_passed"):
                _render_wrap(root, target, shell, spec, destination)
            results.append(row)
            logger.info("material_review event=wrap target=%s mode=%s wrinkle=%s passed=%s", target["id"], mode, wrinkle, row.get("geometry_passed"))
    (root / "wraps" / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    return results


def _tri(faces) -> np.ndarray:
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        else:
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
    return np.asarray(triangles, dtype=np.int64)


def _read_ply_vertices(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    count = 0
    floats = 0
    in_vertex = False
    for line in header.decode("ascii", errors="replace").splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element "):
            in_vertex = False
        elif in_vertex and line.startswith("property float"):
            floats += 1
    record = np.dtype([("values", "<f4", (floats,))])
    values = np.frombuffer(body, dtype=record, count=count)
    return values["values"][:, :3].astype(np.float64)


def _compare_ply(destination: Path, shell: dict) -> dict:
    """Match applied vertices to the PLY. The exporter triangulates and splits normals, so indices differ."""
    ply_files = list(destination.rglob("*.ply"))
    if not ply_files:
        return {"matched": False, "reason": "no_ply"}
    exported = np.vstack([_read_ply_vertices(path) for path in ply_files])
    applied = np.asarray(shell["vertices"], dtype=np.float64)
    # Measured on the box shell: the exporter writes (x, y, z) as (x, -z, y).
    mapped = np.stack([applied[:, 0], -applied[:, 2], applied[:, 1]], axis=1)
    exported_key = {tuple(np.round(vertex, 4)) for vertex in exported}
    found = sum(tuple(np.round(vertex, 4)) in exported_key for vertex in mapped)
    coverage = found / max(len(applied), 1)
    return {
        "matched": coverage >= 0.99,
        "position_coverage": coverage,
        "face_index_identical": False,
        "axis_map": "(x, y, z) -> (x, -z, y)",
        "face_index_note": "The exporter triangulates faces, splits vertices by normal, and applies that axis map, so face indices differ from the applied OBJ.",
        "ply_vertices": int(len(exported)),
        "applied_vertices": int(len(applied)),
        "ply_sha256": content_hash(ply_files[0]),
    }


def _render_wrap(root: Path, target: dict, shell: dict, spec: dict, destination: Path) -> None:
    from sensor_sim.scene_gen.plastic_wrap import classify_faces

    groups = classify_faces(shell["vertices"], _tri(shell["faces"]), np.asarray(target["mesh"]["vertices"]).mean(axis=0))
    film = to_mitsuba(preset("glass_145"))
    substrate = to_mitsuba(preset("matte_red"))
    ground = to_mitsuba(preset("matte_gray"))
    import mitsuba as mi
    require_variant()
    ground_shape = {"id": "ground", "mitsuba": {"type": "rectangle", "to_world": mi.ScalarTransform4f.translate([0.0, 0.0, -0.02]) @ mi.ScalarTransform4f.scale([1.5, 1.5, 1.0]), "bsdf": ground}}
    shapes = [ground_shape, _shape_obj("target", target["path"], substrate)]
    require_scene_context(
        [
            {"id": "ground", "asset_id": "ground", "kind": "plane"},
            {"id": "target", "asset_id": target["id"], "role": "subject"},
        ],
        room_kind="probe",
        allow_bare_plane=True,
        allow_without_context_objects=True,
        exemption="controlled_material_probe",
    )
    for name, faces in groups["groups"].items():
        if len(faces) == 0:
            continue
        path = destination / f"{name}.obj"
        path.write_bytes(write_obj(shell["vertices"], [list(map(int, face)) for face in faces], name=name))
        shapes.append(_shape_obj(name, path, film))
    label = destination.name
    png = root / "gallery_src" / f"{target['id']}_{spec['mode']}_{label}.png"
    image = _render(shapes, png)
    close = _render(shapes, root / "gallery_src" / f"{target['id']}_{spec['mode']}_{label}_close.png", resolution=(640, 480))
    (png.with_suffix(".json")).write_text(json.dumps({"spec": spec, "render": image, "close": close, "shell_hash": mesh_hash(shell["vertices"], _tri(shell["faces"])), "target_hash": file_hash(target["path"])}, indent=2, default=str) + "\n", encoding="utf-8")


def write_gallery(root: Path) -> None:
    import html

    review = root / "review" / "index.html"
    review.parent.mkdir(parents=True, exist_ok=True)
    body = ["<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Material and wrap qualification</title>",
            "<style>body{font-family:sans-serif;margin:2rem;background:#111;color:#eee} img{max-width:640px;background:#222} figure{margin:1rem 0} .notice{border:1px solid #c90;padding:1rem}</style></head><body>",
            "<h1>Material randomization and conformal wrap</h1>",
            "<p class=\"notice\">Mitsuba RGB is the evidence. These scenes are not visually reviewed, admitted, or registered. Polarization accuracy was not measured.</p>"]
    body.append("<h2>Materials</h2><p>The sphere, camera, light, exposure, resolution, and sample count stay fixed. Only the material specification changes.</p>")
    for png in sorted((root / "materials").glob("*.png")):
        href = Path(os.path.relpath(png, review.parent)).as_posix()
        meta = json.loads(png.with_suffix(".json").read_text(encoding="utf-8")) if png.with_suffix(".json").is_file() else {}
        spec = meta.get("spec", {})
        caption = f"{png.stem} family={spec.get('family', meta.get('components'))} id={spec.get('material_id', '')}"
        body.append(f"<figure><img src=\"{html.escape(href)}\" alt=\"{html.escape(png.stem)}\"><figcaption>{html.escape(caption)}</figcaption></figure>")
    body.append("<h2>Wraps</h2><p>Unwrapped targets are the red substrate. A shell is shown only when its geometry check passed. Full wrap is per target class.</p>")
    gallery = root / "gallery_src"
    if gallery.is_dir():
        for png in sorted(gallery.glob("*.png")):
            href = Path(os.path.relpath(png, review.parent)).as_posix()
            body.append(f"<figure><img src=\"{html.escape(href)}\" alt=\"{html.escape(png.stem)}\"><figcaption>{html.escape(png.stem)} Mitsuba RGB</figcaption></figure>")
    body.append("</body></html>\n")
    review.write_text("\n".join(body), encoding="utf-8")
    logger.info("material_review event=gallery path=%s", review)


def seed_invariants() -> dict:
    """Changing one seed must not change the other streams' outputs."""
    geometry = sample_family("diffuse", 11, "geo")
    material_a = sample_family("pplastic", 21, "mat")
    material_b = sample_family("pplastic", 22, "mat")
    wrap_a = make_wrap_spec("w", "box", "partial", offset_m=0.004, thickness_m=0.002, seed=31)
    wrap_b = make_wrap_spec("w", "box", "partial", offset_m=0.004, thickness_m=0.002, seed=32)
    replay = replay_spec(material_a)
    return {
        "material_changes": material_a["parameters"] != material_b["parameters"],
        "replay_exact": replay["parameters"] == material_a["parameters"],
        "wrap_seed_does_not_edit_target_fields": wrap_a["target_id"] == wrap_b["target_id"] and wrap_a["offset_m"] == wrap_b["offset_m"],
        "geometry_spec_independent": geometry["seed"] == 11 and material_a["seed"] == 21,
        "streams": list(STREAMS),
        "schema_version": SCHEMA_VERSION,
    }


def _signature(spec: dict) -> str:
    return json.dumps({"family": spec["family"], "parameters": spec["parameters"]}, sort_keys=True)


def _approval_materials() -> tuple[dict, list[dict]]:
    """Fixed assignment, then three draws where every object changes material."""
    fixed = {
        "sphere": preset("matte_red"),
        "box": preset("plastic_smooth"),
        "apple": preset("aluminum"),
    }
    green = make_spec("apple_green_plastic", "pplastic", {"diffuse_reflectance": [0.15, 0.55, 0.18], "int_ior": 1.5, "ext_ior": 1.0, "alpha": 0.22, "distribution": "beckmann"})
    yellow = make_spec("box_yellow", "diffuse", {"reflectance": [0.82, 0.62, 0.12]})
    copper = make_spec("box_copper", "conductor", {"material": "Cu"})
    draws = [
        {"sphere": preset("matte_red"), "box": preset("plastic_smooth"), "apple": preset("aluminum")},
        {"sphere": preset("matte_blue"), "box": copper, "apple": green},
        {"sphere": preset("aluminum_rough"), "box": yellow, "apple": preset("matte_gray")},
    ]
    for name in ("sphere", "box", "apple"):
        signatures = [_signature(draw[name]) for draw in draws]
        if len(set(signatures)) != 3:
            raise RuntimeError(f"reason=material_repeated object={name}")
        if _signature(fixed[name]) != signatures[0]:
            raise RuntimeError(f"reason=fixed_mismatch object={name}")
    for draw in draws:
        families = [draw[name]["family"] + json.dumps(draw[name]["parameters"], sort_keys=True) for name in ("sphere", "box", "apple")]
        if len(set(families)) != 3:
            raise RuntimeError("reason=objects_share_a_material")
    return fixed, draws


def _placed(instance_id: str, path: Path, spec: dict, translation: list[float]) -> dict:
    import mitsuba as mi

    require_variant()
    return _shape_obj(instance_id, path, to_mitsuba(spec), mi.ScalarTransform4f.translate(translation))


def _rest_translation(path: Path, xy: list[float], support_z: float) -> list[float]:
    """Put the mesh bottom on the support. Z is support height minus the mesh minimum Z."""
    vertices, _faces = read_obj(path)
    bottom = float(np.min(np.asarray(vertices)[:, 2]))
    return [float(xy[0]), float(xy[1]), float(support_z) - bottom]


def _room_shapes(loaded: dict) -> list[dict]:
    from sensor_sim.scene_gen.asset_package import verify_package
    from sensor_sim.scene_gen.indoor_transform import deserialize_matrix

    shapes = []
    for record in loaded["bundle"]["instances"]:
        package = Path(record["catalog"]) / record["asset_id"] / record["version"]
        mesh = package / "geometry" / "render.obj"
        material = verify_package(package)["materials"][0]
        shapes.append(_shape_obj(record["instance_id"], mesh, mitsuba_bsdf(material), deserialize_matrix(record["matrix"])))
    return shapes


def _approval_shapes(materials: dict, layout: str, paths: dict, room_kind: str) -> tuple[list[dict], dict]:
    """Subjects on a free part of a furnished support. A ground plane is not accepted."""
    from sensor_sim.scene_gen.asset_validate import triangulate
    from sensor_sim.scene_gen.placement_separation import blockers_for, place_clear, support_frame

    loaded = compose_scene_records(room_kind, ["sphere", "box", "apple"])
    frame = support_frame(loaded)
    blockers = blockers_for(loaded, frame)
    if layout == "row":
        offsets = {"sphere": (-0.16, 0.0), "box": (0.0, 0.0), "apple": (0.16, 0.0)}
    elif layout == "triangle":
        offsets = {"sphere": (-0.12, -0.08), "box": (0.12, -0.08), "apple": (0.0, 0.10)}
    else:
        offsets = {"sphere": (-0.08, -0.12), "box": (0.10, 0.0), "apple": (-0.02, 0.12)}
    subjects = []
    for name in ("sphere", "box", "apple"):
        vertices, polygons = read_obj(paths[name])
        subjects.append({"name": name, "vertices": vertices, "faces": triangulate(polygons)})
    placed = place_clear(subjects, blockers, frame["polygon"], offsets=offsets)
    shapes = _room_shapes(loaded)
    for subject, translation in zip(subjects, placed["translations"], strict=True):
        shapes.append(_placed(subject["name"], paths[subject["name"]], materials[subject["name"]], translation))
    loaded["placement"] = {
        "pattern_kept": placed["pattern_kept"],
        "minimum_clearance_m": placed["report"]["minimum_clearance_m"],
        "pairs_measured": placed["report"]["pairs_measured"],
    }
    logger.info(
        "material_review event=subject_placement room_kind=%s layout=%s pattern_kept=%s",
        room_kind,
        layout,
        placed["pattern_kept"],
    )
    return shapes, loaded


def _room_camera(loaded: dict, view: str) -> tuple[list[float], list[float], float]:
    camera = loaded["bundle"]["cameras"][view]
    return list(camera["origin"]), list(camera["target"]), float(camera["fov_deg"])


def pack_visual_approval(zip_path: Path = Path("local_review/material_wrap_review.zip")) -> Path:
    """RGB zip for visual approval. Two sections: randomization, then wrap."""
    import html
    import shutil
    import zipfile

    output = ROOT / "approval"
    if output.exists():
        shutil.rmtree(output)
    files = output / "files"
    files.mkdir(parents=True)
    paths = {
        "sphere": Path("assets/candidates/fixture_sphere/1.0.0/geometry/render.obj"),
        "box": ROOT / "targets" / "box.obj",
        "apple": Path("assets/candidates/prop_apple/1.0.0/geometry/render.obj"),
    }
    if not paths["box"].is_file():
        _write_box(paths["box"])
    fixed, draws = _approval_materials()
    resolution = (960, 720)
    import random

    room_kinds = sample_room_kinds(random.Random(RENDER_SEED), 4)
    layout_kinds = room_kinds[:3]
    draw_kind = room_kinds[3]
    captions = []
    for (layout, title), room_kind in zip((("row", "Scene A, objects in a row"), ("triangle", "Scene B, objects in a triangle"), ("column", "Scene C, objects in a column")), layout_kinds, strict=True):
        png = files / f"fixed_{layout}.png"
        shapes, loaded = _approval_shapes(fixed, layout, paths, room_kind)
        _render_room(shapes, png, loaded, "primary", 48, resolution)
        captions.append(("random_fixed", f"{title} in {loaded['room_kind']}", png.name, f"Same three subject materials. Sphere is red diffuse, box is smooth plastic, apple is aluminum. geometry={loaded['geometry_kind']} context_objects={loaded['report']['context_objects']}. pattern_kept={loaded['placement']['pattern_kept']}. Subjects stay at least 4 mm, triangle to triangle, from objects already in the room."))
    for index, draw in enumerate(draws, start=1):
        png = files / f"random_draw_{index}.png"
        shapes, loaded = _approval_shapes(draw, "row", paths, draw_kind)
        _render_room(shapes, png, loaded, "primary", 48, resolution)
        labels = ", ".join(f"{name}={draw[name]['family']}:{draw[name]['material_id']}" for name in ("sphere", "box", "apple"))
        captions.append(("random_draws", f"Same {draw_kind} row, material draw {index}", png.name, f"{labels}. geometry={loaded['geometry_kind']} context_objects={loaded['report']['context_objects']}. pattern_kept={loaded['placement']['pattern_kept']}. Minimum clearance is 4 mm."))
    tight = files / "random_draw_2_close.png"
    shapes, loaded = _approval_shapes(draws[1], "row", paths, draw_kind)
    _render_room(shapes, tight, loaded, "task", 64, resolution)
    captions.append(("random_close", f"Close view of material draw 2 in {draw_kind}", tight.name, "Blue diffuse sphere, copper box, green rough plastic apple, on the room support."))
    shell_dir = ROOT / "wraps" / "sphere_full" / "smooth"
    shell = _load_mesh(shell_dir / "shell_applied.obj")
    target = {"id": "sphere", "path": paths["sphere"], "mesh": _load_mesh(paths["sphere"])}
    from sensor_sim.scene_gen.plastic_wrap import classify_faces

    groups = classify_faces(shell["vertices"], _tri(shell["faces"]), np.asarray(target["mesh"]["vertices"]).mean(axis=0))
    film = to_mitsuba(preset("glass_145"))
    red = to_mitsuba(preset("matte_red"))
    import mitsuba as mi

    require_variant()
    from sensor_sim.scene_gen.asset_validate import triangulate
    from sensor_sim.scene_gen.placement_separation import blockers_for, place_clear, support_frame

    wrap_room = compose_scene_records("lab", ["sphere", "plastic_shell"])
    frame = support_frame(wrap_room)
    blockers = blockers_for(wrap_room, frame)
    sphere_vertices, sphere_polygons = read_obj(paths["sphere"])
    shell_vertices, shell_polygons = read_obj(shell_dir / "shell_applied.obj")
    bare_subject = {"name": "sphere_unwrapped", "vertices": sphere_vertices, "faces": triangulate(sphere_polygons)}
    shell_subject = {"name": "plastic_shell", "vertices": shell_vertices, "faces": triangulate(shell_polygons)}
    placed = place_clear(
        [bare_subject, shell_subject],
        blockers,
        frame["polygon"],
        offsets={"sphere_unwrapped": (-0.18, 0.0), "plastic_shell": (0.18, 0.0)},
    )
    bare_translation, shell_translation = placed["translations"]
    room_shapes = _room_shapes(wrap_room)
    bare = [*room_shapes, _shape_obj("sphere", paths["sphere"], red, mi.ScalarTransform4f.translate(bare_translation))]
    wrapped = [*room_shapes, _shape_obj("sphere", paths["sphere"], red, mi.ScalarTransform4f.translate(shell_translation))]
    for name, faces in groups["groups"].items():
        if len(faces) == 0:
            continue
        part = shell_dir / f"approval_{name}.obj"
        part.write_bytes(write_obj(shell["vertices"], [list(map(int, face)) for face in faces], name=name))
        wrapped.append(_shape_obj(name, part, film, mi.ScalarTransform4f.translate(shell_translation)))
    bare_png = files / "wrap_sphere_unwrapped_close.png"
    wrapped_png = files / "wrap_sphere_full_close.png"
    _render_room(bare, bare_png, wrap_room, "task", 64, resolution)
    _render_room(wrapped, wrapped_png, wrap_room, "task", 64, resolution)
    captions.append(("wrap", f"Unwrapped sphere in {wrap_room['room_kind']}", bare_png.name, f"Red diffuse sphere in the {wrap_room['geometry_kind']} shell. No film mesh. context_objects={wrap_room['report']['context_objects']}."))
    captions.append(("wrap", f"Full plastic shell in {wrap_room['room_kind']}", wrapped_png.name, "Same sphere plus a separate dielectric shell. The room and its other objects stay in the scene."))
    body = [
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Material wrap review</title>",
        "<style>body{font-family:sans-serif;margin:2rem;background:#111;color:#eee} img{max-width:960px;background:#222} figure{margin:1.2rem 0} h2{border-top:1px solid #444;padding-top:1rem}</style>",
        "</head><body><h1>Visual approval</h1>",
        "<p>Open this file after unzipping. Images use relative links. These scenes are not marked visually reviewed, admitted, or registered.</p>",
        "<h2>Material randomization</h2>",
        "<p>The first three images are three furnished rooms. The subject materials stay fixed: red diffuse sphere, smooth plastic box, aluminum apple. Placement of those three subjects changes. The next three images are one room and one row placement, rendered three times. Every subject gets a different material in each draw, and the three subjects in one image do not share a material. The room shell and the room's other objects stay in every image.</p>",
    ]
    for group, heading in (("random_fixed", "Three distinct scenes, same objects and materials"), ("random_draws", "Same scene, three material randomizations"), ("random_close", "Close view")):
        body.append(f"<h3>{heading}</h3>")
        for key, title, filename, note in captions:
            if key != group:
                continue
            body.append(f"<figure><img src=\"files/{html.escape(filename)}\" alt=\"{html.escape(title)}\"><figcaption><strong>{html.escape(title)}</strong><br>{html.escape(note)}</figcaption></figure>")
    body.append("<h2>Material wrap</h2>")
    body.append("<p>The close pair uses one task camera inside the lab scene. The lab label uses the workshop shell. The second image adds the plastic shell as its own mesh and its own dielectric material. The room and its other objects are in both images.</p>")
    for key, title, filename, note in captions:
        if key != "wrap":
            continue
        body.append(f"<figure><img src=\"files/{html.escape(filename)}\" alt=\"{html.escape(title)}\"><figcaption><strong>{html.escape(title)}</strong><br>{html.escape(note)}</figcaption></figure>")
    body.append("</body></html>\n")
    (output / "index.html").write_text("\n".join(body), encoding="utf-8")
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in output.rglob("*"):
            if file.is_file() and file.suffix.lower() in {".html", ".png"}:
                archive.write(file, file.relative_to(output))
    logger.info("material_review event=approval_zip path=%s", zip_path)
    return zip_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ROOT.mkdir(parents=True, exist_ok=True)
    audit = {"head": "e5ef2a3403e87d9977eef70ed98ae0db80075df5", "addon_commit": ADDON_COMMIT, "variant": "cuda_ad_spectral_polarized", "spp": SPP, "resolution": list(RESOLUTION)}
    (ROOT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (ROOT / "seed_invariants.json").write_text(json.dumps(seed_invariants(), indent=2) + "\n", encoding="utf-8")
    transfer = run_transfer(ROOT / "transfer")
    polarized = run_polarized(ROOT)
    supported = _supported_families(polarized)
    for row in transfer.get("cases", []):
        family = row["case_id"].replace("preset_", "")
        spec_family = None
        for item in polarized:
            if item["material_id"] == f"preset_{family}" or item["family"] == family:
                spec_family = item
                break
        if spec_family is not None:
            row["polarized_renderer_compatible"] = spec_family["polarized_renderer_compatible"]
    if transfer.get("cases"):
        (ROOT / "transfer" / "transfer_table.json").write_text(json.dumps(transfer["cases"], indent=2) + "\n", encoding="utf-8")
    materials = run_material_gallery(ROOT, supported)
    wraps = run_wraps(ROOT)
    # Unwrapped reference under the same camera.
    for name, path in (("box", ROOT / "targets" / "box.obj"), ("sphere", Path("assets/candidates/fixture_sphere/1.0.0/geometry/render.obj")), ("apple", Path("assets/candidates/prop_apple/1.0.0/geometry/render.obj"))):
        _render([_shape_obj("target", path, to_mitsuba(preset("matte_red")))], ROOT / "gallery_src" / f"{name}_unwrapped.png")
    write_gallery(ROOT)
    summary = {"transfer": transfer, "polarized": polarized, "materials": materials, "wraps": [{key: value for key, value in row.items() if key != "spec"} | {"wrap_id": row.get("spec", {}).get("wrap_id")} for row in wraps]}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("material_review event=done root=%s", ROOT)


if __name__ == "__main__":
    main()

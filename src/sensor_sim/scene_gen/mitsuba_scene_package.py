"""Rebuild a Mitsuba scene from a frozen manifest. The blend file is not opened."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.indoor_mitsuba import require_variant
from sensor_sim.scene_gen.material_spec import to_mitsuba


def scene_inventory(manifest: dict) -> dict:
    """Classify every instance. An exclusion without a reason is a failure."""
    rows = []
    for instance in manifest.get("instances", []):
        category = str(instance.get("category") or "")
        name = str(instance.get("id"))
        if category in {"calibration"} or "bbox_placeholder" in name or name.startswith("hoof"):
            kind = "helper"
        elif any(token in name for token in ("floor", "wall", "ceiling", "exterior", "Door", "Window", "skirting")):
            kind = "architecture"
        elif category.lower() in {"sofa", "chair", "table", "desk", "bed", "bookcase", "shelf", "cabinet", "tvstand", "rug"}:
            kind = "major_furniture"
        elif any(token in category for token in ("Lamp", "Light", "Fridge", "Oven", "Microwave")):
            kind = "appliance"
        elif any(token in category for token in ("NatureShelf", "LargePlant", "BookStack", "Trinket")):
            kind = "scatter"
        else:
            kind = "small_object"
        included = True
        reason = ""
        if instance.get("excluded"):
            included = False
            reason = str(instance.get("exclude_reason") or "")
            if not reason:
                raise RuntimeError(f"reason=exclusion_without_cause id={name}")
        rows.append({"id": name, "kind": kind, "included": included, "reason": reason})
    lights = []
    for light in manifest.get("lights", []):
        lights.append({
            "id": light.get("id"),
            "included": bool(light.get("included")),
            "reason": light.get("reason", ""),
            "conversion": light.get("conversion", ""),
            "polarization": light.get("polarization", "unpolarized"),
        })
    return {"instances": rows, "lights": lights, "instance_count": len(rows)}


def load_manifest(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    root = Path(path).parent
    for asset in payload.get("assets", []):
        mesh = root / asset["mesh"]
        if not mesh.is_file():
            raise FileNotFoundError(f"reason=missing_mesh path={asset['mesh']}")
    return payload


def build_scene(manifest: dict, root: Path) -> dict:
    """Construct a Mitsuba dict. Each material region gets its own BSDF."""
    require_variant()
    import mitsuba as mi

    scene = {
        "type": "scene",
        "integrator": {"type": "stokes", "integrator": {"type": "path", "max_depth": 5}},
        "sensor": _sensor(manifest["camera"], mi),
    }
    environment = manifest.get("environment")
    if environment and environment.get("type") == "constant":
        scene["environment"] = {"type": "constant", "radiance": {"type": "rgb", "value": float(environment["radiance"])}}
    for light in manifest["lights"]:
        if light.get("included") is False:
            continue
        light["polarization"] = light.get("polarization", "unpolarized")
        if light.get("type") == "point":
            scene[light["id"]] = {"type": "point", "position": light["position"], "intensity": {"type": "rgb", "value": light["intensity"]}}
            continue
        if light.get("type") == "sphere":
            scene[light["id"]] = {
                "type": "sphere",
                "center": light["position"],
                "radius": float(light["radius_m"]),
                "emitter": {"type": "area", "radiance": {"type": "rgb", "value": light["radiance"]}},
            }
            continue
        if light.get("type") == "area" and light.get("radiance") is not None:
            scene[light["id"]] = {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f(np.asarray(light["matrix"], dtype=np.float64)),
                "emitter": {"type": "area", "radiance": {"type": "rgb", "value": light["radiance"]}},
            }
            continue
        scene[light["id"]] = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f(np.asarray(light["matrix"], dtype=np.float64)),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": light["radiance"]}},
        }
    for instance in manifest["instances"]:
        asset = next(item for item in manifest["assets"] if item["id"] == instance["asset_id"])
        regions = instance.get("regions") or asset["regions"]
        if manifest.get("schema") == "infinigen_mitsuba_scene_v1":
            for region in regions:
                spec = region.get("spec") or {}
                if spec.get("schema_version") != "material_spec_v1":
                    raise RuntimeError(f"reason=missing_material_spec instance={instance['id']} slot={region.get('slot')}")
        if len(regions) == 1 and not asset.get("split_meshes"):
            mesh = regions[0].get("mesh") or asset["mesh"]
            scene[instance["id"]] = _shape(root / mesh, instance["matrix"], regions[0]["spec"], mi, root)
            continue
        for region in regions:
            scene[f"{instance['id']}__{region['slot']}"] = _shape(root / region["mesh"], instance["matrix"], region["spec"], mi, root)
    return scene


def _shape(mesh: Path, matrix, spec: dict, mi, root: Path) -> dict:
    return {
        "type": "obj" if mesh.suffix.lower() == ".obj" else "ply",
        "filename": str(mesh.resolve()),
        "to_world": mi.ScalarTransform4f(np.asarray(matrix, dtype=np.float64)),
        "bsdf": to_mitsuba(spec, package_root=root) if spec.get("schema_version") == "material_spec_v1" else spec,
    }


def _sensor(camera: dict, mi) -> dict:
    return {
        "type": "perspective",
        "fov": float(camera["fov_deg"]),
        "to_world": mi.ScalarTransform4f.look_at(origin=camera["origin"], target=camera["target"], up=camera.get("up", [0, 0, 1])),
        "sampler": {"type": "independent"},
        "film": {"type": "hdrfilm", "width": int(camera["width"]), "height": int(camera["height"]), "pixel_format": "rgb", "rfilter": {"type": "box"}},
    }


def replay(manifest_path: Path, png: Path, *, spp: int = 4, seed: int = 1, width: int | None = None, height: int | None = None) -> dict:
    from sensor_sim.scene_gen.indoor_mitsuba import render_stokes, write_rgb_png

    manifest = load_manifest(manifest_path)
    if width is not None and height is not None:
        manifest["camera"] = dict(manifest["camera"])
        manifest["camera"]["width"] = int(width)
        manifest["camera"]["height"] = int(height)
    scene = build_scene(manifest, manifest_path.parent)
    exr = png.with_suffix(".exr")
    render_stokes(scene, exr, spp, seed)
    write_rgb_png(exr, png)
    included = [item["id"] for item in manifest["lights"] if item.get("included", True)]
    return {
        "png": str(png),
        "exr": str(exr),
        "variant": "cuda_ad_spectral_polarized",
        "seed": seed,
        "spp": spp,
        "instances": [item["id"] for item in manifest["instances"]],
        "lights": included,
        "camera": manifest["camera"]["id"] if isinstance(manifest["camera"], dict) and "id" in manifest["camera"] else "camera",
        "film": [manifest["camera"]["width"], manifest["camera"]["height"]],
    }


def construction_report(manifest_path: Path) -> dict:
    """Check that the package is self-contained. This does not open a blend."""
    import hashlib

    manifest = load_manifest(manifest_path)
    root = Path(manifest_path).resolve().parent
    incomplete = []
    outside = []
    texture_hashes = 0
    for instance in manifest.get("instances", []):
        for key in ("id", "asset_id", "matrix", "parent", "regions"):
            if key not in instance:
                incomplete.append(f"{instance.get('id')}:{key}")
        matrix = np.asarray(instance.get("matrix", []), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            incomplete.append(f"{instance.get('id')}:matrix")
        for region in instance.get("regions") or []:
            mesh = (root / region["mesh"]).resolve()
            if root != mesh and root not in mesh.parents:
                outside.append(region["mesh"])
            for texture in (region.get("spec") or {}).get("textures") or []:
                path = Path(texture["path"])
                if path.is_absolute() or ".." in path.parts:
                    outside.append(texture["path"])
                    continue
                file_path = (root / path).resolve()
                if root != file_path and root not in file_path.parents:
                    outside.append(texture["path"])
                    continue
                digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
                texture_hashes += 1
                if digest != texture.get("sha256"):
                    raise RuntimeError(f"reason=texture_hash path={texture['path']}")
    return {
        "instances": len(manifest.get("instances", [])),
        "lights": len(manifest.get("lights", [])),
        "cameras": 1 if isinstance(manifest.get("camera"), dict) else len(manifest.get("cameras") or []),
        "texture_hashes": texture_hashes,
        "incomplete_instances": incomplete,
        "outside_package": outside,
        "passed": not incomplete and not outside,
        "variant": manifest.get("variant"),
    }


def replay_infinigen_scene(manifest_path: Path, png: Path, *, spp: int = 4, seed: int = 1, width: int | None = None, height: int | None = None) -> dict:
    """Rebuild from the frozen manifest. This does not call the generator or open a blend."""
    manifest = load_manifest(manifest_path)
    for key in ("camera", "lights", "instances", "assets", "variant", "units", "world_up"):
        if key not in manifest:
            raise FileNotFoundError(f"reason=manifest_field field={key}")
    if manifest["variant"] != "cuda_ad_spectral_polarized":
        raise RuntimeError(f"reason=variant value={manifest['variant']}")
    constructed = construction_report(manifest_path)
    if not constructed["passed"]:
        raise RuntimeError(f"reason=package_not_self_contained incomplete={constructed['incomplete_instances'][:3]} outside={constructed['outside_package'][:3]}")
    first = replay(manifest_path, png, spp=spp, seed=seed, width=width, height=height)
    png.unlink()
    png.with_suffix(".exr").unlink()
    second = replay(manifest_path, png, spp=spp, seed=seed, width=width, height=height)
    second["replay_matches"] = first["instances"] == second["instances"] and first["lights"] == second["lights"] and first["film"] == second["film"]
    second["construction"] = constructed
    return second


def identity_key(mesh_bytes: bytes) -> str:
    import hashlib

    return hashlib.sha256(mesh_bytes).hexdigest()

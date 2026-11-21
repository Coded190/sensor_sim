"""Reopen a USDC export and record which scene elements survived. Does not save."""

from __future__ import annotations

import json
import sys

import bpy


def main() -> None:
    usdc = sys.argv[sys.argv.index("--") + 1]
    output = sys.argv[sys.argv.index("--") + 2]
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.wm.usd_import(filepath=usdc)
    type_counts = {}
    cameras = []
    lights = []
    mesh_names = []
    materials = []
    for obj in bpy.data.objects:
        type_counts[obj.type] = type_counts.get(obj.type, 0) + 1
        if obj.type == "CAMERA":
            cameras.append(obj.name)
        elif obj.type == "LIGHT":
            lights.append({"name": obj.name, "light_type": obj.data.type})
        elif obj.type == "MESH":
            mesh_names.append(obj.name)
    for material in bpy.data.materials:
        materials.append(material.name)
    images = [image.name for image in bpy.data.images]
    payload = {
        "usdc": usdc,
        "object_types": type_counts,
        "camera_count": len(cameras),
        "cameras": cameras[:20],
        "light_count": len(lights),
        "lights": lights[:30],
        "mesh_count": len(mesh_names),
        "mesh_names": mesh_names[:40],
        "material_count": len(materials),
        "image_count": len(images),
        "hierarchy": len(bpy.data.collections),
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"event=inspect_usdc cameras={len(cameras)} lights={len(lights)} meshes={len(mesh_names)} materials={len(materials)} images={len(images)}")


if __name__ == "__main__":
    main()

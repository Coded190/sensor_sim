"""Assemble qualification scenes from project-owned indoor assets.

Normal assembly is not used here because no human has admitted a package.
Qualification mode loads technically validated candidates, writes that fact
into the manifest, and does not copy them into ``assets/library``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import read_index, set_candidate_status, verify_package
from sensor_sim.scene_gen.asset_validate import triangulate
from sensor_sim.scene_gen.indoor_contact import measure_support, support_graph_ok, unintended_intersection, world_vertices
from sensor_sim.scene_gen.indoor_mitsuba import (
    load_shapes,
    render_stokes,
    require_variant,
    write_rgb_png,
)
from sensor_sim.scene_gen.indoor_transform import identity_matrix, serialize_matrix, translation_matrix
from sensor_sim.scene_gen.mesh_io import read_obj
from sensor_sim.scene_gen.structural import (
    GENERATOR_VERSION,
    build_room,
    build_shelf,
    build_table,
    json_number,
    package_procedural_catalog,
)

logger = logging.getLogger("sensor_sim.scene_gen.indoor_cli")

SEED = 17011


def ensure_catalog(catalog: Path, seed: int = SEED) -> list[dict]:
    catalog = Path(catalog)
    index_path = catalog / "index.json"
    if index_path.is_file():
        index = read_index(index_path)
        if "room_floor" in index.get("assets", {}):
            return [
                {
                    "asset_id": asset_id,
                    "package_sha256": versions["1.0.0"]["package_sha256"],
                    "geometry_passed": versions["1.0.0"]["status"] == "technically_validated",
                }
                for asset_id, versions in sorted(index["assets"].items())
            ]
    return package_procedural_catalog(catalog, seed=seed)


def assemble_demonstration(catalog: Path, output: Path, *, render: bool = True, spp: int = 16) -> dict:
    """Write scenes A, B, and C. Rendering is optional so contact tests can run alone."""
    catalog = Path(catalog)
    output = Path(output)
    ensure_catalog(catalog)
    room = build_room()
    table = build_table(SEED)
    shelf = build_shelf(SEED)
    floor_z = float(room["interior"]["floor_top_z_m"])
    placements = _placements(room, table, shelf, floor_z)
    summaries = {}
    for scene_id, names in (
        ("scene_a", ["room_floor", "room_ceiling", "room_wall_south", "room_wall_north", "room_wall_west", "room_wall_east"]),
        ("scene_b", ["room_floor", "room_ceiling", "room_wall_south", "room_wall_north", "room_wall_west", "room_wall_east", "table_rect", "shelf_two"]),
        (
            "scene_c",
            [
                "room_floor",
                "room_ceiling",
                "room_wall_south",
                "room_wall_north",
                "room_wall_west",
                "room_wall_east",
                "table_rect",
                "shelf_two",
                "prop_crate",
                "prop_cup",
                "prop_book",
            ],
        ),
    ):
        selected = {name: placements[name] for name in names}
        summaries[scene_id] = _write_scene(catalog, output / scene_id, scene_id, selected, room, render=render, spp=spp)
    return summaries


def _placements(room, table, shelf, floor_z: float) -> dict:
    table_top = max(float(piece["maximum"][2]) for piece in table["pieces"])
    shelf_level = float(shelf["support_surfaces"][0]["polygon"][0][2])
    return {
        "room_floor": {"matrix": identity_matrix(), "reflectance": [0.55, 0.52, 0.46], "contact": None},
        "room_ceiling": {"matrix": identity_matrix(), "reflectance": [0.9, 0.9, 0.88], "contact": ("room_floor", "wall_tops")},
        "room_wall_south": {"matrix": identity_matrix(), "reflectance": [0.82, 0.82, 0.78], "contact": ("room_floor", "interior")},
        "room_wall_north": {"matrix": identity_matrix(), "reflectance": [0.82, 0.82, 0.78], "contact": ("room_floor", "interior")},
        "room_wall_west": {"matrix": identity_matrix(), "reflectance": [0.82, 0.82, 0.78], "contact": ("room_floor", "interior")},
        "room_wall_east": {"matrix": identity_matrix(), "reflectance": [0.82, 0.82, 0.78], "contact": ("room_floor", "interior")},
        "table_rect": {
            "matrix": translation_matrix([1.15, 1.25, floor_z]),
            "reflectance": [0.45, 0.28, 0.16],
            "contact": ("room_floor", "interior"),
            "local_parts": table["pieces"],
            "support_polygon": table["support_surfaces"][0]["polygon"],
            "top_z": floor_z + table_top,
        },
        "shelf_two": {
            "matrix": translation_matrix([2.55, 1.15, floor_z]),
            "reflectance": [0.36, 0.24, 0.14],
            "contact": ("room_floor", "interior"),
            "local_parts": shelf["pieces"],
            "support_polygon": shelf["support_surfaces"][0]["polygon"],
            "level_z": floor_z + shelf_level,
        },
        "prop_crate": {"matrix": translation_matrix([0.55, 2.35, floor_z]), "reflectance": [0.62, 0.24, 0.14], "contact": ("room_floor", "interior")},
        "prop_cup": {"matrix": translation_matrix([1.55, 1.55, floor_z + table_top]), "reflectance": [0.75, 0.82, 0.85], "contact": ("table_rect", "top"), "dielectric": True},
        "prop_book": {"matrix": translation_matrix([2.70, 1.22, floor_z + shelf_level]), "reflectance": [0.15, 0.35, 0.22], "contact": ("shelf_two", "level_0")},
    }


def _write_scene(catalog, directory, scene_id, placements, room, *, render: bool, spp: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    instances = []
    contacts = []
    edges = []
    meshes = {}
    for asset_id, placement in placements.items():
        package = catalog / asset_id / "1.0.0"
        manifest = verify_package(package)
        obj_path = package / "geometry" / "render.obj"
        vertices, polygons = read_obj(obj_path)
        matrix = np.asarray(placement["matrix"], dtype=np.float64)
        world = world_vertices(vertices, matrix)
        meshes[asset_id] = {
            "vertices": world,
            "polygons": polygons,
            "manifest": manifest,
            "matrix": matrix,
            "filename": obj_path,
            "reflectance": placement["reflectance"],
            "dielectric": bool(placement.get("dielectric", False)),
        }
        instances.append(
            {
                "instance_id": asset_id,
                "asset_id": asset_id,
                "version": "1.0.0",
                "package_sha256": manifest["package_sha256"],
                "matrix": serialize_matrix(matrix),
                "reflectance": placement["reflectance"],
                "dielectric": bool(placement.get("dielectric", False)),
                "license_status": manifest["provenance"]["license_status"],
                "registered": False,
            }
        )
    floor_polygon = _surface_polygon(meshes["room_floor"]["manifest"], "interior")
    for asset_id, placement in placements.items():
        contact = placement["contact"]
        if contact is None:
            continue
        supporter, surface = contact
        if surface == "interior":
            polygon = floor_polygon
        elif surface == "wall_tops":
            for wall_id in ("room_wall_south", "room_wall_north", "room_wall_west", "room_wall_east"):
                if wall_id not in meshes:
                    continue
                record = measure_support(meshes[asset_id]["vertices"], meshes[asset_id]["polygons"], _top_polygon(meshes[wall_id]["vertices"]))
                record.update({"supported_instance": asset_id, "supporter_instance": wall_id, "surface": "top"})
                contacts.append(record)
                edges.append((wall_id, asset_id))
            continue
        elif surface == "top":
            polygon = _transformed_polygon(placements["table_rect"]["support_polygon"], placements["table_rect"]["matrix"])
        elif surface == "level_0":
            polygon = _transformed_polygon(placements["shelf_two"]["support_polygon"], placements["shelf_two"]["matrix"])
        else:
            raise RuntimeError(surface)
        parts = placement.get("local_parts")
        if parts:
            for part in parts:
                if not str(part["id"]).startswith("leg_") and not str(part["id"]).startswith("side_"):
                    continue
                local = np.asarray(part["vertices"], dtype=np.float64)
                world = (placement["matrix"] @ np.hstack([local, np.ones((len(local), 1))]).T).T[:, :3]
                record = measure_support(world, part["polygons"], polygon)
                record.update({"supported_instance": f"{asset_id}:{part['id']}", "supporter_instance": supporter, "surface": surface})
                contacts.append(record)
                edges.append((supporter, asset_id))
        else:
            record = measure_support(meshes[asset_id]["vertices"], meshes[asset_id]["polygons"], polygon)
            record.update({"supported_instance": asset_id, "supporter_instance": supporter, "surface": surface})
            contacts.append(record)
            edges.append((supporter, asset_id))
    if any(not record["passed"] for record in contacts):
        failed = [record for record in contacts if not record["passed"]]
        raise RuntimeError(f"{scene_id} support failed: {failed[0]}")
    if not support_graph_ok(edges):
        raise RuntimeError(f"{scene_id} support graph has a cycle")
    collision = _collision_report(meshes)
    identity = _identity_report(meshes) if render or True else []
    manifest = {
        "scene_id": scene_id,
        "qualification": True,
        "qualification_note": "These packages are technically validated candidates. They are not visually reviewed, admitted, or registered.",
        "seed": SEED,
        "generator_version": GENERATOR_VERSION,
        "exporter": "indoor_mitsuba",
        "non_registered_packages": [
            {"asset_id": item["asset_id"], "version": item["version"], "package_sha256": item["package_sha256"]} for item in instances
        ],
        "instances": instances,
        "interior_m": room["interior"],
        "supports": _round_records(contacts),
        "support_acyclic": True,
        "collision": collision,
        "identity": identity,
    }
    _write_json(directory / "scene.json", manifest)
    rendered = {}
    if render:
        rendered = _render_scene(directory, meshes, scene_id, spp)
        for asset_id in placements:
            set_candidate_status(catalog, asset_id, "1.0.0", "renderer_validated")
    summary = {
        "scene_id": scene_id,
        "contacts_passed": all(record["passed"] for record in contacts),
        "contact_count": len(contacts),
        "support_acyclic": True,
        "identity_count": len(identity),
        "collision_unintended": collision["unintended_count"],
        "render": rendered,
        "scene_json": str(directory / "scene.json"),
    }
    _write_json(directory / "validation.json", summary)
    logger.info(
        "indoor_cli event=scene scene_id=%s contacts=%d identity=%d",
        scene_id,
        len(contacts),
        len(identity),
    )
    return summary


def _identity_report(meshes: dict) -> list[dict]:
    import mitsuba as mi

    require_variant()
    report = []
    for asset_id, mesh in meshes.items():
        filename = mesh["filename"]
        scene = {
            "type": "scene",
            asset_id: {
                "type": "obj",
                "filename": str(filename),
                "to_world": mi.ScalarTransform4f(mesh["matrix"]),
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]}},
            },
        }
        loaded = load_shapes(scene)
        report.append(
            {
                "instance_id": asset_id,
                "loaded_shapes": len(loaded),
                "loaded_id": loaded[0]["id"] if loaded else "",
                "vertex_count": loaded[0]["vertex_count"] if loaded else 0,
                "position_sha256": loaded[0]["position_sha256"] if loaded else "",
            }
        )
    return report


def _render_scene(directory: Path, meshes: dict, scene_id: str, spp: int) -> dict:
    import mitsuba as mi

    require_variant()
    cameras = _cameras(scene_id, meshes)
    outputs = {}
    for name, camera in cameras.items():
        scene = _scene_dict(meshes, camera, (256, 160) if name == "overview" else (192, 144))
        exr = directory / f"{name}.exr"
        stokes = render_stokes(scene, exr, spp=spp, seed=SEED)
        png = directory / f"{name}.png"
        write_rgb_png(exr, png)
        loaded_scene = mi.load_dict(scene)
        shape_count = len([shape for shape in loaded_scene.shapes() if str(shape.id()) not in {"light", "fill"}])
        outputs[name] = {
            "exr": str(exr),
            "png": str(png),
            "stokes_shape": stokes["shape"],
            "exr_names": stokes["exr_names"],
            "loaded_shape_count": shape_count,
            "s3_sign_independently_remeasured": False,
        }
        _write_json(directory / f"{name}_stokes.json", stokes)
    return outputs


def _scene_dict(meshes: dict, camera: dict, resolution: tuple[int, int]) -> dict:
    import mitsuba as mi

    require_variant()
    width, height = resolution
    scene = {
        "type": "scene",
        "integrator": {"type": "stokes", "integrator": {"type": "path", "max_depth": 5}},
        "sensor": {
            "type": "perspective",
            "fov": float(camera["fov"]),
            "to_world": mi.ScalarTransform4f.look_at(origin=camera["origin"], target=camera["target"], up=[0.0, 0.0, 1.0]),
            "sampler": {"type": "independent"},
            "film": {"type": "hdrfilm", "width": int(width), "height": int(height), "pixel_format": "rgb", "rfilter": {"type": "box"}},
        },
        "light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(origin=[2.1, 1.7, 2.35], target=[2.1, 1.7, 1.0], up=[0.0, 1.0, 0.0])
            @ mi.ScalarTransform4f.scale([0.45, 0.35, 1.0]),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 18.0}},
        },
        "fill": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(origin=[1.6, 0.8, 1.8], target=[1.2, 2.2, 1.2], up=[0.0, 0.0, 1.0])
            @ mi.ScalarTransform4f.scale([0.35, 0.25, 1.0]),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 8.0}},
        },
    }
    for asset_id, mesh in meshes.items():
        reflectance = mesh["reflectance"]
        bsdf = (
            {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}
            if mesh.get("dielectric")
            else {"type": "diffuse", "reflectance": {"type": "rgb", "value": reflectance}}
        )
        scene[asset_id] = {
            "type": "obj",
            "filename": str(mesh["filename"]),
            "to_world": mi.ScalarTransform4f(mesh["matrix"]),
            "bsdf": bsdf,
        }
    return scene


def _cameras(scene_id: str, meshes: dict) -> dict:
    overview = {"origin": [2.15, 0.55, 1.65], "target": [2.05, 2.4, 1.05], "fov": 60.0}
    door = {"origin": [0.35, 1.15, 1.35], "target": [0.72, 0.12, 1.15], "fov": 48.0}
    window = {"origin": [0.35, 2.15, 1.45], "target": [0.85, 3.32, 1.4], "fov": 42.0}
    cameras = {"overview": overview, "door": door, "window": window}
    if scene_id in {"scene_b", "scene_c"} and "table_rect" in meshes:
        table_point = meshes["table_rect"]["vertices"].min(axis=0)
        cameras["table_contact"] = {
            "origin": [float(table_point[0] - 0.15), float(table_point[1] - 0.55), float(table_point[2] + 0.22)],
            "target": [float(table_point[0] + 0.15), float(table_point[1] + 0.1), float(table_point[2] + 0.02)],
            "fov": 35.0,
        }
        shelf_point = meshes["shelf_two"]["vertices"].min(axis=0)
        cameras["shelf_contact"] = {
            "origin": [float(shelf_point[0] - 0.2), float(shelf_point[1] - 0.55), float(shelf_point[2] + 0.25)],
            "target": [float(shelf_point[0] + 0.1), float(shelf_point[1] + 0.05), float(shelf_point[2] + 0.02)],
            "fov": 35.0,
        }
    if scene_id == "scene_c":
        for name in ("prop_crate", "prop_cup", "prop_book"):
            point = meshes[name]["vertices"].min(axis=0)
            cameras[f"{name}_contact"] = {
                "origin": [float(point[0] - 0.05), float(point[1] - 0.35), float(point[2] + 0.18)],
                "target": [float(point[0] + 0.05), float(point[1] + 0.02), float(point[2] + 0.02)],
                "fov": 30.0,
            }
    return cameras


def _collision_report(meshes: dict) -> dict:
    unintended = []
    pairs = [pair for pair in (("prop_crate", "room_wall_west"), ("prop_cup", "prop_book"), ("table_rect", "shelf_two")) if pair[0] in meshes and pair[1] in meshes]
    for left, right in pairs:
        result = unintended_intersection(_triangle_mesh(meshes[left]), _triangle_mesh(meshes[right]))
        result["pair"] = [left, right]
        if result["unintended"]:
            unintended.append(result)
    return {"unintended_count": len(unintended), "checked_pairs": [list(pair) for pair in pairs], "unintended": unintended}


def _triangle_mesh(mesh: dict) -> dict:
    return {"vertices": mesh["vertices"], "faces": triangulate(mesh["polygons"])}


def _surface_polygon(manifest: dict, surface_id: str) -> np.ndarray:
    for surface in manifest["support_surfaces"]:
        if surface["id"] == surface_id:
            return np.asarray(surface["polygon"], dtype=np.float64)
    raise KeyError(surface_id)


def _top_polygon(vertices: np.ndarray) -> np.ndarray:
    z_value = float(np.max(vertices[:, 2]))
    on_top = vertices[np.abs(vertices[:, 2] - z_value) <= 1e-6]
    low = on_top.min(axis=0)
    high = on_top.max(axis=0)
    return np.array(
        [[low[0], low[1], z_value], [high[0], low[1], z_value], [high[0], high[1], z_value], [low[0], high[1], z_value]],
        dtype=np.float64,
    )


def _transformed_polygon(polygon, matrix: np.ndarray) -> np.ndarray:
    return world_vertices(np.asarray(polygon, dtype=np.float64), np.asarray(matrix, dtype=np.float64))


def _round_records(records: list[dict]) -> list[dict]:
    rounded = []
    for record in records:
        item = dict(record)
        for key in ("gap_m", "penetration_m"):
            item[key] = json_number(item[key])
        item["normal"] = [json_number(value) for value in item["normal"]]
        item["contact_point_m"] = [json_number(value) for value in item["contact_point_m"]]
        rounded.append(item)
    return rounded


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Assemble indoor qualification scenes")
    parser.add_argument("--catalog", type=Path, default=Path("assets/candidates"))
    parser.add_argument("--output", type=Path, default=Path("dataset/qualification/indoor_scenes"))
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summaries = assemble_demonstration(args.catalog, args.output, render=not args.no_render, spp=args.spp)
    print(json.dumps(summaries, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

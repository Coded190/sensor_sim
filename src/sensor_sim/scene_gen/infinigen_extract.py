"""Package bakeoff extracts as license-unresolved candidates.

This does not decide whether those meshes may be redistributed. Stable-path
assets record that ``infinigen_gpl`` is GPL-3.0 on the import path. Infinigen 2
source is BSD-3-Clause, and the extracted mesh redistribution decision is also
unresolved. None of these packages are registered.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import create_package, publish_candidate, set_candidate_status
from sensor_sim.scene_gen.asset_validate import validate_mesh
from sensor_sim.scene_gen.indoor_transform import deserialize_matrix, inspect_linear, serialize_matrix
from sensor_sim.scene_gen.mesh_io import read_obj
from sensor_sim.scene_gen.structural import CLEARANCE_TRIANGLE_LIMIT, GENERATOR_VERSION, json_number, json_vector

logger = logging.getLogger("sensor_sim.scene_gen.infinigen_extract")

BAKEOFF = Path("dataset/qualification/indoor_generator_bakeoff")
STABLE_COMMIT = "b11700ebac77695d8f53538705cfdc0cd890c54d"
INFINIGEN2_COMMIT = "53033dd55c7191cbc436be6ffbb63c026935a509"
STABLE_LICENSE = (
    "Infinigen indoors-stable source is BSD-3-Clause. infinigen_gpl at 10c1d76f is GPL-3.0 "
    "and is imported while indoors code loads. Extracted-mesh redistribution was not decided."
)
INFINIGEN2_LICENSE = (
    "Infinigen 2.0 preview source is BSD-3-Clause at 53033dd. Extracted-mesh redistribution "
    "was not decided, so this candidate stays out of the production library."
)

SOURCES = (
    {
        "asset_id": "infinigen_stable_floor",
        "commit": STABLE_COMMIT,
        "license": STABLE_LICENSE,
        "obj": BAKEOFF / "candidates/infinigen_indoors_stable/extracted_assets/living-room_0/0.floor/geometry/mesh.obj",
        "meta": BAKEOFF / "candidates/infinigen_indoors_stable/extracted_assets/living-room_0/0.floor/asset.json",
    },
    {
        "asset_id": "infinigen_stable_desk",
        "commit": STABLE_COMMIT,
        "license": STABLE_LICENSE,
        "obj": BAKEOFF / "candidates/infinigen_indoors_stable/extracted_assets/SimpleDeskFactory(361208).spawn_asset(3395843)/geometry/mesh.obj",
        "meta": BAKEOFF / "candidates/infinigen_indoors_stable/extracted_assets/SimpleDeskFactory(361208).spawn_asset(3395843)/asset.json",
    },
    {
        "asset_id": "infinigen_stable_shelf",
        "commit": STABLE_COMMIT,
        "license": STABLE_LICENSE,
        "obj": BAKEOFF / "candidates/infinigen_indoors_stable/extracted_assets/LargeShelfFactory(4380085).spawn_asset(215187)/geometry/mesh.obj",
        "meta": BAKEOFF / "candidates/infinigen_indoors_stable/extracted_assets/LargeShelfFactory(4380085).spawn_asset(215187)/asset.json",
    },
    {
        "asset_id": "infinigen2_table",
        "commit": INFINIGEN2_COMMIT,
        "license": INFINIGEN2_LICENSE,
        "obj": BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/table_dining/geometry/mesh.obj",
        "meta": BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/table_dining/asset.json",
    },
    {
        "asset_id": "infinigen2_bowl",
        "commit": INFINIGEN2_COMMIT,
        "license": INFINIGEN2_LICENSE,
        "obj": BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/bowl.001/geometry/mesh.obj",
        "meta": BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/bowl.001/asset.json",
    },
    {
        "asset_id": "infinigen2_room_shape",
        "commit": INFINIGEN2_COMMIT,
        "license": INFINIGEN2_LICENSE,
        "obj": BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/room_shape.001/geometry/mesh.obj",
        "meta": BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/room_shape.001/asset.json",
    },
    {
        "asset_id": "infinigen2_room_walls",
        "commit": INFINIGEN2_COMMIT,
        "license": INFINIGEN2_LICENSE,
        "obj": BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/room_walls/geometry/mesh.obj",
        "meta": BAKEOFF / "candidates/infinigen_2_preview/extracted_assets/room_walls/asset.json",
        "reject_empty": True,
    },
)


def package_extracts(catalog: Path, bakeoff: Path | None = None) -> list[dict]:
    """Copy each extract into the candidate catalog. Empty geometry is a rejection record."""
    catalog = Path(catalog)
    catalog.mkdir(parents=True, exist_ok=True)
    root = Path(bakeoff) if bakeoff is not None else Path(".")
    results = []
    for source in SOURCES:
        obj_path = root / source["obj"]
        meta_path = root / source["meta"]
        if not obj_path.is_file() or not meta_path.is_file():
            results.append({"asset_id": source["asset_id"], "status": "missing", "issues": ["missing_source"]})
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vertices, polygons = read_obj(obj_path)
        if source.get("reject_empty") or len(vertices) == 0 or len(polygons) == 0:
            reason = {
                "asset_id": source["asset_id"],
                "status": "rejected",
                "issues": ["empty"],
                "upstream_object": meta.get("object_name"),
                "upstream_commit": source["commit"],
                "license_status": "unresolved",
            }
            rejection = catalog / "_rejections" / f"{source['asset_id']}.json"
            rejection.parent.mkdir(parents=True, exist_ok=True)
            rejection.write_text(json.dumps(reason, sort_keys=True) + "\n", encoding="utf-8")
            logger.info("infinigen_extract event=reject asset_id=%s reason=empty", source["asset_id"])
            results.append(reason)
            continue
        report = _best_profile(vertices, polygons)
        if source["asset_id"] == "infinigen2_room_shape":
            report = dict(report)
            report["passed"] = False
            report["issues"] = [*report["issues"], "unapproved_room_shell"]
        matrix = np.asarray(meta.get("matrix_world"), dtype=np.float64).reshape(4, 4)
        transform = inspect_linear(matrix)
        payload = {"geometry/render.obj": obj_path.read_bytes()}
        roles = {"geometry/render.obj": "render"}
        collision = []
        if report["triangle_count"] <= CLEARANCE_TRIANGLE_LIMIT and report["passed"]:
            payload["geometry/collision.obj"] = payload["geometry/render.obj"]
            roles["geometry/collision.obj"] = "collision"
            collision = [
                {
                    "id": "proxy",
                    "path": "geometry/collision.obj",
                    "render_path": "geometry/render.obj",
                    "triangle_count": report["triangle_count"],
                    "qualified": True,
                }
            ]
        manifest = {
            "asset_id": source["asset_id"],
            "version": "1.0.0",
            "units": "m",
            "frame": {
                "up_axis": "Z",
                "handedness": "right",
                "transform_convention": "row_major_column_vector",
                "matrix": serialize_matrix(deserialize_matrix(transform["matrix"])),
                "mirrored": transform["mirrored"],
            },
            "provenance": {
                "source": "infinigen",
                "generator": "infinigen_extract",
                "generator_version": GENERATOR_VERSION,
                "license": source["license"],
                "license_status": "unresolved",
                "upstream_commit": source["commit"],
                "upstream_object": str(meta.get("object_name")),
                "parameters": {
                    "source_vertices": int(meta.get("vertices") or 0),
                    "source_polygons": int(meta.get("polygons") or 0),
                    "parsed_vertices": int(len(vertices)),
                    "parsed_polygons": int(len(polygons)),
                    "material_slots": [str(name) for name in meta.get("material_slots") or []],
                },
            },
            "geometry": {
                "profile": report["profile"],
                "units": "m",
                "polygon_count": report["polygon_count"],
                "triangle_count": report["triangle_count"],
                "bounds_min": json_vector(report["bounds_min"]),
                "bounds_max": json_vector(report["bounds_max"]),
                "clearance_triangle_limit": CLEARANCE_TRIANGLE_LIMIT,
                "clearance_from_render_mesh": False,
                "validation_issues": report["issues"],
            },
            "collision": collision,
            "materials": [{"id": f"slot_{index}", "name": str(name)} for index, name in enumerate(meta.get("material_slots") or [])],
        }
        destination = catalog / f".stage-{source['asset_id']}"
        if destination.exists():
            import shutil

            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        package_dir = destination / "package"
        sealed = create_package(package_dir, manifest, payload, roles)
        publish_candidate(catalog, package_dir)
        import shutil

        shutil.rmtree(destination, ignore_errors=True)
        if report["passed"]:
            set_candidate_status(catalog, source["asset_id"], "1.0.0", "technically_validated")
        results.append(
            {
                "asset_id": source["asset_id"],
                "status": "technically_validated" if report["passed"] else "candidate",
                "package_sha256": sealed["package_sha256"],
                "polygon_count": report["polygon_count"],
                "triangle_count": report["triangle_count"],
                "issues": report["issues"],
                "license_status": "unresolved",
            }
        )
        logger.info(
            "infinigen_extract event=package asset_id=%s status=%s triangles=%s package_sha256=%s",
            source["asset_id"],
            results[-1]["status"],
            report["triangle_count"],
            sealed["package_sha256"],
        )
    return results


def _best_profile(vertices: np.ndarray, polygons: list[list[int]]) -> dict:
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    axis = [0.0, 0.0, 0.0]
    axis[int(np.argmin(extent))] = 1.0
    structural = validate_mesh(vertices, polygons, "structural_solid", thickness_axis=axis)
    if structural["passed"]:
        return structural
    closed = validate_mesh(vertices, polygons, "closed_solid")
    if closed["passed"]:
        return closed
    opened = validate_mesh(vertices, polygons, "open_surface")
    if opened["passed"]:
        return opened
    structural["passed"] = False
    return structural

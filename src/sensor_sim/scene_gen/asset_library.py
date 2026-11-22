"""Catalog and retrieval for packaged Infinigen assets.

The catalog is not part of ``asset_package_v1``. Package hashes come from
``verify_package``. ``catalog_hash`` is the SHA-256 of the catalog file bytes
and is never stored inside that file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_extension import load_extension, validate_extension
from sensor_sim.scene_gen.asset_package import (
    PackageError,
    canonical_json_bytes,
    normalize_payload_path,
    parse_strict_json,
    sha256_regular,
    verify_package,
)
from sensor_sim.scene_gen.mesh_io import read_mesh

SCHEMA_VERSION = "asset_catalog_v1"
SERIALIZATION = "canonical_json_v1"

_FACTORY_ROLE = {
    "TableDiningFactory": {"category": "table", "tags": ["dining", "support", "table"], "support_role": True},
    "CanFactory": {"category": "can", "tags": ["can", "container"], "support_role": False},
}


class AssetLibraryError(ValueError):
    """A catalog entry or a retrieval filter failed closed."""


def catalog_file_hash(path: Path) -> str:
    """SHA-256 of the exact catalog file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_catalog(path: Path) -> dict:
    document = parse_strict_json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssetLibraryError("reason=catalog_not_object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise AssetLibraryError("reason=schema_version")
    if "catalog_hash" in document:
        raise AssetLibraryError("reason=catalog_hash_inside_file")
    if not isinstance(document.get("entries"), list) or not document["entries"]:
        raise AssetLibraryError("reason=entries")
    return document


def write_catalog(path: Path, document: dict) -> str:
    """Write canonical catalog JSON. The returned hash is of those file bytes."""
    if "catalog_hash" in document:
        raise AssetLibraryError("reason=catalog_hash_inside_file")
    payload = canonical_json_bytes(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def build_catalog(library_root: Path, *, reports_root: Path | None = None) -> dict:
    """Build a catalog from package directories and sidecars. Hashes are recomputed."""
    library_root = Path(library_root)
    entries = []
    for manifest_path in sorted(library_root.glob("*/*/asset.json")):
        package_root = manifest_path.parent
        asset_id = package_root.parent.name
        version = package_root.name
        if asset_id.startswith(".") or version.startswith("."):
            continue
        entries.append(_entry_from_package(library_root, asset_id, version, reports_root))
    if not entries:
        raise AssetLibraryError(f"reason=no_packages root={library_root}")
    entries.sort(key=lambda item: (item["asset_id"], item["package_version"], item["package_hash"]))
    return {"schema_version": SCHEMA_VERSION, "serialization": SERIALIZATION, "entries": entries}


def validate_catalog(document: dict, library_root: Path) -> list[dict]:
    """Verify every entry against the package bytes. Return the verified records."""
    library_root = Path(library_root)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise AssetLibraryError("reason=schema_version")
    verified = []
    seen = set()
    for entry in document["entries"]:
        record = _verify_entry(entry, library_root)
        key = (record["asset_id"], record["package_version"], record["package_hash"])
        if key in seen:
            raise AssetLibraryError(f"reason=duplicate_entry asset_id={record['asset_id']}")
        seen.add(key)
        verified.append(record)
    return verified


def retrieve(
    document: dict,
    library_root: Path,
    *,
    asset_id: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    support_role: bool | None = None,
    require_render: bool = True,
    require_collision: bool = True,
    min_size_m: list[float] | None = None,
    max_size_m: list[float] | None = None,
    collision_qualification: str | None = None,
    redistribution_status: str | None = None,
    selection_seed: int = 0,
) -> dict:
    """Return one verified record. Selection uses SHA-256, not Python ``hash``."""
    matches = []
    for record in validate_catalog(document, library_root):
        if asset_id is not None and record["asset_id"] != asset_id:
            continue
        if category is not None and record["category"] != category:
            continue
        if tags and not set(tags) <= set(record["tags"]):
            continue
        if support_role is not None and bool(record["support_role"]) != bool(support_role):
            continue
        if require_render and not record["render_geometry"]:
            continue
        if require_collision and not record["collision_geometry"]:
            continue
        size = record["bounds_m"]["size_m"]
        if min_size_m is not None and any(float(size[index]) < float(limit) for index, limit in enumerate(min_size_m)):
            continue
        if max_size_m is not None and any(float(size[index]) > float(limit) for index, limit in enumerate(max_size_m)):
            continue
        if collision_qualification is not None and record["collision_qualification"]["status"] != collision_qualification:
            continue
        if redistribution_status is not None and record["redistribution_status"] != redistribution_status:
            continue
        matches.append(record)
    if not matches:
        raise AssetLibraryError(
            f"reason=no_match asset_id={asset_id} category={category} tags={tags} support_role={support_role}"
        )
    matches.sort(key=lambda item: (item["asset_id"], item["package_version"], item["package_hash"]))
    index = _selection_index(int(selection_seed), matches)
    chosen = dict(matches[index])
    chosen["selection"] = {
        "seed": int(selection_seed),
        "candidate_count": len(matches),
        "candidate_ids": [item["asset_id"] for item in matches],
        "index": index,
    }
    return chosen


def composer_asset(record: dict, library_root: Path) -> dict:
    """Adapt a verified catalog record to the dict ``compose_scene`` already accepts."""
    library_root = Path(library_root)
    package_root = library_root / record["asset_id"] / record["package_version"]
    mesh = read_mesh(package_root / record["render_geometry"])
    polygon = record.get("support_polygon_m")
    if polygon is None:
        polygon = _footprint(mesh["vertices"])
    return {
        "category": record["category"],
        "package": record["package"],
        "extension": record["extension"],
        "render_vertices": mesh["vertices"],
        "render_faces": _triangles(mesh["corners"]),
        "support_polygon_local": np.asarray(polygon, dtype=np.float64),
    }


def _entry_from_package(library_root: Path, asset_id: str, version: str, reports_root: Path | None) -> dict:
    package_root = library_root / asset_id / version
    package = verify_package(package_root)
    sidecar_path = library_root / asset_id / f"{version}.asset_extension.json"
    if not sidecar_path.is_file():
        raise AssetLibraryError(f"reason=missing_sidecar path={sidecar_path}")
    extension = load_extension(sidecar_path.read_text(encoding="utf-8"))
    validate_extension(extension, package)
    provenance = package["provenance"]
    factory = provenance.get("generator")
    if factory not in _FACTORY_ROLE:
        raise AssetLibraryError(f"reason=unknown_factory factory={factory}")
    role = _FACTORY_ROLE[factory]
    geometries = {item["role"]: item for item in extension["geometries"]}
    render = geometries["render"]
    collision = geometries["collision"]
    if render["path"] == collision["path"]:
        raise AssetLibraryError("reason=render_collision_same_path")
    render_hash = sha256_regular(package_root, render["path"])
    collision_hash = sha256_regular(package_root, collision["path"])
    if render_hash != render["sha256"] or collision_hash != collision["sha256"]:
        raise AssetLibraryError("reason=geometry_hash_mismatch")
    vertices = read_mesh(package_root / render["path"])["vertices"]
    bounds = _bounds(vertices)
    materials = [
        item["path"]
        for item in package["files"]
        if item["role"] in {"material", "texture"}
    ]
    for relative in materials:
        normalize_payload_path(relative)
        if not (package_root / relative).is_file():
            raise AssetLibraryError(f"reason=missing_material path={relative}")
    blend_path, blend_hash = _blend(library_root, asset_id)
    qualification = _qualification(reports_root, asset_id, collision["method"])
    return {
        "asset_id": package["asset_id"],
        "package_version": package["version"],
        "package_hash": package["package_sha256"],
        "extension_path": f"{asset_id}/{version}.asset_extension.json",
        "source_generator": "Infinigen Indoors",
        "source_factory": factory,
        "source_commit": provenance.get("upstream_commit"),
        "source_seed": provenance.get("seed"),
        "source_config": provenance.get("parameters"),
        "source_blend_path": blend_path,
        "source_blend_hash": blend_hash,
        "render_geometry": render["path"],
        "render_geometry_hash": render_hash,
        "collision_geometry": collision["path"],
        "collision_geometry_hash": collision_hash,
        "bounds_m": bounds,
        "category": role["category"],
        "tags": list(role["tags"]),
        "support_role": role["support_role"],
        "collision_method": collision["method"],
        "collision_qualification": qualification,
        "material_sources": materials,
        "material_fidelity": "approximated",
        "material_fidelity_method": "infinigen_individual_asset_obj_bake",
        "redistribution_status": provenance.get("license_status"),
        "qualification_status": qualification["status"],
        "manipulation_qualified": False,
        "support_polygon_m": _support_polygon(package),
    }


def _verify_entry(entry: dict, library_root: Path) -> dict:
    required = {
        "asset_id",
        "package_version",
        "package_hash",
        "extension_path",
        "source_generator",
        "source_factory",
        "source_commit",
        "source_seed",
        "source_config",
        "source_blend_path",
        "source_blend_hash",
        "render_geometry",
        "render_geometry_hash",
        "collision_geometry",
        "collision_geometry_hash",
        "bounds_m",
        "category",
        "tags",
        "support_role",
        "collision_method",
        "collision_qualification",
        "material_sources",
        "material_fidelity",
        "redistribution_status",
        "qualification_status",
    }
    missing = sorted(required - set(entry))
    if missing:
        raise AssetLibraryError(f"reason=missing_field fields={missing}")
    if entry["redistribution_status"] not in {"unresolved", "allowed", "denied"}:
        raise AssetLibraryError("reason=redistribution_status")
    if not entry["source_factory"] or not entry["source_commit"] or not entry["source_blend_hash"]:
        raise AssetLibraryError("reason=provenance_incomplete")
    try:
        for relative in (
            entry["extension_path"],
            entry["render_geometry"],
            entry["collision_geometry"],
            entry["source_blend_path"],
            *entry["material_sources"],
        ):
            normalize_payload_path(str(relative))
    except PackageError as exc:
        raise AssetLibraryError(str(exc)) from exc
    if entry["render_geometry"] == entry["collision_geometry"]:
        raise AssetLibraryError("reason=render_collision_same_path")
    package_root = library_root / entry["asset_id"] / entry["package_version"]
    if not package_root.is_dir():
        raise AssetLibraryError(f"reason=missing_package path={package_root}")
    package = verify_package(package_root)
    if package["package_sha256"] != entry["package_hash"] or package["asset_id"] != entry["asset_id"] or package["version"] != entry["package_version"]:
        raise AssetLibraryError("reason=package_hash_mismatch")
    sidecar = library_root / entry["extension_path"]
    if not sidecar.is_file():
        raise AssetLibraryError(f"reason=missing_sidecar path={sidecar}")
    extension = load_extension(sidecar.read_text(encoding="utf-8"))
    validate_extension(extension, package)
    render_hash = sha256_regular(package_root, entry["render_geometry"])
    collision_hash = sha256_regular(package_root, entry["collision_geometry"])
    if render_hash != entry["render_geometry_hash"] or collision_hash != entry["collision_geometry_hash"]:
        raise AssetLibraryError("reason=geometry_hash_mismatch")
    for relative in entry["material_sources"]:
        if not (package_root / relative).is_file():
            raise AssetLibraryError(f"reason=missing_material path={relative}")
    blend = _resolve_blend(library_root, entry["source_blend_path"])
    if blend is not None:
        digest = hashlib.sha256(blend.read_bytes()).hexdigest()
        if digest != entry["source_blend_hash"]:
            raise AssetLibraryError("reason=blend_hash_mismatch")
    record = dict(entry)
    record["package"] = package
    record["extension"] = extension
    record["package_root"] = str(package_root)
    return record


def _selection_index(seed: int, matches: list[dict]) -> int:
    if len(matches) == 1:
        return 0
    lines = [f"{int(seed)}"]
    lines.extend(f"{item['asset_id']}|{item['package_version']}|{item['package_hash']}" for item in matches)
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % len(matches)


def _bounds(vertices: np.ndarray) -> dict:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    return {
        "min_m": [float(value) for value in low],
        "max_m": [float(value) for value in high],
        "size_m": [float(value) for value in (high - low)],
    }


def _qualification(reports_root: Path | None, asset_id: str, method: str) -> dict:
    report = {}
    if reports_root is not None:
        path = Path(reports_root) / f"{asset_id}_qualification.json"
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
    distance = report.get("surface_distance") or {}
    render_to = (distance.get("render_to_collision") or {}).get("max_m")
    collision_to = (distance.get("collision_to_render") or {}).get("max_m")
    status = "partially_qualified" if report.get("distance_within_tolerance") is True else "pending"
    note = "Measured against the packaged collision mesh. This is not a manipulation qualification."
    if method == "primitive_proxy" and collision_to is not None:
        note = (
            "The table collision mesh is a slab plus a leg proxy. "
            f"collision_to_render max is {float(collision_to):.6f} m. "
            "That leg error is not a high-fidelity collider."
        )
    return {
        "status": status,
        "method": method,
        "render_to_collision_max_m": None if render_to is None else float(render_to),
        "collision_to_render_max_m": None if collision_to is None else float(collision_to),
        "note": note,
    }


def _blend(library_root: Path, asset_id: str) -> tuple[str, str]:
    relative = f"../source_blends/{asset_id}.blend"
    # The blend is provenance next to the library, not a file inside the package.
    # The stored path is relative to the task_scene directory and is checked with
    # a lexical rejection of escape beyond that directory.
    path = (library_root / relative).resolve()
    task_root = library_root.parent.resolve()
    if task_root not in path.parents and path != task_root:
        raise AssetLibraryError(f"reason=blend_escapes path={path}")
    if not path.is_file():
        raise AssetLibraryError(f"reason=missing_blend path={path}")
    stored = str(Path("source_blends") / f"{asset_id}.blend")
    return stored, hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_blend(library_root: Path, stored: str) -> Path | None:
    normalize_payload_path(stored)
    path = library_root.parent / stored
    if path.is_file():
        return path
    return None


def _support_polygon(package: dict) -> list[list[float]] | None:
    surfaces = package.get("support_surfaces") or []
    if not surfaces:
        return None
    polygon = surfaces[0].get("polygon_m")
    return polygon


def _footprint(vertices: np.ndarray) -> np.ndarray:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    z = float(low[2])
    return np.array([[low[0], low[1], z], [high[0], low[1], z], [high[0], high[1], z], [low[0], high[1], z]])


def _triangles(corners: list) -> np.ndarray:
    faces = []
    for face in corners:
        for index in range(1, len(face) - 1):
            faces.append([int(face[0][0]), int(face[index][0]), int(face[index + 1][0])])
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)

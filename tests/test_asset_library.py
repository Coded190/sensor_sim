"""Catalog validation and deterministic retrieval for packaged assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sensor_sim.scene_gen.asset_extension import extension_bytes
from sensor_sim.scene_gen.asset_library import (
    AssetLibraryError,
    build_catalog,
    catalog_file_hash,
    load_catalog,
    retrieve,
    validate_catalog,
    write_catalog,
)
from sensor_sim.scene_gen.asset_package import create_package, sha256_regular, verify_package

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "dataset" / "task_scene" / "library"
REPORTS = ROOT / "dataset" / "task_scene" / "reports"


def _obj(scale: float) -> bytes:
    return (
        f"v 0 0 0\nv {scale} 0 0\nv 0 {scale} 0\nv 0 0 {scale}\n"
        "f 1 2 3\nf 1 3 4\n"
    ).encode("utf-8")


def _package(library: Path, asset_id: str, factory: str, seed: int, scale: float) -> None:
    render = _obj(scale)
    collision = _obj(scale * 0.9)
    material = b"newmtl default\nKd 0.5 0.5 0.5\n"
    payload = {
        "geometry/render.obj": render,
        "geometry/collision.obj": collision,
        "geometry/asset.mtl": material,
    }
    roles = {
        "geometry/render.obj": "render",
        "geometry/collision.obj": "collision",
        "geometry/asset.mtl": "material",
    }
    manifest = {
        "asset_id": asset_id,
        "version": "1.0.0",
        "units": "m",
        "frame": {"handedness": "right", "up": [0, 0, 1]},
        "geometry": {"frame": "asset_root", "render_mesh": "geometry/render.obj", "units": "m"},
        "collision": {"mesh": "geometry/collision.obj", "method": "convex_mesh" if factory == "CanFactory" else "primitive_proxy"},
        "support_surfaces": [
            {
                "frame": "support_surface" if factory == "TableDiningFactory" else "base_contact",
                "id": "surface",
                "polygon_m": [[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [scale, scale, 0.0], [0.0, scale, 0.0]],
            }
        ],
        "provenance": {
            "source": "infinigen_indoors",
            "generator": factory,
            "generator_version": "indoors-stable",
            "seed": seed,
            "parameters": {"export": "obj"},
            "license": "generated mesh redistribution unresolved",
            "license_status": "unresolved",
            "upstream_commit": "b11700ebac77695d8f53538705cfdc0cd890c54d",
            "upstream_object": factory,
        },
    }
    destination = library / asset_id / "1.0.0"
    create_package(destination, manifest, payload, roles)
    package = verify_package(destination)
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    frame = "support_surface" if factory == "TableDiningFactory" else "base_contact"
    extension = {
        "schema_version": "asset_extension_v1",
        "asset_id": asset_id,
        "package_version": "1.0.0",
        "package_sha256": package["package_sha256"],
        "frames": [
            {"name": "asset_root", "purpose": "origin", "origin_m": [0, 0, 0], "axes": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}, "transform_to_asset": identity, "units": "m", "provenance": "estimated"},
            {"name": frame, "purpose": frame, "origin_m": [0, 0, 0], "axes": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}, "transform_to_asset": identity, "units": "m", "provenance": "estimated"},
        ],
        "regions": [
            {"id": "region", "region_type": frame, "frame": frame, "geometry": "rectangle", "normal": [0, 0, 1], "units": "m", "provenance": "estimated", "method": "fixture"}
        ],
        "geometries": [
            {"role": "render", "path": "geometry/render.obj", "sha256": sha256_regular(destination, "geometry/render.obj"), "frame": "asset_root", "units": "m", "method": "fixture", "validation": "fixture"},
            {"role": "collision", "path": "geometry/collision.obj", "sha256": sha256_regular(destination, "geometry/collision.obj"), "frame": "asset_root", "units": "m", "method": manifest["collision"]["method"], "validation": "fixture"},
        ],
    }
    (library / asset_id / "1.0.0.asset_extension.json").write_bytes(extension_bytes(extension))
    blend_dir = library.parent / "source_blends"
    blend_dir.mkdir(parents=True, exist_ok=True)
    (blend_dir / f"{asset_id}.blend").write_bytes(f"blend-{asset_id}".encode("utf-8"))


def _library(tmp_path: Path) -> tuple[Path, dict]:
    library = tmp_path / "library"
    library.mkdir()
    _package(library, "table_dining", "TableDiningFactory", 7, 1.0)
    _package(library, "can_a", "CanFactory", 11, 0.1)
    _package(library, "can_b", "CanFactory", 12, 0.12)
    document = build_catalog(library, reports_root=None)
    return library, document


def test_retrieval_is_deterministic_and_fail_closed(tmp_path: Path):
    library, document = _library(tmp_path)
    table = retrieve(document, library, category="table", support_role=True)
    assert table["asset_id"] == "table_dining"
    assert table["redistribution_status"] == "unresolved"
    assert table["manipulation_qualified"] is False
    first = retrieve(document, library, category="can", selection_seed=1)
    second = retrieve(document, library, category="can", selection_seed=1)
    assert first["asset_id"] == second["asset_id"]
    assert first["selection"]["candidate_count"] == 2
    found = {retrieve(document, library, category="can", selection_seed=seed)["asset_id"] for seed in range(8)}
    assert found == {"can_a", "can_b"}
    exact = retrieve(document, library, asset_id="can_a")
    assert exact["asset_id"] == "can_a"
    with pytest.raises(AssetLibraryError, match="no_match"):
        retrieve(document, library, category="bottle")
    stale = json.loads(json.dumps(document))
    stale["entries"][0]["package_hash"] = "ab" * 32
    with pytest.raises(AssetLibraryError, match="package_hash_mismatch"):
        validate_catalog(stale, library)
    escaped = json.loads(json.dumps(document))
    escaped["entries"][0]["render_geometry"] = "../outside.obj"
    with pytest.raises(AssetLibraryError, match="path_traversal"):
        validate_catalog(escaped, library)


def test_catalog_hash_is_the_file_bytes(tmp_path: Path):
    library, document = _library(tmp_path)
    path = tmp_path / "catalog.json"
    digest = write_catalog(path, document)
    assert "catalog_hash" not in path.read_text(encoding="utf-8")
    assert digest == catalog_file_hash(path)
    loaded = load_catalog(path)
    assert loaded["schema_version"] == "asset_catalog_v1"
    assert {item["asset_id"] for item in loaded["entries"]} == {"table_dining", "can_a", "can_b"}


def test_live_packages_build_a_two_asset_catalog(tmp_path: Path):
    document = build_catalog(LIBRARY, reports_root=REPORTS)
    path = tmp_path / "catalog.json"
    write_catalog(path, document)
    loaded = load_catalog(path)
    table = retrieve(loaded, LIBRARY, category="table", support_role=True)
    can = retrieve(loaded, LIBRARY, category="can", support_role=False)
    assert table["asset_id"] == "table_dining"
    assert can["asset_id"] == "can"
    assert table["package_hash"] == verify_package(LIBRARY / "table_dining" / "1.0.0")["package_sha256"]
    assert can["collision_qualification"]["render_to_collision_max_m"] == pytest.approx(0.0015612326461850734, rel=0, abs=1e-12)
    assert table["collision_qualification"]["collision_to_render_max_m"] > 0.1
    assert table["redistribution_status"] == "unresolved"
    assert hashlib.sha256((LIBRARY.parent / table["source_blend_path"]).read_bytes()).hexdigest() == table["source_blend_hash"]

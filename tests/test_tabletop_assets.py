"""Package checks for the new tabletop assets. Catalog writes stay in a temporary directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import read_index, verify_package
from sensor_sim.scene_gen.asset_validate import validate_mesh
from sensor_sim.scene_gen.structural import box_mesh
from sensor_sim.scene_gen.tabletop_assets import ASSET_IDS, RESERVED_TOKENS, build_asset, material_for, package_catalog
from sensor_sim.scene_gen.task_retrieve import retrieve

CATALOG = Path("assets/candidates")
SOURCE_STATUS = {
    "prop_book": "renderer_validated",
    "prop_cup": "renderer_validated",
    "prop_crate": "renderer_validated",
    "prop_crate_open": "renderer_validated",
    "table_rect": "renderer_validated",
    "room_floor": "renderer_validated",
    "fixture_sphere": "technically_validated",
    "infinigen2_room_shape": "candidate",
}


def test_meshes_match_the_requested_sizes():
    expected = {
        "prop_bowl": (0.11, 0.11, 0.045),
        "prop_plate": (0.14, 0.14, 0.012),
        "prop_bottle": (0.056, 0.056, 0.16),
        "prop_can": (0.068, 0.068, 0.115),
        # The spoon head begins at x=0.004 and the handle ends at x=0.140, so the extent is 0.136 m.
        "prop_spoon": (0.1358, 0.03, 0.0089),
        "prop_marker": (0.13, 0.016, 0.016),
        "prop_phone": (0.145, 0.07, 0.009),
        "prop_tape_roll": (0.076, 0.076, 0.0234),
        "prop_small_box": (0.07, 0.045, 0.04),
    }
    for asset_id, size in expected.items():
        built = build_asset(asset_id)
        measured = built["vertices"].max(axis=0) - built["vertices"].min(axis=0)
        assert np.allclose(measured, size, atol=0.002), (asset_id, measured)
        assert built["report"]["passed"]
        assert built["report"]["triangle_count"] <= 2500
        assert float(built["vertices"][:, 2].min()) == 0.0


def test_material_variants_share_geometry_and_use_pplastic(tmp_path: Path):
    package_catalog(tmp_path)
    bowl = verify_package(tmp_path / "prop_bowl" / "1.0.0")
    glossy = verify_package(tmp_path / "prop_bowl_glossy" / "1.0.0")
    can = verify_package(tmp_path / "prop_can" / "1.0.0")
    rough = verify_package(tmp_path / "prop_can_rough" / "1.0.0")
    assert bowl["geometry"]["geometry_sha256"] == glossy["geometry"]["geometry_sha256"]
    assert can["geometry"]["geometry_sha256"] == rough["geometry"]["geometry_sha256"]
    assert bowl["package_sha256"] != glossy["package_sha256"]
    assert can["package_sha256"] != rough["package_sha256"]
    assert (tmp_path / "prop_bowl" / "1.0.0" / "geometry" / "render.obj").read_bytes() == (
        tmp_path / "prop_bowl_glossy" / "1.0.0" / "geometry" / "render.obj"
    ).read_bytes()
    assert glossy["materials"][0]["type"] == "pplastic"
    assert bowl["materials"][0]["type"] == "diffuse"
    assert bowl["materials"][0]["research_role"] == "depolarizing_reference"
    assert can["materials"][0] == material_for("prop_can")
    assert rough["materials"][0]["type"] == "roughconductor"
    assert rough["materials"][0]["alpha"] == 0.1
    assert "plastic" != glossy["materials"][0]["type"]
    index = read_index(tmp_path / "index.json")
    for asset_id in ASSET_IDS:
        assert index["assets"][asset_id]["1.0.0"]["status"] == "technically_validated"
        root = tmp_path / asset_id / "1.0.0"
        for path in root.rglob("*"):
            assert not path.is_symlink()


def test_package_hash_is_stable(tmp_path: Path):
    first = package_catalog(tmp_path / "a")
    second = package_catalog(tmp_path / "b")
    assert [item["package_sha256"] for item in first] == [item["package_sha256"] for item in second]


def test_existing_validator_still_rejects_a_degenerate_triangle():
    vertices, polygons = box_mesh([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    report = validate_mesh(vertices, polygons, "closed_solid")
    assert not report["passed"]


def test_new_catalog_text_does_not_change_retrieval_winners():
    for asset_id in ASSET_IDS:
        built = build_asset(asset_id)
        text = " ".join([asset_id.replace("_", " "), built["category"], *built["aliases"]])
        assert not (set(text.split()) & RESERVED_TOKENS)
    book = retrieve(CATALOG, "book", top_k=10, require_renderer=True)
    mug = retrieve(CATALOG, "mug", top_k=10, require_renderer=True)
    assert book["results"][0]["asset_id"] == "prop_book"
    assert book["returned_count"] < 10
    assert mug["results"][0]["asset_id"] == "prop_cup"


def test_existing_candidate_statuses_stay_put():
    index = json.loads((CATALOG / "index.json").read_text(encoding="utf-8"))
    for asset_id, status in SOURCE_STATUS.items():
        assert index["assets"][asset_id]["1.0.0"]["status"] == status

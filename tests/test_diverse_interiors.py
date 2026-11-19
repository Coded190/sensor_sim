"""Diverse furnished interiors: sizes, appliance parts, cameras, replay, and RGB."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from sensor_sim.scene_gen.interior_assets import build_asset
from sensor_sim.scene_gen.interior_scenes import (
    RESOLUTION,
    ROOMS,
    ROOT,
    _aims_through_opening,
    _inside_room,
    replay_interior,
)
from tests.test_semantic_scenes import DATASET_V1, LOCKED

INDEX_SHA256 = "081bf07291a4cc3f8f2af1fe27b8f8beb0164153100244aefa25df1ba209c157"


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def _bundles() -> list[Path]:
    return [next((ROOT / name).glob("*/bundle.json")).parent for name in ROOMS]


def test_six_room_sizes_are_distinct():
    sizes = [(row["width"], row["depth"], row["height"]) for row in ROOMS.values()]
    assert len(sizes) == 6
    assert len(set(sizes)) == 6
    saved = []
    for path in _bundles():
        spec = json.loads((path / "room_spec.json").read_text(encoding="utf-8"))
        saved.append((spec["interior_width_m"], spec["interior_depth_m"], spec["interior_height_m"]))
    assert len(set(saved)) == 6


def test_refrigerator_and_oven_have_separate_doors():
    oven = {component["id"] for component in build_asset("furn_oven")["components"]}
    fridge = {component["id"] for component in build_asset("furn_fridge")["components"]}
    assert "door" in oven and "panel" in oven
    assert {"door_left", "door_right", "handle_left", "handle_right"} <= fridge
    catalog = ROOT / "catalog"
    oven_package = json.loads((catalog / "furn_oven" / "1.0.0" / "asset.json").read_text(encoding="utf-8"))
    fridge_package = json.loads((catalog / "furn_fridge" / "1.0.0" / "asset.json").read_text(encoding="utf-8"))
    assert "door" in {item["id"] for item in oven_package["geometry"]["components"]}
    assert {"door_left", "door_right"} <= {item["id"] for item in fridge_package["geometry"]["components"]}


def test_primary_cameras_are_inside_and_not_aimed_through_an_opening():
    for path in _bundles():
        bundle = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
        spec = json.loads((path / "room_spec.json").read_text(encoding="utf-8"))
        camera = bundle["cameras"]["primary"]
        assert _inside_room(camera["origin"], spec)
        assert not _aims_through_opening(camera["origin"], camera["target"], spec)
        assert camera["origin"] != bundle["cameras"]["secondary"]["origin"]


def test_replay_does_not_remesh(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("replay remeshed or called another replay")

    monkeypatch.setattr("sensor_sim.scene_gen.interior_assets.build_asset", fail)
    monkeypatch.setattr("sensor_sim.scene_gen.asset_package.create_package", fail)
    monkeypatch.setattr("sensor_sim.scene_gen.task_replay.replay", fail)
    monkeypatch.setattr("sensor_sim.scene_gen.room_generator.replay_room", fail)
    for path in _bundles():
        before = (path / "scene.json").read_bytes()
        report = replay_interior(path)
        assert report["valid"] is True
        assert report["remeshed"] is False
        assert (path / "scene.json").read_bytes() == before


def test_rgb_pngs_are_1280_by_720_and_gallery_is_local():
    review = (ROOT / "review" / "index.html").read_text(encoding="utf-8")
    assert "http://" not in review and "https://" not in review
    for path in _bundles():
        for view in ("primary", "secondary", "task"):
            png = path / "render_rgb_v1" / f"{view}.png"
            assert _png_size(png) == RESOLUTION
            assert view in review


def test_previous_qualification_hashes_are_unchanged():
    for relative, digest in LOCKED.items():
        if relative.endswith(".py"):
            assert hashlib.sha256(Path("src/sensor_sim/scene_gen", relative).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(Path("assets/candidates/index.json").read_bytes()).hexdigest() == INDEX_SHA256
    dataset = Path("dataset/dataset_v1")
    files = [path for path in dataset.rglob("*") if path.is_file()]
    assert len(files) == DATASET_V1["files"]
    assert sum(path.stat().st_size for path in files) == DATASET_V1["bytes"]
    for name in ("run_manifest.json", "audit.json", "production_config.yaml"):
        assert hashlib.sha256((dataset / name).read_bytes()).hexdigest() == DATASET_V1[name]

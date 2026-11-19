"""Room diversity: convention, openings, furniture, replay, and locked inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sensor_sim.scene_gen.indoor_mitsuba import _bitmap_names
from sensor_sim.scene_gen.indoor_transform import deserialize_matrix
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES
from sensor_sim.scene_gen.room_generator import (
    ROOT,
    SOURCE_BUNDLE,
    assemble,
    compare_reference,
    place_arrangement,
    probe_openings,
    replay_room,
    table_local_poses,
    write_gallery,
)
from sensor_sim.scene_gen.room_schema import RoomSpecError, diversity_specifications, make_specification, reference_specification

CATALOG = Path("dataset/qualification/room_diversity/catalog")
INDEX_SHA256 = "081bf07291a4cc3f8f2af1fe27b8f8beb0164153100244aefa25df1ba209c157"
PROTECTED = {
    "src/sensor_sim/film/sfp_scene.py": "3287fc531e50069d3fb912292f3a3bc907fa1e80a39d1c1a30398a8940ea3ea0",
    "src/sensor_sim/film/mitsuba_scene.py": "6bdd40ac8ed12a243b0693da78c87219f3c5d51d66466b8772cd42c89d9b0653",
    "src/sensor_sim/scenes.py": "752e7c5c7aa067eaaa77559b307066a8c57d72a481bd8853c1a0194d80aa0022",
    "src/sensor_sim/scene_gen/xml_patch.py": "6b8ced6842897aa462eb1d855d4f7fe8cf8702e9f796af5bebe497d4b3942e32",
    "src/sensor_sim/scene_gen/generate.py": "78ee05ab9afe551436b0f35836880e8c6b1680e8c559f642083269ff3931ddf8",
    "src/sensor_sim/scene_gen/cli.py": "29a5ddc431bc4fa13887afed2fab774f485331d7aa5e97d0732b7d42a806763f",
    "src/sensor_sim/scene_gen/meshcheck.py": "051676b437f4ae4394d201b8470cf179ddeebbbb3d25f3b96479f3fbeb684198",
    "src/sensor_sim/scene_gen/placement.py": "650e9b36e9fb4422877b4cb392b9d7253c258cc1385329be502661a9d6ca466d",
    "src/sensor_sim/scene_gen/indoor_mitsuba.py": "dfbd5e9019be0d512f787ce2ec1ebffd78cf60cf52fff427d00a8a1e6c26d2b3",
    "src/sensor_sim/scene_gen/structural.py": "6e839ad9debb3bcbc4b16c1016f1eac77b6af58e8f1dc93b49c5302e007db1a7",
}
SCENE_JSON = {
    "dataset/qualification/task_scenes/book_on_table/e54a5e18636df8e7/scene.json": "fd783a414450af9130225eb6a7d6dc345544cbe0798f8d4b6bfc59b5829879c0",
    "dataset/qualification/task_scenes/cup_on_shelf/8a0a6b98b048ef83/scene.json": "464bb6215a98dc0f9421646588e882332f8d3a7008c161633a11573c551fc6fb",
    "dataset/qualification/task_scenes/book_inside_crate/e547b5c803391dbe/scene.json": "46cf481bc6113876a7d6f599d6e2d6ba9f3e83164f2e1b6885f60a7f74c6437e",
    "dataset/qualification/task_scenes/crate_and_book_on_table/44ac50f76b66afcb/scene.json": "8cd3f9a43d86e9a12c6d8d186957aaf28ad45cc7f4984dbc9e417063dff64591",
    "dataset/qualification/task_scenes/cup_near_book/95d8b3f496b78836/scene.json": "51a2bdaca8d2bd5d86dc384b6ea1a313275d548b9d9b4804d72f07294fa5dca9",
    "dataset/qualification/task_scenes/objects_in_workspace/3cf07be40ab68252/scene.json": "07137d96cd7a86a5e96a788cf5ac974b668642fdaf3f279408fec9fb78a9fe48",
    "dataset/qualification/indoor_scenes/scene_a/scene.json": "3aabf6862bf8c24de6d4e96975aa11ff3b93c07e4c9acc3d5115311a0228005a",
    "dataset/qualification/indoor_scenes/scene_b/scene.json": "248b6f7db28d3eb93ed192d788ec46db1f49d33214ba0ea024724a2673b5358b",
    "dataset/qualification/indoor_scenes/scene_c/scene.json": "d4d5f17d1c0d4641cb96060e55fa1c5abda861493eb6060a064ee082d3fb57a7",
}


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _specs():
    return diversity_specifications((3.2, 3.0, 2.4))


def test_convention_rejects_an_opening_that_leaves_the_wall():
    spec = reference_specification()
    assert spec["convention"]["finished_floor_z"] == 0.0
    assert spec["faces"]["west"]["interior_m"] == 0.0
    assert spec["exterior_bounds_m"]["minimum_m"][2] == -spec["floor_thickness_m"]
    with pytest.raises(RoomSpecError, match="opening_pier"):
        make_specification(
            room_id="room_bad",
            width=4.0,
            depth=3.2,
            height=2.5,
            seed=1,
            openings=[{"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.0, "width_m": 0.9, "height_m": 2.1, "sill_m": 0.0}],
        )


def test_reference_matches_build_room_component_bounds():
    report = compare_reference()
    assert report["matches"] is True
    assert report["mismatches"] == []
    assert len(report["pairs"]) == 11
    assert np.allclose(report["shift_m"], [0.12, 0.12, 0.05])


def test_profiles_have_distinct_bounds_and_real_openings():
    specs = _specs()
    bounds = []
    for spec in specs.values():
        built = assemble(spec)
        floor = next(piece for piece in built["pieces"] if piece["id"] == "floor")
        assert floor["minimum"][2] == pytest.approx(-spec["floor_thickness_m"])
        assert floor["maximum"][2] == pytest.approx(0.0)
        ceiling = next(piece for piece in built["pieces"] if piece["id"] == "ceiling")
        assert ceiling["minimum"][2] == pytest.approx(spec["interior_height_m"])
        walls = [piece for piece in built["pieces"] if piece["id"].endswith("_left") or piece["id"] in {"east", "west"}]
        assert min(piece["minimum"][2] for piece in walls) == pytest.approx(0.0)
        assert not any("corner" in piece["id"] for piece in built["pieces"])
        bounds.append(tuple(spec["exterior_bounds_m"]["maximum_m"]))
        probe_openings(spec)
    assert len(set(bounds)) == 3


def test_frames_leave_the_aperture_empty_on_every_wall():
    spec = make_specification(
        room_id="room_cross",
        width=5.0,
        depth=4.6,
        height=2.7,
        seed=2,
        openings=[
            {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.4, "width_m": 0.9, "height_m": 2.1, "sill_m": 0.0},
            {"id": "window_north", "kind": "window", "wall": "north", "offset_m": 0.5, "width_m": 1.2, "height_m": 1.0, "sill_m": 0.9},
            {"id": "window_east", "kind": "window", "wall": "east", "offset_m": 0.6, "width_m": 0.8, "height_m": 1.0, "sill_m": 1.0},
            {"id": "window_west", "kind": "window", "wall": "west", "offset_m": 0.7, "width_m": 1.0, "height_m": 1.0, "sill_m": 0.9},
        ],
    )
    rays = probe_openings(spec)
    assert {item["wall"] for item in rays} == {"south", "north", "east", "west"}
    assert all(item["through_clear"] and item["control_hit"] for item in rays)


def test_furniture_faces_the_table_and_keeps_the_passage():
    spec = _specs()["room_compact"]
    for arrangement in ("central", "perimeter"):
        placed = place_arrangement(spec, arrangement)
        assert placed["valid"] is True
        assert placed["failed"] == []
        chair = placed["bodies"]["chair_1"]["matrix"]
        seating = chair[:3, :3] @ np.array([0.0, -1.0, 0.0])
        assert np.dot(seating, placed["seating_direction"]) > 0.99


def test_tabletop_local_matrices_match_the_source_run():
    source = table_local_poses(SOURCE_BUNDLE)
    bundle = json.loads(next(ROOT.glob("room_compact/*/bundle.json")).read_text(encoding="utf-8"))
    assert bundle["source_tabletop"]["run_id"] == "ed940a5f855d1588"
    assert bundle["source_tabletop"]["bundle_sha256"] == source["bundle_sha256"]
    table = deserialize_matrix(next(item["matrix"] for item in bundle["instances"] if item["instance_id"] == "table_1"))
    inverse = np.linalg.inv(table)
    for instance_id, local in source["local"].items():
        world = deserialize_matrix(next(item["matrix"] for item in bundle["instances"] if item["instance_id"] == instance_id))
        assert np.allclose(inverse @ world, local, atol=1e-8)


def test_a_room_that_cannot_hold_the_furniture_reports_the_failure():
    spec = make_specification(
        room_id="room_tiny",
        width=2.2,
        depth=2.0,
        height=2.4,
        seed=3,
        openings=[
            {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.2, "width_m": 0.9, "height_m": 2.1, "sill_m": 0.0},
            {"id": "window_north", "kind": "window", "wall": "north", "offset_m": 0.3, "width_m": 0.8, "height_m": 0.8, "sill_m": 0.9},
        ],
    )
    try:
        placed = place_arrangement(spec, "central")
    except RoomSpecError as exc:
        assert "reason=" in str(exc)
        return
    assert placed["valid"] is False
    assert placed["failed"]


def test_replay_reproduces_scene_json_without_meshing(monkeypatch):
    def fail_assemble(*_args, **_kwargs):
        raise AssertionError("assemble was called")

    monkeypatch.setattr("sensor_sim.scene_gen.room_generator.assemble", fail_assemble)
    bundle_dir = next(ROOT.glob("room_standard/*/bundle.json")).parent
    before = (bundle_dir / "scene.json").read_bytes()
    replay_room(bundle_dir, CATALOG)
    assert (bundle_dir / "scene.json").read_bytes() == before


def test_saved_exrs_match_the_channel_contract():
    import mitsuba as mi

    from sensor_sim.scene_gen.indoor_mitsuba import require_variant

    require_variant()
    paths = sorted(ROOT.glob("*/*/render_*/*.exr"))
    assert len(paths) >= 12
    for path in paths:
        bitmap = mi.Bitmap(str(path))
        image = np.array(bitmap)
        assert image.dtype == np.float32
        assert image.shape[2] == 15
        assert np.isfinite(image).all()
        assert _bitmap_names(bitmap) == list(CHANNEL_NAMES)


def test_gallery_links_resolve(tmp_path: Path):
    review = tmp_path / "review" / "index.html"
    write_gallery(review, ROOT)
    text = review.read_text(encoding="utf-8")
    assert "Human visual review, admission, and registration are pending." in text
    for token in text.split('href="')[1:]:
        href = token.split('"', 1)[0]
        assert not href.startswith("http")
        assert (review.parent / href).resolve().exists(), href


def test_previous_scenes_and_the_candidate_index_are_unchanged():
    for path, digest in {**PROTECTED, **SCENE_JSON}.items():
        assert _sha(path) == digest
    assert _sha("assets/candidates/index.json") == INDEX_SHA256
    old = json.loads(Path("dataset/qualification/scene_quality_v2/old_bundle_hashes.json").read_text(encoding="utf-8"))
    for path, digest in old.items():
        assert _sha(path) == digest
    quality = sorted(Path("dataset/qualification/scene_quality_v2").glob("*/*/bundle.json"))
    assert len(quality) == 3
    assert {path.parent.name for path in quality} == {"ed940a5f855d1588", "b05e275699b4654c", "2aa23b16d561407a"}

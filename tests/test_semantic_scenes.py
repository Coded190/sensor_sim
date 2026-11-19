"""Semantic scene qualification: packages, zones, clutter, replay, and locked inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import verify_package
from sensor_sim.scene_gen.indoor_contact import unintended_intersection
from sensor_sim.scene_gen.indoor_mitsuba import _bitmap_names
from sensor_sim.scene_gen.indoor_transform import apply_points, deserialize_matrix
from sensor_sim.scene_gen.mesh_io import write_obj
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES
from sensor_sim.scene_gen.semantic_assets import BUILDERS, build_asset, interior_is_open
from sensor_sim.scene_gen.semantic_quality import assess
from sensor_sim.scene_gen.semantic_composition import (
    ROOT,
    VERSION,
    package_version,
    place_recipe,
    replay_semantic,
    write_failure,
    write_gallery,
    _place_object,
    _place_workbench,
)
from sensor_sim.scene_gen.semantic_schema import require_recipe
from sensor_sim.scene_gen.tabletop_layout import load_mesh
from tests.test_room_diversity import INDEX_SHA256, PROTECTED, SCENE_JSON

CATALOG = ROOT / "catalog"
ASSET_CATALOG = Path("assets/candidates")
ROOM_CATALOG = Path("dataset/qualification/room_diversity/catalog")
ALLOWED_BSDF = {"diffuse", "pplastic", "conductor", "roughconductor"}
HOLLOW = {"prop_pot", "prop_serving_dish", "prop_fastener_tray", "prop_parts_bin"}
LOCKED = {
    "room_bundles": "fdff86896dfce89eb822b7a46a101f73f25fd561950d20684155a91d5baaa4ec",
    "room_specs": "96d818140a4264c1c9bd3d9680bfb834f0c98e2bdcf56da9b10de27abd313936",
    "room_packages": "e07a4cc9b9873f5cf11b1f28abdeae2ac077e1c532e86ae8e7e32d7be343b181",
    "quality_bundles": "851173653a3e1a77a13fefa2ee3a25768d48abffdd9b7dc5db6bcfbf42ae6486",
    "tabletop_bundles": "5cf2d69950ac58908b8832481a3d9a95c1b8d49c8c14f15e0b312d7c7d2b231c",
    "candidate_assets": "10634d43db2480bea85f42d6aad1f9892ecd90512dfec26a5c4b03ef7fcfa1e6",
    "tabletop_assets.py": "d47c24dff6b096a5295b3a31520fee1431b085966da3ae9cbee06a858706a990",
    "room_generator.py": "75abb14ed8569ac17fc8d9232102e9d04c7c83b67518d01cb9f2094f8ea4b118",
    "room_schema.py": "626d19794ca73cebbd22f14e15754998a9e871294056dfd4e20651685efb4a88",
    "task_replay.py": "d9e6e1ac4103cb59c67d077977815632239ec94637365609fdf5c2f1fec1bcbf",
}
DATASET_V1 = {"files": 1533, "bytes": 457265977, "run_manifest.json": "9475a86ae204d9b2840b23e558f8043272ad8c6f6e689f00541688879b08c196", "audit.json": "2bc69f6f45849d30c7eee283da1093649c3560f3050d936105e479f779d0a95e", "production_config.yaml": "9fac8b70d7e663427593db4fe871b1a3270621baeb07fa92da87a4efd51872d8"}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _aggregate(paths) -> str:
    rows = {str(path): _sha(path) for path in paths}
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _bundles() -> list[tuple[Path, dict]]:
    found = []
    for path in sorted(ROOT.glob("*/*/bundle.json")):
        found.append((path.parent, json.loads(path.read_text(encoding="utf-8"))))
    return found


def _support(bundle: dict) -> dict:
    return next(item for item in bundle["furniture_instances"] if item.get("support_polygon") is not None)


def _mesh(asset_id: str, version: str):
    semantic = CATALOG / asset_id / version
    root = CATALOG if semantic.is_dir() else ASSET_CATALOG
    loaded = load_mesh(root, asset_id, version)
    return loaded["vertices"], loaded["faces"]


def _local_bounds(instance: dict, support_matrix: np.ndarray):
    vertices, _faces = _mesh(instance["asset_id"], instance["version"])
    local = np.linalg.inv(support_matrix) @ deserialize_matrix(instance["matrix"])
    transformed = apply_points(local, vertices)
    return transformed.min(axis=0), transformed.max(axis=0)


def test_packages_are_closed_recognized_and_use_four_bsdfs():
    catalog = json.loads((CATALOG / "semantic_catalog.json").read_text(encoding="utf-8"))
    by_id = {item["asset_id"]: item for item in catalog["packages"]}
    assert set(by_id) == set(BUILDERS)
    for asset_id in BUILDERS:
        built = build_asset(asset_id)
        version = package_version(asset_id)
        package = verify_package(CATALOG / asset_id / version)
        assert package["materials"][0]["type"] in ALLOWED_BSDF
        assert built["triangle_count"] <= 2500
        assert built["report"]["passed"]
        obj = write_obj(built["vertices"], built["polygons"], name=asset_id)
        assert obj == write_obj(built["vertices"], built["polygons"], name=asset_id)
        assert hashlib.sha256(obj).hexdigest() == hashlib.sha256((CATALOG / asset_id / version / "geometry" / "render.obj").read_bytes()).hexdigest()
        assert by_id[asset_id]["package_sha256"] == package["package_sha256"]
        assert by_id[asset_id]["status"] == "renderer_validated"
        assert by_id[asset_id]["visible_pixels"] > 0
        evidence = by_id[asset_id]["evidence"]
        assert Path(evidence["exr"]).is_file()
        assert evidence["package_sha256"] == package["package_sha256"]
    for asset_id in HOLLOW:
        assert interior_is_open(build_asset(asset_id))
    for asset_id in ("furniture_workbench", "furniture_counter"):
        old = (CATALOG / asset_id / "1.0.0" / "geometry" / "render.obj").read_bytes()
        new = (CATALOG / asset_id / "1.0.1" / "geometry" / "render.obj").read_bytes()
        assert old == new
        verify_package(CATALOG / asset_id / "1.0.0")
        assert verify_package(CATALOG / asset_id / "1.0.1")["provenance"]["parameters"]["revision_reason"]
    assert (CATALOG / "furniture_workbench" / "1.0.2" / "geometry" / "render.obj").read_bytes() == (
        CATALOG / "furniture_workbench" / "1.0.1" / "geometry" / "render.obj"
    ).read_bytes()


def test_workbench_shelf_does_not_interpenetrate_the_legs():
    built = build_asset("furniture_workbench")
    offset = 0
    spans = []
    for piece in built["pieces"]:
        count = len(piece["vertices"])
        spans.append((piece["id"], offset, offset + count))
        offset += count
    faces = built["faces"]
    shelf_span = next(span for span in spans if span[0] == "lower_shelf")
    shelf_body = {"vertices": built["vertices"], "faces": faces[np.all((faces >= shelf_span[1]) & (faces < shelf_span[2]), axis=1)]}
    for leg_id, start, stop in spans:
        if not leg_id.startswith("leg_"):
            continue
        leg_body = {"vertices": built["vertices"], "faces": faces[np.all((faces >= start) & (faces < stop), axis=1)]}
        found = unintended_intersection(leg_body, shelf_body)
        assert not found["unintended"]
        assert found["clearance_m"] > 0.0
    size = np.asarray(built["measured_m"])
    assert np.allclose(size, [1.8, 0.8, 0.9], atol=1e-6)
    assert built["support"]["rectangle_m"] == [0.0, 0.0, 1.8, 0.8]


def test_candidate_index_does_not_list_semantic_assets():
    index = json.loads(Path("assets/candidates/index.json").read_text(encoding="utf-8"))
    assert _sha("assets/candidates/index.json") == INDEX_SHA256
    assert set(BUILDERS).isdisjoint(index["assets"])


def test_recipes_keep_one_zone_unique_ids_and_the_clutter_order():
    bundles = {bundle["task_id"]: bundle for _path, bundle in _bundles()}
    assert set(bundles) == {"kitchen", "dining", "desk", "workbench_low", "workbench_moderate", "workbench_high"}
    fractions = []
    bands = {"workbench_low": (0.10, 0.15), "workbench_moderate": (0.15, 0.22), "workbench_high": (0.22, 0.30)}
    for task_id, _expected in (("workbench_low", 9), ("workbench_moderate", 14), ("workbench_high", 19)):
        bundle = bundles[task_id]
        clutter = bundle["clutter"]
        assert clutter["instance_count"] == _expected
        assert bundle["renderer"]["variant"] == "cuda_ad_spectral_polarized"
        assert bundle["renderer"]["seed"] == 17421
        fractions.append(clutter["union_fraction"])
        support = _support(bundle)
        zones = {zone["id"]: zone for zone in support["zones"]}
        assert set(zones) == {"assembly", "tools", "staging", "small_parts", "free_workspace"}
        assert "free_workspace" not in {item["zone_id"] for item in bundle["object_instances"]}
        ids = [item["instance_id"] for item in [*bundle["room_components"], *bundle["furniture_instances"], *bundle["object_instances"]]]
        assert len(ids) == len(set(ids))
        require_recipe(bundle["recipe"])
        matrix = deserialize_matrix(support["matrix"])
        for item in bundle["object_instances"]:
            assert item["zone_id"] in zones
            assert item["role"] in zones[item["zone_id"]]["roles"]
            low, high = _local_bounds(item, matrix)
            rect = zones[item["zone_id"]]["rectangle_m"]
            assert low[0] >= rect[0] - 1e-4 and high[0] <= rect[2] + 1e-4
            assert low[1] >= rect[1] - 1e-4 and high[1] <= rect[3] + 1e-4
            relation = next(entry for entry in bundle["relations"] if entry["instances"][0] == item["instance_id"])
            assert relation["surface"] == "top"
            assert relation["passed"] is True
            assert relation["measured_m"] <= 0.001
        assert clutter["nearest_clearance_m"] >= 0.004
    assert fractions[0] < fractions[1] < fractions[2]
    for task_id, band in bands.items():
        assert band[0] <= bundles[task_id]["clutter"]["union_fraction"] <= band[1]
    high = bundles["workbench_high"]["clutter"]
    assert min(high["usable_free_rectangle"]["size_m"]) >= 0.25
    assert high["usable_free_rectangle"]["origin_m"][0] > 1.2
    assert "prop_screwdriver" in high["repeated_assets"]
    assert "prop_pliers" in high["repeated_assets"]
    low_camera = bundles["workbench_low"]["cameras"]
    assert low_camera == bundles["workbench_moderate"]["cameras"] == bundles["workbench_high"]["cameras"]
    assert bundles["workbench_low"]["lighting"] == bundles["workbench_moderate"]["lighting"] == bundles["workbench_high"]["lighting"]
    high_dir = next(path for path, bundle in _bundles() if bundle["task_id"] == "workbench_high")
    rows = json.loads((high_dir / "render_ceiling_v1" / "instances.json").read_text(encoding="utf-8"))
    partial = [row for row in rows if row["instance_id"].startswith("prop_") and row["visible_pixels"] > 0 and row["visible_to_projected_ratio"] < 1]
    assert partial
    for row in rows:
        if row["shorter_pixel_side"] < 32:
            assert row["image_support_flag"] is False
        if row["instance_id"].startswith("prop_screwdriver"):
            assert row["role"] == "tool"


def test_dining_has_two_chairs_and_the_other_scenes_use_their_zones():
    bundles = {bundle["task_id"]: bundle for _path, bundle in _bundles()}
    dining = bundles["dining"]
    chairs = [item for item in dining["furniture_instances"] if item["asset_id"] == "prop_chair"]
    assert len(chairs) >= 2
    table = _support(dining)
    table_center = apply_points(deserialize_matrix(table["matrix"]), np.asarray(table["support_polygon"], dtype=np.float64)).mean(axis=0)
    for chair in chairs:
        matrix = deserialize_matrix(chair["matrix"])
        facing = -(matrix[:3, :3] @ np.array([0.0, 1.0, 0.0]))
        toward = table_center - matrix[:3, 3]
        assert float(np.dot(facing[:2], toward[:2])) > 0.0
    zones = {item["zone_id"] for item in dining["object_instances"]}
    assert {"place_a", "place_b", "serving", "drinks"} <= zones
    for zone_id in ("place_a", "place_b"):
        assets = {item["asset_id"] for item in dining["object_instances"] if item["zone_id"] == zone_id}
        assert "prop_plate" in assets
    for task_id, required in (("kitchen", {"preparation", "ingredients", "storage"}), ("desk", {"main_work", "input_devices", "writing"})):
        used = {item["zone_id"] for item in bundles[task_id]["object_instances"]}
        assert required <= used


def test_a_bad_slot_is_a_diagnostic_failure(tmp_path: Path):
    built = build_asset("furniture_workbench")
    from sensor_sim.scene_gen.semantic_composition import load_room

    body = _place_workbench(built, load_room("room_spacious")["spec"])
    assets = {"prop_screwdriver": build_asset("prop_screwdriver"), "furniture_workbench": built}
    placed = _place_object("prop_screwdriver__1", "prop_screwdriver", "assembly", (9.0, 9.0), 0.0, body, assets, ASSET_CATALOG)
    assert placed["inside_zone"] is False
    failure = {"recipe": {"scene_type": "workbench", "seed": 17411}, "failed": [{"name": "support:prop_screwdriver__1", "passed": False}]}
    path = tmp_path / "workbench_low" / "failure.json"
    write_failure(path, failure)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["failed"][0]["name"] == "support:prop_screwdriver__1"


def test_the_same_seed_repeats_and_a_different_seed_changes_poses():
    assets = {asset_id: build_asset(asset_id) for asset_id in BUILDERS}
    first = place_recipe("workbench_low", assets)
    second = place_recipe("workbench_low", assets)
    other = place_recipe("workbench_low", assets, seed=17499)
    assert first["valid"] and second["valid"] and other["valid"]
    first_pose = [item["matrix"].round(8).tolist() for item in first["objects"]]
    second_pose = [item["matrix"].round(8).tolist() for item in second["objects"]]
    other_pose = [item["matrix"].round(8).tolist() for item in other["objects"]]
    assert first_pose == second_pose
    assert first_pose != other_pose


def test_replay_reads_the_bundle_and_does_not_rebuild(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("replay rebuilt geometry")

    monkeypatch.setattr("sensor_sim.scene_gen.semantic_composition.build_asset", fail)
    monkeypatch.setattr("sensor_sim.scene_gen.semantic_composition.create_package", fail)
    monkeypatch.setattr("sensor_sim.scene_gen.task_replay.replay", fail)
    monkeypatch.setattr("sensor_sim.scene_gen.room_generator.replay_room", fail)
    for bundle_dir, _bundle in _bundles():
        before = (bundle_dir / "scene.json").read_bytes()
        replay_semantic(bundle_dir, CATALOG, ROOM_CATALOG)
        assert (bundle_dir / "scene.json").read_bytes() == before


def test_renders_match_the_exr_contract_and_the_gallery_links(tmp_path: Path):
    import mitsuba as mi

    from sensor_sim.scene_gen.indoor_mitsuba import require_variant

    require_variant()
    exrs = sorted(ROOT.glob("*/*/render_ceiling_v1/*.exr"))
    assert len(exrs) == 24
    for path in exrs:
        bitmap = mi.Bitmap(str(path))
        image = np.array(bitmap)
        assert image.dtype == np.float32
        assert image.shape[2] == 15
        assert np.isfinite(image).all()
        assert _bitmap_names(bitmap) == list(CHANNEL_NAMES)
        if path.name == "tabletop.exr":
            mask = np.load(path.parent / "instance_ids.npy")
            assert list(mask.shape) == list(image.shape[:2])
    review = ROOT / "review" / "index.html"
    text = review.read_text(encoding="utf-8")
    assert "Human visual review, admission, and registration are pending." in text
    assert "share room_spacious" in text
    for token in text.split('href="')[1:]:
        href = token.split('"', 1)[0]
        assert (review.parent / href).resolve().exists(), href
    copied = tmp_path / "review" / "index.html"
    write_gallery(copied, ROOT)
    assert "pending" in copied.read_text(encoding="utf-8")


def test_previous_datasets_and_protected_modules_are_unchanged():
    for path, digest in {**PROTECTED, **SCENE_JSON}.items():
        assert _sha(path) == digest
    assert _aggregate(sorted(Path("dataset/qualification/room_diversity").glob("*/*/bundle.json"))) == LOCKED["room_bundles"]
    assert _aggregate(sorted(Path("dataset/qualification/room_diversity").glob("*/*/room_spec.json"))) == LOCKED["room_specs"]
    assert _aggregate(sorted(Path("dataset/qualification/room_diversity/catalog").glob("*/*/asset.json"))) == LOCKED["room_packages"]
    assert _aggregate(sorted(Path("dataset/qualification/scene_quality_v2").glob("*/*/bundle.json"))) == LOCKED["quality_bundles"]
    assert _aggregate(sorted(Path("dataset/qualification/tabletop_scenes").glob("*/*/bundle.json"))) == LOCKED["tabletop_bundles"]
    assert _aggregate(sorted(Path("assets/candidates").glob("*/*/asset.json"))) == LOCKED["candidate_assets"]
    assert len(list(Path("dataset/qualification/tabletop_scenes").glob("*/*/bundle.json"))) == 15
    assert {path.parent.name for path in Path("dataset/qualification/scene_quality_v2").glob("*/*/bundle.json")} == {"ed940a5f855d1588", "b05e275699b4654c", "2aa23b16d561407a"}
    assert {path.parent.name for path in Path("dataset/qualification/room_diversity").glob("*/*/bundle.json")} == {
        "7f774e080dde15d6",
        "459e09531c7e8e52",
        "0ea3fd7596033729",
        "764f63997834f6c5",
        "dd985a60befaf87d",
        "4a6929a751f3eee8",
    }
    for name, digest in (
        ("src/sensor_sim/scene_gen/tabletop_assets.py", LOCKED["tabletop_assets.py"]),
        ("src/sensor_sim/scene_gen/room_generator.py", LOCKED["room_generator.py"]),
        ("src/sensor_sim/scene_gen/room_schema.py", LOCKED["room_schema.py"]),
        ("src/sensor_sim/scene_gen/task_replay.py", LOCKED["task_replay.py"]),
    ):
        assert _sha(name) == digest
    root = Path("dataset/dataset_v1")
    files = [path for path in root.rglob("*") if path.is_file()]
    assert len(files) == DATASET_V1["files"]
    assert sum(path.stat().st_size for path in files) == DATASET_V1["bytes"]
    for name in ("run_manifest.json", "audit.json", "production_config.yaml"):
        assert _sha(root / name) == DATASET_V1[name]
    assert VERSION == "1.0.0"


def test_quality_check_separates_hard_failures_from_warnings():
    bundles = {bundle["task_id"]: (path, bundle) for path, bundle in _bundles()}
    for task_id, (path, bundle) in bundles.items():
        report = assess(bundle, CATALOG, Path("dataset/qualification/room_diversity/catalog"), ASSET_CATALOG, path / "render_ceiling_v1")
        assert report["passed"], report["hard_failures"]
    broken = json.loads(json.dumps(bundles["workbench_high"][1]))
    broken["object_instances"][0]["matrix"] = [float("nan")] * 16
    report = assess(broken, CATALOG, Path("dataset/qualification/room_diversity/catalog"), ASSET_CATALOG)
    assert not report["passed"]
    assert any(item["name"].startswith("transform:") for item in report["hard_failures"])

"""Package hash, path security, approval records, and library recovery."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import pytest

from sensor_sim.scene_gen.asset_package import (
    PackageError,
    canonical_json_bytes,
    create_package,
    file_hashes,
    format_catalog,
    manifest_body_for_hash,
    normalize_payload_path,
    package_digest,
    package_directory,
    parse_strict_json,
    publish_candidate,
    read_index,
    reconcile_library,
    register_package,
    set_candidate_status,
    validate_payload_paths,
    verify_package,
)
from sensor_sim.scene_gen.asset_validate import validate_package

MANIFEST = {
    "asset_id": "box_prop",
    "version": "1.0.0",
    "provenance": {
        "source": "sensor_sim",
        "generator": "test",
        "generator_version": "1.0.0",
        "license": "project-owned",
        "license_status": "permitted",
    },
}
PAYLOAD = b"hello\n"
PATH = "payload/a.bin"


def _package(root: Path, name: str = "package", payload: bytes = PAYLOAD, manifest: dict | None = None) -> Path:
    destination = root / name
    create_package(destination, manifest or MANIFEST, {PATH: payload}, {PATH: "payload"})
    return destination


def _approval(package: Path, **overrides) -> dict:
    manifest = verify_package(package)
    approval = {
        "approval_schema_version": 1,
        "asset_id": manifest["asset_id"],
        "version": manifest["version"],
        "package_sha256": manifest["package_sha256"],
        "decision": "admitted",
        "reviewer_id": "fixture-reviewer",
        "approved_utc": "2026-11-18T21:00:00Z",
        "visual_evidence": "fixture://not-a-production-review",
        "license_decision": "permitted",
    }
    approval.update(overrides)
    return approval


def _register_job(library: str, package: str, approval: dict, queue) -> None:
    try:
        register_package(Path(library), Path(package), approval)
        queue.put({"ok": True, "error": ""})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def test_package_is_canonical_and_stable(tmp_path: Path):
    first = _package(tmp_path, "one")
    second = _package(tmp_path, "two")
    manifest = validate_package(first)
    assert manifest["package_sha256"] == verify_package(second)["package_sha256"]
    assert (first / "asset.json").read_bytes() == (second / "asset.json").read_bytes()
    stored = (first / "asset.json").read_text(encoding="utf-8")
    assert stored == canonical_json_bytes(parse_strict_json(stored)).decode("utf-8")
    assert "status" not in manifest
    body = json.loads(stored)
    body.pop("package_sha256")
    asset_hash = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    payload_hash = hashlib.sha256(PAYLOAD).hexdigest()
    entries = [
        {"path": "asset.json", "sha256": asset_hash},
        {"path": PATH, "sha256": payload_hash},
    ]
    preimage = (
        json.dumps({"files": entries}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(preimage).hexdigest() == manifest["package_sha256"]
    assert package_digest(entries) == manifest["package_sha256"]


def test_modified_and_missing_files(tmp_path: Path):
    package = _package(tmp_path)
    target = package / "payload" / "a.bin"
    target.write_bytes(b"hello!\n")
    with pytest.raises(PackageError, match="file_hash_mismatch"):
        verify_package(package)
    target.unlink()
    with pytest.raises(PackageError, match="missing_file"):
        verify_package(package)


def test_path_rejections():
    with pytest.raises(PackageError, match="absolute_path"):
        normalize_payload_path("/tmp/a.bin")
    with pytest.raises(PackageError, match="path_traversal"):
        normalize_payload_path("../a.bin")
    with pytest.raises(PackageError, match="backslash"):
        normalize_payload_path("payload\\a.bin")
    with pytest.raises(PackageError, match="drive_or_colon"):
        normalize_payload_path("C:/a.bin")
    with pytest.raises(PackageError, match="control_character"):
        normalize_payload_path("payload/a\nb.bin")
    with pytest.raises(PackageError, match="trailing_space_or_dot"):
        normalize_payload_path("payload/a.bin ")
    with pytest.raises(PackageError, match="reserved_name"):
        normalize_payload_path("payload/CON")
    with pytest.raises(PackageError, match="unicode_not_nfc"):
        normalize_payload_path("payload/cafe\u0301.bin")
    with pytest.raises(PackageError, match="duplicate_path"):
        validate_payload_paths(["payload/a.bin", "payload/a.bin"])
    with pytest.raises(PackageError, match="casefold_collision"):
        validate_payload_paths(["payload/A.bin", "payload/a.bin"])


def test_symlink_special_and_duplicate_directory_entries(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").mkdir()
    (source / "payload" / "a.bin").write_bytes(PAYLOAD)
    (source / "payload" / "link.bin").symlink_to(source / "payload" / "a.bin")
    with pytest.raises(PackageError, match="symlink"):
        package_directory(tmp_path / "linked-file", MANIFEST, source)

    nested = tmp_path / "nested"
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.bin").write_bytes(PAYLOAD)
    nested.mkdir()
    (nested / "payload").symlink_to(real, target_is_directory=True)
    with pytest.raises(PackageError, match="symlink"):
        package_directory(tmp_path / "linked-dir", MANIFEST, nested)

    fifo_root = tmp_path / "fifo"
    fifo_root.mkdir()
    os.mkfifo(fifo_root / "pipe")
    with pytest.raises(PackageError, match="special_file"):
        package_directory(tmp_path / "fifo-package", MANIFEST, fifo_root)


def test_json_rejections(tmp_path: Path):
    with pytest.raises(PackageError, match="duplicate_json_key"):
        parse_strict_json('{"a": 1, "a": 2}\n')
    with pytest.raises(PackageError, match="non_finite"):
        parse_strict_json('{"a": NaN}\n')
    with pytest.raises(PackageError, match="invalid_json"):
        parse_strict_json("{")
    with pytest.raises(PackageError, match="non_finite"):
        create_package(
            tmp_path / "bad",
            {**MANIFEST, "provenance": {**MANIFEST["provenance"], "parameters": {"scale": float("nan")}}},
            {PATH: PAYLOAD},
        )
    package = _package(tmp_path)
    (package / "asset.json").write_bytes(b"\xff")
    with pytest.raises(PackageError, match="utf8"):
        verify_package(package)


def test_status_whitespace_and_legacy_hash(tmp_path: Path):
    with pytest.raises(PackageError, match="status_in_manifest"):
        create_package(tmp_path / "status", {**MANIFEST, "status": "admitted"}, {PATH: PAYLOAD})
    package = _package(tmp_path)
    original = (package / "asset.json").read_text(encoding="utf-8")
    (package / "asset.json").write_text(original.replace("{", "{ ", 1), encoding="utf-8")
    with pytest.raises(PackageError, match="manifest_not_canonical"):
        verify_package(package)
    body = {"schema_version": 1, "asset_id": "box_prop"}
    assert manifest_body_for_hash({**body, "status": "admitted", "package_sha256": "abc"}) == manifest_body_for_hash(body)


def test_exclusion_and_unexpected_file(tmp_path: Path):
    package = _package(tmp_path)
    digest = verify_package(package)["package_sha256"]
    (package / "admission_report.json").write_text("{}\n", encoding="utf-8")
    (package / "notes.log").write_text("log\n", encoding="utf-8")
    assert verify_package(package)["package_sha256"] == digest
    (package / "extra.txt").write_text("no\n", encoding="utf-8")
    with pytest.raises(PackageError, match="unexpected_file"):
        verify_package(package)


def test_registration_keeps_bytes_and_rejects_wrong_approval(tmp_path: Path):
    package = _package(tmp_path)
    before = file_hashes(package)
    library = tmp_path / "library"
    registered = register_package(library, package, _approval(package))
    published = library / "box_prop" / "1.0.0"
    assert file_hashes(published) == before
    assert verify_package(published)["package_sha256"] == registered["package_sha256"]
    assert "status" not in json.loads((published / "asset.json").read_text(encoding="utf-8"))
    with pytest.raises(PackageError, match="version_exists"):
        register_package(library, package, _approval(package))
    other = _package(tmp_path, "other", payload=b"other\n")
    wrong = _approval(other)
    wrong["asset_id"] = "box_prop"
    wrong["version"] = "1.0.0"
    with pytest.raises(PackageError, match="approval_hash"):
        register_package(tmp_path / "library-two", package, wrong)
    assert not (tmp_path / "library-two" / "box_prop").exists()


def test_candidate_status_does_not_admit_or_rewrite_bytes(tmp_path: Path):
    package = _package(tmp_path)
    catalog = tmp_path / "candidates"
    publish_candidate(catalog, package)
    before = file_hashes(catalog / "box_prop" / "1.0.0")
    with pytest.raises(PackageError, match="human_status_refused"):
        set_candidate_status(catalog, "box_prop", "1.0.0", "admitted")
    with pytest.raises(PackageError, match="human_status_refused"):
        set_candidate_status(catalog, "box_prop", "1.0.0", "visually_reviewed")
    set_candidate_status(catalog, "box_prop", "1.0.0", "technically_validated")
    set_candidate_status(catalog, "box_prop", "1.0.0", "renderer_validated")
    assert file_hashes(catalog / "box_prop" / "1.0.0") == before
    text = format_catalog(catalog, tmp_path / "empty-library")
    assert "catalog=candidates" in text
    assert "catalog=library" in text
    assert "renderer_validated" in text
    assert "count=0" in text.split("catalog=library", 1)[1]


def test_index_failure_leaves_a_reconciled_orphan(tmp_path: Path, monkeypatch):
    package = _package(tmp_path)
    library = tmp_path / "library"
    real = __import__("sensor_sim.scene_gen.asset_package", fromlist=["commit_index"]).commit_index

    def fail(path, payload):
        if Path(path).name == "index.json":
            raise PackageError("reason=index_write_failed")
        return real(path, payload)

    monkeypatch.setattr("sensor_sim.scene_gen.asset_package.commit_index", fail)
    with pytest.raises(PackageError, match="index_write_failed"):
        register_package(library, package, _approval(package))
    published = library / "box_prop" / "1.0.0"
    assert published.is_dir()
    assert file_hashes(published) == file_hashes(package)
    monkeypatch.setattr("sensor_sim.scene_gen.asset_package.commit_index", real)
    first = reconcile_library(library)
    second = reconcile_library(library)
    assert first == second
    assert first[0]["asset_id"] == "box_prop"
    assert first[0]["version"] == "1.0.0"
    assert published.is_dir()
    assert read_index(library / "index.json")["assets"] == {}


def test_concurrent_registration(tmp_path: Path):
    first = _package(tmp_path, "first", b"one\n")
    second_manifest = {
        **MANIFEST,
        "asset_id": "shelf_prop",
    }
    second = tmp_path / "second"
    create_package(second, second_manifest, {PATH: b"two\n"}, {PATH: "payload"})
    library = tmp_path / "library"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    jobs = (
        (library, first, _approval(first)),
        (library, second, _approval(second)),
    )
    processes = [
        context.Process(target=_register_job, args=(str(item[0]), str(item[1]), item[2], queue)) for item in jobs
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0
    results = [queue.get(timeout=10) for _ in jobs]
    assert [item["ok"] for item in results] == [True, True]
    index = read_index(library / "index.json")
    assert index["assets"]["box_prop"]["1.0.0"]["package_sha256"] == verify_package(first)["package_sha256"]
    assert index["assets"]["shelf_prop"]["1.0.0"]["package_sha256"] == verify_package(second)["package_sha256"]

    collision = tmp_path / "collision"
    queue = context.Queue()
    processes = [
        context.Process(target=_register_job, args=(str(collision), str(first), _approval(first), queue))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0
    results = [queue.get(timeout=10) for _ in range(2)]
    assert sorted(item["ok"] for item in results) == [False, True]
    assert "version_exists" in results[0]["error"] + results[1]["error"]
    assert file_hashes(collision / "box_prop" / "1.0.0") == file_hashes(first)


def test_cli_round_trip(tmp_path: Path):
    source = tmp_path / "source" / "payload"
    source.mkdir(parents=True)
    (source / "a.bin").write_bytes(PAYLOAD)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    destination = tmp_path / "package"
    from sensor_sim.scene_gen.asset_cli import main

    assert main(["package", "--manifest", str(manifest_path), "--source", str(tmp_path / "source"), "--destination", str(destination), "--role", f"{PATH}=payload"]) == 0
    assert main(["validate", "--package", str(destination)]) == 0
    assert main(["verify", "--package", str(destination)]) == 0
    assert main(["inspect", "--package", str(destination)]) == 0
    catalog = tmp_path / "candidates"
    publish_candidate(catalog, destination)
    library = tmp_path / "library"
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps(_approval(destination)), encoding="utf-8")
    assert main(["list", "--candidates", str(catalog), "--library", str(library)]) == 0
    assert main(["register", "--library", str(library), "--package", str(destination), "--approval", str(approval)]) == 0
    assert main(["reconcile", "--library", str(library)]) == 0

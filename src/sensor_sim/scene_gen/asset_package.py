"""Immutable indoor asset packages and the production library index.

The package hash is the SHA-256 of one canonical JSON document. ``asset.json``
is stored in that same canonical form. Lifecycle status is not stored in the
immutable manifest. It lives in the candidate index or in ``assets/library/index.json``.

Exclusion policy version 1 omits only these regular files from the hash:

- a file named ``admission_report.json``
- a file whose name ends in ``.blend1``, ``.log``, or ``.tmp``
- a file under a top-level directory named ``logs``, ``cache``, ``caches``, ``tmp``, or ``temp``

Any other regular file is payload. Symbolic links and special files are rejected
everywhere, including inside an excluded directory.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("sensor_sim.scene_gen.asset_package")

SCHEMA_VERSION = 1
EXCLUSION_POLICY_VERSION = 1
APPROVAL_SCHEMA_VERSION = 1
CANONICAL_SEPARATORS = (",", ":")
_EXCLUDED_ROOTS = frozenset({"logs", "cache", "caches", "tmp", "temp"})
_RESERVED_WINDOWS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_ASSET_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "exclusion_policy_version",
        "asset_id",
        "version",
        "package_sha256",
        "provenance",
        "files",
        "geometry",
        "materials",
        "textures",
        "collision",
        "support_surfaces",
        "contact_regions",
        "components",
        "transform",
        "units",
        "frame",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "source",
        "generator",
        "generator_version",
        "seed",
        "parameters",
        "license",
        "license_status",
        "upstream_commit",
        "upstream_object",
    }
)
_LICENSE_STATUS = frozenset({"permitted", "unresolved", "restricted"})
_FILE_ROLES = frozenset({"render", "collision", "support", "texture", "material", "payload", "provenance"})
AUTOMATED_STATUSES = frozenset({"technically_validated", "renderer_validated"})
HUMAN_STATUSES = frozenset({"visually_reviewed", "admitted"})
CANDIDATE_STATUSES = frozenset({"candidate"}) | AUTOMATED_STATUSES


class PackageError(ValueError):
    """A package, path, approval, or library operation failed closed."""


def canonical_json_bytes(value: object) -> bytes:
    """UTF-8 canonical JSON with a trailing newline and no insignificant whitespace."""
    reject_nonfinite(value)
    text = json.dumps(value, sort_keys=True, separators=CANONICAL_SEPARATORS, ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("utf-8")


def parse_strict_json(text: str) -> object:
    """Parse JSON, rejecting duplicate keys and non-finite constants."""

    def pairs(items: list[tuple[str, object]]) -> dict:
        parsed: dict[str, object] = {}
        for key, value in items:
            if key in parsed:
                raise PackageError(f"reason=duplicate_json_key key={key}")
            parsed[key] = value
        return parsed

    def constant(name: str) -> object:
        raise PackageError(f"reason=non_finite constant={name}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except PackageError:
        raise
    except json.JSONDecodeError as exc:
        raise PackageError(f"reason=invalid_json detail={exc.msg}") from exc


def reject_nonfinite(value: object, place: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PackageError(f"reason=non_finite path={place}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PackageError(f"reason=invalid_key path={place}")
            reject_nonfinite(item, f"{place}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            reject_nonfinite(item, f"{place}[{index}]")
        return
    raise PackageError(f"reason=unsupported_json path={place} type={type(value).__name__}")


def is_excluded(relative: str) -> bool:
    """Return True when policy version 1 omits this regular file from the hash."""
    parts = relative.split("/")
    name = parts[-1]
    if name == "admission_report.json":
        return True
    if name.endswith(".blend1") or name.endswith(".log") or name.endswith(".tmp"):
        return True
    return parts[0] in _EXCLUDED_ROOTS


def normalize_payload_path(path: str) -> str:
    """Return the canonical relative POSIX path or reject an ambiguous path."""
    if not isinstance(path, str) or path == "":
        raise PackageError("reason=invalid_path")
    if "\\" in path:
        raise PackageError(f"reason=backslash path={path!r}")
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise PackageError("reason=control_character")
    if ":" in path:
        raise PackageError(f"reason=drive_or_colon path={path!r}")
    normalized = unicodedata.normalize("NFC", path)
    if normalized != path:
        raise PackageError(f"reason=unicode_not_nfc path={path!r}")
    if normalized.startswith("/"):
        raise PackageError(f"reason=absolute_path path={normalized}")
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise PackageError(f"reason=path_traversal path={normalized}")
    for part in parts:
        if part.endswith(" ") or part.endswith("."):
            raise PackageError(f"reason=trailing_space_or_dot path={normalized}")
        stem = part.split(".", 1)[0].upper()
        if stem in _RESERVED_WINDOWS:
            raise PackageError(f"reason=reserved_name path={normalized}")
    if parts == ["asset.json"]:
        raise PackageError("reason=reserved_manifest_path")
    return normalized


def validate_payload_paths(paths: list[str]) -> list[str]:
    """Normalize paths and reject duplicates, including case-folded collisions."""
    accepted: list[str] = []
    seen: set[str] = set()
    folded: dict[str, str] = {}
    for raw in paths:
        path = normalize_payload_path(raw)
        if path in seen:
            raise PackageError(f"reason=duplicate_path path={path}")
        fold = path.casefold()
        if fold in folded:
            raise PackageError(f"reason=casefold_collision path={path} other={folded[fold]}")
        seen.add(path)
        folded[fold] = path
        accepted.append(path)
    return accepted


def package_digest(entries: list[dict[str, str]]) -> str:
    """SHA-256 of the documented ``{"files": ...}`` preimage."""
    files = [{"path": item["path"], "sha256": item["sha256"]} for item in entries]
    files.sort(key=lambda item: item["path"])
    preimage = canonical_json_bytes({"files": files})
    return hashlib.sha256(preimage).hexdigest()


def create_package(
    destination: Path,
    manifest: dict,
    payload: dict[str, bytes],
    roles: dict[str, str] | None = None,
) -> dict:
    """Write a new immutable package directory. The destination must not exist."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise PackageError(f"reason=destination_exists path={destination}")
    prepared = prepare_manifest(manifest)
    paths = validate_payload_paths(list(payload))
    role_map = _roles_for(paths, roles)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pkg-", dir=destination.parent))
    try:
        for path in paths:
            _write_bytes_nofollow(temporary, path, payload[path])
        sealed = seal_package(temporary, prepared, paths, role_map)
        fsync_tree(temporary)
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    logger.info(
        "asset_package event=create asset_id=%s version=%s package_sha256=%s",
        sealed["asset_id"],
        sealed["version"],
        sealed["package_sha256"],
    )
    return sealed


def package_directory(
    destination: Path,
    manifest: dict,
    source: Path,
    roles: dict[str, str] | None = None,
) -> dict:
    """Package regular files from ``source``. Excluded files are omitted. Links are rejected."""
    payload: dict[str, bytes] = {}
    source = Path(source)
    for relative in walk_files(source):
        if relative == "asset.json":
            raise PackageError("reason=source_contains_manifest")
        if is_excluded(relative):
            continue
        payload[relative] = read_regular_bytes(source, relative)
    return create_package(destination, manifest, payload, roles)


def verify_package(root: Path) -> dict:
    """Recompute hashes from file bytes and require the stored manifest to match."""
    root = Path(root)
    raw = read_utf8_nofollow(root, "asset.json")
    parsed = parse_strict_json(raw)
    if not isinstance(parsed, dict):
        raise PackageError("reason=manifest_not_object")
    canonical = canonical_json_bytes(parsed).decode("utf-8")
    if raw != canonical:
        raise PackageError("reason=manifest_not_canonical")
    if "status" in parsed:
        raise PackageError("reason=status_in_manifest")
    unknown = set(parsed) - _MANIFEST_KEYS
    if unknown:
        raise PackageError(f"reason=unknown_manifest_key keys={sorted(unknown)}")
    if parsed.get("schema_version") != SCHEMA_VERSION:
        raise PackageError("reason=schema_version")
    if parsed.get("exclusion_policy_version") != EXCLUSION_POLICY_VERSION:
        raise PackageError("reason=exclusion_policy_version")
    check_identity(parsed)
    check_provenance(parsed.get("provenance"))
    stored = parsed.get("package_sha256")
    if not isinstance(stored, str) or _HEX64.fullmatch(stored) is None:
        raise PackageError("reason=package_sha256")
    body = dict(parsed)
    body.pop("package_sha256")
    asset_digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    declared = parsed.get("files")
    if not isinstance(declared, list):
        raise PackageError("reason=files")
    entries = [{"path": "asset.json", "sha256": asset_digest}]
    declared_paths: list[str] = []
    for item in declared:
        path, digest = check_file_entry(item)
        actual = sha256_regular(root, path)
        if actual != digest:
            raise PackageError(f"reason=file_hash_mismatch path={path}")
        declared_paths.append(path)
        entries.append({"path": path, "sha256": actual})
    validate_payload_paths(declared_paths)
    recomputed = package_digest(entries)
    if recomputed != stored:
        raise PackageError("reason=package_sha256_mismatch")
    present = set(walk_files(root))
    expected = {item["path"] for item in entries}
    if "asset.json" not in present:
        raise PackageError("reason=missing_file path=asset.json")
    for path in expected:
        if path not in present:
            raise PackageError(f"reason=missing_file path={path}")
    for path in present:
        if path in expected or is_excluded(path):
            continue
        raise PackageError(f"reason=unexpected_file path={path}")
    logger.info(
        "asset_package event=verify result=ok asset_id=%s version=%s package_sha256=%s",
        parsed["asset_id"],
        parsed["version"],
        stored,
    )
    return parsed


def prepare_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise PackageError("reason=manifest_not_object")
    if "status" in manifest:
        raise PackageError("reason=status_in_manifest")
    unknown = set(manifest) - _MANIFEST_KEYS
    if unknown:
        raise PackageError(f"reason=unknown_manifest_key keys={sorted(unknown)}")
    if "package_sha256" in manifest:
        raise PackageError("reason=package_sha256_supplied")
    if "files" in manifest:
        raise PackageError("reason=files_supplied")
    prepared = {
        "schema_version": SCHEMA_VERSION,
        "exclusion_policy_version": EXCLUSION_POLICY_VERSION,
        "asset_id": manifest.get("asset_id"),
        "version": manifest.get("version"),
        "provenance": manifest.get("provenance"),
    }
    for key in (
        "geometry",
        "materials",
        "textures",
        "collision",
        "support_surfaces",
        "contact_regions",
        "components",
        "transform",
        "units",
        "frame",
    ):
        if key in manifest:
            prepared[key] = manifest[key]
    check_identity(prepared)
    check_provenance(prepared["provenance"])
    reject_nonfinite(prepared)
    return prepared


def check_identity(manifest: dict) -> None:
    asset_id = manifest.get("asset_id")
    version = manifest.get("version")
    if not isinstance(asset_id, str) or _ASSET_ID.fullmatch(asset_id) is None:
        raise PackageError(f"reason=asset_id value={asset_id!r}")
    if not isinstance(version, str) or _VERSION.fullmatch(version) is None:
        raise PackageError(f"reason=version value={version!r}")


def check_provenance(provenance: object) -> None:
    if not isinstance(provenance, dict):
        raise PackageError("reason=provenance")
    unknown = set(provenance) - _PROVENANCE_KEYS
    if unknown:
        raise PackageError(f"reason=unknown_provenance_key keys={sorted(unknown)}")
    for key in ("source", "generator", "generator_version", "license", "license_status"):
        if not isinstance(provenance.get(key), str) or provenance[key] == "":
            raise PackageError(f"reason=provenance_field field={key}")
    if provenance["license_status"] not in _LICENSE_STATUS:
        raise PackageError("reason=license_status")
    reject_nonfinite(provenance)


def check_file_entry(item: object) -> tuple[str, str]:
    if not isinstance(item, dict):
        raise PackageError("reason=file_entry")
    allowed = {"path", "sha256", "role"}
    if set(item) - allowed:
        raise PackageError("reason=file_entry_key")
    path = normalize_payload_path(item.get("path") if isinstance(item.get("path"), str) else "")
    digest = item.get("sha256")
    role = item.get("role")
    if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
        raise PackageError(f"reason=file_sha256 path={path}")
    if role not in _FILE_ROLES:
        raise PackageError(f"reason=file_role path={path}")
    return path, digest


def seal_package(root: Path, manifest: dict, paths: list[str], roles: dict[str, str]) -> dict:
    """Hash payload files, then write canonical ``asset.json`` including the package digest."""
    files = []
    for path in sorted(paths):
        files.append({"path": path, "role": roles[path], "sha256": sha256_regular(root, path)})
    body = dict(manifest)
    body["files"] = files
    asset_digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    entries = [{"path": "asset.json", "sha256": asset_digest}, *[{"path": item["path"], "sha256": item["sha256"]} for item in files]]
    sealed = dict(body)
    sealed["package_sha256"] = package_digest(entries)
    written = canonical_json_bytes(sealed)
    _write_bytes_nofollow(root, "asset.json", written)
    reread = sha256_regular(root, "asset.json")
    if reread != hashlib.sha256(written).hexdigest():
        raise PackageError("reason=manifest_write_mismatch")
    return sealed


def manifest_body_for_hash(manifest: dict) -> bytes:
    """Canonical bytes hashed for ``asset.json``, with the legacy status field removed."""
    body = dict(manifest)
    body.pop("package_sha256", None)
    body.pop("status", None)
    return canonical_json_bytes(body)


def publish_candidate(catalog: Path, package: Path) -> dict:
    """Copy a verified package into the candidate catalog without admitting it."""
    catalog = Path(catalog)
    verified = verify_package(package)
    with library_lock(catalog):
        destination = catalog / verified["asset_id"] / verified["version"]
        index = read_index(catalog / "index.json")
        if destination.exists() or index_entry(index, verified["asset_id"], verified["version"]) is not None:
            raise PackageError(f"reason=version_exists asset_id={verified['asset_id']} version={verified['version']}")
        _publish_directory(package, destination)
        insert_index(
            index,
            verified,
            status="candidate",
            approval_sha256=None,
            registered_utc=None,
        )
        commit_index(catalog / "index.json", index)
    logger.info(
        "asset_package event=publish_candidate asset_id=%s version=%s status=candidate package_sha256=%s",
        verified["asset_id"],
        verified["version"],
        verified["package_sha256"],
    )
    return verified


def set_candidate_status(catalog: Path, asset_id: str, version: str, status: str) -> dict:
    """Set an automated technical status. Human review and admission are refused."""
    if status in HUMAN_STATUSES or status == "registered":
        raise PackageError(f"reason=human_status_refused status={status}")
    if status not in AUTOMATED_STATUSES:
        raise PackageError(f"reason=status status={status}")
    catalog = Path(catalog)
    with library_lock(catalog):
        package = catalog / asset_id / version
        before = file_hashes(package)
        verified = verify_package(package)
        index = read_index(catalog / "index.json")
        current = index_entry(index, asset_id, version)
        if current is None:
            raise PackageError("reason=candidate_missing")
        if current["package_sha256"] != verified["package_sha256"]:
            raise PackageError("reason=index_hash_mismatch")
        if status == "renderer_validated" and current["status"] not in {"technically_validated", "renderer_validated"}:
            raise PackageError("reason=renderer_before_technical")
        if status == "technically_validated" and current["status"] not in {"candidate", "technically_validated"}:
            raise PackageError("reason=technical_regression")
        current["status"] = status
        commit_index(catalog / "index.json", index)
        after = file_hashes(package)
        if before != after:
            raise PackageError("reason=package_bytes_changed")
    logger.info(
        "asset_package event=candidate_status asset_id=%s version=%s status=%s package_sha256=%s",
        asset_id,
        version,
        status,
        verified["package_sha256"],
    )
    return current


def register_package(library: Path, package: Path, approval: dict) -> dict:
    """Publish an admitted package. The approval record stays outside the version directory."""
    library = Path(library)
    verified = verify_package(package)
    check_approval(approval, verified)
    if verified["provenance"]["license_status"] != "permitted":
        raise PackageError("reason=license_not_permitted")
    asset_id = verified["asset_id"]
    version = verified["version"]
    published = False
    destination = library / asset_id / version
    with library_lock(library):
        index = read_index(library / "index.json")
        if destination.exists() or index_entry(index, asset_id, version) is not None:
            raise PackageError(f"reason=version_exists asset_id={asset_id} version={version}")
        try:
            _publish_directory(package, destination)
            published = True
            republished = verify_package(destination)
            if republished["package_sha256"] != verified["package_sha256"]:
                raise PackageError("reason=copy_hash_mismatch")
            approval_digest = store_approval(library, approval)
            index = read_index(library / "index.json")
            if index_entry(index, asset_id, version) is not None or not destination.exists():
                raise PackageError("reason=version_exists")
            insert_index(
                index,
                republished,
                status="registered",
                approval_sha256=approval_digest,
                registered_utc=utc_now(),
            )
            commit_index(library / "index.json", index)
        except Exception as exc:
            if published:
                record_orphan(library, asset_id, version, verified["package_sha256"], str(exc))
            raise
    logger.info(
        "asset_package event=register asset_id=%s version=%s package_sha256=%s approval_sha256=%s",
        asset_id,
        version,
        verified["package_sha256"],
        approval_digest,
    )
    return verified


def reconcile_library(library: Path) -> list[dict]:
    """Record version directories missing from the index. Do not delete or register them."""
    library = Path(library)
    with library_lock(library):
        index = read_index(library / "index.json")
        found: list[dict] = []
        if library.exists():
            for asset_dir in sorted(library.iterdir()):
                if not asset_dir.is_dir() or asset_dir.is_symlink():
                    continue
                if _ASSET_ID.fullmatch(asset_dir.name) is None:
                    continue
                for version_dir in sorted(asset_dir.iterdir()):
                    if not version_dir.is_dir() or version_dir.is_symlink():
                        continue
                    if _VERSION.fullmatch(version_dir.name) is None:
                        continue
                    if index_entry(index, asset_dir.name, version_dir.name) is not None:
                        continue
                    manifest_path = version_dir / "asset.json"
                    digest = ""
                    if manifest_path.is_file() and not manifest_path.is_symlink():
                        try:
                            digest = str(verify_package(version_dir)["package_sha256"])
                        except PackageError:
                            digest = ""
                    found.append(
                        {
                            "asset_id": asset_dir.name,
                            "version": version_dir.name,
                            "package_sha256": digest,
                            "reason": "missing_from_index",
                        }
                    )
        payload = {"schema_version": SCHEMA_VERSION, "orphans": found}
        commit_index(library / "_orphans.json", payload)
    logger.info("asset_package event=reconcile orphans=%d", len(found))
    return found


def check_approval(approval: dict, manifest: dict) -> None:
    if not isinstance(approval, dict):
        raise PackageError("reason=approval")
    required = {
        "approval_schema_version",
        "asset_id",
        "version",
        "package_sha256",
        "decision",
        "reviewer_id",
        "approved_utc",
        "visual_evidence",
        "license_decision",
    }
    if set(approval) != required:
        raise PackageError("reason=approval_fields")
    if approval["approval_schema_version"] != APPROVAL_SCHEMA_VERSION:
        raise PackageError("reason=approval_schema_version")
    if approval["decision"] != "admitted":
        raise PackageError("reason=approval_decision")
    if approval["asset_id"] != manifest["asset_id"] or approval["version"] != manifest["version"]:
        raise PackageError("reason=approval_identity")
    if approval["package_sha256"] != manifest["package_sha256"]:
        raise PackageError("reason=approval_hash")
    for key in ("reviewer_id", "approved_utc", "visual_evidence", "license_decision"):
        if not isinstance(approval[key], str) or approval[key] == "":
            raise PackageError(f"reason=approval_field field={key}")
    if not parse_utc(approval["approved_utc"]):
        raise PackageError("reason=approval_time")
    reject_nonfinite(approval)


def store_approval(library: Path, approval: dict) -> str:
    """Write an append-only approval outside every version directory."""
    body = canonical_json_bytes(approval)
    digest = hashlib.sha256(body).hexdigest()
    relative = f"{approval['asset_id']}/{approval['version']}__{approval['package_sha256']}.json"
    destination = library / "_approvals" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.read_bytes()
        if existing != body:
            raise PackageError("reason=approval_conflict")
        return digest
    temporary = destination.with_suffix(".json.tmp")
    write_fsync(temporary, body)
    os.replace(temporary, destination)
    fsync_directory(destination.parent)
    return digest


def record_orphan(library: Path, asset_id: str, version: str, package_sha256: str, reason: str) -> None:
    path = library / "_orphans.json"
    current = {"schema_version": SCHEMA_VERSION, "orphans": []}
    if path.is_file():
        loaded = parse_strict_json(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("orphans"), list):
            current = loaded
    entry = {
        "asset_id": asset_id,
        "version": version,
        "package_sha256": package_sha256,
        "reason": reason,
    }
    orphans = [item for item in current["orphans"] if not (item.get("asset_id") == asset_id and item.get("version") == version)]
    orphans.append(entry)
    orphans.sort(key=lambda item: (item.get("asset_id", ""), item.get("version", "")))
    try:
        commit_index(path, {"schema_version": SCHEMA_VERSION, "orphans": orphans})
    except Exception:
        logger.exception(
            "asset_package event=orphan_record_failed asset_id=%s version=%s package_sha256=%s",
            asset_id,
            version,
            package_sha256,
        )


def read_index(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "assets": {}}
    parsed = parse_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("assets"), dict):
        raise PackageError(f"reason=index path={path}")
    return parsed


def index_entry(index: dict, asset_id: str, version: str) -> dict | None:
    asset = index.get("assets", {}).get(asset_id)
    if not isinstance(asset, dict):
        return None
    entry = asset.get(version)
    return entry if isinstance(entry, dict) else None


def insert_index(index: dict, manifest: dict, *, status: str, approval_sha256: str | None, registered_utc: str | None) -> None:
    assets = index.setdefault("assets", {})
    versions = assets.setdefault(manifest["asset_id"], {})
    if manifest["version"] in versions:
        raise PackageError("reason=version_exists")
    entry = {
        "package_sha256": manifest["package_sha256"],
        "status": status,
    }
    if approval_sha256 is not None:
        entry["approval_sha256"] = approval_sha256
    if registered_utc is not None:
        entry["registered_utc"] = registered_utc
    versions[manifest["version"]] = entry


def commit_index(path: Path, payload: dict) -> None:
    """Atomically replace an index file. Tests may patch this function to simulate a failed commit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_fsync(temporary, canonical_json_bytes(payload))
    os.replace(temporary, path)
    fsync_directory(path.parent)


def format_catalog(candidates: Path | None = None, library: Path | None = None) -> str:
    """Text listing that keeps the candidate catalog distinct from the production library."""
    lines = [
        "catalog=candidates",
        "meaning=packages that may be technically or renderer validated and are not admitted production assets",
    ]
    lines.extend(format_index(Path(candidates) if candidates is not None else None))
    lines.append("catalog=library")
    lines.append("meaning=admitted registered production versions only")
    lines.extend(format_index(Path(library) if library is not None else None))
    return "\n".join(lines) + "\n"


def format_index(root: Path | None) -> list[str]:
    if root is None or not (root / "index.json").is_file():
        return ["count=0"]
    index = read_index(root / "index.json")
    lines = []
    for asset_id in sorted(index["assets"]):
        versions = index["assets"][asset_id]
        if not isinstance(versions, dict):
            continue
        for version in sorted(versions):
            entry = versions[version]
            lines.append(
                f"asset_id={asset_id} version={version} status={entry.get('status')} package_sha256={entry.get('package_sha256')}"
            )
    return [f"count={len(lines)}", *lines]


def file_hashes(root: Path) -> dict[str, str]:
    return {relative: sha256_regular(root, relative) for relative in walk_files(root) if not is_excluded(relative)}


def walk_files(root: Path) -> list[str]:
    root = Path(root)
    if root.is_symlink():
        raise PackageError("reason=symlink path=.")
    found: list[str] = []
    for relative in _walk(root, root):
        found.append(relative)
    found.sort()
    validate_payload_paths([item for item in found if item != "asset.json"])
    return found


def sha256_regular(root: Path, relative: str) -> str:
    """Hash one regular file through the descriptor returned by a no-follow open."""
    if relative != "asset.json":
        relative = normalize_payload_path(relative)
    path = component_path(root, relative)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PackageError(f"reason=open path={relative} errno={exc.errno}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise PackageError(f"reason=special_file path={relative}")
        linked = path.lstat()
        if stat.S_ISLNK(linked.st_mode) or (linked.st_dev, linked.st_ino) != (info.st_dev, info.st_ino):
            raise PackageError(f"reason=symlink path={relative}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def read_regular_bytes(root: Path, relative: str) -> bytes:
    relative = normalize_payload_path(relative)
    path = component_path(root, relative)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise PackageError(f"reason=special_file path={relative}")
        return os.read(descriptor, info.st_size)
    finally:
        os.close(descriptor)


def _roles_for(paths: list[str], roles: dict[str, str] | None) -> dict[str, str]:
    supplied = roles or {}
    unknown = set(supplied) - set(paths)
    if unknown:
        raise PackageError(f"reason=role_path paths={sorted(unknown)}")
    mapped = {}
    for path in paths:
        role = supplied.get(path, "payload")
        if role not in _FILE_ROLES:
            raise PackageError(f"reason=file_role path={path}")
        mapped[path] = role
    return mapped


def read_utf8_nofollow(root: Path, relative: str) -> str:
    path = component_path(root, relative if relative == "asset.json" else normalize_payload_path(relative))
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PackageError(f"reason=open path={relative} errno={exc.errno}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise PackageError(f"reason=special_file path={relative}")
        data = os.read(descriptor, info.st_size)
    finally:
        os.close(descriptor)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PackageError("reason=utf8") from exc


def component_path(root: Path, relative: str) -> Path:
    """Join a relative path, rejecting a symlink at every component."""
    current = Path(root)
    if current.is_symlink():
        raise PackageError("reason=symlink path=.")
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise PackageError(f"reason=missing_file path={relative}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise PackageError(f"reason=symlink path={relative}")
        final = index == len(parts) - 1
        if not final and not stat.S_ISDIR(info.st_mode):
            raise PackageError(f"reason=not_directory path={relative}")
        if final and not stat.S_ISREG(info.st_mode):
            raise PackageError(f"reason=special_file path={relative}")
    lexical = Path(root, *parts)
    if Path(os.path.commonpath((str(Path(root)), str(lexical)))) != Path(root):
        raise PackageError(f"reason=path_escape path={relative}")
    return current


def fsync_tree(root: Path) -> None:
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            descriptor = os.open(os.path.join(directory, name), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        fsync_directory(Path(directory))


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_fsync(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> bool:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return parsed.tzinfo is None


@contextmanager
def library_lock(root: Path) -> Iterator[None]:
    """Exclusive cross-process lock held for the whole check-and-publish critical section."""
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / ".library.lock", os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _walk(root: Path, current: Path) -> Iterator[str]:
    try:
        entries = list(os.scandir(current))
    except OSError as exc:
        raise PackageError(f"reason=scan path={current} errno={exc.errno}") from exc
    for entry in entries:
        info = entry.stat(follow_symlinks=False)
        relative = Path(entry.path).relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise PackageError(f"reason=symlink path={relative}")
        if stat.S_ISDIR(info.st_mode):
            yield from _walk(root, Path(entry.path))
            continue
        if not stat.S_ISREG(info.st_mode):
            raise PackageError(f"reason=special_file path={relative}")
        if relative != "asset.json":
            normalize_payload_path(relative)
        yield relative


def _write_bytes_nofollow(root: Path, relative: str, data: bytes) -> None:
    parts = relative.split("/")
    current = Path(root)
    for part in parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise PackageError(f"reason=symlink path={relative}")
        current.mkdir(exist_ok=True)
    destination = current / parts[-1]
    if destination.is_symlink() or destination.exists():
        raise PackageError(f"reason=destination_exists path={relative}")
    write_fsync(destination, data)


def _publish_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        raise PackageError(f"reason=version_exists path={destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pub-", dir=destination.parent))
    try:
        for relative in walk_files(source):
            if is_excluded(relative):
                continue
            data = read_regular_bytes(source, relative) if relative != "asset.json" else open_manifest_bytes(source)
            _write_bytes_nofollow(temporary, relative, data)
        fsync_tree(temporary)
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def open_manifest_bytes(root: Path) -> bytes:
    return read_utf8_nofollow(root, "asset.json").encode("utf-8")

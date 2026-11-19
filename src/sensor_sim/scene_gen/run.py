"""One dataset run: successes are validated samples, failures stay out of the count."""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen import GENERATOR_VERSION
from sensor_sim.scene_gen.dataset import validate_sample


def config_fingerprint(spec: dict) -> str:
    """Hash of the config that defines a run. Profiles are included."""
    payload = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def stream_seed(base: int, slot: int, attempt: int, name: str) -> int:
    """One deterministic stream. Geometry, material, and render use different names."""
    tag = {"geometry": 1, "material": 2, "render": 3, "scene": 4}.get(name)
    if tag is None:
        tag = int(np.frombuffer(name.encode(), dtype=np.uint32)[:1][0])
    sequence = np.random.SeedSequence([int(base), int(slot), int(attempt), int(tag)])
    return int(sequence.generate_state(1)[0])


def manifest_path(sample_root: Path) -> Path:
    return Path(sample_root).parent / "run_manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sample_id(run_id: str, slot: int) -> str:
    """Stable id inside one run. A new run id keeps the same base seed from colliding."""
    return f"scene_{str(run_id)[:12]}_{int(slot):04d}"


def expand_schedule(strata: list[dict] | None, requested: int) -> list[dict | None]:
    """One target stratum per sample slot. Failed attempts retry this same slot."""
    if not strata:
        return [None] * int(requested)
    slots: list[dict | None] = []
    for item in strata:
        count = int(item["count"])
        slots.extend([copy.deepcopy(item) for _copy in range(count)])
    if len(slots) != int(requested):
        raise RuntimeError(f"strata sum to {len(slots)} slots, requested {requested}")
    return slots


def new_manifest(spec: dict, *, requested: int, base_seed: int, split_mode: str | None, split_seed: int) -> dict:
    return {
        "run_id": uuid.uuid4().hex,
        "config_hash": config_fingerprint(spec),
        "generator_version": GENERATOR_VERSION,
        "base_seed": int(base_seed),
        "requested_successes": int(requested),
        "completed_ids": [],
        "attempted_seeds": [],
        "failures": [],
        "split_mode": split_mode,
        "split_seed": int(split_seed),
        "status": "running",
        "created": _now(),
        "completed": None,
    }


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save_manifest(path: Path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def sample_is_valid(directory: Path, expected_hash: str | None = None) -> bool:
    """True only when the files validate. A status flag alone does not count."""
    from sensor_sim.scene_gen.provenance import publication_issues

    directory = Path(directory)
    if publication_issues(directory):
        return False
    try:
        validate_sample(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError, RuntimeError):
        return False
    if metadata.get("status") != "complete":
        return False
    if expected_hash is not None and metadata.get("config_hash") not in (None, expected_hash):
        return False
    return True


def resume_manifest(path: Path, spec: dict, *, requested: int, base_seed: int, split_mode: str | None, split_seed: int, sample_root: Path) -> dict:
    """Load a matching run and drop ids whose files no longer validate."""
    fingerprint = config_fingerprint(spec)
    if not Path(path).is_file():
        return new_manifest(spec, requested=requested, base_seed=base_seed, split_mode=split_mode, split_seed=split_seed)
    manifest = load_manifest(path)
    if manifest.get("config_hash") != fingerprint:
        raise RuntimeError(f"run config hash changed; refusing to resume {path}")
    kept = []
    for scene_id in manifest.get("completed_ids", []):
        if sample_is_valid(Path(sample_root) / scene_id, fingerprint):
            kept.append(scene_id)
    manifest["completed_ids"] = kept
    manifest["requested_successes"] = int(requested)
    manifest["status"] = "running"
    return manifest

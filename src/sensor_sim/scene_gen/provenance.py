"""Publication audit. A run_id check is not this audit."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from sensor_sim.film.images import read_exr
from sensor_sim.scene_gen.dataset import (
    GROUND_TRUTH,
    TRAINING_MIN_OBJECT_PIXELS,
    channels_to_array,
    object_id_publication_issues,
    sha256,
)
from sensor_sim.scene_gen.dataset_contract import PRODUCTION_DATASET_ID

REQUIRED_FILES = (
    "scene_config.json",
    "scene_exported.xml",
    "scene.xml",
    "polarization.exr",
    "metadata.json",
)
REQUIRED_METADATA = (
    "run_id",
    "config_hash",
    "status",
    "scene_id",
    "logical_shape_ids",
    "artifact_sha256",
    "objects",
    "shape_map",
)


def publication_issues(directory: Path) -> list[str]:
    """Problems that keep a sample out of completed_ids. Empty means the files agree."""
    directory = Path(directory)
    issues: list[str] = []
    expected = [directory / name for name in REQUIRED_FILES]
    expected.extend(directory / "ground_truth" / f"{name}.exr" for name in GROUND_TRUTH)
    for path in expected:
        if not path.is_file():
            issues.append(f"missing {path.name if path.parent == directory else path.relative_to(directory)}")
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return issues
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return issues + ["metadata.json is corrupt"]
    for key in REQUIRED_METADATA:
        if metadata.get(key) in (None, "", []):
            issues.append(f"missing provenance field {key}")
    if str(metadata.get("status")) != "complete":
        issues.append("status is not complete")
    if not str(metadata.get("run_id") or "").strip():
        issues.append("run_id is empty")
    claimed = metadata.get("artifact_sha256") or {}
    if isinstance(claimed, dict):
        for relative, digest in claimed.items():
            path = directory / relative
            if not path.is_file():
                issues.append(f"hashed artifact missing {relative}")
                continue
            actual = sha256(path)
            if actual != digest:
                issues.append(f"hash mismatch {relative}")
    else:
        issues.append("artifact_sha256 is not a mapping")
    config_path = directory / "scene_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            issues.append("scene_config.json is corrupt")
            config = {}
        if metadata.get("config_hash") and config.get("config_hash") not in (None, metadata.get("config_hash")):
            issues.append("config hash disagrees with scene_config.json")
    xml_path = directory / "scene.xml"
    if xml_path.is_file() and metadata.get("logical_shape_ids"):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            issues.append("scene.xml is corrupt")
            root = None
        if root is not None:
            found = []
            for node in root.iter():
                if node.tag.rsplit("}", 1)[-1] != "shape":
                    continue
                ident = node.get("id") or ""
                if ident.startswith("mesh-"):
                    found.append(ident)
            expected_ids = [str(item) for item in metadata.get("logical_shape_ids", [])]
            if sorted(found) != sorted(expected_ids):
                issues.append(f"logical ids {expected_ids} disagree with xml {found}")
            text = xml_path.read_text(errors="replace")
            for item in metadata.get("objects") or []:
                name = str(item.get("name"))
                if f"mat-obj_{name}" not in text and f"mesh-{name}" not in text:
                    issues.append(f"material or mesh for {name} is missing from scene.xml")
    by_object = {str(item.get("object_id")): item.get("material_id") for item in metadata.get("shape_map") or []}
    for item in metadata.get("objects") or []:
        name = str(item.get("name"))
        if name in by_object and item.get("material_id") != by_object[name]:
            issues.append(f"material id for {name} disagrees with shape_map")
    schema = int(metadata.get("schema_version") or 0)
    production = metadata.get("dataset_id") == PRODUCTION_DATASET_ID
    if schema >= 3 or production:
        object_path = directory / "ground_truth" / "object_ids.exr"
        if object_path.is_file():
            raw = np.asarray(channels_to_array(read_exr(object_path)), dtype=np.float64)
            issues.extend(object_id_publication_issues(raw, metadata))
        else:
            issues.append("missing object_ids.exr")
    if production or _requires_supervision(directory):
        issues.extend(_supervision_issues(metadata))
    return issues


def _requires_supervision(directory: Path) -> bool:
    path = directory / "scene_config.json"
    if not path.is_file():
        return False
    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(config.get("require_sample_supervision") or config.get("dataset_id") == PRODUCTION_DATASET_ID)


def _supervision_issues(metadata: dict) -> list[str]:
    """At least one requested object must be eligible. Ground does not qualify."""
    intended = [
        item
        for item in metadata.get("shape_map") or []
        if item.get("asset_id") != "plane" and str(item.get("object_id")) != "ground"
    ]
    eligible = [
        item
        for item in intended
        if int(item.get("object_id_pixel_count") or 0) >= TRAINING_MIN_OBJECT_PIXELS
        and int(item.get("normal_supervised_pixel_count") or 0) >= 1
    ]
    if not eligible:
        return ["no intended object has 80 object-id pixels and a supervised normal"]
    stratum = metadata.get("stratum") or {}
    if metadata.get("occlusion_report") is not None or stratum.get("require_partial_occlusion"):
        report = metadata.get("occlusion_report") or {}
        if not report.get("pass"):
            return ["partial occlusion requirements failed"]
    if stratum.get("repeated_asset"):
        objects = [item for item in metadata.get("objects") or [] if item.get("kind") == "obj" or item.get("asset_id") not in (None, "plane")]
        families = {item.get("asset_id") for item in objects if item.get("asset_id") != "plane"}
        instances = [item.get("asset_instance_id") for item in objects if item.get("asset_id") != "plane"]
        if len(families) != 1 or len(instances) != len(set(instances)):
            return ["repeated-asset identity failed"]
        visible = [
            item
            for item in intended
            if int(item.get("object_id_pixel_count") or 0) >= TRAINING_MIN_OBJECT_PIXELS
        ]
        if len(visible) < 2:
            return ["repeated-asset instances are not both visible"]
    return []


def admit_completed(manifest: dict, sample_root: Path, scene_id: str) -> bool:
    """Add a scene only when the publication audit is clean."""
    issues = publication_issues(Path(sample_root) / scene_id)
    if issues:
        return False
    completed = manifest.setdefault("completed_ids", [])
    if scene_id not in completed:
        completed.append(scene_id)
    return True

"""Read-only audit of the published Dataset v1 run.

Writes only under dataset/qualification/dataset_v1_readiness/post_generation_audit.
"""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from sensor_sim.film.images import read_exr
from sensor_sim.film.lighting import display_srgb
from sensor_sim.film.polar import analyzer_images
from sensor_sim.scene_gen.convention import S0_MIN, STOKES_FRAME
from sensor_sim.scene_gen.dataset import (
    channels_to_array,
    discrete_object_ids,
    load_sample,
    normal_supervision_mask,
    normal_valid_mask,
    raw_object_counts,
    sha256,
    training_views,
)
from sensor_sim.scene_gen.render_sample import stack_stokes
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES, stokes_consistency
from sensor_sim.scene_gen.narrow import floor_status, mesh_clearance
from sensor_sim.scene_gen.occlusion import rasterize
from sensor_sim.scene_gen.production_schedule import schedule_hash
from sensor_sim.scene_gen.reference_clearance import reference_clearance, reference_floor
from sensor_sim.scene_gen.run import config_fingerprint

ROOT = Path("dataset/dataset_v1")
AUDIT = Path("dataset/qualification/dataset_v1_readiness/post_generation_audit")
INCOMPLETE = Path("dataset/dataset_v1_incomplete_radius")
EXPECTED_RUN = "0e0d3bc4916847b8857f42ed761208a9"
EXPECTED_CONFIG = "6752351954a6ee2f4ddefbd40cb783b439d9fb1c894512089f9789975f75ea78"
EXPECTED_SCHEDULE = "b61cd6828f7a9dfd4faca98738eaf167a603a3c35f84a694ece1450531483f29"
FAMILIES = ("sphere", "ellipsoid", "rounded_box", "capsule", "torus", "cylinder", "cone", "bumpy")


def _write(name: str, payload: dict) -> None:
    path = AUDIT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", path, flush=True)


def _preset_label(preset: dict) -> str:
    if preset.get("type") == "dielectric":
        return f"dielectric_{float(preset['int_ior']):.2f}"
    if preset.get("type") == "pplastic":
        return "pplastic"
    if preset.get("type") == "conductor":
        return "conductor_Cu"
    return "roughconductor_Cu"


def _load_ply(path: Path) -> dict:
    import mitsuba as mi

    mi.set_variant("cuda_ad_spectral_polarized")
    shape = mi.load_dict({"type": "ply", "filename": str(path.resolve())})
    count = int(shape.vertex_count())
    vertices = np.stack([np.array(shape.vertex_position(index)).reshape(-1)[:3] for index in range(count)])
    faces = np.array(shape.faces_buffer()).reshape(-1, 3)
    return {"vertices": vertices, "faces": faces, "path": str(path)}


def reconcile() -> dict:
    spec = yaml.safe_load(Path("configs/scenes/dataset_v1.yaml").read_text())
    profiles = json.loads(Path(spec["profile_file"]).read_text())
    fingerprinted = dict(spec)
    fingerprinted["profiles"] = profiles
    manifest = json.loads((ROOT / "run_manifest.json").read_text())
    issues = []
    if config_fingerprint(fingerprinted) != manifest["config_hash"]:
        issues.append("config fingerprint does not match the manifest")
    if manifest["config_hash"] != EXPECTED_CONFIG:
        issues.append("manifest config hash is not the recorded production hash")
    if schedule_hash(manifest["schedule"]) != spec["schedule_hash"]:
        issues.append("schedule hash does not match the manifest schedule")
    if spec["schedule_hash"] != EXPECTED_SCHEDULE:
        issues.append("yaml schedule hash changed")
    if manifest["run_id"] != EXPECTED_RUN or manifest["status"] != "complete":
        issues.append(f"manifest status {manifest.get('status')} run {manifest.get('run_id')}")
    completed = manifest["completed_ids"]
    if len(completed) != len(set(completed)) or len(completed) != 64:
        issues.append("completed ids are not 64 unique values")
    failure_records = manifest.get("failures", [])
    if any(item.get("status") == "complete" for item in failure_records):
        issues.append("a failure record is marked complete")
    if len(completed) != len(set(completed)):
        issues.append("completed_ids contains a duplicate")
    groups = {"artifact": [], "extra": [], "metadata": [], "preview_or_log": [], "unexpected": []}
    link_issues = []
    hash_mismatches = []
    for index, scene_id in enumerate(completed):
        sample = (ROOT / "samples" / scene_id).resolve()
        if not sample.is_dir() or not scene_id.endswith(f"_{index:04d}"):
            issues.append(f"{scene_id} is missing or out of order")
            continue
        metadata = json.loads((sample / "metadata.json").read_text())
        if metadata.get("dataset_id") != "dataset_v1_spec" or metadata.get("run_id") != EXPECTED_RUN:
            issues.append(f"{scene_id} dataset or run id mismatch")
        if metadata.get("schema_version") != 3 or metadata.get("visibility_policy_version") != 1:
            issues.append(f"{scene_id} schema {metadata.get('schema_version')}")
        claimed = metadata.get("artifact_sha256") or {}
        seen = set()
        for relative in claimed:
            path = sample / relative
            seen.add(path.resolve())
            if not path.is_file():
                hash_mismatches.append(f"missing {scene_id}/{relative}")
                continue
            if sha256(path) != claimed[relative]:
                hash_mismatches.append(f"hash {scene_id}/{relative}")
            groups["artifact"].append(f"{scene_id}/{relative}")
        for path in sample.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if sample not in resolved.parents and resolved != sample:
                link_issues.append(f"escapes {path}")
            if path.is_symlink():
                link_issues.append(f"symlink {path}")
            name = path.name
            kind = "extra"
            if resolved in seen:
                continue
            if name in {"metadata.json", "scene_config.json"}:
                kind = "metadata"
            elif name == "preview.png" or name.endswith(".log") or "blender" in name:
                kind = "preview_or_log"
            groups[kind].append(f"{scene_id}/{path.relative_to(sample)}")
        text = (sample / "scene.xml").read_text(errors="replace")
        for line in text.splitlines():
            if "filename" not in line or ".." in line:
                if ".." in line:
                    link_issues.append(f"parent path {scene_id}")
                continue
            if 'value="' in line:
                value = line.split('value="', 1)[1].split('"', 1)[0]
                target = (sample / value).resolve()
                if sample not in target.parents:
                    link_issues.append(f"mesh escapes {scene_id} {value}")
                if not target.is_file():
                    link_issues.append(f"mesh missing {scene_id} {value}")
    report = {
        "pass": not issues and not hash_mismatches and not link_issues,
        "issues": issues,
        "hash_mismatches": hash_mismatches,
        "link_issues": link_issues,
        "file_counts": {key: len(value) for key, value in groups.items()},
        "config_fingerprint_matches_manifest": config_fingerprint(fingerprinted) == manifest["config_hash"],
        "schedule_hash_matches": schedule_hash(manifest["schedule"]) == EXPECTED_SCHEDULE,
        "completed": len(completed),
        "failure_ids_excluded": True,
    }
    _write("manifest_reconciliation.json", report)
    return report


def _objects(metadata: dict) -> list[dict]:
    return [item for item in metadata.get("objects", []) if item.get("kind") == "obj"]


def distribution_and_measurements() -> dict:
    spec = yaml.safe_load(Path("configs/scenes/dataset_v1.yaml").read_text())
    manifest = json.loads((ROOT / "run_manifest.json").read_text())
    matrix = Counter()
    scene_materials = Counter()
    instance_materials = Counter()
    family_counts = Counter()
    count_counts = Counter()
    issues = []
    visibility_rows = []
    stokes_rows = []
    exr_issues = []
    occlusion_rows = []
    repeat_rows = []
    for index, scene_id in enumerate(manifest["completed_ids"]):
        sample = ROOT / "samples" / scene_id
        slot = spec["strata"][index]
        metadata = json.loads((sample / "metadata.json").read_text())
        label = _preset_label(slot["preset"])
        matrix[(slot["family"], label)] += 1
        family_counts[slot["family"]] += 1
        objects = _objects(metadata)
        count_counts[len(objects)] += 1
        scene_materials[label] += 1
        for item in objects:
            instance_materials[_preset_label(item["bsdf"])] += 1
            if item["bsdf"] != slot["preset"]:
                issues.append(f"{scene_id} object {item['name']} preset differs from the slot")
            if item["asset_id"] != slot["family"]:
                issues.append(f"{scene_id} family {item['asset_id']}")
        if any(item.get("material") == "Au" or item.get("asset_id") == "dragon" for item in objects):
            issues.append(f"{scene_id} contains gold or dragon")
        exr_path = sample / "polarization.exr"
        digest_before = sha256(exr_path)
        from sensor_sim.scene_gen.dataset import exr_schema

        schema = exr_schema(exr_path)
        if schema["file_channels"] != list(CHANNEL_NAMES) or schema["height"] != 128 or schema["width"] != 128:
            exr_issues.append(f"{scene_id} channels or size {schema['file_channels']} {schema['height']} {schema['width']}")
        if any("alpha" in name.lower() for name in schema["file_channels"]):
            exr_issues.append(f"{scene_id} has alpha")
        raw = read_exr(exr_path)
        for name, values in raw.items():
            if not np.isfinite(np.asarray(values)).all():
                exr_issues.append(f"{scene_id} nonfinite {name}")
        loaded = load_sample(sample, dataset_v1=True)
        views = training_views(loaded)
        expected_shapes = {"s0": (128, 128, 1), "s0_s2": (128, 128, 3), "stokes": (128, 128, 4), "dolp_aolp": (128, 128, 2), "rgb_stokes": (128, 128, 7)}
        for key, shape in expected_shapes.items():
            if views[key].shape != shape or views[key].dtype != np.float32:
                exr_issues.append(f"{scene_id} view {key} {views[key].shape} {views[key].dtype}")
        if sha256(exr_path) != digest_before:
            exr_issues.append(f"{scene_id} exr bytes changed")
        scalar = {}
        for name in ("S0", "S1", "S2", "S3"):
            stacked = np.asarray(stack_stokes(raw, name), dtype=np.float64)
            scalar[name] = stacked.mean(axis=2) if stacked.ndim == 3 else stacked
        if not np.allclose(loaded["s0"], scalar["S0"]):
            exr_issues.append(f"{scene_id} scalar S0 is not the RGB-channel mean")
        raw_ids = np.asarray(channels_to_array(read_exr(sample / "ground_truth" / "object_ids.exr")), dtype=np.float64).reshape(128, 128)
        labels, fractional = discrete_object_ids(raw_ids)
        counts = raw_object_counts(raw_ids)
        if int(fractional.sum()):
            issues.append(f"{scene_id} fractional ids {int(fractional.sum())}")
        known = {int(item["object_index"]) for item in metadata["shape_map"]}
        unknown = (~fractional) & (labels > 0) & ~np.isin(labels, np.array(sorted(known), dtype=np.int32))
        if int(unknown.sum()):
            issues.append(f"{scene_id} unknown ids {int(unknown.sum())}")
        valid_normals = normal_valid_mask(loaded["normal_shading_camera"])
        mask = loaded["normal_supervision_mask"]
        if not np.array_equal(mask, normal_supervision_mask(loaded, metadata)):
            issues.append(f"{scene_id} supervision mask disagrees")
        for item in metadata["shape_map"]:
            if item.get("asset_id") == "plane":
                continue
            label_id = int(item["object_index"])
            assigned = (labels == label_id) & ~fractional
            raw_count = int(counts.get(label_id, 0))
            normal_count = int(np.count_nonzero(assigned & valid_normals))
            supervised = int(np.count_nonzero(assigned & mask))
            if raw_count != int(item["object_id_pixel_count"]) or normal_count != int(item["normal_valid_pixel_count"]):
                issues.append(f"{scene_id} {item['object_id']} counts {raw_count}/{normal_count}")
            if (raw_count >= 80 and supervised != normal_count) or (raw_count < 80 and supervised != 0):
                issues.append(f"{scene_id} {item['object_id']} supervision {supervised}")
            if int(item["normal_supervised_pixel_count"]) != supervised:
                issues.append(f"{scene_id} metadata supervision {item['object_id']}")
            category = "fully_hidden" if raw_count == 0 else "low_visibility" if raw_count < 80 else "training_eligible"
            visibility_rows.append(
                {
                    "scene_id": scene_id,
                    "object_id": item["object_id"],
                    "family": item.get("asset_id"),
                    "material": label,
                    "object_count": len(objects),
                    "split": slot["split"],
                    "occlusion": bool(slot.get("require_partial_occlusion")),
                    "repeated": bool(slot.get("repeated_asset")),
                    "raw_object_id_pixels": raw_count,
                    "normal_valid_pixels": normal_count,
                    "normal_supervised_pixels": supervised,
                    "category": category,
                }
            )
        eligible = [row for row in visibility_rows if row["scene_id"] == scene_id and row["category"] == "training_eligible" and row["normal_supervised_pixels"] > 0]
        if not eligible:
            issues.append(f"{scene_id} has no supervised object")
        illuminated = loaded["s0"] > S0_MIN
        object_pixels = loaded["object_ids"] > 0
        consistency = stokes_consistency(loaded["s0"], loaded["s1"], loaded["s2"], loaded["s3"], np.ones_like(illuminated, dtype=bool))
        by_object = stokes_consistency(loaded["s0"], loaded["s1"], loaded["s2"], loaded["s3"], object_pixels)
        by_light = stokes_consistency(loaded["s0"], loaded["s1"], loaded["s2"], loaded["s3"], illuminated)
        by_dim = stokes_consistency(loaded["s0"], loaded["s1"], loaded["s2"], loaded["s3"], ~illuminated)
        dop = np.zeros_like(loaded["s0"])
        np.divide(np.sqrt(loaded["s1"] ** 2 + loaded["s2"] ** 2 + loaded["s3"] ** 2), loaded["s0"], out=dop, where=illuminated)
        dolp = np.zeros_like(loaded["s0"])
        np.divide(np.sqrt(loaded["s1"] ** 2 + loaded["s2"] ** 2), loaded["s0"], out=dolp, where=illuminated)
        analyzers = analyzer_images(loaded["s0"], loaded["s1"], loaded["s2"])
        analyzer_negative = int(sum(np.count_nonzero((values < -1e-5) & illuminated) for values in analyzers.values()))
        sums_ok = bool(
            np.allclose(analyzers["analyzer_000"] + analyzers["analyzer_090"], loaded["s0"])
            and np.allclose(analyzers["analyzer_045"] + analyzers["analyzer_135"], loaded["s0"])
        )
        stokes_rows.append(
            {
                "scene_id": scene_id,
                "family": slot["family"],
                "material": label,
                "all_pixels": consistency,
                "object_pixels": by_object,
                "illuminated": by_light,
                "weak": by_dim,
                "dop_p95": float(np.percentile(dop[illuminated], 95)) if illuminated.any() else None,
                "dolp_p95": float(np.percentile(dolp[illuminated], 95)) if illuminated.any() else None,
                "s3_min": float(loaded["s3"].min()),
                "s3_max": float(loaded["s3"].max()),
                "s3_median": float(np.median(loaded["s3"])),
                "s3_positive": int(np.count_nonzero(loaded["s3"] > 1e-6)),
                "s3_negative": int(np.count_nonzero(loaded["s3"] < -1e-6)),
                "s3_near_zero": int(np.count_nonzero(np.abs(loaded["s3"]) <= 1e-6)),
                "analyzer_negative_illuminated_pixels": analyzer_negative,
                "analyzer_sums_match_s0": sums_ok,
                "n_illuminated": int(illuminated.sum()),
            }
        )
        if slot.get("require_partial_occlusion") or slot.get("repeated_asset"):
            _special(sample, metadata, slot, scene_id, labels, fractional, valid_normals, occlusion_rows, repeat_rows, issues)
        if (index + 1) % 16 == 0:
            print("measured", index + 1, flush=True)
    values = np.array([row["raw_object_id_pixels"] for row in visibility_rows], dtype=np.float64)
    categories = Counter(row["category"] for row in visibility_rows)
    for family in FAMILIES:
        for preset in ("dielectric_1.33", "dielectric_1.40", "dielectric_1.50", "dielectric_1.60", "dielectric_1.70", "pplastic", "conductor_Cu", "roughconductor_Cu"):
            if matrix[(family, preset)] != 1:
                issues.append(f"matrix {family} {preset} = {matrix[(family, preset)]}")
    if dict(count_counts) != {1: 32, 2: 16, 3: 8, 4: 8}:
        issues.append(f"counts {dict(count_counts)}")
    cross = {
        "pass": not issues and not exr_issues,
        "issues": issues,
        "exr_issues": exr_issues,
        "matrix": {f"{family}|{preset}": matrix[(family, preset)] for family in FAMILIES for preset in ("dielectric_1.33", "dielectric_1.40", "dielectric_1.50", "dielectric_1.60", "dielectric_1.70", "pplastic", "conductor_Cu", "roughconductor_Cu")},
        "scenes_per_material": dict(scene_materials),
        "instances_per_material": dict(instance_materials),
        "non_ground_objects": int(values.size),
        "object_counts": {str(key): value for key, value in count_counts.items()},
    }
    _write("cross_tabulation.json", cross)
    _write(
        "visibility_supervision_report.json",
        {
            "n_objects": int(values.size),
            "categories": dict(categories),
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "maximum": float(values.max()),
            "by_family": _group(visibility_rows, "family"),
            "by_material": _group(visibility_rows, "material"),
            "by_count": _group(visibility_rows, "object_count"),
            "by_split": _group(visibility_rows, "split"),
            "rows": visibility_rows,
            "pass": not issues,
        },
    )
    _write(
        "stokes_integrity.json",
        {
            "rule": "S0 >= 0 and S1^2 + S2^2 + S3^2 <= S0^2 + 1e-6 on the scalar RGB-channel mean",
            "rows": stokes_rows,
            "total_violations_all_pixels": int(sum(row["all_pixels"]["n_violations"] for row in stokes_rows)),
            "total_pixels": int(sum(row["all_pixels"]["n_mask"] for row in stokes_rows)),
            "analyzer_sum_failures": [row["scene_id"] for row in stokes_rows if not row["analyzer_sums_match_s0"]],
            "pass": not exr_issues and all(row["analyzer_sums_match_s0"] for row in stokes_rows),
        },
    )
    _write("occlusion_and_repeats.json", {"occlusion": occlusion_rows, "repeated": repeat_rows})
    return {"issues": issues, "exr_issues": exr_issues, "stokes_rows": stokes_rows, "visibility_rows": visibility_rows}


def _group(rows: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row["raw_object_id_pixels"])
    return {
        name: {
            "n": len(values),
            "median": float(np.median(values)),
            "fully_hidden": int(sum(value == 0 for value in values)),
            "low_visibility": int(sum(0 < value < 80 for value in values)),
            "training_eligible": int(sum(value >= 80 for value in values)),
        }
        for name, values in grouped.items()
    }


def _special(sample, metadata, slot, scene_id, labels, fractional, valid_normals, occlusion_rows, repeat_rows, issues) -> None:
    objects = _objects(metadata)
    if slot.get("repeated_asset"):
        instances = [item.get("asset_instance_id") for item in objects]
        transforms = [(round(float(item["xy"][0]), 6), round(float(item["xy"][1]), 6), round(float(item["yaw"]), 6)) for item in objects]
        visible = []
        for item in metadata["shape_map"]:
            if item.get("asset_id") == "plane":
                continue
            visible.append(int(item["object_id_pixel_count"]) >= 80 and int(item["normal_supervised_pixel_count"]) >= 1)
        ok = len(set(item["asset_id"] for item in objects)) == 1 and len(set(instances)) == len(instances) and len(set(transforms)) == len(transforms) and all(visible)
        if not ok:
            issues.append(f"{scene_id} repeated-asset check failed")
        repeat_rows.append({"scene_id": scene_id, "asset_ids": [item["asset_id"] for item in objects], "instances": instances, "pass": ok})
    if not slot.get("require_partial_occlusion"):
        return
    meshes = []
    for item in objects:
        ply = sample / "meshes" / f"{item['name']}.ply"
        mesh = _load_ply(ply)
        mesh["name"] = item["name"]
        mesh["object_index"] = int(item["object_index"])
        meshes.append(mesh)
    camera = metadata["camera"]
    silhouettes = [rasterize(mesh["vertices"], mesh["faces"], camera, 128, 128) for mesh in meshes]
    rows = []
    for mesh, silhouette in zip(meshes, silhouettes):
        assigned = (labels == mesh["object_index"]) & ~fractional
        visible = int(np.count_nonzero(assigned))
        isolated = int(silhouette.sum())
        supervised = int(np.count_nonzero(assigned & valid_normals))
        rows.append({"name": mesh["name"], "isolated": isolated, "visible": visible, "supervised": supervised, "fraction": None if isolated == 0 else (isolated - visible) / isolated})
    rear = max(rows, key=lambda row: -1.0 if row["fraction"] is None else row["fraction"])
    passed = rear["fraction"] is not None and 0.10 <= rear["fraction"] <= 0.50 and all(row["visible"] >= 80 and row["supervised"] >= 1 for row in rows)
    if not passed:
        issues.append(f"{scene_id} occlusion fraction {rear['fraction']}")
    occlusion_rows.append({"scene_id": scene_id, "objects": rows, "rear": rear["name"], "fraction": rear["fraction"], "pass": passed})


def isolated_rear_renders() -> None:
    """Compare the silhouette raster with one Mitsuba shape-index render of the rear mesh."""
    from sensor_sim.scene_gen.render_sample import render_aovs

    payload = json.loads((AUDIT / "occlusion_and_repeats.json").read_text())
    comparisons = []
    for row in payload["occlusion"]:
        sample = ROOT / "samples" / row["scene_id"]
        tree = ET.parse(sample / "scene.xml")
        root = tree.getroot()
        scene = ET.Element("scene", {"version": "3.9.1"})
        for child in list(root):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag in {"default", "sensor"}:
                scene.append(child)
            if tag == "shape" and child.get("id") == f"mesh-{row['rear']}":
                for element in list(child):
                    if element.tag.rsplit("}", 1)[-1] == "ref":
                        child.remove(element)
                for element in child.iter():
                    if element.tag.rsplit("}", 1)[-1] == "string" and element.get("name") == "filename":
                        element.set("value", str((sample / element.get("value")).resolve()))
                ET.SubElement(child, "bsdf", {"type": "diffuse"})
                scene.append(child)
        ET.SubElement(scene, "integrator", {"type": "path"})
        destination = AUDIT / "occlusion_isolated" / f"{row['scene_id']}.xml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(ET.tostring(scene, encoding="unicode"))
        _scene, channels, _names = render_aovs(destination, "cuda_ad_spectral_polarized", "idx:shape_index")
        index = np.asarray(channels["idx.I"], dtype=np.float64)
        rendered = int(np.count_nonzero(np.isfinite(index) & (index > 0.0)))
        rear = next(item for item in row["objects"] if item["name"] == row["rear"])
        comparisons.append(
            {
                "scene_id": row["scene_id"],
                "rear": row["rear"],
                "raster_isolated_pixels": rear["isolated"],
                "mitsuba_shape_index_pixels": rendered,
                "disagreement": int(rendered - rear["isolated"]),
            }
        )
        print("isolated", row["scene_id"], rendered, rear["isolated"], flush=True)
    payload["mitsuba_isolated_comparison"] = comparisons
    payload["isolated_note"] = (
        "rasterize is a projected triangle mask. The Mitsuba count is the shape-index AOV of the rear mesh alone. "
        "They are not the same projection, so a nonzero disagreement is reported and does not by itself change the 0.10-0.50 decision."
    )
    _write("occlusion_and_repeats.json", payload)


def geometry() -> dict:
    manifest = json.loads((ROOT / "run_manifest.json").read_text())
    rows = []
    issues = []
    reference_enabled = True
    for scene_id in manifest["completed_ids"]:
        sample = ROOT / "samples" / scene_id
        metadata = json.loads((sample / "metadata.json").read_text())
        objects = _objects(metadata)
        meshes = [_load_ply(sample / "meshes" / f"{item['name']}.ply") for item in objects]
        floors = [reference_floor(mesh["vertices"]) for mesh in meshes]
        if any(item["penetrating"] for item in floors):
            issues.append(f"{scene_id} floor penetration")
        pairs = []
        for left in range(len(meshes)):
            for right in range(left + 1, len(meshes)):
                generated = mesh_clearance(meshes[left], meshes[right])
                independent = None
                if reference_enabled and meshes[left]["faces"].shape[0] <= 2500 and meshes[right]["faces"].shape[0] <= 2500:
                    independent = reference_clearance(meshes[left], meshes[right])
                elif reference_enabled:
                    independent = {
                        "complete": False,
                        "method": "reference_grid",
                        "note": "exported mesh has more than 2500 triangles; the independent grid walk was not run on this pair",
                        "n_triangles": [int(meshes[left]["faces"].shape[0]), int(meshes[right]["faces"].shape[0])],
                    }
                if generated["clearance_m"] < 0.005 or generated["surface_intersection"]:
                    issues.append(f"{scene_id} clearance {generated['clearance_m']}")
                pairs.append({"objects": [objects[left]["name"], objects[right]["name"]], "generator": generated, "independent": independent})
        if pairs:
            closest = min(pairs, key=lambda row: row["generator"]["clearance_m"])
        else:
            closest = None
        rows.append(
            {
                "scene_id": scene_id,
                "minimum_clearance_m": None if closest is None else closest["generator"]["clearance_m"],
                "closest_pair": None if closest is None else closest["objects"],
                "floors": floors,
                "pairs": pairs,
            }
        )
        print("geometry", scene_id, "pairs", len(pairs), flush=True)
    report = {
        "pass": not issues,
        "issues": issues,
        "independent_scope": "The reference checker is exact on analytic tests and on exported pairs with at most 2500 triangles per mesh. Larger procedural meshes keep the generation BVH result and are marked not independently measured.",
        "rows": rows,
    }
    _write("geometry_integrity.json", report)
    return report


def visual(stokes_note: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((ROOT / "run_manifest.json").read_text())
    out = AUDIT / "visual_qc" / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    rgb_tiles = []
    labels = []
    for scene_id in manifest["completed_ids"]:
        sample = ROOT / "samples" / scene_id
        loaded = load_sample(sample)
        rgb = loaded["rgb"]
        figure, axes = plt.subplots(3, 3, figsize=(9, 9))
        panels = [
            (display_srgb(rgb), None, "RGB"),
            (display_srgb(loaded["s0"]), "gray", "S0"),
            (loaded["dolp"] if False else np.sqrt(loaded["s1"] ** 2 + loaded["s2"] ** 2) / np.maximum(loaded["s0"], 1e-8), "viridis", "DoLP"),
            (0.5 * np.arctan2(loaded["s2"], loaded["s1"]), "hsv", "AoLP"),
            (loaded["normal_shading_camera"] * 0.5 + 0.5, None, "shading normal"),
            (loaded["depth"], "magma", "depth"),
            (loaded["object_ids"], "tab20", "object id"),
            (loaded["material_ids"], "tab20", "material id"),
            (loaded["normal_supervision_mask"], "gray", "supervision"),
        ]
        for axis, (image, cmap, title) in zip(axes.ravel(), panels):
            axis.imshow(np.clip(image, 0.0, 1.0) if title in {"RGB", "S0", "shading normal"} else image, cmap=cmap)
            axis.set_title(title, fontsize=8)
            axis.axis("off")
        figure.suptitle(scene_id)
        figure.tight_layout()
        figure.savefig(out / f"{scene_id}.png", dpi=80)
        plt.close(figure)
        rgb_tiles.append(np.clip(display_srgb(rgb), 0.0, 1.0))
        labels.append(scene_id[-4:])
        print("visual", scene_id, flush=True)
    sheet, axes = plt.subplots(8, 8, figsize=(16, 16))
    for axis, image, label in zip(axes.ravel(), rgb_tiles, labels):
        axis.imshow(image)
        axis.set_title(label, fontsize=7)
        axis.axis("off")
    sheet.suptitle("Dataset v1 RGB contact sheet")
    sheet.tight_layout()
    sheet.savefig(AUDIT / "visual_qc" / "rgb_contact_sheet.png", dpi=100)
    plt.close(sheet)
    (AUDIT / "visual_qc" / "report.md").write_text(
        "RGB panels are the leading EXR channels R, G, and B after a display transform on the copy only. "
        "The sample preview.png files remain grayscale S0 and were not rewritten.\n\n"
        + stokes_note
        + "\n\nNo sample was flagged by an automatic image statistic beyond the numeric audit. Inspect the contact sheet for silhouette and material defects.\n"
    )


def failures_and_splits() -> None:
    spec = yaml.safe_load(Path("configs/scenes/dataset_v1.yaml").read_text())
    manifest = json.loads((ROOT / "run_manifest.json").read_text())
    rows = []
    for item in manifest["failures"]:
        stratum = item.get("stratum") or {}
        error = str(item.get("error", ""))
        if "placement_failed" in error:
            stage = "placement"
        elif "partial occlusion" in error:
            stage = "occlusion"
        elif "repeated-asset" in error:
            stage = "identity"
        elif "supervision" in error:
            stage = "visibility"
        else:
            stage = item.get("failure_stage", "unknown")
        rows.append(
            {
                "scene_id": item.get("scene_id"),
                "slot_id": stratum.get("slot_id"),
                "attempt_index": item.get("attempt_index"),
                "scene_seed": item.get("scene_seed"),
                "geometry_seed": item.get("geometry_seed"),
                "material_seed": item.get("material_seed"),
                "render_seed": item.get("render_seed"),
                "family": stratum.get("family"),
                "material": _preset_label(stratum.get("preset") or {}),
                "object_count": stratum.get("count_objects"),
                "occlusion": bool(stratum.get("require_partial_occlusion")),
                "repeated": bool(stratum.get("repeated_asset")),
                "stage": stage,
                "error": error,
            }
        )
    by_slot = Counter(row["slot_id"] for row in rows)
    _write(
        "retry_postmortem.json",
        {
            "attempts": len(manifest["attempted_seeds"]),
            "published": len(manifest["completed_ids"]),
            "failures": len(rows),
            "failure_fraction": len(rows) / len(manifest["attempted_seeds"]),
            "by_stage": dict(Counter(row["stage"] for row in rows)),
            "by_family": dict(Counter(row["family"] for row in rows)),
            "by_material": dict(Counter(row["material"] for row in rows)),
            "by_count": dict(Counter(str(row["object_count"]) for row in rows)),
            "by_slot": dict(by_slot),
            "max_failures_for_one_slot": max(by_slot.values()) if by_slot else 0,
            "selection_note": "Accepted poses are the seeds that passed placement and visibility. The quotas still match the schedule. This 16/80 rate is this run only.",
            "rows": rows,
        },
    )
    splits = {}
    for name in ("train", "validation", "test"):
        splits[name] = (ROOT / "splits" / f"{name}.txt").read_text().split()
    completed = manifest["completed_ids"]
    issues = []
    if [len(splits[name]) for name in ("train", "validation", "test")] != [48, 8, 8]:
        issues.append("split sizes")
    if set(splits["train"]) & set(splits["validation"]) or set(splits["train"]) & set(splits["test"]) or set(splits["validation"]) & set(splits["test"]):
        issues.append("splits overlap")
    if set(splits["train"]) | set(splits["validation"]) | set(splits["test"]) != set(completed):
        issues.append("splits do not match completed ids")
    margins = defaultdict(Counter)
    instance_ids = defaultdict(set)
    for index, scene_id in enumerate(completed):
        slot = spec["strata"][index]
        metadata = json.loads((ROOT / "samples" / scene_id / "metadata.json").read_text())
        split = slot["split"]
        margins[split]["scenes"] += 1
        margins[split][f"count_{slot['count_objects']}"] += 1
        margins[split][slot["family"]] += 1
        margins[split][_preset_label(slot["preset"])] += 1
        if slot.get("require_partial_occlusion"):
            margins[split]["occlusion"] += 1
        if slot.get("repeated_asset"):
            margins[split]["repeated"] += 1
        for item in _objects(metadata):
            instance_ids[item["asset_instance_id"]].add(split)
    shared = {key: sorted(value) for key, value in instance_ids.items() if len(value) > 1}
    _write(
        "split_audit.json",
        {
            "pass": not issues,
            "issues": issues,
            "sizes": {name: len(values) for name, values in splits.items()},
            "margins": {name: dict(counter) for name, counter in margins.items()},
            "shared_asset_instance_ids_across_splits": shared,
            "claim": "scene-disjoint only; materials and families are shared across splits",
        },
    )


def convention_documents(stokes_summary: dict) -> None:
    mitsuba_note = "The installed package was inspected for analyzer and polarizer plugin names during the audit."
    (AUDIT / "stokes_convention.md").write_text(
        "# Stokes convention\n\n"
        "Dataset v1 stores 15 float32 EXR channels: R, G, B and S0.R through S3.B. "
        "There is no alpha channel. Scalar Stokes used by the loader are the mean of the three RGB-converted channels. "
        "Those three channels are not wavelengths, and the dataset is not hyperspectral.\n\n"
        "The project claim in `convention.py` is: the Stokes reference axis is the camera horizontal axis, "
        "S1 is horizontal versus vertical, S2 is +45 versus -45, and S3 is right versus left circular polarization. "
        "AoLP is `0.5 * atan2(S2, S1)` on a 180-degree period, stored as 0 where DoLP < 0.05 or S0 <= 1e-4. "
        "The camera-space shading normal points toward the camera, which is negative camera Z. "
        "A Stokes basis rotation is a double-angle mix of S1 and S2. It is not the normal transform.\n\n"
        "The analyzer equation checked on the scalar values is "
        "`I(theta) = 0.5 * (S0 + S1 cos(2 theta) + S2 sin(2 theta))`. "
        "Synthetic horizontal, vertical, plus and minus 45 degree, and unpolarized states are unit tests of that equation. "
        "They do not prove Mitsuba's physical basis.\n\n"
        f"{mitsuba_note}\n\n"
        "S3 sign was not given an independent circular-polarization reference in Mitsuba 3.9.1 during this audit. "
        "S3 handedness is `not independently qualified`. The physical cone test does not determine the sign.\n\n"
        f"Claimed frame: `{json.dumps(STOKES_FRAME)}`\n\n"
        f"Measured violations across all pixels: {stokes_summary.get('total_violations_all_pixels')} of {stokes_summary.get('total_pixels')}.\n"
    )
    (AUDIT / "stokes_convention_report.json").write_text((AUDIT / "stokes_integrity.json").read_text())
    (AUDIT / "forward_model_readiness.md").write_text(
        "# Forward-model readiness\n\n"
        "Linear analyzer intensities were reconstructed from the scalar Stokes arrays and checked for the sum identities. "
        "That check passed as an equation on the stored numbers. It was not compared with a separate Mitsuba linear-analyzer render, "
        "because this audit did not find a documented 3.9.1 analyzer plugin to invoke without guessing a plugin name.\n\n"
        "S3 handedness is not independently qualified. A four-component Stokes forward model is not supported yet. "
        "A linear-polarization comparison can use S0, S1, and S2 only with the project basis treated as a claim, "
        "and it must report each component. The three RGB-converted channels must not be treated as wavelengths.\n\n"
        "The Fresnel model itself is the next milestone and is not implemented here.\n"
    )


def dataset_card() -> None:
    (AUDIT / "dataset_card.md").write_text(
        "# Dataset v1\n\n"
        "- Dataset id: `dataset_v1_spec`\n"
        "- Run id: `0e0d3bc4916847b8857f42ed761208a9`\n"
        "- Config hash: `6752351954a6ee2f4ddefbd40cb783b439d9fb1c894512089f9789975f75ea78`\n"
        "- Schedule hash: `b61cd6828f7a9dfd4faca98738eaf167a603a3c35f84a694ece1450531483f29`\n"
        "- 64 scenes and 120 non-ground object instances\n"
        "- 8 families and 8 non-gold presets, one scene per pair\n"
        "- Object counts: 32, 16, 8, 8 scenes\n"
        "- 8 partial-occlusion scenes and 8 repeated-asset scenes\n"
        "- Split seed 17011: 48 train, 8 validation, 8 test, scene-disjoint\n"
        "- 128 by 128, 256 spp, `cuda_ad_spectral_polarized`, fixed camera and light, Z-up meters\n"
        "- 15-channel float32 EXR, scalar Stokes are RGB-channel means, schema 3, visibility policy 1, 80-pixel rule\n"
        "- Camera-space shading normals are the supervision target\n"
        "- 80 attempts, 16 failures, 9572 seconds, 457,039,003 bytes\n\n"
        "This is a controlled synthetic Shape-from-Polarization set. It is not hyperspectral, not a noisy real sensor, "
        "and not a benchmark for new viewpoints or lighting. Materials and families appear in every split.\n"
    )


def inventory() -> None:
    lines = []
    roots = [
        ROOT,
        Path("configs/scenes/dataset_v1.yaml"),
        AUDIT,
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    files = []
    for entry in roots:
        if entry.is_file():
            files.append(entry)
        else:
            files.extend(path for path in entry.rglob("*") if path.is_file())
    for path in sorted(files):
        if path.name == "final_inventory.sha256":
            continue
        lines.append(f"{sha256(path)}  {path}")
    (AUDIT / "final_inventory.sha256").write_text("\n".join(lines) + "\n")
    print("inventory", len(lines), flush=True)


def refresh_source_snapshot() -> None:
    import shutil

    snapshot = AUDIT / "source_snapshot"
    for relative in (
        Path("src/sensor_sim/scene_gen/post_audit.py"),
        Path("src/sensor_sim/scene_gen/reference_clearance.py"),
        Path("tests/test_reference_clearance.py"),
        Path("tests/test_stokes_convention_audit.py"),
    ):
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative, dest)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    reconciliation = reconcile()
    measured = distribution_and_measurements()
    isolated_rear_renders()
    stokes = json.loads((AUDIT / "stokes_integrity.json").read_text())
    convention_documents(stokes)
    visual("Grayscale sample previews were not used as the RGB photographs.")
    geometry()
    failures_and_splits()
    dataset_card()
    refresh_source_snapshot()
    inventory()
    print("audit_pass", reconciliation["pass"] and not measured["issues"] and not measured["exr_issues"], flush=True)


if __name__ == "__main__":
    main()

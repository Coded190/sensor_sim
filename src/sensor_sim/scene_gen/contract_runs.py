"""Qualification renders for the Dataset v1 contract. These are not production samples."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from sensor_sim.film.images import read_exr
from sensor_sim.scene_gen.dataset import channels_to_array, load_sample, raw_object_counts, training_views
from sensor_sim.scene_gen.generate import PROTOTYPE_CAMERA, PROTOTYPE_LIGHT, generate_sample
from sensor_sim.scene_gen.narrow import floor_status, mesh_clearance
from sensor_sim.scene_gen.placement import FLOOR_Z

ROOT = Path("dataset/qualification")
FINAL = ROOT / "dataset_v1_readiness" / "contract_finalization"
DIELECTRIC = {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0}


def visibility_histogram() -> dict:
    """Image-count histogram at 128x128. Ownership metadata is not the measurement."""
    rows = []
    for path in sorted(ROOT.rglob("ground_truth/object_ids.exr")):
        if "contract_finalization" in path.parts:
            continue
        raw = np.asarray(channels_to_array(read_exr(path)), dtype=np.float64)
        if raw.shape[:2] != (128, 128):
            continue
        metadata_path = path.parents[1] / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
        counts = raw_object_counts(raw)
        if metadata is None:
            for label, count in counts.items():
                rows.append({"sample": path.parents[1].name, "object_index": int(label), "object_id_pixel_count": int(count), "source": "image_only"})
            continue
        for item in metadata.get("shape_map") or metadata.get("objects") or []:
            if item.get("asset_id") == "plane" or str(item.get("object_id") or item.get("name")) == "ground":
                continue
            label = int(item.get("object_index") or 0)
            if label <= 0:
                continue
            rows.append(
                {
                    "sample": metadata.get("scene_id", path.parents[1].name),
                    "object_id": item.get("object_id") or item.get("name"),
                    "object_index": label,
                    "object_id_pixel_count": int(counts.get(label, 0)),
                    "legacy_visible_pixels": item.get("visible_pixels"),
                    "schema_version": metadata.get("schema_version"),
                }
            )
    values = np.array([row["object_id_pixel_count"] for row in rows], dtype=np.float64)
    summary = {
        "definition": "pixels in the saved object-id image assigned to a non-ground logical id; fractional ids are excluded",
        "legacy_definition_not_used": "visible_pixels was the position-ownership count on schema 2 samples and does not validate this histogram",
        "resolution": [128, 128],
        "threshold_pixels": 80,
        "threshold_changed": False,
        "n_objects": int(values.size),
        "n_samples": len({row["sample"] for row in rows}),
        "minimum": None if values.size == 0 else float(values.min()),
        "median": None if values.size == 0 else float(np.median(values)),
        "p10": None if values.size == 0 else float(np.percentile(values, 10)),
        "p90": None if values.size == 0 else float(np.percentile(values, 90)),
        "fully_hidden": int(np.count_nonzero(values <= 0)),
        "below_80": int(np.count_nonzero((values >= 1) & (values < 80))),
        "at_least_80": int(np.count_nonzero(values >= 80)),
    }
    destination = FINAL / "visibility_histogram.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print("histogram", json.dumps(summary), flush=True)
    return summary


def _scene(scene_id: str, objects: list[dict], spp: int, run_id: str, dataset_id: str) -> dict:
    from sensor_sim.scene_gen.qualify_v1 import _config

    config = _config(scene_id, objects, resolution=(128, 128), spp=spp)
    config["run_id"] = run_id
    config["dataset_id"] = dataset_id
    config["generation_authorized"] = False
    config["camera"] = dict(PROTOTYPE_CAMERA)
    config["light"] = dict(PROTOTYPE_LIGHT)
    return config


def hidden_diagnostic() -> dict:
    from sensor_sim.film.shape_mesh import place_mesh
    from sensor_sim.scene_gen.assets import build_asset
    from sensor_sim.scene_gen.qualify_v1 import _object

    front_mesh, _spec = build_asset("sphere", np.random.default_rng(2), radius=0.07)
    back_mesh, _spec = build_asset("sphere", np.random.default_rng(3), radius=0.015)
    front = place_mesh(front_mesh["vertices"], 0.0, (0.0, -0.01))
    back = place_mesh(back_mesh["vertices"], 0.0, (0.0, 0.12))
    back = back.copy()
    back[:, 2] += float(front[:, 2].mean() - back[:, 2].mean())
    objects = [
        _object("front", 1, "sphere", front, front_mesh["faces"], DIELECTRIC),
        _object("back", 2, "sphere", back, back_mesh["faces"], {"type": "pplastic", "int_ior": 1.5, "ext_ior": 1.0, "diffuse_reflectance": 0.5, "alpha": 0.1}),
    ]
    out = FINAL / "hidden_diagnostic"
    config = _scene("schema3_hidden", objects, spp=4, run_id="schema3-hidden", dataset_id="visibility_schema3")
    started = time.perf_counter()
    sample = generate_sample(config, out)
    elapsed = time.perf_counter() - started
    loaded = load_sample(sample)
    metadata = json.loads((sample / "metadata.json").read_text())
    back_row = next(item for item in metadata["shape_map"] if item["object_id"] == "back")
    report = {
        "sample": str(sample),
        "schema_version": metadata["schema_version"],
        "visibility_policy_version": metadata["visibility_policy_version"],
        "back_in_metadata": True,
        "back_object_id_pixel_count": back_row["object_id_pixel_count"],
        "back_normal_supervised_pixel_count": back_row["normal_supervised_pixel_count"],
        "back_legacy_visible_pixels": back_row["visible_pixels"],
        "image_counts": {str(key): int(value) for key, value in raw_object_counts(loaded["object_ids"]).items()},
        "supervised_back_pixels": int(np.count_nonzero(loaded["normal_supervision_mask"] & (loaded["object_ids"] == 2))),
        "view_keys": sorted(training_views(loaded)),
        "seconds": elapsed,
        "pass": metadata["schema_version"] == 3 and back_row["object_id_pixel_count"] == 0 and back_row["normal_supervised_pixel_count"] == 0,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("hidden", json.dumps(report), flush=True)
    if not report["pass"]:
        raise RuntimeError("schema 3 hidden diagnostic failed")
    return report


def _opacity_table(xml_path: Path) -> list[dict]:
    """Map each logical mesh to the BSDF it references.

    The opacity wrapper is a separate BSDF element. It is not nested inside the
    shape, so the shape's ``ref`` has to be followed.
    """
    root = ET.parse(xml_path).getroot()
    bsdfs = {}
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == "bsdf" and node.get("id"):
            bsdfs[node.get("id")] = node
    rows = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "shape":
            continue
        ident = node.get("id") or ""
        if not ident.startswith("mesh-") or ident == "mesh-ground":
            continue
        ref = next((child.get("id") for child in list(node) if child.tag.rsplit("}", 1)[-1] == "ref" and child.get("name") == "bsdf"), None)
        material = bsdfs.get(ref)
        treatment = "unwrapped"
        opacity = None
        if material is not None and material.get("type") == "mask":
            for value in list(material):
                if value.get("name") == "opacity":
                    treatment = value.tag.rsplit("}", 1)[-1]
                    opacity = value.get("value")
        rows.append({"shape_id": ident, "bsdf_id": ref, "treatment": treatment, "opacity": opacity})
    return rows


def _wrap_dielectric(source: Path, destination: Path, kind: str) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    target = None
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == "bsdf" and node.get("type") == "dielectric" and node.get("id"):
            target = node
            break
    if target is None:
        raise RuntimeError("single-object control has no dielectric")
    parent = next(parent for parent in root.iter() if target in list(parent))
    index = list(parent).index(target)
    parent.remove(target)
    mask = ET.Element("bsdf", {"type": "mask", "id": target.get("id"), "name": target.get("name") or target.get("id")})
    if kind == "float":
        ET.SubElement(mask, "float", {"name": "opacity", "value": "1.0"})
    elif kind == "spectrum":
        ET.SubElement(mask, "spectrum", {"name": "opacity", "value": "1"})
    elif kind == "rgb":
        ET.SubElement(mask, "rgb", {"name": "opacity", "value": "1 1 1"})
    else:
        raise RuntimeError(kind)
    target.attrib.pop("id", None)
    target.attrib.pop("name", None)
    mask.append(target)
    parent.insert(index, mask)
    destination.write_text(ET.tostring(root, encoding="unicode"))


def _compare_exr(left: Path, right: Path) -> dict:
    from sensor_sim.scene_gen.convergence_v2 import named_exr
    from sensor_sim.scene_gen.metric_audit import angular_difference_deg, polarization_from_stokes, scalar_stokes, summarize
    from sensor_sim.scene_gen.patch_compare import ABSOLUTE_TOLERANCE, AOLP_TOLERANCE_DEG, DOLP_TOLERANCE

    left_channels, _info = named_exr(left)
    right_channels, _info = named_exr(right)
    left_scalar = scalar_stokes(left_channels)
    right_scalar = scalar_stokes(right_channels)
    mask = left_scalar["s0"] > 1e-4
    metrics = {}
    passed = True
    for name in ("R", "G", "B", "S0.R", "S0.G", "S0.B", "S1.R", "S1.G", "S1.B", "S2.R", "S2.G", "S2.B", "S3.R", "S3.G", "S3.B"):
        error = np.abs(left_channels[name] - right_channels[name])
        metrics[name] = summarize(error[mask])
        passed = passed and metrics[name]["n"] > 0 and metrics[name]["max"] <= ABSOLUTE_TOLERANCE
    for name in ("s0", "s1", "s2", "s3"):
        error = np.abs(left_scalar[name] - right_scalar[name])
        metrics[name] = summarize(error[mask])
        passed = passed and metrics[name]["max"] <= ABSOLUTE_TOLERANCE
    left_pol = polarization_from_stokes(left_scalar["s0"], left_scalar["s1"], left_scalar["s2"])
    right_pol = polarization_from_stokes(right_scalar["s0"], right_scalar["s1"], right_scalar["s2"])
    metrics["dolp"] = summarize(np.abs(left_pol["dolp"] - right_pol["dolp"])[mask])
    passed = passed and metrics["dolp"]["max"] <= DOLP_TOLERANCE
    aolp_mask = mask & (left_pol["dolp"] >= 0.05) & (right_pol["dolp"] >= 0.05)
    metrics["aolp_deg"] = summarize(angular_difference_deg(left_pol["aolp"], right_pol["aolp"])[aolp_mask]) if aolp_mask.any() else {"median": None, "p95": None, "max": None, "n": 0}
    if metrics["aolp_deg"]["n"]:
        passed = passed and metrics["aolp_deg"]["max"] <= AOLP_TOLERANCE_DEG
    return {
        "pass": bool(passed),
        "n_s0": int(mask.sum()),
        "n_aolp": int(metrics["aolp_deg"]["n"]),
        "tolerances": {"absolute": ABSOLUTE_TOLERANCE, "dolp": DOLP_TOLERANCE, "aolp_deg": AOLP_TOLERANCE_DEG},
        "metrics": metrics,
    }


def four_copy() -> dict:
    from sensor_sim.scene_gen.assets import build_asset
    from sensor_sim.scene_gen.placement import place_objects
    from sensor_sim.scene_gen.qualify_v1 import _object
    from sensor_sim.scene_gen.render_sample import render_stokes

    out = FINAL / "four_copy_256"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2564)
    meshes = []
    for index in range(4):
        mesh, _spec = build_asset("sphere", rng, radius=0.03 + 0.002 * index)
        meshes.append(mesh)
    placed = place_objects(meshes, rng, PROTOTYPE_CAMERA, seed=2564, max_attempts=80)
    objects = [
        _object(f"obj_{index:02d}", index + 1, "sphere", pose["vertices"], meshes[index]["faces"], DIELECTRIC)
        for index, pose in enumerate(placed)
    ]
    config = _scene("four_dielectric", objects, spp=256, run_id="four-copy-256", dataset_id="patch_qualification")
    sample = out / "four_dielectric"
    started = time.perf_counter()
    if not (sample / "metadata.json").is_file():
        sample = generate_sample(config, out)
    four_seconds = time.perf_counter() - started
    loaded = load_sample(sample)
    metadata = json.loads((sample / "metadata.json").read_text())
    treatments = _opacity_table(sample / "scene.xml")
    expected = [
        ("mesh-obj_00", "unwrapped", None),
        ("mesh-obj_01", "float", "1.0"),
        ("mesh-obj_02", "spectrum", "1"),
        ("mesh-obj_03", "rgb", "1 1 1"),
    ]
    identity_ok = [(row["shape_id"], row["treatment"], row["opacity"]) for row in treatments] == expected
    shape_ids = [row["shape_id"] for row in treatments]
    single_mesh, _spec = build_asset("sphere", np.random.default_rng(11), radius=0.05)
    from sensor_sim.film.shape_mesh import place_mesh

    single = _object("ball", 1, "sphere", place_mesh(single_mesh["vertices"], 0.0, (0.0, 0.02)), single_mesh["faces"], DIELECTRIC)
    control = _scene("single_dielectric", [single], spp=256, run_id="four-copy-control", dataset_id="patch_qualification")
    control_sample = out / "single_dielectric"
    if not (control_sample / "metadata.json").is_file():
        control_sample = generate_sample(control, out)
    kinds = {"float": "float", "spectrum": "spectrum", "rgb": "rgb"}
    comparisons = {}
    for label in kinds:
        xml = control_sample / f"control_{label}.xml"
        path = control_sample / f"control_{label}.exr"
        if not path.is_file():
            _wrap_dielectric(control_sample / "scene.xml", xml, label)
            _scene_obj, _channels, names, bitmap = render_stokes(xml, "cuda_ad_spectral_polarized", 256, int(control["seed"]))
            bitmap.write(str(path))
        else:
            names = []
        comparisons[label] = _compare_exr(control_sample / "polarization.exr", path)
        comparisons[label]["channel_names"] = names
    report = {
        "sample": str(sample),
        "seconds": None if four_seconds < 1.0 else four_seconds,
        "seconds_note": "The four-object and single-object samples were published in the first pass. This field is the rerun time when those files are reused, not a new render.",
        "shape_ids": shape_ids,
        "distinct_nonempty": len(shape_ids) == 4 and all(shape_ids) and len(set(shape_ids)) == 4,
        "opacity_matches_logical_order": identity_ok,
        "treatments": treatments,
        "object_counts": {str(key): int(value) for key, value in raw_object_counts(loaded["object_ids"]).items()},
        "schema_version": metadata["schema_version"],
        "view_shapes": {key: list(value.shape) for key, value in training_views(loaded).items()},
        "controls": comparisons,
        "pass": identity_ok and all(item["pass"] for item in comparisons.values()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("four_copy", report["pass"], report["treatments"], {key: value["pass"] for key, value in comparisons.items()}, flush=True)
    if not report["pass"]:
        raise RuntimeError("four-copy 256 spp qualification failed")
    return report


def audit_published_sample(sample: Path) -> dict:
    """Counts, clearance, EXR, and supervision for one qualification sample."""
    import mitsuba as mi

    sample = Path(sample)
    metadata = json.loads((sample / "metadata.json").read_text())
    loaded = load_sample(sample)
    mi.set_variant("cuda_ad_spectral_polarized")
    scene = mi.load_file(str(sample / "scene.xml"))
    meshes = {}
    for shape in scene.shapes():
        ident = str(shape.id())
        if not ident.startswith("mesh-") or ident == "mesh-ground":
            continue
        vertices = np.stack([np.array(shape.vertex_position(index)).reshape(-1)[:3] for index in range(int(shape.vertex_count()))])
        faces = np.array(shape.faces_buffer()).reshape(-1, 3)
        meshes[ident] = {"vertices": vertices, "faces": faces, "floor": floor_status(vertices, FLOOR_Z)}
    names = sorted(meshes)
    pairs = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            clearance = mesh_clearance(meshes[left_name], meshes[right_name])
            pairs.append({"pair": [left_name, right_name], **clearance})
    return {
        "scene_id": metadata["scene_id"],
        "schema_version": metadata["schema_version"],
        "object_count": sum(1 for item in metadata["objects"] if item.get("kind") == "obj"),
        "assets": [item.get("asset_id") for item in metadata["objects"] if item.get("kind") == "obj"],
        "asset_instance_ids": [item.get("asset_instance_id") for item in metadata["objects"] if item.get("kind") == "obj"],
        "logical_shape_ids": metadata["logical_shape_ids"],
        "loaded_presets": metadata.get("loaded_presets"),
        "shape_map": [
            {
                "object_id": item["object_id"],
                "object_id_pixel_count": item.get("object_id_pixel_count"),
                "normal_valid_pixel_count": item.get("normal_valid_pixel_count"),
                "normal_supervised_pixel_count": item.get("normal_supervised_pixel_count"),
                "visible_pixels_legacy_ownership": item.get("visible_pixels"),
            }
            for item in metadata["shape_map"]
        ],
        "pairs": pairs,
        "floors": {name: item["floor"] for name, item in meshes.items()},
        "separation_pass": all(pair["clearance_m"] + 1e-6 >= 0.005 and not pair["surface_intersection"] for pair in pairs)
        and all(not item["floor"]["penetrating"] for item in meshes.values()),
        "view_shapes": {key: list(value.shape) for key, value in training_views(loaded).items()},
        "mask_true": int(loaded["normal_supervision_mask"].sum()),
        "bytes": sum(path.stat().st_size for path in sample.rglob("*") if path.is_file()),
    }


if __name__ == "__main__":
    import sys

    command = sys.argv[1]
    if command == "histogram":
        visibility_histogram()
    elif command == "hidden":
        hidden_diagnostic()
    elif command == "four-copy":
        four_copy()
    else:
        raise SystemExit(f"unknown command {command}")

"""Hard failures and composition warnings for a saved semantic scene.

Occupancy and free-space numbers are warnings. A missing package, a failed
support relation, or a non-finite transform is a hard failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.asset_package import PackageError, verify_package
from sensor_sim.scene_gen.indoor_transform import deserialize_matrix

COVERAGE_BANDS = {
    "workbench_low": (0.10, 0.15),
    "workbench_moderate": (0.15, 0.22),
    "workbench_high": (0.22, 0.30),
}
REQUIRED_ASSETS = {
    "kitchen": {"prop_pot", "prop_cutting_board", "prop_plate"},
    "dining": {"prop_plate", "prop_bowl", "prop_fork"},
    "desk": {"prop_keyboard", "prop_notebook", "prop_phone"},
    "workbench": {"prop_work_panel", "prop_screwdriver", "prop_hammer"},
}


def assess(bundle: dict, semantic_catalog: Path, room_catalog: Path, asset_catalog: Path, render_dir: Path | None = None) -> dict:
    hard: list[dict] = []
    warnings: list[dict] = []
    _matrices(bundle, hard)
    _relations(bundle, hard)
    _clearance(bundle, hard)
    _packages(bundle, semantic_catalog, room_catalog, asset_catalog, hard)
    _required(bundle, hard)
    _cameras(bundle, hard)
    _coverage(bundle, warnings)
    _workspace(bundle, warnings)
    if render_dir is not None:
        _renders(bundle, render_dir, hard)
    return {"passed": not hard, "hard_failures": hard, "warnings": warnings, "task_id": bundle.get("task_id")}


def _matrices(bundle: dict, hard: list[dict]) -> None:
    for item in [*bundle["room_components"], *bundle["furniture_instances"], *bundle["object_instances"]]:
        try:
            matrix = np.asarray(deserialize_matrix(item["matrix"]), dtype=np.float64)
        except (PackageError, ValueError, TypeError):
            hard.append({"name": f"transform:{item['instance_id']}", "passed": False})
            continue
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            hard.append({"name": f"transform:{item['instance_id']}", "passed": False})


def _relations(bundle: dict, hard: list[dict]) -> None:
    for relation in bundle["relations"]:
        if relation.get("type") == "supported_on" and not relation.get("passed", False):
            hard.append({"name": f"support:{relation['instances'][0]}", "passed": False, "reason": relation.get("reason")})
        measured = relation.get("measured_m")
        if measured is not None and abs(float(measured)) > 0.001 and relation.get("type") == "supported_on":
            hard.append({"name": f"support_gap:{relation['instances'][0]}", "passed": False, "measured_m": measured})


def _clearance(bundle: dict, hard: list[dict]) -> None:
    nearest = bundle.get("clutter", {}).get("nearest_clearance_m")
    if nearest is None:
        return
    if float(nearest) < 0.004:
        hard.append({"name": "clearance", "passed": False, "nearest_clearance_m": nearest})


def _packages(bundle, semantic_catalog, room_catalog, asset_catalog, hard) -> None:
    for package in bundle["room_packages"]:
        _one_package(room_catalog / package["path"], package, hard)
    for package in bundle["semantic_packages"]:
        _one_package(semantic_catalog / package["path"], package, hard)
    for item in bundle["furniture_instances"] + bundle["object_instances"]:
        if item["asset_id"] in {package["asset_id"] for package in bundle["semantic_packages"]}:
            continue
        path = Path(asset_catalog) / item["asset_id"] / item["version"]
        if not (path / "asset.json").is_file():
            hard.append({"name": f"missing_asset:{item['asset_id']}", "passed": False})


def _one_package(path: Path, package: dict, hard: list[dict]) -> None:
    if not path.is_dir():
        hard.append({"name": f"missing_package:{package['asset_id']}", "passed": False})
        return
    manifest = verify_package(path)
    if manifest["package_sha256"] != package["package_sha256"]:
        hard.append({"name": f"package_hash:{package['asset_id']}", "passed": False})


def _required(bundle: dict, hard: list[dict]) -> None:
    scene_type = bundle["recipe"]["scene_type"]
    present = {item["asset_id"] for item in bundle["object_instances"]}
    missing = REQUIRED_ASSETS[scene_type] - present
    if missing:
        hard.append({"name": "missing_required", "passed": False, "assets": sorted(missing)})


def _cameras(bundle: dict, hard: list[dict]) -> None:
    cameras = bundle.get("cameras") or {}
    for name in ("overview", "close"):
        camera = cameras.get(name)
        origin = np.asarray(camera["origin"], dtype=np.float64) if camera else None
        target = np.asarray(camera["target"], dtype=np.float64) if camera else None
        if camera is None or not np.isfinite(origin).all() or not np.isfinite(target).all():
            hard.append({"name": f"camera:{name}", "passed": False})
    renderer = bundle.get("renderer") or {}
    if renderer.get("variant") != "cuda_ad_spectral_polarized":
        hard.append({"name": "renderer", "passed": False, "variant": renderer.get("variant")})


def _coverage(bundle: dict, warnings: list[dict]) -> None:
    band = COVERAGE_BANDS.get(bundle["task_id"])
    if band is None:
        return
    fraction = float(bundle["clutter"]["union_fraction"])
    if not band[0] <= fraction <= band[1]:
        warnings.append({"name": "coverage", "union_fraction": fraction, "band": list(band)})


def _workspace(bundle: dict, warnings: list[dict]) -> None:
    if bundle["recipe"]["scene_type"] != "workbench":
        return
    usable = bundle["clutter"].get("usable_free_rectangle") or {}
    size = usable.get("size_m") or [0.0, 0.0]
    if min(size) < 0.25:
        warnings.append({"name": "usable_workspace", "size_m": size, "origin_m": usable.get("origin_m")})


def _renders(bundle: dict, render_dir: Path, hard: list[dict]) -> None:
    render_dir = Path(render_dir)
    for name in ("room", "tabletop"):
        if not (render_dir / f"{name}.exr").is_file() or not (render_dir / f"{name}.png").is_file():
            hard.append({"name": f"render:{name}", "passed": False})
    if not (render_dir / "validation.json").is_file():
        hard.append({"name": "render:validation", "passed": False})
        return
    validation = json.loads((render_dir / "validation.json").read_text(encoding="utf-8"))
    if validation.get("variant") != bundle["renderer"]["variant"]:
        hard.append({"name": "render:variant", "passed": False})

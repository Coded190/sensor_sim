"""Measured evidence for the Dataset v1 gates. This does not generate Dataset v1."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from sensor_sim.scene_gen.export_audit import xml_shape_report
from sensor_sim.scene_gen.generate import generate_sample, random_config

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "dataset" / "qualification" / "final"
SEEDS = (269837089, 1681914905, 2751786026, 987396961, 1237807964)


def _save(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


def _tabletop_spec() -> dict:
    spec = yaml.safe_load((ROOT / "configs" / "scenes" / "tabletop.yaml").read_text())
    spec["profiles"] = json.loads((ROOT / spec["profile_file"]).read_text())
    return spec


def rerun_empty_ids() -> dict:
    spec = _tabletop_spec()
    rows = {}
    for seed in SEEDS:
        config = random_config(spec, int(seed))
        config["scene_id"] = f"rerun_{seed}"
        config["resolution"] = [int(v) for v in spec.get("resolution", [128, 128])]
        config["spp"] = int(spec.get("spp", 16))
        started = time.perf_counter()
        try:
            sample = generate_sample(config, OUT / "rerun", overwrite=True)
            xml = xml_shape_report(sample / "scene.xml")
            ids = [item["id"] for item in xml["shapes"]]
            rows[str(seed)] = {
                "pass": "" not in ids and all(f"mesh-{item['name']}" in ids for item in config["objects"]),
                "scene_id": config["scene_id"],
                "object_names": [item["name"] for item in config["objects"]],
                "families": [item["asset_id"] for item in config["objects"]],
                "materials": [item["bsdf"]["type"] for item in config["objects"]],
                "geometry_seed": config["geometry_seed"],
                "material_seed": config["material_seed"],
                "xml_ids": ids,
                "empty_ids": xml["empty"],
                "seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            rows[str(seed)] = {"pass": False, "error": str(exc), "seconds": time.perf_counter() - started}
        print("rerun", seed, rows[str(seed)].get("pass"), rows[str(seed)].get("error", ""), flush=True)
    return rows


def special_name_sphere() -> dict:
    """An object named sphere with IOR 1.33 must not be forced to 1.5."""
    from sensor_sim.film.shape_mesh import place_mesh
    from sensor_sim.scene_gen.assets import build_asset
    from sensor_sim.scene_gen.qualify_v1 import _config, _object

    mesh, _spec = build_asset("sphere", np.random.default_rng(0), radius=0.04)
    vertices = place_mesh(mesh["vertices"], 0.0, (0.0, 0.04))
    config = _config(
        "named_sphere",
        [_object("sphere", 1, "sphere", vertices, mesh["faces"], {"type": "dielectric", "int_ior": 1.33, "ext_ior": 1.0})],
        resolution=(48, 48),
        spp=2,
    )
    sample = generate_sample(config, OUT / "names", overwrite=True)
    text = (sample / "scene.xml").read_text()
    return {"pass": 'name="int_ior" value="1.33"' in text or "1.33" in text, "scene_id": "named_sphere", "has_1_33": "1.33" in text}


def main() -> None:
    report = {}
    if (OUT / "evidence.json").is_file():
        report = json.loads((OUT / "evidence.json").read_text())
    if not report.get("empty_id_reruns") or not all(row.get("pass") for row in report["empty_id_reruns"].values()):
        report["empty_id_reruns"] = rerun_empty_ids()
        _save(report)
    report["named_sphere"] = special_name_sphere()
    _save(report)
    print("wrote", OUT / "evidence.json", flush=True)


if __name__ == "__main__":
    main()

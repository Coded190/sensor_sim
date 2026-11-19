"""Pipeline-closure measurements. Does not generate Dataset v1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "dataset" / "qualification" / "pipeline_closure"
ORIGINAL = ROOT / "dataset" / "qualification" / "final" / "rerun" / "rerun_269837089"


def _report() -> dict:
    path = OUT / "evidence.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def _save(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


def _mesh_arrays(shape) -> tuple[np.ndarray, np.ndarray]:
    count = int(shape.vertex_count())
    vertices = np.stack([np.array(shape.vertex_position(i)).reshape(-1)[:3] for i in range(count)])
    faces = np.array(shape.faces_buffer()).reshape(-1, 3)
    return vertices, faces


def compare_shapes(original_scene, rebuilt_scene) -> dict:
    original = {shape.id(): shape for shape in original_scene.shapes()}
    rebuilt = {shape.id(): shape for shape in rebuilt_scene.shapes()}
    rows = {}
    for ident, shape in original.items():
        if not str(ident).startswith("mesh-"):
            continue
        other = rebuilt.get(ident)
        if other is None:
            rows[ident] = {"pass": False, "error": "missing in rebuild"}
            continue
        left, left_faces = _mesh_arrays(shape)
        right, right_faces = _mesh_arrays(other)
        if left.shape == right.shape and left_faces.shape == right_faces.shape and np.array_equal(left_faces, right_faces):
            error = np.linalg.norm(left - right, axis=1)
            paired = True
        else:
            step = max(1, left.shape[0] // 400)
            error = np.array([np.linalg.norm(right - point, axis=1).min() for point in left[::step]])
            paired = False
        center = float(np.linalg.norm(left.mean(0) - right.mean(0)))
        rows[ident] = {
            "pass": bool(float(error.max()) <= 1e-4 and center <= 1e-4),
            "paired_vertices": paired,
            "n_compared": int(error.size),
            "max_m": float(error.max()),
            "p95_m": float(np.percentile(error, 95)),
            "median_m": float(np.median(error)),
            "center_error_m": center,
            "vertex_count": [int(left.shape[0]), int(right.shape[0])],
            "triangle_count": [int(left_faces.shape[0]), int(right_faces.shape[0])],
        }
    return rows


def stokes_of(xml: Path, spp: int, seed: int):
    from sensor_sim.scene_gen.render_sample import render_stokes, stack_stokes

    _scene, channels, _names, _bitmap = render_stokes(xml, "cuda_ad_spectral_polarized", spp, seed)
    return [stack_stokes(channels, name) for name in ("S0", "S1", "S2", "S3")]


def component_delta(reference, other, mask: np.ndarray) -> dict:
    def avg(value):
        array = np.asarray(value, dtype=np.float64)
        return array.mean(axis=2) if array.ndim == 3 else array

    ref = [avg(item) for item in reference]
    cur = [avg(item) for item in other]
    use = np.asarray(mask, dtype=bool) & np.isfinite(ref[0]) & (ref[0] > 1e-4)
    out = {"n": int(use.sum()), "mask": "measurement_mask and reference S0 > 1e-4"}
    for index, name in enumerate(("S0", "S1", "S2", "S3")):
        error = np.abs(cur[index] - ref[index])
        out[name] = {
            "median_abs": float(np.median(error[use])) if use.any() else None,
            "p95_abs": float(np.percentile(error[use], 95)) if use.any() else None,
        }
    return out


def reproduce() -> dict:
    import mitsuba as mi
    import yaml
    from sensor_sim.scene_gen.dataset import load_sample
    from sensor_sim.scene_gen.generate import generate_sample, random_config

    saved = json.loads((ORIGINAL / "scene_config.json").read_text())
    meta = json.loads((ORIGINAL / "metadata.json").read_text())
    spec = yaml.safe_load((ROOT / "configs/scenes/tabletop.yaml").read_text())
    spec["profiles"] = json.loads((ROOT / spec["profile_file"]).read_text())
    rebuilt_config = random_config(spec, int(saved["seed"]))
    same_seed = int(rebuilt_config["geometry_seed"]) == int(saved["geometry_seed"])
    rebuilt_config["scene_id"] = "rebuild_269837089"
    rebuilt_config["run_id"] = "pipeline-closure-repro"
    rebuilt_config["resolution"] = saved["resolution"]
    rebuilt_config["spp"] = int(saved["spp"])
    sample = generate_sample(rebuilt_config, OUT / "reproducibility", overwrite=True)
    mi.set_variant("cuda_ad_spectral_polarized")
    original_scene = mi.load_file(str(ORIGINAL / "scene.xml"))
    rebuilt_scene = mi.load_file(str(sample / "scene.xml"))
    geometry = compare_shapes(original_scene, rebuilt_scene)
    loaded = load_sample(ORIGINAL)
    same = component_delta(stokes_of(ORIGINAL / "scene.xml", 16, int(meta["render_seed"])), stokes_of(sample / "scene.xml", 16, int(meta["render_seed"])), loaded["measurement_mask"])
    other = component_delta(stokes_of(sample / "scene.xml", 16, int(meta["render_seed"])), stokes_of(sample / "scene.xml", 16, int(meta["render_seed"]) + 1), loaded["measurement_mask"])
    return {
        "status": "mitsuba_render_tested",
        "seed": int(saved["seed"]),
        "geometry_seed_matches_saved": same_seed,
        "saved_geometry_seed": int(saved["geometry_seed"]),
        "rebuilt_geometry_seed": int(rebuilt_config["geometry_seed"]),
        "render_seed": int(meta["render_seed"]),
        "geometry": geometry,
        "same_seed_render": same,
        "different_seed_render": other,
        "note": "different_seed_render is Monte Carlo variability, not an export mismatch",
    }


def main() -> None:
    report = _report()
    report["reproducibility"] = reproduce()
    _save(report)
    print("repro geometry seeds", report["reproducibility"]["geometry_seed_matches_saved"], flush=True)
    print("max errors", {key: row.get("max_m") for key, row in report["reproducibility"]["geometry"].items()}, flush=True)


if __name__ == "__main__":
    main()

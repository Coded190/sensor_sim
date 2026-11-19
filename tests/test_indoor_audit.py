"""Audit checks for identity, Stokes diagnostics, and non-overwriting replay."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.indoor_cli import assemble_demonstration
from sensor_sim.scene_gen.indoor_mitsuba import (
    grid_shapes,
    identity_buffer,
    neutral_stokes_scene,
    render_stokes,
    stokes_cone_diagnostic,
    stokes_scene,
)
from sensor_sim.scene_gen.metric_audit import CHANNEL_NAMES


def _shape(name, center, scale, index_name):
    return {
        "shape_id": name,
        "logical_object": index_name,
        "component": "body",
        "material_slot": "default",
        "instance": name,
        "center": center,
        "scale": scale,
        "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]}},
    }


def test_occlusion_and_equal_depth():
    shapes = [
        _shape("front", [0.0, 0.0, 0.40], [0.25, 0.25, 0.02], "front"),
        _shape("hidden", [0.0, 0.0, 0.10], [0.08, 0.08, 0.02], "hidden"),
        _shape("side", [0.70, 0.0, 0.40], [0.06, 0.06, 0.02], "side"),
    ]
    image = identity_buffer(shapes, width=32, height=32)
    visible = set(int(value) for value in np.unique(image) if int(value) != 0)
    assert 1 in visible
    assert 2 not in visible
    assert 3 in visible
    assert image.dtype == np.int32
    same = [
        _shape("a", [0.0, 0.0, 0.2], [0.1, 0.1, 0.05], "a"),
        _shape("b", [0.0, 0.0, 0.2], [0.1, 0.1, 0.05], "b"),
    ]
    tied = identity_buffer(same, width=24, height=24)
    assert set(int(value) for value in np.unique(tied)) <= {0, 1}
    assert 1 in set(int(value) for value in np.unique(tied))


def test_saved_scene_identity_counts_still_match_grid():
    report_shapes = grid_shapes(10)
    image = identity_buffer(report_shapes, width=24, height=24)
    visible = {int(value) for value in np.unique(image) if int(value) != 0}
    assert visible == set(range(1, 11))


def test_stokes_rgb_cone_is_diagnostic(tmp_path: Path):
    polarized = render_stokes(stokes_scene((16, 16)), tmp_path / "polarized.exr", spp=4, seed=7)
    neutral = render_stokes(neutral_stokes_scene((16, 16)), tmp_path / "neutral.exr", spp=4, seed=7)
    import mitsuba as mi

    for report, path in ((polarized, tmp_path / "polarized.exr"), (neutral, tmp_path / "neutral.exr")):
        image = np.array(mi.Bitmap(str(path)))
        diagnostic = stokes_cone_diagnostic(image, report["exr_names"])
        assert report["exr_names"] == list(CHANNEL_NAMES)
        assert diagnostic["hard_failure"] is False
        assert diagnostic["representation"] == "mitsuba_hdrfilm_rgb_stokes"
        assert diagnostic["nonfinite_count"] == 0
    assert abs(neutral["channel_mean"]["S1.R"]) < abs(polarized["channel_mean"]["S1.R"])


def test_replay_does_not_overwrite_scene_c(tmp_path: Path):
    original = Path("dataset/qualification/indoor_scenes/scene_c/scene.json")
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    assemble_demonstration(Path("assets/candidates"), tmp_path / "replay", render=False)
    after = hashlib.sha256(original.read_bytes()).hexdigest()
    assert before == after
    replayed = (tmp_path / "replay" / "scene_c" / "scene.json").read_bytes()
    assert replayed == original.read_bytes()
    assert "indoor_scenes/scene_c" not in str(tmp_path / "replay")

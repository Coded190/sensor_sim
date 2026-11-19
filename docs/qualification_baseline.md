# Qualification baseline

Recorded before the normal, depth, and material changes in this qualification pass.

## Git

- Commit: `0609c61124005144060a8f9eb673d73510b57ff5`
- Subject: Prepare the held-out film experiments so a machine with a free GPU can render and train them.
- The working tree was already dirty. This baseline is that tree, not a clean checkout.

Locked files were not in the dirty list:

- `src/sensor_sim/film/mitsuba_scene.py`
- `src/sensor_sim/film/sfp_scene.py`
- `material_bsdf` in `src/sensor_sim/scenes.py`

Dirty paths at the time of the run included `.gitignore`, `README.md`, film dataset and training modules, `pyproject.toml`, `uv.lock`, and untracked `src/sensor_sim/scene_gen/`, `third_party/`, `configs/scenes/`, `docs/`, and `tests/test_scene_gen.py`.

## Command

```bash
uv run pytest -q --tb=line
```

## Result

83 passed, 2 failed, in 61.87 s. 85 tests were collected.

`tests/test_scene_gen.py` passed, including the Blender export of the prototype sphere.

The two failures read `outputs/phase3/frames/000/stokes/render.exr`, which is not in the tree:

- `tests/test_phase4.py::test_dolp_on_a_known_pixel_matches_polarization_maps`
- `tests/test_phase4.py::test_feature_stack_widths_and_order`

Both raise `FileStream` from `sensor_sim.phase4.load_frame`. They are missing-output failures, not scene-generator failures. Later qualification runs should treat them as pre-existing unless a change in this pass touches Phase 4 loading.

Film tests that do not need that EXR passed, including `tests/test_film.py`, `tests/test_heldout.py`, `tests/test_sfp.py`, and `tests/test_phase0.py`.

# Task-conditioned scene generation

This is a deterministic local generator. It does not call GPT-4, Gemini, or a vision-language model, and it is not a RoboGen reproduction. Visual review and admission are pending. The six tasks are a functional check on the procedural catalog, not a result about Objaverse, articulation, robot reachability, dynamic stability, or soft bodies.

## Modules and commands

- `src/sensor_sim/scene_gen/task_schema.py`
- `src/sensor_sim/scene_gen/task_retrieve.py`
- `src/sensor_sim/scene_gen/task_size.py`
- `src/sensor_sim/scene_gen/task_layout.py`
- `src/sensor_sim/scene_gen/task_replay.py`
- `src/sensor_sim/scene_gen/task_cli.py`

```text
python -m sensor_sim.scene_gen.task_cli validate --task task.json
python -m sensor_sim.scene_gen.task_cli generate --task task.json --catalog assets/candidates --output dataset/qualification/task_scenes
python -m sensor_sim.scene_gen.task_cli inspect --bundle dataset/qualification/task_scenes/<task_id>/<run_id>
python -m sensor_sim.scene_gen.task_cli replay --bundle dataset/qualification/task_scenes/<task_id>/<run_id> --catalog assets/candidates
python -m sensor_sim.scene_gen.task_cli evaluate --catalog assets/candidates --output dataset/qualification/task_scenes --seed 17011
```

`sentence_transformers` does not import in this environment, and no local checkpoint was used. Retrieval is scikit-learn TF-IDF with lowercase tokens matching `[a-z0-9_]+`, unigrams, no stop words, `min_df` 1, `max_df` 1, smooth IDF, L2 normalization, and float64. The score type is `tfidf_cosine`. Query expansion is `deterministic_local_template_v1`. A `mug` query expands to the cup tokens. Replay does not fit the vectorizer again. It reloads the saved ranking and checks the selected package hashes.

Rendered selections require `renderer_validated` and `license_status` `permitted`. A technically validated asset can appear in discovery. `fixture_sphere` does that for the query `sphere` and is excluded when renderer qualification is required. Unresolved Infinigen assets are rejected for license before ranking. Visual verification is `pending`.

## Relations

Required relations are checked after every correction. `supported_on` calls `measure_support` and keeps the 1 mm and 0.1 mm limits. `inside` tests the transformed vertices against the named interior, including the margin. `near` and `keep_clear_of` use triangle-to-triangle distance. `reachable_from` fails when no robot is configured. Unsupported numeric relations fail closed. Doors and windows are not articulated. Joint state is `not_applicable`.

Scale stays on the instance transform. The source mesh is not rewritten. Size priors for these assets are `asset_declared` from the package bounds. That is not a claim of an external anthropometric survey. The book fits the open crate at scale 1. A 0.50 by 0.50 by 0.20 m object does not.

## Six tasks

Seed 17011, top-k 10, all valid:

| Task | Run | Result |
| --- | --- | --- |
| `book_on_table` | `dataset/qualification/task_scenes/book_on_table/e54a5e18636df8e7` | valid, no correction |
| `cup_on_shelf` | `dataset/qualification/task_scenes/cup_on_shelf/8a0a6b98b048ef83` | valid, no correction |
| `book_inside_crate` | `dataset/qualification/task_scenes/book_inside_crate/e547b5c803391dbe` | valid, open crate, book in the cavity |
| `crate_and_book_on_table` | `dataset/qualification/task_scenes/crate_and_book_on_table/44ac50f76b66afcb` | valid after 10 planar corrections |
| `cup_near_book` | `dataset/qualification/task_scenes/cup_near_book/95d8b3f496b78836` | valid after 12 corrections, distance limit 0.001 m to 0.30 m |
| `objects_in_workspace` | `dataset/qualification/task_scenes/objects_in_workspace/3cf07be40ab68252` | valid inside the declared box, no robot |

Seed 17012 also produced six valid bundles in different run directories. The scene manifests differ from seed 17011 because the support-plane nudge changes. Replay of a saved bundle reproduces `scene.json` and does not search the catalog.

Retrieval at k = 10, renderer-qualified procedural catalog, 12 eligible assets:

| Query | Acceptable | Precision | Recall | Reciprocal rank | Returned |
| --- | --- | --- | --- | --- | --- |
| book | `prop_book` | 1 | 1 | 1 | `prop_book` |
| cup | `prop_cup` | 1 | 1 | 1 | `prop_cup` |
| mug | `prop_cup` | 1 | 1 | 1 | `prop_cup` |
| table | `table_rect` | 1 | 1 | 1 | `table_rect` |
| shelf | `shelf_two` | 1 | 1 | 1 | `shelf_two` |
| crate | `prop_crate_open` | 0.5 | 1 | 1 | `prop_crate_open`, `prop_crate` |

The solid crate is a text match and a geometry rejection for containment. The inside task skips it because it has no interior. Required-object coverage for the six tasks is complete after that check.

Negative results: a book larger than the cavity is rejected by `fits_interior`; a workspace smaller than the crate is stored as invalid; a 5 m clearance demand on the table is invalid; `reachable_from` without a robot raises `robot_required`.

## Limits

The generator does not call an external model, does not rank Objaverse, does not articulate the door or window, and does not claim reachability or dynamic stability. Human visual review is pending.

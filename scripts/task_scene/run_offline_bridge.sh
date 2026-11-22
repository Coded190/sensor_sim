#!/usr/bin/env bash
# Run the offline bridge with separate project and Isaac interpreters.
# The project .venv never imports Isaac Lab.
set -euo pipefail
ROOT="/home/cmoore/sensor_sim"
PROJECT_PY="${ROOT}/.venv/bin/python"
ISAAC_PY="/home/cmoore/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11"
ISAAC_SITE="/home/cmoore/moore/isaac_so_arm101/.venv/lib/python3.11/site-packages"
export PYTHONPATH="${ISAAC_SITE}/isaaclab/source/isaaclab:${ISAAC_SITE}:/home/cmoore/moore/isaac_so_arm101/src:${ROOT}/src:${ROOT}/scripts/isaac"

cd "${ROOT}"
env -u PYTHONPATH "${PROJECT_PY}" -m pytest tests/test_asset_library.py tests/test_task_scene_connector.py tests/test_scene_state.py -q

"${ISAAC_PY}" "${ROOT}/scripts/isaac/bridge_render_scene.py"
"${ISAAC_PY}" "${ROOT}/scripts/isaac/bridge_simulate_offline.py"

env -u PYTHONPATH "${PROJECT_PY}" "${ROOT}/scripts/mitsuba/replay_isaac_states.py"
env -u PYTHONPATH "${PROJECT_PY}" "${ROOT}/scripts/task_scene/validate_offline_bridge.py"

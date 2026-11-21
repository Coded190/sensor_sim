#!/usr/bin/env bash
# Run the pinned Infinigen checkout. This script is not imported by sensor_sim.
set -euo pipefail
ROOT="${INFINIGEN_ROOT:-}"
EXPECTED="${INFINIGEN_COMMIT:-53033dd55c7191cbc436be6ffbb63c026935a509}"
if [[ -z "${ROOT}" || ! -d "${ROOT}/.git" ]]; then
  echo "reason=infinigen_checkout_missing set INFINIGEN_ROOT to a git checkout. sensor_sim does not install Infinigen." >&2
  exit 2
fi
ACTUAL="$(git -C "${ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL}" != "${EXPECTED}" ]]; then
  echo "reason=commit_mismatch expected=${EXPECTED} actual=${ACTUAL}" >&2
  exit 3
fi
echo "infinigen_commit=${ACTUAL}"
echo "reason=generator_command_not_invoked The checkout is present, but the exporter arguments must be read from this commit before a run. Refusing a guessed command."
exit 4

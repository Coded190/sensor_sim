"""Shell out to an isolated Infinigen install. This module does not import Infinigen."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

INDOORS_ROOT = Path(os.environ.get("INFINIGEN_INDOORS_ROOT", "/home/cmoore/infinigen_src/indoors-stable"))
INDOORS_PYTHON = Path(os.environ.get("INFINIGEN_INDOORS_PYTHON", "/home/cmoore/infinigen_envs/indoors/bin/python"))
SUPPORTED_ROOMS = ("LivingRoom", "Kitchen", "Bedroom", "DiningRoom", "Bathroom", "Office")
HISTORICAL_STABLE = "b11700ebac77695d8f53538705cfdc0cd890c54d"


class InfinigenBackendError(RuntimeError):
    """The isolated Infinigen install cannot run the requested generation."""


def installation_status(root: Path | None = None, python: Path | None = None) -> dict:
    root = Path(root or INDOORS_ROOT)
    python = Path(python or INDOORS_PYTHON)
    commit = None
    if (root / ".git").is_dir():
        try:
            commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            commit = None
    return {
        "root": str(root),
        "python": str(python),
        "installed": (root / "infinigen_examples").is_dir() and python.is_file(),
        "commit": commit,
        "matches_historical_pin": commit == HISTORICAL_STABLE,
    }


def require_installation(root: Path | None = None, python: Path | None = None) -> dict:
    status = installation_status(root, python)
    if not status["installed"]:
        raise InfinigenBackendError("reason=infinigen_not_installed")
    return status


def registry_rooms(tags_path: Path) -> list[str]:
    text = Path(tags_path).read_text(encoding="utf-8", errors="replace")
    return [name for name in SUPPORTED_ROOMS if name in text]


def coarse_command(room: str, seed: int, output: Path, *, fast: bool, python: Path) -> list[str]:
    if room not in SUPPORTED_ROOMS:
        raise InfinigenBackendError(f"reason=generator_missing room={room}")
    gin = ["fast_solve.gin", "singleroom.gin"] if fast else ["singleroom.gin"]
    return [
        str(python),
        "-m",
        "infinigen_examples.generate_indoors",
        "--seed",
        str(int(seed)),
        "--task",
        "coarse",
        "--output_folder",
        str(output),
        "-g",
        *gin,
        "-p",
        "compose_indoors.terrain_enabled=False",
        f'restrict_solving.restrict_parent_rooms=["{room}"]',
    ]


def render_command(seed: int, run_dir: Path, *, python: Path | None = None) -> list[str]:
    return [
        str(python or INDOORS_PYTHON),
        "-m",
        "infinigen_examples.generate_indoors",
        "--seed",
        str(int(seed)),
        "--task",
        "render",
        "--input_folder",
        str(run_dir),
        "--output_folder",
        str(Path(run_dir) / "frames"),
    ]


def run_generation(room: str, seed: int, output: Path, *, fast: bool, root: Path | None = None, python: Path | None = None) -> dict:
    """Invoke the upstream coarse generator. This is not a replay."""
    status = require_installation(root, python)
    command = coarse_command(room, seed, output, fast=fast, python=Path(status["python"]))
    completed = subprocess.run(command, cwd=status["root"], check=False, capture_output=True, text=True)
    record = {
        "source": "infinigen_indoors",
        "commit": status["commit"],
        "room": room,
        "seed": int(seed),
        "fast_solve": bool(fast),
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    output.mkdir(parents=True, exist_ok=True)
    (Path(output) / "generation_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise InfinigenBackendError(f"reason=generate_failed room={room} code={completed.returncode}")
    return record

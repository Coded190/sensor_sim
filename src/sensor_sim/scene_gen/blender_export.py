"""Run Blender and the vendored Mitsuba add-on."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from sensor_sim.film import ROOT
from sensor_sim.film.blender_build import find_blender

ADDON_SOURCE = ROOT / "third_party" / "mitsuba_blender"
ADDON_COMMIT = ROOT / "third_party" / "mitsuba_blender_COMMIT"
SCRIPT = ROOT / "assets" / "blender" / "create_dataset_scene.py"
ADDON_MODULE = "bl_ext.user_default.mitsuba_blender"


def addon_commit() -> str:
    return ADDON_COMMIT.read_text().splitlines()[0].strip()


def blender_python(binary: Path) -> Path:
    matches = sorted(binary.parent.glob("*/python/bin/python3.11"))
    if not matches:
        raise FileNotFoundError(f"no embedded Python next to {binary}")
    return matches[0]


def install_addon(binary: Path) -> Path:
    """Copy the vendored add-on into this Blender's user extensions."""
    candidates = [path.name for path in binary.parent.iterdir() if path.is_dir() and path.name[:1].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"{binary.parent} has no version directory")
    folder = max(candidates)
    destination = Path.home() / ".config" / "blender" / folder / "extensions" / "user_default" / "mitsuba_blender"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(ADDON_SOURCE, destination)
    return destination


def ensure_blender_mitsuba(binary: Path) -> str:
    """Return the Mitsuba version inside Blender. Install 3.9.1 if it is missing."""
    python = blender_python(binary)
    probe = subprocess.run(
        [str(python), "-c", "import mitsuba as mi; print(mi.__version__)"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return probe.stdout.strip()
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "mitsuba==3.9.1"],
        check=True,
    )
    probe = subprocess.run(
        [str(python), "-c", "import mitsuba as mi; print(mi.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    )
    return probe.stdout.strip()


def export_scene(params: dict) -> None:
    """Write ``params['xml']`` and ``params['rays']`` from Blender."""
    binary = find_blender()
    if binary is None:
        raise FileNotFoundError("blender binary was not found")
    version = ensure_blender_mitsuba(binary)
    if not version.startswith("3.9.1"):
        raise RuntimeError(f"Blender's Mitsuba is {version}, expected 3.9.1")
    install_addon(binary)
    payload = Path(params["params_json"])
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text(json.dumps(params) + "\n")
    env = os.environ.copy()
    env["DRJIT_NO_RTLD_DEEPBIND"] = "1"
    completed = subprocess.run(
        [str(binary), "--background", "--python", str(SCRIPT), "--", str(payload)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    log = Path(params["log"])
    log.write_text(completed.stdout + "\n" + completed.stderr)
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-30:])
        raise RuntimeError(f"blender export failed: {tail}")
    if not Path(params["xml"]).is_file():
        raise RuntimeError(f"blender did not write {params['xml']}")

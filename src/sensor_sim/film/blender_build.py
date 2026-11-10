"""Find Blender and build one blend file from a scene description."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from sensor_sim.film import ROOT

logger = logging.getLogger("sensor_sim.film")

SCRIPT = ROOT / "assets" / "blender" / "create_film_scene.py"
SEARCH_DIRS = (
    Path("/usr/bin"),
    Path("/usr/local/bin"),
    Path("/opt"),
    Path("/snap/bin"),
    Path.home(),
    ROOT / "tools",
)


def find_blender() -> Path | None:
    """Return a blender binary from PATH, tools/, or the usual install prefixes."""
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)
    candidates = [ROOT / "tools" / "blender-4.5.9-linux-x64" / "blender"]
    for directory in SEARCH_DIRS:
        if not directory.exists():
            continue
        candidates.extend(directory.glob("blender"))
        candidates.extend(directory.glob("blender-*-linux-x64/blender"))
    binaries = [path for path in candidates if path.is_file() and os.access(path, os.X_OK)]
    if not binaries:
        return None
    binaries.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return binaries[0]


def build_blend(params: dict, blender: Path | None = None) -> dict:
    """Run Blender in the background and return the pose record it writes."""
    binary = blender or find_blender()
    if binary is None or not binary.is_file():
        raise FileNotFoundError("blender binary was not found and apt install needs a password")
    out_dir = Path(params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = out_dir / "blender_params.json"
    payload.write_text(json.dumps(params) + "\n")
    command = [
        str(binary),
        "--background",
        "--python",
        str(SCRIPT),
        "--",
        str(payload),
    ]
    logger.info("film blender=%s blend=%s", binary, params["blend"])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    log_path = out_dir / "blender.log"
    log_path.write_text(completed.stdout + "\n" + completed.stderr)
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-40:])
        logger.error("film blender status=failed code=%s tail=%s", completed.returncode, tail)
        raise RuntimeError(f"blender failed with code {completed.returncode}")
    poses = json.loads(Path(params["poses"]).read_text())
    logger.info(
        "film blender status=ok blend=%s objects=%s",
        params["blend"],
        ",".join(poses["objects"]),
    )
    return poses

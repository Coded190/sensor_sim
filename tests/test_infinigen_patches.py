"""The Indoors checkout patches are part of the generator, not the upstream SHA."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

INDOORS = Path("/home/cmoore/infinigen_envs/indoors/bin/python")
SCRIPT = Path(__file__).with_name("infinigen_patch_checks.py")
PATCH = Path("dataset/qualification/asset_realism/reports/indoors_local.patch")


@pytest.mark.skipif(not INDOORS.is_file(), reason="indoors interpreter is not installed")
def test_indoors_patches_stop_the_known_hangs():
    completed = subprocess.run([str(INDOORS), str(SCRIPT)], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "event=patch_checks_passed" in completed.stdout
    assert PATCH.is_file()
    assert (PATCH.parent / "indoors_local.patch.sha256").read_text(encoding="utf-8").strip()

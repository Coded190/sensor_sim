"""Record that Infinigen export stays outside the sensor_sim environment.

The preserved bakeoff blends are the source scenes for the pinned commits.
This script does not import Infinigen. A missing checkout is an error.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

STABLE = "b11700ebac77695d8f53538705cfdc0cd890c54d"
PREVIEW = "53033dd55c7191cbc436be6ffbb63c026935a509"


def main() -> int:
    root = os.environ.get("INFINIGEN_ROOT", "")
    expected = os.environ.get("INFINIGEN_COMMIT", PREVIEW)
    if expected not in {STABLE, PREVIEW}:
        print(f"reason=unpinned_commit value={expected}", file=sys.stderr)
        return 5
    if not root or not (Path(root) / ".git").is_dir():
        print("reason=infinigen_checkout_missing", file=sys.stderr)
        return 2
    actual = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        print(f"reason=commit_mismatch expected={expected} actual={actual}", file=sys.stderr)
        return 3
    print(f"infinigen_commit={actual}")
    print("reason=exporter_interface_must_be_read_from_this_commit")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())

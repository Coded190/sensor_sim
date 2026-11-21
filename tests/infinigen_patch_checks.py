"""Run the Indoors checkout patch checks with that environment's Python.

This file is executed by ``/home/cmoore/infinigen_envs/indoors/bin/python``.
It does not import ``sensor_sim``.
"""

from __future__ import annotations

import ast
import signal
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path("/home/cmoore/infinigen_src/indoors-stable")


def _load(path: Path, name: str):
    return _load_many(path, [name])[name]


def _load_many(path: Path, names: list[str]):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = set(names)
    nodes = [
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.ClassDef, ast.Assign)) and (
            getattr(item, "name", None) in wanted
            or (isinstance(item, ast.Assign) and any(getattr(target, "id", None) in wanted for target in item.targets))
        )
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "signal": signal,
        "np": np,
        "trimesh": trimesh,
        "logger": __import__("logging").getLogger("patch_check"),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def check_stable_pose() -> None:
    stable = _load(ROOT / "infinigen/assets/objects/elements/nature_shelf_trinkets/generate.py", "_stable_pose")
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 5, 4]])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    before = np.array(mesh.vertices, copy=True)

    class Hung(Exception):
        pass

    def alarm(signum, frame):
        raise Hung()

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        trimesh.poses.compute_stable_poses(mesh)
        raise SystemExit("unpatched compute_stable_poses returned")
    except Hung:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    started = time.perf_counter()
    pose = stable(mesh)
    elapsed = time.perf_counter() - started
    if pose is None or not np.isfinite(pose).all() or pose.shape != (4, 4):
        raise SystemExit(f"stable pose invalid {pose}")
    if elapsed > 2.0:
        raise SystemExit(f"stable pose too slow {elapsed}")
    if not np.allclose(mesh.vertices, before):
        raise SystemExit("stable pose replaced the render mesh")
    z = pose[:3, 2]
    if abs(float(np.linalg.norm(z)) - 1.0) > 1e-6:
        raise SystemExit("pose rotation is not orthonormal in z")
    print(f"event=stable_pose_ok elapsed_s={elapsed:.3f}")


def check_skeleton() -> None:
    walk = _load(ROOT / "infinigen/assets/objects/creatures/util/part_util.py", "walk_directed_edges")
    coordinates = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0)]
    original = list(coordinates)
    chain = walk(coordinates, [(0, 1), (1, 2)])
    if coordinates != original:
        raise SystemExit("walk mutated coordinates")
    if chain != [coordinates[0], coordinates[0], coordinates[1]]:
        raise SystemExit(f"chain walk changed {chain}")
    cycle = walk(coordinates, [(0, 1), (1, 0)])
    if len(cycle) != 3:
        raise SystemExit(f"cycle did not stop {cycle}")
    print(f"event=skeleton_walk_ok chain={len(chain)} cycle={len(cycle)}")


def check_timeout_keeps_placeholder() -> None:
    text = (ROOT / "infinigen/core/constraints/example_solver/populate.py").read_text(encoding="utf-8")
    if "SPAWN_LIMIT_S = 180" not in text or "spawn_timeout" not in text:
        raise SystemExit("spawn timeout is not recorded")
    if "continue" not in text.split("spawn_timeout", 1)[1][:400]:
        raise SystemExit("timeout does not keep the placeholder")
    loaded = _load_many(
        ROOT / "infinigen/core/constraints/example_solver/populate.py",
        ["_spawn_limited", "_alarm", "_SpawnTimeout", "SPAWN_LIMIT_S"],
    )
    spawn = loaded["_spawn_limited"]
    alarm = loaded["_alarm"]
    timeout = loaded["_SpawnTimeout"]
    if loaded["SPAWN_LIMIT_S"] != 180:
        raise SystemExit("spawn limit is not 180")

    class Generator:
        def spawn_asset(self, i, loc, rot):
            return {"id": i, "loc": tuple(loc)}

        def finalize_assets(self, objs):
            return None

    class Placeholder:
        location = (1, 2, 3)
        rotation_euler = (0, 0, 0)

    result = spawn(Generator(), 7, Placeholder())
    if result["id"] != 7 or result["loc"] != (1, 2, 3):
        raise SystemExit("fast spawn did not return the asset")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 0.05)
    try:
        time.sleep(0.3)
        raise SystemExit("timeout was not raised")
    except timeout:
        print("event=spawn_timeout_ok limit_s=180")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


if __name__ == "__main__":
    check_stable_pose()
    check_skeleton()
    check_timeout_keeps_placeholder()
    print("event=patch_checks_passed")

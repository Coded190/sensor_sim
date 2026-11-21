"""The expansion queue keeps three CPU solves and does not restart finished scenes."""

import importlib.util
from pathlib import Path

from sensor_sim.scene_gen.infinigen_backend import coarse_command

_SPEC = importlib.util.spec_from_file_location(
    "run_full_expansion",
    Path(__file__).resolve().parents[1] / "scripts" / "infinigen" / "run_full_expansion.py",
)
_QUEUE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_QUEUE)

COARSE_WORKERS = _QUEUE.COARSE_WORKERS
RENDER_SLOTS = _QUEUE.RENDER_SLOTS
ROOMS = _QUEUE.ROOMS
partition_rooms = _QUEUE.partition_rooms
require_full_command = _QUEUE.require_full_command
take_coarse_slots = _QUEUE.take_coarse_slots


def test_livingroom_21_is_outside_the_expansion_queue():
    assert ("LivingRoom", 21) not in ROOMS
    assert COARSE_WORKERS == 3
    assert RENDER_SLOTS == 1


def test_finished_scenes_stay_out_of_both_queues(tmp_path: Path):
    finished = tmp_path / "source_runs" / "Kitchen_full_31"
    frame = finished / "frames" / "Image" / "camera_0"
    frame.mkdir(parents=True)
    (finished / "scene.blend").write_bytes(b"blend")
    (frame / "Image_0_0_0001_0.png").write_bytes(b"png")
    rendered = tmp_path / "source_runs" / "Bedroom_full_41"
    rendered.mkdir(parents=True)
    (rendered / "scene.blend").write_bytes(b"blend")
    coarse, render, done = partition_rooms([("Kitchen", 31), ("Bedroom", 41), ("DiningRoom", 51)], tmp_path)
    assert done == [("Kitchen", 31)]
    assert render == [("Bedroom", 41)]
    assert coarse == [("DiningRoom", 51)]


def test_three_slots_count_the_scene_already_running():
    pending = [("Bathroom", 61), ("Office", 71), ("LivingRoom", 22), ("LivingRoom", 23)]
    start, rest = take_coarse_slots(pending, running=[("DiningRoom", 51)], limit=3)
    assert start == [("Bathroom", 61), ("Office", 71)]
    assert rest == [("LivingRoom", 22), ("LivingRoom", 23)]


def test_full_command_rejects_fast_solve():
    command = coarse_command("Bathroom", 61, Path("/tmp/Bathroom_full_61"), fast=False, python=Path("/usr/bin/python"))
    require_full_command(command)
    assert "fast_solve.gin" not in command
    assert command[command.index("--output_folder") + 1] == "/tmp/Bathroom_full_61"

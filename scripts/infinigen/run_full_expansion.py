"""Full-quality room expansion. Fast solves are not queued.

Three coarse solves run at once. Each solve is the Infinigen simulated-annealing
loop, which is Python and NumPy on the CPU. The Cycles frame uses the one
OptiX device, so frames run one at a time. LivingRoom seed 21 is not regenerated.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from sensor_sim.scene_gen.infinigen_backend import coarse_command, render_command, require_installation

ROOT = Path(__file__).resolve().parents[2] / "dataset" / "qualification" / "asset_realism"
ROOMS = [
    ("Kitchen", 31),
    ("Bedroom", 41),
    ("DiningRoom", 51),
    ("Bathroom", 61),
    ("Office", 71),
    ("LivingRoom", 22),
    ("LivingRoom", 23),
    ("Kitchen", 32),
    ("Kitchen", 33),
    ("Bedroom", 42),
    ("Bedroom", 43),
    ("DiningRoom", 52),
    ("DiningRoom", 53),
    ("Bathroom", 62),
    ("Bathroom", 63),
    ("Office", 72),
    ("Office", 73),
]
COARSE_WORKERS = 3
RENDER_SLOTS = 1
POLL_SECONDS = 5
HEARTBEAT_SECONDS = 300


def output_dir(root: Path, room: str, seed: int) -> Path:
    return root / "source_runs" / f"{room}_full_{seed}"


def blend_path(root: Path, room: str, seed: int) -> Path:
    return output_dir(root, room, seed) / "scene.blend"


def frame_path(root: Path, room: str, seed: int) -> Path:
    return output_dir(root, room, seed) / "frames" / "Image" / "camera_0" / "Image_0_0_0001_0.png"


def partition_rooms(rooms: list[tuple[str, int]], root: Path) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """Split the queue into coarse work, render-only work, and finished scenes."""
    coarse: list[tuple[str, int]] = []
    render: list[tuple[str, int]] = []
    done: list[tuple[str, int]] = []
    for room, seed in rooms:
        has_blend = blend_path(root, room, seed).is_file()
        has_frame = frame_path(root, room, seed).is_file()
        if has_blend and has_frame:
            done.append((room, seed))
        elif has_blend:
            render.append((room, seed))
        else:
            coarse.append((room, seed))
    return coarse, render, done


def take_coarse_slots(
    pending: list[tuple[str, int]],
    running: list[tuple[str, int]],
    limit: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Start enough solves to fill ``limit`` CPU slots. Running scenes stay put."""
    occupied = set(running)
    start: list[tuple[str, int]] = []
    rest: list[tuple[str, int]] = []
    open_slots = max(0, limit - len(occupied))
    for item in pending:
        if item in occupied:
            continue
        if len(start) < open_slots:
            start.append(item)
        else:
            rest.append(item)
    return start, rest


def require_full_command(command: list[str]) -> None:
    if any("fast_solve.gin" in part for part in command):
        raise RuntimeError("reason=fast_solve_in_full_queue")


def live_generators() -> dict[int, str]:
    found: dict[int, str] = {}
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"generate_indoors" not in raw:
            continue
        found[int(entry.name)] = raw.replace(b"\0", b" ").decode(errors="replace")
    return found


def process_elapsed_seconds(pid: int) -> float | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        btime = next(int(line.split()[1]) for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines() if line.startswith("btime "))
    except (OSError, IndexError, StopIteration, ValueError):
        return None
    ticks = os.sysconf("SC_CLK_TCK")
    return max(0.0, time.time() - (btime + start_ticks / ticks))


def _adopt(root: Path, pending: list[tuple[str, int]]) -> list[dict]:
    adopted: list[dict] = []
    commands = live_generators()
    for room, seed in list(pending):
        folder = str(output_dir(root, room, seed))
        matches = [pid for pid, command in commands.items() if folder in command and "--task coarse" in command]
        if not matches:
            continue
        pid = matches[0]
        pending.remove((room, seed))
        adopted.append({
            "room": room,
            "seed": seed,
            "pid": pid,
            "process": None,
            "log": None,
            "started": time.perf_counter() - (process_elapsed_seconds(pid) or 0.0),
            "adopted": True,
        })
    return adopted


def _start_coarse(status: dict, room: str, seed: int, log) -> dict:
    output = output_dir(ROOT, room, seed)
    output.mkdir(parents=True, exist_ok=True)
    command = coarse_command(room, seed, output, fast=False, python=Path(status["python"]))
    require_full_command(command)
    log_path = output / "solver_logs" / "coarse.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=status["root"], stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    log.write(f"event=coarse_start room={room} seed={seed} pid={process.pid} workers={COARSE_WORKERS}\n")
    log.flush()
    return {"room": room, "seed": seed, "pid": process.pid, "process": process, "log": handle, "started": time.perf_counter(), "adopted": False, "command": command}


def _start_render(status: dict, room: str, seed: int, log) -> dict:
    output = output_dir(ROOT, room, seed)
    command = render_command(seed, output, python=Path(status["python"]))
    handle = (output / "solver_logs" / "render.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=status["root"], stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    log.write(f"event=render_start room={room} seed={seed} pid={process.pid} device=optix slots={RENDER_SLOTS}\n")
    log.flush()
    return {"room": room, "seed": seed, "pid": process.pid, "process": process, "log": handle, "started": time.perf_counter()}


def _write_manifest(room: str, seed: int, command: list[str] | None) -> None:
    output = output_dir(ROOT, room, seed)
    record = {
        "room": room,
        "seed": seed,
        "fast_solve": False,
        "coarse_workers": COARSE_WORKERS,
        "render_slots": RENDER_SLOTS,
        "command": command,
        "blend": str(blend_path(ROOT, room, seed)),
        "blend_exists": blend_path(ROOT, room, seed).is_file(),
        "frame_exists": frame_path(ROOT, room, seed).is_file(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "generation_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _coarse_finished(job: dict) -> bool:
    process = job["process"]
    if process is not None:
        return process.poll() is not None
    return not Path(f"/proc/{job['pid']}").exists()


def main() -> None:
    status = require_installation()
    log_path = ROOT / "reports" / "expansion_queue.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    coarse_pending, render_pending, done = partition_rooms(ROOMS, ROOT)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"event=parallel_start commit={status['commit']} fast=false "
            f"coarse_workers={COARSE_WORKERS} render_slots={RENDER_SLOTS} "
            f"reason=coarse_solver_is_cpu_annealing\n"
        )
        for room, seed in done:
            log.write(f"event=scene_skipped room={room} seed={seed} reason=blend_and_frame\n")
        running = _adopt(ROOT, coarse_pending)
        for job in running:
            elapsed = process_elapsed_seconds(job["pid"])
            log.write(f"event=coarse_adopt room={job['room']} seed={job['seed']} pid={job['pid']} elapsed_s={0 if elapsed is None else elapsed:.1f}\n")
        log.flush()
        renders: list[dict] = []
        last_beat = time.perf_counter()
        while coarse_pending or render_pending or running or renders:
            still_running: list[dict] = []
            for job in running:
                if not _coarse_finished(job):
                    still_running.append(job)
                    continue
                if job["log"] is not None:
                    job["log"].close()
                code = job["process"].returncode if job["process"] is not None else "adopted"
                seconds = time.perf_counter() - job["started"]
                has_blend = blend_path(ROOT, job["room"], job["seed"]).is_file()
                log.write(f"event=coarse_done room={job['room']} seed={job['seed']} code={code} seconds={seconds:.1f} blend={str(has_blend).lower()}\n")
                log.flush()
                if has_blend:
                    render_pending.append((job["room"], job["seed"]))
                else:
                    log.write(f"event=coarse_failed room={job['room']} seed={job['seed']} reason=no_blend\n")
                    _write_manifest(job["room"], job["seed"], job.get("command"))
            running = still_running
            start, coarse_pending = take_coarse_slots(coarse_pending, [(job["room"], job["seed"]) for job in running], COARSE_WORKERS)
            for room, seed in start:
                running.append(_start_coarse(status, room, seed, log))
            still_rendering: list[dict] = []
            for job in renders:
                if job["process"].poll() is None:
                    still_rendering.append(job)
                    continue
                job["log"].close()
                seconds = time.perf_counter() - job["started"]
                has_frame = frame_path(ROOT, job["room"], job["seed"]).is_file()
                log.write(f"event=render_done room={job['room']} seed={job['seed']} code={job['process'].returncode} seconds={seconds:.1f} frame={str(has_frame).lower()}\n")
                log.flush()
                _write_manifest(job["room"], job["seed"], None)
            renders = still_rendering
            while len(renders) < RENDER_SLOTS and render_pending:
                room, seed = render_pending.pop(0)
                if frame_path(ROOT, room, seed).is_file():
                    log.write(f"event=render_exists room={room} seed={seed}\n")
                    continue
                if not blend_path(ROOT, room, seed).is_file():
                    log.write(f"event=render_skipped room={room} seed={seed} reason=blend_missing\n")
                    continue
                renders.append(_start_render(status, room, seed, log))
            now = time.perf_counter()
            if now - last_beat >= HEARTBEAT_SECONDS:
                coarse_names = ",".join(f"{job['room']}:{job['seed']}" for job in running) or "none"
                render_names = ",".join(f"{job['room']}:{job['seed']}" for job in renders) or "none"
                log.write(
                    f"event=heartbeat coarse={coarse_names} render={render_names} "
                    f"pending_coarse={len(coarse_pending)} pending_render={len(render_pending)}\n"
                )
                log.flush()
                last_beat = now
            if running or renders:
                time.sleep(POLL_SECONDS)
        log.write("event=expansion_finished\n")
        log.flush()


if __name__ == "__main__":
    main()

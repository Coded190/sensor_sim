"""Peak GPU memory for one render, the frozen Stokes net, and the policy steps.

A step whose nvidia-smi reading would pass 15 GiB is skipped. SmolVLA is
skipped when its package is not installed, because installing it here would
have to be checked against the pinned CUDA torch wheel.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch

from sensor_sim.film.sfp import OUTPUT
from sensor_sim.policy.nets import PolicyNet

logger = logging.getLogger("sensor_sim.policy")

LIMIT_MIB = 15 * 1024
REPORT = OUTPUT / "policy" / "vram.json"


def used_mib() -> int:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(text.strip().splitlines()[0])


def _reset() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def smoke() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []

    def record(name: str, fn) -> None:
        _reset()
        before = used_mib()
        if before > LIMIT_MIB:
            rows.append({"name": name, "skipped": True, "reason": f"gpu already at {before} MiB"})
            logger.info("policy stage=vram name=%s skipped=yes used_mib=%d", name, before)
            return
        peak_used = {"value": before}
        stop = threading.Event()

        def poll() -> None:
            while not stop.is_set():
                try:
                    peak_used["value"] = max(peak_used["value"], used_mib())
                except subprocess.CalledProcessError:
                    pass
                stop.wait(0.05)

        try:
            watcher = threading.Thread(target=poll)
            watcher.start()
            fn()
            torch.cuda.synchronize()
            stop.set()
            watcher.join()
            peak = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
            used = peak_used["value"]
            skipped = used > LIMIT_MIB
            rows.append({"name": name, "skipped": skipped, "allocated_mib": peak, "nvidia_peak_mib": used, "before_mib": before})
            logger.info(
                "policy stage=vram name=%s allocated_mib=%d nvidia_used_mib=%d skipped=%s",
                name,
                peak,
                used,
                skipped,
            )
        except RuntimeError as exc:
            rows.append({"name": name, "skipped": True, "reason": str(exc)})
            logger.info("policy stage=vram name=%s skipped=yes reason=%s", name, exc)
        finally:
            stop.set()
            torch.cuda.empty_cache()

    def mitsuba_frame() -> None:
        from sensor_sim.policy.collect import _record, viewing_camera
        from sensor_sim.policy.observe import _apply, pose_matrix, render_measurement
        from sensor_sim.policy.tasks import scripted_episode

        record = _record("train", "001_sphere")
        episode = scripted_episode("lift", "sphere", (0.0, 0.02), 0.2, float(record["radius_m"]))
        frame = episode["frames"][0]
        pose = frame["object"]
        matrix = pose_matrix(
            record["floor_xy"], record["yaw_rad"], pose["position"][:2], pose["yaw"], pose["tilt"], record["radius_m"]
        )
        render_measurement(
            record["mesh_dir"], matrix, frame["segments"], viewing_camera(0.2), _apply(matrix, record["probe"])
        )

    def stokes_forward() -> None:
        from sensor_sim.policy.stokes_checkpoint import load_stokes

        model, _mean, _std = load_stokes("cuda")
        image = torch.randn(1, 9, 512, 512, device="cuda")
        with torch.no_grad():
            model(image)
        del model

    def policy_step(kind: str, cameras: int) -> None:
        model = PolicyNet(kind, use_second=cameras == 2).cuda()
        rgb = torch.randn(8, 3, 128, 128, device="cuda")
        state = torch.randn(8, 8, device="cuda")
        extra = torch.randn(8, 3, 128, 128, device="cuda") if cameras == 2 else None
        if kind == "act":
            loss = torch.mean(torch.abs(model(rgb, state, extra)))
        else:
            noisy = torch.randn(8, 8, 5, device="cuda")
            time = torch.rand(8, device="cuda")
            loss = torch.mean(model(rgb, state, extra, noisy, time) ** 2)
        loss.backward()
        del model, loss

    record("mitsuba_stokes_frame", mitsuba_frame)
    record("stokes_unet_512", stokes_forward)
    record("act_one_camera_batch8", lambda: policy_step("act", 1))
    record("act_two_camera_batch8", lambda: policy_step("act", 2))
    record("diffusion_one_camera_batch8", lambda: policy_step("diffusion", 1))
    record("diffusion_two_camera_batch8", lambda: policy_step("diffusion", 2))
    try:
        import lerobot  # noqa: F401

        rows.append({"name": "smolvla_batch1", "skipped": False, "reason": "imported; weights are loaded in the policy run"})
    except ImportError:
        rows.append(
            {
                "name": "smolvla_batch1",
                "skipped": True,
                "reason": "lerobot is not installed. A dry-run of lerobot 0.7.0 would downgrade numpy 2.5.4 to 2.2.6 and add torchvision 0.29.2, which is not paired with torch 2.15.0+cu132. The published envelope is about 10-16 GB, so it was not loaded.",
            }
        )
        logger.info("policy stage=vram name=smolvla_batch1 skipped=yes reason=lerobot_not_installed")
    report = {"limit_mib": LIMIT_MIB, "rows": rows}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("policy stage=vram_done path=%s", REPORT)
    return report


if __name__ == "__main__":
    smoke()

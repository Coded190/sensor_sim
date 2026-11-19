"""Train RGB policies, then finetune a zero-initialized second image.

The RGB run is the baseline. The finetunes add Stokes normals, ground-truth
normals, or the raw DoLP image. The fusion layer starts at zero, so the
finetune begins from the RGB policy's actions. Its learning rate is one tenth
of the RGB run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sensor_sim.film.sfp import OUTPUT
from sensor_sim.policy.nets import PolicyNet

logger = logging.getLogger("sensor_sim.policy")

DATA = OUTPUT / "policy" / "demos"
CKPT = OUTPUT / "policy" / "checkpoints"
CHUNK = 8
RGB_LR = 1e-4
MAX_RGB_EPOCHS = 20
FINETUNE_EPOCHS = 10


def load_index() -> list[dict]:
    return json.loads((DATA / "index.json").read_text())["episodes"]


def load_split(split: str) -> list[dict]:
    rows = []
    for item in load_index():
        if item["split"] != split:
            continue
        arrays = np.load(item["path"])
        row = dict(item)
        row["arrays"] = {key: arrays[key] for key in ("rgb", "normals", "gt_normals", "dolp", "state", "action")}
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no {split} episodes")
    return rows


def _stats(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([row["arrays"][key].astype(np.float32) for row in rows], axis=0)
    mean = values.mean(axis=(0, 1, 2))
    std = values.std(axis=(0, 1, 2))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _action_stats(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([row["arrays"]["action"].astype(np.float32) for row in rows], axis=0)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _batch(rows: list[dict], modality: str, size: int, generator: torch.Generator, device: str, stats: dict) -> dict:
    images = []
    extras = []
    states = []
    actions = []
    for _ in range(size):
        row = rows[int(torch.randint(0, len(rows), (1,), generator=generator))]
        length = row["arrays"]["action"].shape[0]
        start = int(torch.randint(0, length, (1,), generator=generator))
        indexes = [min(start + offset, length - 1) for offset in range(CHUNK)]
        images.append(row["arrays"]["rgb"][start].astype(np.float32))
        if modality != "rgb":
            extras.append(row["arrays"][modality][start].astype(np.float32))
        states.append(row["arrays"]["state"][start].astype(np.float32))
        actions.append(row["arrays"]["action"][indexes].astype(np.float32))

    def norm_image(batch: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
        scaled = (batch - mean) / std
        return torch.tensor(np.transpose(scaled, (0, 3, 1, 2)), dtype=torch.float32, device=device)

    rgb = norm_image(np.stack(images), stats["rgb_mean"], stats["rgb_std"])
    extra = None
    if modality != "rgb":
        extra = norm_image(np.stack(extras), stats["extra_mean"], stats["extra_std"])
    state = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
    action = torch.tensor(np.stack(actions), dtype=torch.float32, device=device)
    action = (action - torch.tensor(stats["action_mean"], device=device)) / torch.tensor(stats["action_std"], device=device)
    return {"rgb": rgb, "extra": extra, "state": state, "action": action}


def _loss(model: PolicyNet, batch: dict) -> torch.Tensor:
    if model.kind == "act":
        prediction = model(batch["rgb"], batch["state"], batch["extra"])
        return torch.mean(torch.abs(prediction - batch["action"]))
    step = torch.randint(1, 9, (batch["action"].shape[0],), device=batch["action"].device)
    scale = step.float().view(-1, 1, 1) / 8.0
    noise = torch.randn_like(batch["action"])
    noisy = batch["action"] + scale * noise
    prediction = model(batch["rgb"], batch["state"], batch["extra"], noisy, scale.view(-1))
    return torch.mean((prediction - noise) ** 2)


def _run_epochs(model, rows, modality, stats, epochs, lr, seed, device, tag: str) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    losses = []
    steps = max(8, len(rows) * 40 // 8)
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for _ in range(steps):
            batch = _batch(rows, modality, 8, generator, device, stats)
            loss = _loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(epoch_loss))
        losses.append(mean_loss)
        logger.info("policy stage=train tag=%s epoch=%d loss=%.6f steps=%d", tag, epoch, mean_loss, steps)
        if epoch >= 5 and losses[-1] > 0.99 * min(losses[-3:-1]):
            logger.info("policy stage=train tag=%s stop=plateau epoch=%d", tag, epoch)
            break
    return losses


def _save(path: Path, model: PolicyNet, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **payload}, path)
    logger.info("policy stage=checkpoint path=%s", path)


def _copy_rgb(model: PolicyNet, state: dict) -> None:
    own = model.state_dict()
    for key, value in state.items():
        if key.startswith("extra."):
            continue
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            own[key] = value
    model.load_state_dict(own)
    nn.init.zeros_(model.fuse.weight)
    nn.init.zeros_(model.fuse.bias)


def assert_second_image_is_connected(device: str = "cpu") -> None:
    """The zero-init fusion matches RGB, then a step makes the second image matter."""
    model = PolicyNet("act", use_second=True).to(device)
    rgb = torch.randn(2, 3, 32, 32, device=device)
    extra = torch.randn(2, 3, 32, 32, device=device)
    state = torch.randn(2, 8, device=device)
    model.eval()
    with torch.no_grad():
        with_image = model(rgb, state, extra)
        without_image = model(rgb, state, torch.zeros_like(extra))
    if not torch.allclose(with_image, without_image, atol=1e-5):
        raise RuntimeError("zero-init fusion changed the action before any update")
    model.train()
    loss = torch.mean(torch.abs(model(rgb, state, extra)))
    loss.backward()
    if model.fuse.weight.grad is None or float(model.fuse.weight.grad.abs().sum()) == 0.0:
        raise RuntimeError("the second-image fusion received no gradient")
    torch.optim.SGD(model.parameters(), lr=1e-3).step()
    model.zero_grad()
    model(rgb, state, extra).sum().backward()
    encoder_grad = model.extra.stem[0].weight.grad
    if encoder_grad is None or float(encoder_grad.abs().sum()) == 0.0:
        raise RuntimeError("the second-image encoder received no gradient")
    model.eval()
    with torch.no_grad():
        changed = model(rgb, state, extra)
        zeroed = model(rgb, state, torch.zeros_like(extra))
    if not torch.isfinite(changed).all() or torch.allclose(changed, zeroed, atol=1e-8):
        raise RuntimeError("zeroing the second image did not change the action after one step")
    logger.info("policy stage=connect_ok")


def train_all(device: str = "cuda") -> None:
    """RGB baselines for ACT and diffusion, then three finetunes, three seeds."""
    assert_second_image_is_connected(device)
    train_rows = load_split("train")
    stats_rgb = _image_pair_stats(train_rows, "rgb")
    for kind in ("act", "diffusion"):
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = PolicyNet(kind, use_second=False).to(device)
            tag = f"{kind}_rgb_seed{seed}"
            losses = _run_epochs(model, train_rows, "rgb", stats_rgb, MAX_RGB_EPOCHS, RGB_LR, seed, device, tag)
            rgb_path = CKPT / f"{tag}.pt"
            _save(
                rgb_path,
                model,
                {
                    "kind": kind,
                    "modality": "rgb",
                    "seed": seed,
                    "losses": losses,
                    "stats": _json_stats(stats_rgb),
                },
            )
            for modality in ("normals", "gt_normals", "dolp"):
                finetune = PolicyNet(kind, use_second=True).to(device)
                _copy_rgb(finetune, model.state_dict())
                stats = _image_pair_stats(train_rows, modality)
                fine_tag = f"{kind}_{modality}_seed{seed}"
                fine_losses = _run_epochs(
                    finetune, train_rows, modality, stats, FINETUNE_EPOCHS, RGB_LR / 10.0, seed, device, fine_tag
                )
                _save(
                    CKPT / f"{fine_tag}.pt",
                    finetune,
                    {
                        "kind": kind,
                        "modality": modality,
                        "seed": seed,
                        "losses": fine_losses,
                        "rgb_checkpoint": str(rgb_path),
                        "stats": _json_stats(stats),
                        "lr": RGB_LR / 10.0,
                    },
                )
            del model
            if device == "cuda":
                torch.cuda.empty_cache()


def _image_pair_stats(rows: list[dict], modality: str) -> dict:
    rgb_mean, rgb_std = _stats(rows, "rgb")
    if modality == "rgb":
        extra_mean, extra_std = rgb_mean, rgb_std
    else:
        extra_mean, extra_std = _stats(rows, modality)
    action_mean, action_std = _action_stats(rows)
    return {
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "extra_mean": extra_mean,
        "extra_std": extra_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


def _json_stats(stats: dict) -> dict:
    return {key: value.tolist() for key, value in stats.items()}


def load_policy(path: Path, device: str) -> tuple[PolicyNet, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PolicyNet(payload["kind"], use_second=payload["modality"] != "rgb").to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in payload["stats"].items()}
    return model, stats

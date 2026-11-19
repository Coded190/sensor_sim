"""ACT and a small diffusion policy. Both read RGB, and optionally a second image.

The second image is fused through a linear layer whose weights start at zero,
so the first forward pass matches the RGB policy. A normal map does not use
ImageNet statistics, so its encoder is trained from scratch.
"""

from __future__ import annotations

import torch
from torch import nn


class _Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.bn1(self.conv1(x)))
        hidden = self.bn2(self.conv2(hidden))
        return torch.relu(hidden + self.skip(x))


class ResNet18Encoder(nn.Module):
    """ResNet-18 trunk. The output is a 128-d pooled vector, not a class logit."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_Block(64, 64, 1), _Block(64, 64, 1))
        self.layer2 = nn.Sequential(_Block(64, 128, 2), _Block(128, 128, 1))
        self.layer3 = nn.Sequential(_Block(128, 256, 2), _Block(256, 256, 1))
        self.layer4 = nn.Sequential(_Block(256, 512, 2), _Block(512, 512, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        hidden = self.layer1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        hidden = self.layer4(hidden)
        return self.proj(self.pool(hidden).flatten(1))


class PolicyNet(nn.Module):
    """Chunk predictor. ``kind`` is ``act`` (L1 regression) or ``diffusion`` (noise prediction)."""

    def __init__(self, kind: str, use_second: bool, chunk: int = 8) -> None:
        super().__init__()
        if kind not in ("act", "diffusion"):
            raise ValueError(f"kind must be act or diffusion, got {kind}")
        self.kind = kind
        self.chunk = int(chunk)
        self.rgb = ResNet18Encoder()
        self.extra = ResNet18Encoder() if use_second else None
        self.fuse = nn.Linear(128, 128)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.state = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128))
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(128, self.chunk * 5)
        self.noise_net = nn.Sequential(
            nn.Linear(128 + 32 + self.chunk * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.chunk * 5),
        )
        self.time_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32))

    def features(self, rgb: torch.Tensor, state: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        visual = self.rgb(rgb)
        if self.extra is not None:
            if extra is None:
                raise ValueError("this checkpoint expects a second image")
            visual = visual + self.fuse(self.extra(extra))
        tokens = torch.stack([visual, self.state(state)], dim=1)
        return self.encoder(tokens)[:, 0]

    def forward(self, rgb: torch.Tensor, state: torch.Tensor, extra: torch.Tensor | None = None, noisy: torch.Tensor | None = None, time: torch.Tensor | None = None) -> torch.Tensor:
        cond = self.features(rgb, state, extra)
        if self.kind == "act":
            return self.head(cond).view(-1, self.chunk, 5)
        if noisy is None or time is None:
            raise ValueError("diffusion forward needs the noisy action and the timestep")
        step = self.time_net(time.view(-1, 1))
        flat = noisy.view(noisy.shape[0], -1)
        return self.noise_net(torch.cat([cond, step, flat], dim=-1)).view(-1, self.chunk, 5)

    def act(self, rgb: torch.Tensor, state: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """First action of the chunk. Diffusion uses 8 deterministic denoising steps."""
        if self.kind == "act":
            return self.forward(rgb, state, extra)[:, 0]

        def predict(sample: torch.Tensor, scale: float) -> torch.Tensor:
            time = torch.full((sample.shape[0],), scale, device=sample.device)
            return self.forward(rgb, state, extra, sample, time)

        # Variance of ``clean + 1 * epsilon`` is 2 when the clean action is standardized.
        sample = torch.randn(rgb.shape[0], self.chunk, 5, device=rgb.device) * (2.0 ** 0.5)
        return denoise_epsilon(sample, predict)[:, 0]


def denoise_epsilon(sample: torch.Tensor, predict) -> torch.Tensor:
    """Invert ``noisy = clean + t * epsilon`` over t = 1, 7/8, ..., 1/8.

    ``predict(sample, t)`` returns epsilon. The same epsilon is reused at the
    next smaller t, so a perfect predictor returns the clean sample.
    """
    for step in range(8, 0, -1):
        scale = step / 8.0
        epsilon = predict(sample, scale)
        clean = sample - scale * epsilon
        next_scale = (step - 1) / 8.0
        sample = clean + next_scale * epsilon
    return sample

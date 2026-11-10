"""Ronneberger U-Net with four pools and a base width of 32.

The module returns one logit channel. The probability is the sigmoid of
that logit. Model A uses 3 input channels and Model B uses 7. The first
convolution is the only layer whose weight count changes.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FilmUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        if in_channels not in (3, 7):
            raise ValueError(f"in_channels must be 3 or 7, got {in_channels}")
        widths = [base_channels * (2**level) for level in range(5)]
        self.encoders = nn.ModuleList()
        incoming = in_channels
        for width in widths[:4]:
            self.encoders.append(ConvBlock(incoming, width))
            incoming = width
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = ConvBlock(widths[3], widths[4])
        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose2d(widths[4], widths[3], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                ConvBlock(widths[3] * 2, widths[3]),
                ConvBlock(widths[2] * 2, widths[2]),
                ConvBlock(widths[1] * 2, widths[1]),
                ConvBlock(widths[0] * 2, widths[0]),
            ]
        )
        self.head = nn.Conv2d(widths[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 16 or x.shape[-2] % 16:
            raise ValueError(f"height and width must be divisible by 16, got {tuple(x.shape[-2:])}")
        skips = []
        hidden = x
        for block in self.encoders:
            hidden = block(hidden)
            skips.append(hidden)
            hidden = self.pool(hidden)
        hidden = self.bottleneck(hidden)
        for index, (up, block) in enumerate(zip(self.ups, self.decoders, strict=True)):
            hidden = up(hidden)
            skip = skips[-(index + 1)]
            if hidden.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(f"skip {skip.shape} does not match upsampled {hidden.shape}")
            hidden = block(torch.cat([hidden, skip], dim=1))
        return self.head(hidden)


def parameter_count(model: nn.Module) -> int:
    """Number of trained scalars, including batch-norm scale and shift."""
    return sum(parameter.numel() for parameter in model.parameters())

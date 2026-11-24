"""Lightweight Mobile-UNet for semantic segmentation of Territorial.io screenshots.

Classes:
  0: UI / Overlay / Background
  1: Water
  2: Mountain / Impassable
  3: Neutral Territory
  4: Occupied Player Territory Interior
  5: Territory Darkened Inner Border (critical for separating same-hue neighbors)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SEG_CLASSES: tuple[str, ...] = (
    "ui_bg",
    "water",
    "mountain",
    "neutral",
    "player_interior",
    "player_border",
)


class _DSConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VisionUNetMobile(nn.Module):
    """Fast multi-scale UNet (~160k params) for pure-pixel territory & border segmentation."""

    def __init__(self, in_ch: int = 3, num_classes: int = len(SEG_CLASSES), base: int = 24) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True),
        )
        self.enc1 = _DSConvBlock(base, base)
        self.enc2 = _DSConvBlock(base, base * 2)
        self.enc3 = _DSConvBlock(base * 2, base * 4)
        self.bottleneck = _DSConvBlock(base * 4, base * 4)

        self.dec3 = _DSConvBlock(base * 8, base * 2)
        self.dec2 = _DSConvBlock(base * 4, base)
        self.dec1 = _DSConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)

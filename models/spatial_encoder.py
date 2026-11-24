"""Multi-scale IMPALA-style Residual Visual Encoder with CoordConv for Territorial.io CV tensors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.silu(x + out)


class _ImpalaStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(8, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = _ResBlock(out_ch)
        self.res2 = _ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.silu(self.gn(self.conv(x))))
        x = self.res1(x)
        return self.res2(x)


class SpatialVisualEncoder(nn.Module):
    """Encodes [B, C, 128, 128] pure-CV visual observation maps into:
    - global_embed: [B, out_dim]
    - spatial_feat: [B, spatial_ch, 32, 32] (for 2D spatial pointer decoding)
    """

    def __init__(self, in_channels: int = 11, out_dim: int = 128, spatial_ch: int = 64) -> None:
        super().__init__()
        # +2 channels for normalized CoordConv (x, y)
        self.stage1 = _ImpalaStage(in_channels + 2, 32)  # 128 -> 64
        self.stage2 = _ImpalaStage(32, spatial_ch)       # 64 -> 32 (kept for spatial pointer!)
        self.stage3 = _ImpalaStage(spatial_ch, 96)       # 32 -> 16
        self.stage4 = _ImpalaStage(96, 128)              # 16 -> 8
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _append_coords(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xs, ys], dim=1)

    def forward(self, obs_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._append_coords(obs_map)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)  # [B, spatial_ch, 32, 32]
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        global_embed = self.proj(s4)
        return global_embed, s2

"""Fast Canvas-Font Glyph/Digit CNN for Territorial.io HUD & Map Troop OCR.

Designed to classify cropped white-on-dark / outlined sans-serif glyphs
('0'-'9', '.', 'K', 'M', '%', '+', ':') in < 1.5ms per batch on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

OCR_VOCAB: tuple[str, ...] = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ".", "K", "M", "%", "+", ":", "?",
)
CHAR_TO_IDX: dict[str, int] = {ch: idx for idx, ch in enumerate(OCR_VOCAB)}
IDX_TO_CHAR: dict[int, str] = {idx: ch for idx, ch in enumerate(OCR_VOCAB)}


class CanvasOCRNet(nn.Module):
    """Lightweight depthwise-separable CNN for 24x24 grayscale glyph crops."""

    def __init__(self, num_classes: int = len(OCR_VOCAB)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),  # 12x12
            nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24, bias=False),
            nn.Conv2d(24, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),  # 6x6
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),
            nn.Conv2d(48, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

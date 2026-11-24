"""Fast pure-pixel Canvas OCR for Territorial.io HUD numbers, percentages, clocks, and map troop counts.

Combines:
1. Fast white-glyph connected-component segmentation (handles outlined white/pink/yellow canvas text)
2. Trained `CanvasOCRNet` depthwise-separable CNN (with pure-numpy glyph template fallback if untrained)
3. Exact numeric parser supporting integers ('12 450'), abbreviations ('14.2K', '1.35M'),
   percentages ('7.00%'), and match clocks ('2:45').
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch

from models.ocr_net import IDX_TO_CHAR, OCR_VOCAB, CanvasOCRNet


def parse_troop_string(text: str) -> float | None:
    """Parse Territorial.io troop strings like '850', '12 400', '14.2K', '2.1M'."""
    cleaned = text.strip().upper().replace(" ", "").replace(",", "")
    if not cleaned:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*([KM])?", cleaned)
    if not m:
        return None
    val = float(m.group(1))
    suffix = m.group(2)
    if suffix == "K":
        val = round(val * 1_000.0, 2)
    elif suffix == "M":
        val = round(val * 1_000_000.0, 2)
    return val


def parse_percent_string(text: str) -> float | None:
    cleaned = text.strip().replace(" ", "")
    m = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if not m:
        return None
    val = float(m.group(1))
    return val if 0.0 <= val <= 100.0 else None


def parse_clock_string(text: str) -> int | None:
    m = re.search(r"(\d{1,2}):(\d{2})", text.strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def render_synthetic_glyph(char: str, size: int = 24, font_scale: float = 0.68, thickness: int = 2) -> np.ndarray:
    """Render a clean 24x24 grayscale glyph matching bold canvas sans-serif style."""
    canvas = np.zeros((size, size), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(char, font, font_scale, thickness)
    x = max(1, (size - tw) // 2)
    y = min(size - 2, (size + th) // 2)
    cv2.putText(canvas, char, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
    return canvas


class FastCanvasOCR:
    """Sub-2ms CPU canvas OCR reader backed by CanvasOCRNet + template fallback."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = "models/checkpoints/ocr_net.pt",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.net = CanvasOCRNet().to(self.device).eval()
        self.neural_ready = False
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.net.load_state_dict(state["model_state"] if "model_state" in state else state)
                self.neural_ready = True
            except Exception:
                self.neural_ready = False

        # Pre-rendered multi-scale templates as zero-dependency fallback before Stage 1 completes
        self._templates: dict[str, np.ndarray] = {}
        for ch in OCR_VOCAB[:-1]:
            tmpl = render_synthetic_glyph(ch, size=24).astype(np.float32) / 255.0
            norm = float(np.linalg.norm(tmpl)) + 1e-6
            self._templates[ch] = tmpl / norm

    def _extract_glyph_crops(self, rgb_patch: np.ndarray, bright_thresh: int = 175) -> list[tuple[int, np.ndarray]]:
        """Segment individual bright canvas glyphs (white, pink red-interest, yellow bonus) left-to-right."""
        if rgb_patch.size == 0 or rgb_patch.shape[0] < 6 or rgb_patch.shape[1] < 4:
            return []
        # Max channel picks up white (255,255,255), red-interest (255,120,100), yellow (255,255,100)
        max_c = rgb_patch.max(axis=2)
        min_c = rgb_patch.min(axis=2)
        # Keep bright neutral/warm foreground text while ignoring saturated dark backgrounds
        fg = ((max_c >= bright_thresh) & ((min_c >= 90) | (rgb_patch[:, :, 0] >= 200))).astype(np.uint8) * 255

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        h_img = rgb_patch.shape[0]
        raw_boxes: list[tuple[int, int, int, int]] = []
        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            if area < 4 or h < max(3, int(h_img * 0.16)) or w > h_img * 1.4:
                continue
            raw_boxes.append((int(x), int(y), int(w), int(h)))

        # Merge vertically stacked dots/colons (e.g. ':') sharing horizontal overlap
        raw_boxes.sort(key=lambda b: b[0])
        merged: list[tuple[int, int, int, int]] = []
        for bx, by, bw, bh in raw_boxes:
            if merged:
                px, py, pw, ph = merged[-1]
                overlap = min(px + pw, bx + bw) - max(px, bx)
                if overlap > 0 and max(bw, pw) <= max(4, int(h_img * 0.32)):
                    nx0 = min(px, bx)
                    ny0 = min(py, by)
                    nx1 = max(px + pw, bx + bw)
                    ny1 = max(py + ph, by + bh)
                    merged[-1] = (nx0, ny0, nx1 - nx0, ny1 - ny0)
                    continue
            merged.append((bx, by, bw, bh))

        crops: list[tuple[int, np.ndarray]] = []
        for bx, by, bw, bh in merged:
            glyph = fg[by : by + bh, bx : bx + bw]
            # Pad to square preserving aspect ratio, then resize to 24x24
            side = max(bw, bh) + 4
            sq = np.zeros((side, side), dtype=np.uint8)
            oy = (side - bh) // 2
            ox = (side - bw) // 2
            sq[oy : oy + bh, ox : ox + bw] = glyph
            resized = cv2.resize(sq, (24, 24), interpolation=cv2.INTER_AREA)
            crops.append((bx, resized))
        return crops

    def read_text(self, rgb_patch: np.ndarray, bright_thresh: int = 175) -> tuple[str, float]:
        crops = self._extract_glyph_crops(rgb_patch, bright_thresh=bright_thresh)
        if not crops:
            return "", 0.0

        imgs = [c[1].astype(np.float32) / 255.0 for c in crops]
        if self.neural_ready:
            batch = torch.from_numpy(np.stack(imgs, axis=0)).unsqueeze(1).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.net(batch), dim=-1)
                confs, preds = probs.max(dim=-1)
            chars = [IDX_TO_CHAR[int(p)] for p in preds.cpu().tolist()]
            conf = float(confs.mean().item())
        else:
            chars = []
            scores = []
            for img in imgs:
                norm = float(np.linalg.norm(img)) + 1e-6
                u = img / norm
                best_ch, best_sc = "?", 0.0
                for ch, tmpl in self._templates.items():
                    sc = float(np.sum(u * tmpl))
                    if sc > best_sc:
                        best_ch, best_sc = ch, sc
                chars.append(best_ch)
                scores.append(best_sc)
            conf = float(np.mean(scores)) if scores else 0.0

        text = "".join(c for c in chars if c != "?")
        return text, conf

    def read_troops(self, rgb_patch: np.ndarray) -> tuple[float | None, float]:
        text, conf = self.read_text(rgb_patch, bright_thresh=180)
        return parse_troop_string(text), conf

    def read_percent(self, rgb_patch: np.ndarray) -> tuple[float | None, float]:
        text, conf = self.read_text(rgb_patch, bright_thresh=175)
        return parse_percent_string(text), conf

    def read_clock(self, rgb_patch: np.ndarray) -> tuple[int | None, float]:
        text, conf = self.read_text(rgb_patch, bright_thresh=175)
        return parse_clock_string(text), conf

"""Master visual perception assembler turning raw RGB screenshots into structured PerceivedFrame."""

from __future__ import annotations

import numpy as np

from core.config import VisionConfig
from core.types import PerceivedFrame, RawFrame, ScreenState
from perception.entity_tracker import TerritoryEntityTracker
from perception.self_identifier import SelfTerritoryIdentifier
from perception.temporal_tracker import TemporalChangeTracker
from vision.boat_and_icon_detector import BoatAndIconDetector
from vision.ocr import FastCanvasOCR
from vision.screen_classifier import ScreenStateClassifier
from vision.segmentation import TerritorySegmenter
from vision.ui_detector import ScaleInvariantUIDetector


class VisualPerceptionPipeline:
    """Complete pure-pixel perception pipeline: RawFrame -> PerceivedFrame."""

    def __init__(self, config: VisionConfig | None = None, device: str = "cpu") -> None:
        self.cfg = config or VisionConfig()
        ocr_ckpt = self.cfg.ocr_checkpoint if self.cfg.use_neural_ocr else None
        seg_ckpt = self.cfg.seg_checkpoint if self.cfg.use_neural_seg else None

        self.ocr = FastCanvasOCR(checkpoint_path=ocr_ckpt, device=device)
        self.ui_detector = ScaleInvariantUIDetector(ocr=self.ocr)
        self.screen_classifier = ScreenStateClassifier(ui_detector=self.ui_detector)
        self.segmenter = TerritorySegmenter(
            target_size=self.cfg.obs_grid_size,
            unet_checkpoint=seg_ckpt,
            device=device,
        )
        self.boat_detector = BoatAndIconDetector()
        self.self_id = SelfTerritoryIdentifier()
        self.entity_tracker = TerritoryEntityTracker(ocr=self.ocr, max_entities=self.cfg.max_entities)
        self.temporal = TemporalChangeTracker(grid_size=self.cfg.obs_grid_size)
        self._last_self_troops = 0.0

    def reset(self) -> None:
        self.screen_classifier.reset()
        self.self_id.reset()
        self.entity_tracker.reset()
        self.temporal.reset()
        self._last_self_troops = 0.0

    def process(self, frame: RawFrame) -> PerceivedFrame:
        rgb = frame.rgb
        h, w = frame.height, frame.width
        state, ui = self.screen_classifier.classify(rgb)
        ui_mask = self.ui_detector.build_ui_mask(h, w, ui)

        ds = self.cfg.obs_grid_size
        if state not in (ScreenState.IN_GAME, ScreenState.SPAWN_SELECTION, ScreenState.SPECTATING):
            empty_i32 = np.zeros((ds, ds), dtype=np.int32)
            empty_u8 = np.zeros((ds, ds), dtype=np.uint8)
            empty_f32 = np.zeros((ds, ds), dtype=np.float32)
            return PerceivedFrame(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                width=w,
                height=h,
                screen_state=state,
                ui=ui,
                label_map=empty_i32,
                self_mask=empty_u8,
                neutral_mask=empty_u8,
                water_mask=empty_u8,
                combat_delta_map=empty_f32,
            )

        seg = self.segmenter.segment(rgb, ui_mask=ui_mask)
        if state == ScreenState.IN_GAME:
            self_eid, self_mask = self.self_id.identify(seg)
        else:
            self_eid = None
            self_mask = np.zeros((ds, ds), dtype=np.uint8)

        combat_delta = self.temporal.update(rgb, seg.label_map, ui_mask=ui_mask)
        crown_pt = self.boat_detector.detect_crown_point(rgb, ui_mask=ui_mask)
        boats = self.boat_detector.detect_boats(rgb, seg.water_mask, self_mask)

        entities = self.entity_tracker.extract_entities(
            full_rgb=rgb,
            seg=seg,
            self_entity_id=self_eid,
            self_mask=self_mask,
            hud_self_troops=ui.hud_troops,
            crown_norm=crown_pt,
        )

        # Compute aggregate land ratios (excluding UI and water/mountains)
        valid_land = (seg.label_map >= 0).sum()
        self_land_px = int(self_mask.sum())
        neutral_px = int(seg.neutral_mask.sum())
        denom = float(max(1, valid_land))

        if ui.hud_troops is not None:
            self._last_self_troops = float(ui.hud_troops)
        elif self_eid is not None:
            for e in entities:
                if e.is_self:
                    self._last_self_troops = e.estimated_troops
                    break

        alive_est = ui.hud_players_alive if ui.hud_players_alive is not None else max(1, len(entities))

        return PerceivedFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            width=w,
            height=h,
            screen_state=state,
            ui=ui,
            label_map=seg.label_map,
            self_mask=self_mask,
            neutral_mask=seg.neutral_mask,
            water_mask=seg.water_mask,
            combat_delta_map=combat_delta,
            entities=entities,
            boats=boats,
            self_entity_id=self_eid,
            self_troops=self._last_self_troops,
            self_land_px=self_land_px,
            self_land_ratio=self_land_px / denom,
            neutral_land_ratio=neutral_px / denom,
            alive_count_est=alive_est,
        )

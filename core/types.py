"""Immutable data contracts connecting Vision -> Perception -> Map -> Brain -> Control -> Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np


class ScreenState(StrEnum):
    UNKNOWN = "UNKNOWN"
    MAIN_MENU = "MAIN_MENU"
    SINGLEPLAYER_LOBBY = "SINGLEPLAYER_LOBBY"
    SPAWN_SELECTION = "SPAWN_SELECTION"
    IN_GAME = "IN_GAME"
    SPECTATING = "SPECTATING"
    VICTORY = "VICTORY"
    DEFEAT = "DEFEAT"
    MODAL_OVERLAY = "MODAL_OVERLAY"


class ActionType(IntEnum):
    WAIT = 0
    EXPAND_NEUTRAL = 1
    ATTACK_ENEMY = 2
    LAUNCH_BOAT = 3
    SET_SLIDER = 4
    CAMERA_PAN = 5
    CAMERA_ZOOM = 6
    PEACE_VOTE = 7
    INTERCEPT_BOAT = 8
    UI_CLICK = 9


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def as_slice(self) -> tuple[slice, slice]:
        return slice(max(0, self.y), max(0, self.y + self.h)), slice(max(0, self.x), max(0, self.x + self.w))


@dataclass
class RawFrame:
    rgb: np.ndarray  # [H, W, 3] uint8 RGB screenshot from external display
    timestamp: float
    frame_id: int
    window_bbox: BBox

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


@dataclass
class UILayout:
    screen_state: ScreenState
    balance_box: BBox | None = None
    stats_panel: BBox | None = None
    slider_box: BBox | None = None
    slider_track_x0: int = 0
    slider_track_x1: int = 0
    leaderboard_box: BBox | None = None
    singleplayer_btn: BBox | None = None
    multiplayer_btn: BBox | None = None
    start_match_btn: BBox | None = None
    back_btn: BBox | None = None
    play_again_btn: BBox | None = None
    estimated_slider_pct: float = 0.5
    balance_gaining: bool = True
    red_interest_active: bool = False
    cycle_progress: float = 0.0
    hud_troops: int | None = None
    hud_interest_pct: float | None = None
    hud_land: int | None = None
    hud_elapsed_sec: int | None = None
    hud_players_alive: int | None = None


@dataclass
class TerritoryEntity:
    entity_id: int
    is_self: bool
    is_neutral: bool
    mean_rgb: tuple[int, int, int]
    area_px: int
    centroid_norm: tuple[float, float]  # (x/W, y/H) in [0, 1]
    bbox: BBox
    estimated_troops: float
    troop_confidence: float
    name_hint: str = ""
    border_contact_with_self_px: int = 0
    coastal_px: int = 0
    delta_area_px: float = 0.0
    delta_troops: float = 0.0
    is_under_attack: bool = False
    has_crown: bool = False
    attack_point_norm: tuple[float, float] | None = None


@dataclass
class BoatDetection:
    x_norm: float
    y_norm: float
    vx_norm: float = 0.0
    vy_norm: float = 0.0
    is_hostile: bool = True
    distance_to_self_coast: float = 1.0


@dataclass
class PerceivedFrame:
    frame_id: int
    timestamp: float
    width: int
    height: int
    screen_state: ScreenState
    ui: UILayout
    label_map: np.ndarray  # [H_ds, W_ds] int32: -2=water, -1=mountain, 0=neutral, >0=entity_id
    self_mask: np.ndarray  # [H_ds, W_ds] uint8 binary
    neutral_mask: np.ndarray  # [H_ds, W_ds] uint8 binary
    water_mask: np.ndarray  # [H_ds, W_ds] uint8 binary
    combat_delta_map: np.ndarray  # [H_ds, W_ds] float32 active border flux
    entities: list[TerritoryEntity] = field(default_factory=list)
    boats: list[BoatDetection] = field(default_factory=list)
    self_entity_id: int | None = None
    self_troops: float = 0.0
    self_land_px: int = 0
    self_land_ratio: float = 0.0
    neutral_land_ratio: float = 0.0
    alive_count_est: int = 0


@dataclass
class ActionDecision:
    action_type: ActionType
    target_x_norm: float = 0.5  # normalized viewport coordinate [0, 1]
    target_y_norm: float = 0.5
    slider_pct: float = 0.25  # requested commitment ratio in [0.02, 1.0]
    target_entity_id: int | None = None
    camera_dx: float = 0.0
    camera_dy: float = 0.0
    zoom_delta: int = 0
    confidence: float = 1.0
    rationale: str = ""
    log_prob: float = 0.0
    value_est: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    succeeded: bool
    reason: str
    pixel_diff_score: float
    troop_delta: float
    state_before: ScreenState
    state_after: ScreenState
    latency_ms: float

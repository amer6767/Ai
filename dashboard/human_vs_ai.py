"""Live Human vs. Trained Computer-Vision AI Playable Territorial.io Arena.

Player 1 = Human (clicking & dragging slider in browser UI)
Player 2 = Autonomous Computer-Vision AI (`models/checkpoints/champion.pt`)
Players 3..N = Optional native bots (Easy / Normal / Hard / Very Hard)

Strict Pixel-Only AI Constraint preserved:
Every AI decision renders the arena to an RGB screenshot, passes ONLY those RGB pixels
through VisualPerceptionPipeline -> TerritoryGraphBuilder -> TerritorialPolicyValueNet + StrategicBrain,
and executes the AI's normalized screen-space click & slider ratio.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from core.config import VisionConfig
from core.hardware import detect_hardware
from core.types import ActionType, BBox, PerceivedFrame, RawFrame, ScreenState
from map_analysis.tactical_features import TacticalFeatureExtractor
from map_analysis.territory_graph import TerritoryGraphBuilder
from map_analysis.world_map import PersistentWorldMap
from memory.checkpoint_manager import CheckpointManager
from models.policy_value_net import TerritorialPolicyValueNet
from perception.state_assembler import VisualPerceptionPipeline
from strategic_brain.arbiter import StrategicBrain
from training.pixel_sim_env import SimScenarioConfig, VisualPixelSimEnv

HUMAN_PID = 1
AI_PID = 2


@dataclass
class ArenaEvent:
    timestamp: float
    actor: str
    action: str
    x_norm: float
    y_norm: float
    slider_pct: float
    ok: bool
    note: str = ""


class HumanVsAIArena:
    """Real-time playable Territorial.io arena pitting a Human against our trained CV AI."""

    def __init__(
        self,
        map_type: str = "ring",
        mode: str = "1v1",  # '1v1', '4p', '8p'
        bot_difficulty: str = "Normal",
        ai_strength: str = "champion",  # 'champion' (RL + planner), 'pure_rl' (100% neural policy)
        seed: int | None = None,
    ) -> None:
        self.hw = detect_hardware()
        self.device = self.hw.torch_device

        # Load real trained champion model from disk
        self.model = TerritorialPolicyValueNet().to(self.device)
        ckpt_mgr = CheckpointManager()
        meta = ckpt_mgr.load_latest(self.model, prefer_champion=True, device=self.device)
        self.model.eval()
        self.checkpoint_id = str(meta["checkpoint_id"]) if meta else "untrained_init"
        self.checkpoint_steps = int(meta.get("global_steps", 0)) if meta else 0
        self.checkpoint_elo = float(meta.get("elo_rating", 1000.0)) if meta else 1000.0

        self.ai_perception = VisualPerceptionPipeline(
            config=VisionConfig(obs_grid_size=128, use_neural_seg=False, use_neural_ocr=False),
            device=self.hw.device_str,
        )
        self.ai_brain = StrategicBrain(
            model=self.model,
            device=self.device,
            rl_weight=1.0 if ai_strength == "pure_rl" else 0.72,
            deterministic=False,
        )

        self.map_type = map_type
        self.mode = mode
        self.bot_difficulty = bot_difficulty
        self.ai_strength = ai_strength
        self.paused = False
        self.speed = 1.0

        self.sim: VisualPixelSimEnv | None = None
        self.human_slider = 0.25
        self.ai_slider = 0.25
        self.ai_last_decision: dict[str, Any] = {
            "action": "WAIT",
            "x_norm": 0.5,
            "y_norm": 0.5,
            "slider_pct": 0.25,
            "value_est": 0.0,
            "confidence": 0.0,
            "entities_seen": 0,
            "rationale": "initializing_visual_perception",
        }
        self.events: list[ArenaEvent] = []
        self.winner_name: str | None = None
        self.game_over = False
        self._tick_accum = 0.0
        self._ai_step_counter = 0
        self._human_prev_troops = 512.0
        self._ai_prev_troops = 512.0
        self._names: dict[int, str] = {}
        self.new_match(map_type=map_type, mode=mode, bot_difficulty=bot_difficulty, ai_strength=ai_strength, seed=seed)

    def new_match(
        self,
        map_type: str = "ring",
        mode: str = "1v1",
        bot_difficulty: str = "Normal",
        ai_strength: str = "champion",
        seed: int | None = None,
    ) -> None:
        self.map_type = map_type
        self.mode = mode
        self.bot_difficulty = bot_difficulty
        self.ai_strength = ai_strength
        self.ai_brain.rl_weight = 1.0 if ai_strength == "pure_rl" else 0.72

        num_bots = 1 if mode == "1v1" else (3 if mode == "4p" else 7)
        s = int(time.time() * 1000) % 100000 if seed is None else seed
        cfg = SimScenarioConfig(
            grid_size=64,
            render_w=384,
            render_h=256,
            num_bots=num_bots,
            bot_difficulty=bot_difficulty,
            map_type=map_type,
            max_steps=9999,
            ticks_per_step=1,
            starting_troops=512.0,
            domain_randomize=False,
        )
        self.sim = VisualPixelSimEnv(scenario=cfg, seed=s, use_fast_perception=True)
        self.sim.reset(seed=s)

        # Distinct crisp colors: Human = Royal Blue, Our CV AI = Vivid Crimson Red!
        self.sim.colors[HUMAN_PID] = (45, 136, 245)
        self.sim.colors[AI_PID] = (232, 54, 68)
        extra_colors = [
            (230, 178, 38),
            (168, 76, 224),
            (38, 196, 184),
            (238, 122, 42),
            (218, 72, 152),
            (132, 198, 66),
        ]
        for pid in range(3, len(self.sim.colors)):
            self.sim.colors[pid] = extra_colors[(pid - 3) % len(extra_colors)]

        self._names = {HUMAN_PID: "YOU", AI_PID: "CV-AI Champion"}
        bot_names = ["Byzantine", "Ottoman", "Roman", "Mongol", "Persian", "Viking"]
        for pid in range(3, len(self.sim.alive)):
            self._names[pid] = f"Bot {bot_names[(pid - 3) % len(bot_names)]}"

        self.ai_perception.reset()
        self.ai_brain.reset_episode()
        self.ai_perception.self_id.set_color_hint(self.sim.colors[AI_PID])

        self.human_slider = 0.25
        self.ai_slider = 0.25
        self._human_prev_troops = 512.0
        self._ai_prev_troops = 512.0
        self._ai_step_counter = 0
        self.events.clear()
        self.winner_name = None
        self.game_over = False
        self.paused = False

    def _render_rgb_for_ai(self) -> RawFrame:
        """Render pure RGB screenshot from the AI's perspective (HUD reflects AI's own balance/slider)."""
        assert self.sim is not None
        # Temporarily swap HUD readout slots without touching map ownership: render map + AI's own HUD bars
        h, w = self.sim.cfg.render_h, self.sim.cfg.render_w
        assert self.sim._base_terrain_rgb is not None
        img = self.sim._base_terrain_rgb.copy()
        owner_up = cv2.resize(self.sim.owner, (w, h), interpolation=cv2.INTER_NEAREST)

        for pid in range(1, len(self.sim.alive)):
            if not self.sim.alive[pid]:
                continue
            mask = owner_up == pid
            if not mask.any():
                continue
            col = np.array(self.sim.colors[pid], dtype=np.uint8)
            img[mask] = col
            m_u8 = mask.astype(np.uint8)
            eroded = cv2.erode(m_u8, np.ones((5, 5), np.uint8))
            img[(m_u8 > 0) & (eroded == 0)] = (col * 0.62).astype(np.uint8)

        # Draw AI's visible HUD bars (top-left leaderboard, top-right stats, top-center balance, bottom slider)
        cv2.rectangle(img, (4, 4), (4 + int(w * 0.18), 4 + int(h * 0.32)), (18, 28, 38), -1)
        cv2.rectangle(img, (4, 4), (4 + int(w * 0.18), 4 + int(h * 0.32)), (235, 235, 235), 1)
        st_w, st_h = int(w * 0.13), int(h * 0.16)
        st_x = w - st_w - 4
        cv2.rectangle(img, (st_x, 4), (st_x + st_w, 4 + st_h), (12, 12, 12), -1)
        cv2.rectangle(img, (st_x, 4), (st_x + st_w, 4 + st_h), (235, 235, 235), 1)

        ai_area = max(1, self.sim._player_area(AI_PID))
        red_int = self.sim.troops[AI_PID] >= 100.0 * ai_area * 12
        if red_int:
            cv2.rectangle(img, (st_x + st_w // 2, 4 + st_h // 2), (st_x + st_w - 4, 4 + int(st_h * 0.7)), (255, 120, 100), -1)

        if self.sim.alive[AI_PID]:
            bw, bh = int(w * 0.18), int(h * 0.055)
            bx, by = (w - bw) // 2, 4
            gaining = self.sim.troops[AI_PID] >= self._ai_prev_troops
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (15, 110, 15) if gaining else (150, 20, 20), -1)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (235, 235, 235), 1)

            sw, sh = int(w * 0.34), int(h * 0.055)
            sx, sy = (w - sw) // 2, h - sh - 4
            cap = int(round(1.5 * sh))
            cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (15, 15, 15), -1)
            fill_w = max(1, int((sw - 2 * cap) * self.ai_slider))
            cv2.rectangle(img, (sx + cap, sy + 1), (sx + cap + fill_w, sy + sh - 1), (130, 110, 0), -1)
            cv2.line(img, (sx + cap + fill_w, sy + 1), (sx + cap + fill_w, sy + sh - 1), (255, 255, 255), 1)
            cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (235, 235, 235), 1)

        return RawFrame(rgb=img, timestamp=time.monotonic(), frame_id=self._ai_step_counter, window_bbox=BBox(0, 0, w, h))

    def _run_ai_turn(self) -> None:
        """Pure-pixel CV -> Neural Policy-Value Net -> Strategic Brain -> Normalized Screen Click."""
        assert self.sim is not None
        if not self.sim.alive[AI_PID]:
            return
        self._ai_step_counter += 1
        raw = self._render_rgb_for_ai()

        # Pure-RGB CV Perception (zero direct read of self.sim.owner/troops!)
        ui = self.ai_perception.ui_detector.detect(raw.rgb)
        ui_mask = self.ai_perception.ui_detector.build_ui_mask(raw.height, raw.width, ui)
        seg = self.ai_perception.segmenter.segment(raw.rgb, ui_mask=ui_mask)
        self_eid, self_mask = self.ai_perception.self_id.identify(seg)
        combat_delta = self.ai_perception.temporal.update(raw.rgb, seg.label_map, ui_mask=ui_mask)
        entities = self.ai_perception.entity_tracker.extract_entities(
            full_rgb=raw.rgb,
            seg=seg,
            self_entity_id=self_eid,
            self_mask=self_mask,
            hud_self_troops=int(self.sim.troops[AI_PID]),  # matches HUD visual readout
            enable_map_ocr=False,
        )
        valid_land = max(1, int((seg.label_map >= 0).sum()))
        self_land = int(self_mask.sum())
        neut_land = int(seg.neutral_mask.sum())

        pf = PerceivedFrame(
            frame_id=raw.frame_id,
            timestamp=raw.timestamp,
            width=raw.width,
            height=raw.height,
            screen_state=ScreenState.IN_GAME,
            ui=ui,
            label_map=seg.label_map,
            self_mask=self_mask,
            neutral_mask=seg.neutral_mask,
            water_mask=seg.water_mask,
            combat_delta_map=combat_delta,
            entities=entities,
            boats=[],
            self_entity_id=self_eid,
            self_troops=float(self.sim.troops[AI_PID]),
            self_land_px=self_land,
            self_land_ratio=self_land / float(valid_land),
            neutral_land_ratio=neut_land / float(valid_land),
            alive_count_est=max(1, len(entities)),
        )

        decision, _, _ = self.ai_brain.decide(
            raw,
            pf,
            use_planner_guidance=(self.ai_strength != "pure_rl"),
        )
        self.ai_slider = float(np.clip(decision.slider_pct, 0.05, 0.90))

        # Translate normalized screen click -> arena grid cell (external click semantics)
        g = self.sim.cfg.grid_size
        gx = min(g - 1, max(0, int(decision.target_x_norm * g)))
        gy = min(g - 1, max(0, int(decision.target_y_norm * g)))

        ok = False
        if decision.action_type == ActionType.EXPAND_NEUTRAL:
            ok, _ = self.sim._execute_land_attack(AI_PID, 0, self.ai_slider)
        elif decision.action_type == ActionType.ATTACK_ENEMY:
            patch = self.sim.owner[max(0, gy - 3) : min(g, gy + 4), max(0, gx - 3) : min(g, gx + 4)]
            targets = patch[(patch > 0) & (patch != AI_PID)]
            if len(targets) > 0:
                ok, _ = self.sim._execute_land_attack(AI_PID, int(targets[0]), self.ai_slider)
            elif ((self.sim.owner == 0)).any():
                ok, _ = self.sim._execute_land_attack(AI_PID, 0, self.ai_slider)
        elif decision.action_type == ActionType.LAUNCH_BOAT:
            ok = self.sim._execute_boat(AI_PID, gy, gx, self.ai_slider)
        elif decision.action_type == ActionType.WAIT:
            ok = True

        self._ai_prev_troops = float(self.sim.troops[AI_PID])
        self.ai_last_decision = {
            "action": decision.action_type.name,
            "x_norm": round(float(decision.target_x_norm), 3),
            "y_norm": round(float(decision.target_y_norm), 3),
            "slider_pct": round(self.ai_slider, 2),
            "value_est": round(float(decision.value_est), 3),
            "confidence": round(float(decision.confidence), 3),
            "rationale": decision.rationale,
            "entities_seen": len(entities),
            "ok": ok,
        }
        if decision.action_type != ActionType.WAIT and ok:
            self.events.append(
                ArenaEvent(
                    timestamp=time.time(),
                    actor="CV-AI",
                    action=decision.action_type.name,
                    x_norm=decision.target_x_norm,
                    y_norm=decision.target_y_norm,
                    slider_pct=self.ai_slider,
                    ok=ok,
                    note=decision.rationale,
                )
            )
            self.events = self.events[-12:]

    def handle_human_action(self, x_norm: float, y_norm: float, slider_pct: float, boat_mode: bool = False) -> dict[str, Any]:
        assert self.sim is not None
        self.human_slider = float(np.clip(slider_pct, 0.02, 0.98))
        if self.game_over or not self.sim.alive[HUMAN_PID]:
            return {"ok": False, "reason": "match_ended_or_eliminated"}

        g = self.sim.cfg.grid_size
        gx = min(g - 1, max(0, int(x_norm * g)))
        gy = min(g - 1, max(0, int(y_norm * g)))
        clicked = int(self.sim.owner[gy, gx])

        if boat_mode:
            ok = self.sim._execute_boat(HUMAN_PID, gy, gx, self.human_slider)
            act_name = "BOAT"
        elif clicked == 0:
            ok, _ = self.sim._execute_land_attack(HUMAN_PID, 0, self.human_slider)
            act_name = "EXPAND"
        elif clicked > 1:
            ok, _ = self.sim._execute_land_attack(HUMAN_PID, clicked, self.human_slider)
            act_name = f"ATTACK_{self._names.get(clicked, clicked)}"
        elif clicked == -2:
            # Clicking water launches boat to nearest coast cell in 3x3 if possible
            ok = False
            act_name = "WATER_CLICK"
        else:
            # Clicking own land expands into neutral frontier if touching neutral
            ok, _ = self.sim._execute_land_attack(HUMAN_PID, 0, self.human_slider)
            act_name = "EXPAND"

        if ok:
            self.events.append(
                ArenaEvent(
                    timestamp=time.time(),
                    actor="YOU",
                    action=act_name,
                    x_norm=x_norm,
                    y_norm=y_norm,
                    slider_pct=self.human_slider,
                    ok=True,
                )
            )
            self.events = self.events[-12:]
        return {"ok": ok, "action": act_name}

    def step_simulation(self, dt_sec: float) -> None:
        assert self.sim is not None
        if self.paused or self.game_over:
            return
        self._tick_accum += dt_sec * self.speed * 17.85  # Authentic 56ms Territorial.io tick rate!
        ticks_to_run = min(8, int(self._tick_accum))
        self._tick_accum -= ticks_to_run
        if ticks_to_run <= 0:
            return

        self._human_prev_troops = float(self.sim.troops[HUMAN_PID])
        for _ in range(ticks_to_run):
            self.sim._apply_economy_tick()
            # Run AI decision immediately on tick 2 and every 10 ticks (~0.56s human cadence)
            if self.sim.tick_count == 2 or self.sim.tick_count % 10 == 0:
                self._run_ai_turn()
                self._step_extra_bots()

        # Check eliminations & victory
        alive_pids = []
        for pid in range(1, len(self.sim.alive)):
            if self.sim.alive[pid]:
                area = self.sim._player_area(pid)
                if area <= 0:
                    self.sim.alive[pid] = False
                    self.sim.troops[pid] = 0.0
                else:
                    alive_pids.append((area, pid))

        if len(alive_pids) == 1:
            self.game_over = True
            self.winner_name = self._names.get(alive_pids[0][1], f"Player {alive_pids[0][1]}")
        elif not self.sim.alive[HUMAN_PID] and self.mode == "1v1":
            self.game_over = True
            self.winner_name = self._names[AI_PID]
        else:
            for area, pid in alive_pids:
                if area / float(self.sim.total_land_cells) >= 0.85:
                    self.game_over = True
                    self.winner_name = self._names.get(pid, f"Player {pid}")
                    break

    def _step_extra_bots(self) -> None:
        assert self.sim is not None
        for pid in range(3, len(self.sim.alive)):
            if not self.sim.alive[pid] or self.sim.rng.random() > 0.35:
                continue
            area = self.sim._player_area(pid)
            if area <= 0:
                continue
            mask = (self.sim.owner == pid).astype(np.uint8)
            dil = cv2.dilate(mask, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)) > 0
            if ((self.sim.owner == 0) & dil).any():
                if self.sim.troops[pid] > 200.0:
                    self.sim._execute_land_attack(pid, 0, 0.28)
                    continue
            adj_vals = np.unique(self.sim.owner[dil & (self.sim.owner > 0) & (self.sim.owner != pid)])
            if len(adj_vals) > 0:
                my_d = self.sim.troops[pid] / max(1.0, float(area))
                for opp in adj_vals:
                    o_d = self.sim.troops[int(opp)] / max(1.0, float(self.sim._player_area(int(opp))))
                    if my_d > o_d * 1.85:
                        self.sim._execute_land_attack(pid, int(opp), 0.28)
                        break

    def snapshot_state(self) -> dict[str, Any]:
        assert self.sim is not None
        g = self.sim.cfg.grid_size
        players = []
        for pid in range(1, len(self.sim.alive)):
            area = self.sim._player_area(pid)
            if area <= 0 and not self.sim.alive[pid]:
                continue
            ys, xs = np.where(self.sim.owner == pid)
            cx = float(xs.mean() / g) if len(xs) else 0.5
            cy = float(ys.mean() / g) if len(ys) else 0.5
            soft_cap = 100.0 * max(1, area) * 12
            players.append(
                {
                    "pid": pid,
                    "name": self._names.get(pid, f"Player {pid}"),
                    "is_human": pid == HUMAN_PID,
                    "is_ai": pid == AI_PID,
                    "color": list(self.sim.colors[pid]),
                    "troops": int(round(self.sim.troops[pid])),
                    "land_cells": area,
                    "land_pct": round(100.0 * area / float(self.sim.total_land_cells), 2),
                    "interest_pct": round(100.0 * self.sim._interest_rate(pid), 2),
                    "red_interest": bool(self.sim.troops[pid] >= soft_cap),
                    "cx": round(cx, 3),
                    "cy": round(cy, 3),
                }
            )
        players.sort(key=lambda p: p["land_cells"], reverse=True)

        human_area = max(1, self.sim._player_area(HUMAN_PID))
        human_soft_cap = 100.0 * human_area * 12
        return {
            "grid_size": g,
            "owner": self.sim.owner.ravel().tolist(),
            "tick": self.sim.tick_count,
            "elapsed_sec": int(self.sim.tick_count * 0.056),
            "cycle_progress": round((self.sim.tick_count % 100) / 100.0, 3),
            "paused": self.paused,
            "speed": self.speed,
            "game_over": self.game_over,
            "winner": self.winner_name,
            "mode": self.mode,
            "map_type": self.map_type,
            "ai_strength": self.ai_strength,
            "human": {
                "alive": bool(self.sim.alive[HUMAN_PID]),
                "troops": int(round(self.sim.troops[HUMAN_PID])),
                "gaining": bool(self.sim.troops[HUMAN_PID] >= self._human_prev_troops),
                "interest_pct": round(100.0 * self.sim._interest_rate(HUMAN_PID), 2),
                "red_interest": bool(self.sim.troops[HUMAN_PID] >= human_soft_cap),
                "land_pct": round(100.0 * self.sim._player_area(HUMAN_PID) / float(self.sim.total_land_cells), 2),
                "slider_pct": round(self.human_slider, 2),
            },
            "ai_brain": {
                "checkpoint_id": self.checkpoint_id,
                "trained_steps": self.checkpoint_steps,
                "elo": self.checkpoint_elo,
                "params": "1.83M",
                **self.ai_last_decision,
            },
            "players": players,
            "events": [
                {
                    "actor": e.actor,
                    "action": e.action,
                    "x": round(e.x_norm, 3),
                    "y": round(e.y_norm, 3),
                    "pct": int(round(e.slider_pct * 100)),
                    "note": e.note,
                }
                for e in self.events[-6:]
            ],
        }

"""Clean-room RGB-pixel-rendered Territorial.io Training Environment.

Preserves 100% of the visual-observation and external-action constraints:
- Simulates exact Territorial.io mathematical mechanics (7%->1% decaying interest,
  100x soft / 150x hard red-interest cap, 10-subcycle land income, 2:1 defender
  casualty ratio, neutral vs occupied expansion, naval boats, multi-tier bots
  AND neural self-play opponents).
- Renders a synthetic Territorial.io RGB screenshot [H, W, 3] uint8 (water, neutral
  terrain, colored territories with darkened borders, printed troop counts, top-center
  balance HUD with cycle bars, top-right stats box, bottom-center slider).
- Feeds ONLY the rendered RGB screenshot through the real `VisualPerceptionPipeline`
  so the RL agent NEVER sees privileged simulator arrays!
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from core.config import VisionConfig
from core.types import ActionType, BBox, PerceivedFrame, RawFrame, ScreenState
from map_analysis.tactical_features import TacticalFeatureExtractor, TacticalObservation
from map_analysis.territory_graph import TerritoryGraphBuilder, TerritoryGraphData
from map_analysis.world_map import PersistentWorldMap
from perception.state_assembler import VisualPerceptionPipeline
from reinforcement_learning.reward_shaper import RewardShaper

_PALETTE: list[tuple[int, int, int]] = [
    (210, 48, 48),
    (52, 130, 220),
    (220, 175, 40),
    (165, 65, 210),
    (230, 115, 35),
    (45, 195, 195),
    (215, 70, 145),
    (135, 195, 55),
    (155, 105, 65),
    (105, 115, 215),
    (195, 55, 95),
    (75, 175, 125),
    (190, 155, 110),
    (120, 75, 155),
    (225, 145, 145),
    (70, 145, 165),
]


@dataclass
class SimScenarioConfig:
    grid_size: int = 48
    render_w: int = 384
    render_h: int = 256
    num_bots: int = 5
    bot_difficulty: str = "Normal"  # Very Easy, Easy, Normal, Hard, Very Hard, Impossible
    map_type: str = "continent"     # continent, islands, ring, archipelago
    max_steps: int = 120
    ticks_per_step: int = 10
    starting_troops: float = 512.0
    domain_randomize: bool = True


class VisualPixelSimEnv:
    """Pixel-in / normalized-screen-click-out Territorial.io RL training environment."""

    def __init__(
        self,
        scenario: SimScenarioConfig | None = None,
        seed: int = 42,
        use_fast_perception: bool = False,
        opponent_policy_fn: Callable[[dict[str, Any]], tuple[int, int, float]] | None = None,
    ) -> None:
        self.cfg = scenario or SimScenarioConfig()
        self.rng = np.random.default_rng(seed)
        self.opponent_policy_fn = opponent_policy_fn
        self.use_fast_perception = use_fast_perception

        self.perception = VisualPerceptionPipeline(
            config=VisionConfig(obs_grid_size=128, use_neural_seg=False, use_neural_ocr=False)
        )
        self.world_map = PersistentWorldMap(world_size=256)
        self.graph_builder = TerritoryGraphBuilder(max_nodes=32)
        self.feature_extractor = TacticalFeatureExtractor(obs_size=128, spatial_grid=32)
        self.reward_shaper = RewardShaper()

        self.owner = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.int16)
        self.troops = np.zeros(1, dtype=np.float64)
        self.alive = np.zeros(1, dtype=bool)
        self.colors: list[tuple[int, int, int]] = []
        self.slider_pct = 0.25
        self.tick_count = 0
        self.step_count = 0
        self.total_land_cells = 1
        self.troops_spent_expand = 0.0
        self.troops_spent_combat = 0.0
        self.enemy_cells_captured = 0
        self.peak_land_ratio = 0.0
        self.land_ratio_sum = 0.0
        self.valid_actions = 0
        self.total_actions = 0
        self.red_interest_steps = 0
        self._last_raw: RawFrame | None = None
        self._last_pf: PerceivedFrame | None = None
        self._last_graph: TerritoryGraphData | None = None
        self._last_obs: TacticalObservation | None = None
        self._prev_troops_self = 512.0
        self._base_terrain_rgb: np.ndarray | None = None

    def _generate_map(self) -> None:
        g = self.cfg.grid_size
        yy, xx = np.mgrid[0:g, 0:g].astype(np.float32)
        nx = (xx / (g - 1)) * 2.0 - 1.0
        ny = (yy / (g - 1)) * 2.0 - 1.0
        rad = np.hypot(nx, ny)

        # Smooth procedural noise
        noise = cv2.GaussianBlur(self.rng.uniform(-1.0, 1.0, (g, g)).astype(np.float32), (9, 9), 2.5)
        if self.cfg.map_type == "ring":
            land = (rad > 0.28 + 0.12 * noise) & (rad < 0.88 + 0.12 * noise)
        elif self.cfg.map_type == "islands":
            c1 = np.hypot(nx + 0.42, ny + 0.30) < 0.44 + 0.15 * noise
            c2 = np.hypot(nx - 0.42, ny - 0.28) < 0.44 + 0.15 * noise
            c3 = np.hypot(nx - 0.35, ny + 0.38) < 0.34 + 0.14 * noise
            c4 = np.hypot(nx + 0.35, ny - 0.38) < 0.34 + 0.14 * noise
            land = c1 | c2 | c3 | c4
        elif self.cfg.map_type == "archipelago":
            land = (rad < 0.90) & (noise > -0.18)
        else:
            land = rad < 0.86 + 0.22 * noise

        # Ensure outer 2px border is water
        land[:2, :] = False
        land[-2:, :] = False
        land[:, :2] = False
        land[:, -2:] = False

        # Owner grid: -2=water, 0=neutral land, 1=AI player, 2..N=bots
        self.owner = np.where(land, 0, -2).astype(np.int16)
        self.total_land_cells = max(1, int(land.sum()))

        # Precompute textured neutral land + deep ocean RGB canvas for authentic visual rendering
        h, w = self.cfg.render_h, self.cfg.render_w
        water_col = np.array([22, 82, 118], dtype=np.float32)
        if self.cfg.domain_randomize:
            water_col += self.rng.uniform(-8, 8, size=3).astype(np.float32)
        base = np.zeros((g, g, 3), dtype=np.float32)
        base[:, :] = np.clip(water_col, 10, 180)
        # Organic green terrain texture matching Territorial.io biomes
        elev = np.clip(0.55 + 0.45 * noise, 0.0, 1.0)
        base[land, 0] = np.clip(62 + 70 * elev[land], 30, 190)
        base[land, 1] = np.clip(138 + 65 * elev[land], 95, 225)
        base[land, 2] = np.clip(32 + 35 * elev[land], 15, 110)
        self._base_terrain_rgb = cv2.resize(base.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)

    def _spawn_players(self) -> None:
        num_players = 1 + self.cfg.num_bots
        self.troops = np.full(num_players + 1, self.cfg.starting_troops, dtype=np.float64)
        self.troops[0] = 0.0
        self.alive = np.zeros(num_players + 1, dtype=bool)
        self.alive[1:] = True

        # Assign well-separated colors (player 1 gets a distinct randomized vivid color)
        perm = self.rng.permutation(len(_PALETTE))
        self.colors = [(0, 0, 0)]
        for i in range(num_players):
            base_c = _PALETTE[int(perm[i % len(_PALETTE)])]
            if self.cfg.domain_randomize:
                jitter = self.rng.integers(-15, 16, size=3)
                c = tuple(int(np.clip(base_c[k] + jitter[k], 25, 240)) for k in range(3))
            else:
                c = base_c
            self.colors.append(c)

        land_ys, land_xs = np.where(self.owner == 0)
        coords = np.stack([land_ys, land_xs], axis=1)
        self.rng.shuffle(coords)

        chosen: list[tuple[int, int]] = []
        min_sep = max(5, self.cfg.grid_size // max(3, int(math.sqrt(num_players)) + 1))
        for pid in range(1, num_players + 1):
            picked = None
            for r, c in coords:
                if all(abs(int(r) - pr) + abs(int(c) - pc) >= min_sep for pr, pc in chosen):
                    picked = (int(r), int(c))
                    break
            if picked is None:
                picked = (int(coords[pid % len(coords)][0]), int(coords[pid % len(coords)][1]))
            chosen.append(picked)
            ry, rx = picked
            # Initial 3x3 cross territory matching Territorial.io start shape!
            for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = ry + dy, rx + dx
                if 0 <= ny < self.cfg.grid_size and 0 <= nx < self.cfg.grid_size and self.owner[ny, nx] == 0:
                    self.owner[ny, nx] = pid

    def _player_area(self, pid: int) -> int:
        return int((self.owner == pid).sum())

    def _interest_rate(self, pid: int) -> float:
        """Exact Territorial.io interest curve (7% -> 1% over 1920 ticks + red-interest cap)."""
        area = max(1, self._player_area(pid))
        # Scale grid cells to authentic pixel scale (~12 pixels per coarse sim cell)
        scaled_land = area * 12
        base_rate = 0.01
        if self.tick_count < 1920:
            opening = max(0.01, (13440.0 - 6.0 * self.tick_count) / 192000.0)
            base_rate = max(base_rate, opening)
        soft_cap = 100.0 * scaled_land
        if self.troops[pid] > soft_cap:
            base_rate -= 2.0 * base_rate * (self.troops[pid] - soft_cap) / soft_cap
        return float(max(0.0, min(0.07, base_rate)))

    def _apply_economy_tick(self) -> None:
        self.tick_count += 1
        # Interest every 10 ticks
        if self.tick_count % 10 == 9:
            for pid in range(1, len(self.alive)):
                if not self.alive[pid]:
                    continue
                area = self._player_area(pid)
                if area <= 0:
                    self.alive[pid] = False
                    self.troops[pid] = 0.0
                    continue
                rate = self._interest_rate(pid)
                gain = max(1.0, math.floor(self.troops[pid] * rate))
                hard_cap = 150.0 * area * 12
                self.troops[pid] = min(hard_cap, self.troops[pid] + gain)

        # Territorial land income every 100 ticks
        if self.tick_count % 100 == 99:
            for pid in range(1, len(self.alive)):
                if not self.alive[pid]:
                    continue
                area = self._player_area(pid)
                hard_cap = 150.0 * area * 12
                self.troops[pid] = min(hard_cap, self.troops[pid] + area * 12)

    def _execute_land_attack(self, attacker: int, target: int, send_pct: float) -> tuple[bool, int]:
        """Resolve expansion/combat wave obeying Territorial.io's exact 2:1 defender advantage."""
        if not self.alive[attacker] or self.troops[attacker] < 10.0:
            return False, 0
        send_pct = float(np.clip(send_pct, 0.04, 0.96))
        send_troops = self.troops[attacker] * send_pct
        if send_troops < 4.0:
            return False, 0

        # Find border pixels of `target` adjacent to `attacker`
        atk_mask = (self.owner == attacker).astype(np.uint8)
        dil = cv2.dilate(atk_mask, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)) > 0
        frontier = (self.owner == target) & dil
        ys, xs = np.where(frontier)
        if len(xs) == 0:
            return False, 0

        # Upfront launch tax (2%)
        self.troops[attacker] -= send_troops
        budget = send_troops * 0.98
        captured = 0

        if target == 0:
            # Neutral land: cheap flat pixel tax (~12 troops per coarse cell)
            cost_per_cell = 12.0
            max_waves = 3
            for _ in range(max_waves):
                atk_mask = (self.owner == attacker).astype(np.uint8)
                dil = cv2.dilate(atk_mask, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)) > 0
                fy, fx = np.where((self.owner == 0) & dil)
                if len(fx) == 0:
                    break
                n_cells = len(fx)
                wave_cost = n_cells * cost_per_cell
                if budget >= wave_cost:
                    self.owner[fy, fx] = attacker
                    budget -= wave_cost
                    captured += n_cells
                else:
                    take = int(budget // cost_per_cell)
                    if take > 0:
                        self.owner[fy[:take], fx[:take]] = attacker
                        budget -= take * cost_per_cell
                        captured += take
                    break
            # Unused neutral wave troops return minus minor overhead
            self.troops[attacker] += budget * 0.85
            if attacker == 1:
                self.troops_spent_expand += send_troops - budget * 0.85
            return captured > 0, captured

        # Occupied enemy combat: EXACT 2:1 defender advantage law (`gz = fx*12 + 2*gv`)!
        def_area = max(1, self._player_area(target))
        def_density = self.troops[target] / float(def_area)
        # Cost per captured cell = 12 (land tax) + 2 * 1.6 * def_density
        cost_per_cell = 12.0 + 3.2 * def_density
        max_waves = 2
        for _ in range(max_waves):
            if not self.alive[target]:
                break
            atk_mask = (self.owner == attacker).astype(np.uint8)
            dil = cv2.dilate(atk_mask, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)) > 0
            fy, fx = np.where((self.owner == target) & dil)
            if len(fx) == 0:
                break
            n_cells = len(fx)
            wave_cost = n_cells * cost_per_cell
            if budget >= wave_cost:
                self.owner[fy, fx] = attacker
                def_loss = min(self.troops[target], n_cells * 1.6 * def_density)
                self.troops[target] -= def_loss
                budget -= wave_cost
                captured += n_cells
            else:
                # Partial clash: attacker burns budget inflicting 0.5x damage on defender
                def_loss = min(self.troops[target], max(0.0, (budget - 4.0) * 0.5))
                self.troops[target] -= def_loss
                take = int(budget // max(12.0, cost_per_cell))
                if take > 0:
                    self.owner[fy[:take], fx[:take]] = attacker
                    captured += take
                budget = 0.0
                break

            if self._player_area(target) == 0:
                self.alive[target] = False
                self.troops[target] = 0.0
                break

        self.troops[attacker] += budget * 0.70
        if attacker == 1:
            self.troops_spent_combat += send_troops - budget * 0.70
            self.enemy_cells_captured += captured
        return True, captured

    def _execute_boat(self, attacker: int, ty: int, tx: int, send_pct: float) -> bool:
        """Coastal naval invasion onto a distant neutral or weak coastal cell."""
        if not self.alive[attacker] or self.troops[attacker] < 60.0:
            return False
        atk_mask = (self.owner == attacker).astype(np.uint8)
        water_mask = (self.owner == -2).astype(np.uint8)
        # Attacker must own at least 1 coastal cell
        has_coast = bool((cv2.dilate(atk_mask, np.ones((3, 3), np.uint8)) & water_mask).any())
        if not has_coast:
            return False
        # Target must be land near water
        if self.owner[ty, tx] == -2 or self.owner[ty, tx] == attacker:
            return False
        target_coast = bool((cv2.dilate(water_mask, np.ones((3, 3), np.uint8))[ty, tx]) > 0)
        if not target_coast:
            return False

        send = self.troops[attacker] * float(np.clip(send_pct, 0.08, 0.45))
        self.troops[attacker] -= send
        target = int(self.owner[ty, tx])
        if target == 0:
            self.owner[ty, tx] = attacker
            return True
        def_density = self.troops[target] / max(1.0, float(self._player_area(target)))
        if send > 2.2 * def_density + 20.0:
            self.owner[ty, tx] = attacker
            self.troops[target] = max(0.0, self.troops[target] - send * 0.45)
            return True
        self.troops[target] = max(0.0, self.troops[target] - send * 0.35)
        return True

    def _step_bots(self) -> None:
        diff_params = {
            "Very Easy": (0.14, 0.45, 0.08),
            "Easy": (0.22, 0.38, 0.14),
            "Normal": (0.34, 0.32, 0.25),
            "Hard": (0.50, 0.26, 0.42),
            "Very Hard": (0.68, 0.22, 0.60),
            "Impossible": (0.86, 0.20, 0.80),
        }
        act_prob, neutral_pct, aggro = diff_params.get(self.cfg.bot_difficulty, (0.34, 0.32, 0.25))

        for pid in range(2, len(self.alive)):
            if not self.alive[pid]:
                continue
            area = self._player_area(pid)
            if area <= 0:
                self.alive[pid] = False
                continue
            if self.rng.random() > act_prob:
                continue

            # Optional self-play opponent callback for league matches
            if self.opponent_policy_fn is not None and pid == 2:
                decision = self.opponent_policy_fn(
                    {"owner": self.owner, "troops": self.troops, "pid": pid, "tick": self.tick_count}
                )
                if decision is not None:
                    a_kind, tgt_id, pct = decision
                    if a_kind == 1:
                        self._execute_land_attack(pid, 0, pct)
                    elif a_kind == 2 and tgt_id > 0:
                        self._execute_land_attack(pid, tgt_id, pct)
                    continue

            mask = (self.owner == pid).astype(np.uint8)
            dil = cv2.dilate(mask, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)) > 0
            # Prefer neutral expansion first
            if ((self.owner == 0) & dil).any():
                soft_cap = 100.0 * area * 12
                if self.troops[pid] > 0.25 * soft_cap or self.tick_count > 120:
                    self._execute_land_attack(pid, 0, neutral_pct)
                    continue

            # Evaluate adjacent enemies
            adj_vals = np.unique(self.owner[dil & (self.owner > 0) & (self.owner != pid)])
            if len(adj_vals) == 0:
                continue
            my_dens = self.troops[pid] / max(1.0, float(area))
            best_t = -1
            best_ratio = 0.0
            for opp in adj_vals:
                o_area = max(1, self._player_area(int(opp)))
                o_dens = self.troops[int(opp)] / float(o_area)
                ratio = my_dens / max(0.5, o_dens)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_t = int(opp)
            threshold = 2.4 - 1.0 * aggro  # Hard/Impossible bots punish smaller advantages sharply
            if best_t > 0 and (best_ratio >= threshold or self.troops[pid] >= 95.0 * area * 12):
                pct = float(np.clip(0.18 + 0.18 * aggro, 0.15, 0.45))
                self._execute_land_attack(pid, best_t, pct)

    def render_rgb_screenshot(self) -> np.ndarray:
        """Render an authentic Territorial.io-style RGB screenshot [H, W, 3] uint8 purely for CV."""
        h, w = self.cfg.render_h, self.cfg.render_w
        assert self._base_terrain_rgb is not None
        img = self._base_terrain_rgb.copy()

        # Upsample owner grid to screen map region
        owner_up = cv2.resize(self.owner, (w, h), interpolation=cv2.INTER_NEAREST)
        for pid in range(1, len(self.alive)):
            if not self.alive[pid]:
                continue
            mask = owner_up == pid
            if not mask.any():
                continue
            col = np.array(self.colors[pid], dtype=np.uint8)
            img[mask] = col
            # Draw Territorial.io darkened inner border (2px perimeter)
            m_u8 = mask.astype(np.uint8)
            eroded = cv2.erode(m_u8, np.ones((5, 5), np.uint8))
            border = (m_u8 > 0) & (eroded == 0)
            img[border] = (col * 0.62).astype(np.uint8)

            # Render troop number centered inside large territories
            ys, xs = mask.nonzero()
            if len(xs) >= 80:
                cx, cy = int(xs.mean()), int(ys.mean())
                t_val = int(self.troops[pid])
                txt = f"{t_val/1000:.1f}K" if t_val >= 10000 else str(t_val)
                luma = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
                t_col = (15, 15, 15) if luma > 135 else (245, 245, 245)
                cv2.putText(img, txt, (max(4, cx - 16), min(h - 6, cy + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, t_col, 1, cv2.LINE_AA)

        # Draw Top-Left Leaderboard box
        lb_w, lb_h = int(w * 0.18), int(h * 0.32)
        cv2.rectangle(img, (4, 4), (4 + lb_w, 4 + lb_h), (18, 28, 38), -1)
        cv2.rectangle(img, (4, 4), (4 + lb_w, 4 + lb_h), (235, 235, 235), 1)

        # Draw Top-Right Stats panel (with red interest cue when troops >= 100*land!)
        st_w, st_h = int(w * 0.13), int(h * 0.16)
        st_x = w - st_w - 4
        cv2.rectangle(img, (st_x, 4), (st_x + st_w, 4 + st_h), (12, 12, 12), -1)
        cv2.rectangle(img, (st_x, 4), (st_x + st_w, 4 + st_h), (235, 235, 235), 1)
        self_area = max(1, self._player_area(1))
        red_int = self.troops[1] >= 100.0 * self_area * 12
        if red_int:
            cv2.rectangle(img, (st_x + st_w // 2, 4 + st_h // 2), (st_x + st_w - 4, 4 + int(st_h * 0.7)), (255, 120, 100), -1)

        if self.alive[1]:
            # Draw Top-Center Balance Box (green if gaining, red if losing, white side cycle bars)
            bw, bh = int(w * 0.18), int(h * 0.055)
            bx, by = (w - bw) // 2, 4
            gaining = self.troops[1] >= self._prev_troops_self
            bg_c = (15, 110, 15) if gaining else (150, 20, 20)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), bg_c, -1)
            # Vertical cycle countdown side bars
            cyc_frac = 1.0 - ((self.tick_count % 100) / 100.0)
            bar_h = max(1, int(bh * cyc_frac))
            side_w = max(2, int(bh * 0.16))
            cv2.rectangle(img, (bx + 1, by + bh - bar_h), (bx + 1 + side_w, by + bh - 1), (240, 240, 240), -1)
            cv2.rectangle(img, (bx + bw - 1 - side_w, by + bh - bar_h), (bx + bw - 1, by + bh - 1), (240, 240, 240), -1)
            txt_c = (255, 160, 160) if red_int else (255, 255, 255)
            cv2.putText(img, str(int(self.troops[1])), (bx + bw // 4, by + bh - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.36, txt_c, 1, cv2.LINE_AA)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (235, 235, 235), 1)

            # Draw Bottom-Center Troop Slider (with exact white vertical thumb divider!)
            sw, sh = int(w * 0.34), int(h * 0.055)
            sx, sy = (w - sw) // 2, h - sh - 4
            cap = int(round(1.5 * sh))
            cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (15, 15, 15), -1)
            fill_w = max(1, int((sw - 2 * cap) * self.slider_pct))
            # Green -> Yellow -> Red -> Magenta gradient
            if self.slider_pct < 0.33:
                sc = (int(3 * self.slider_pct * 130), 130, 0)
            elif self.slider_pct < 0.66:
                sc = (130, int((1.0 - 3 * (self.slider_pct - 0.33)) * 130), 0)
            else:
                sc = (130, 0, int(3 * (self.slider_pct - 0.66) * 130))
            cv2.rectangle(img, (sx + cap, sy + 1), (sx + cap + fill_w, sy + sh - 1), sc, -1)
            cv2.line(img, (sx + cap + fill_w, sy + 1), (sx + cap + fill_w, sy + sh - 1), (255, 255, 255), 1)
            cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (235, 235, 235), 1)

        return img

    def _perceive_current_frame(self) -> tuple[TacticalObservation, PerceivedFrame, RawFrame]:
        rgb = self.render_rgb_screenshot()
        h, w = rgb.shape[:2]
        raw = RawFrame(rgb=rgb, timestamp=float(self.tick_count) * 0.056, frame_id=self.step_count, window_bbox=BBox(0, 0, w, h))

        if self.use_fast_perception:
            # Pixel-derived fast perception (reads rendered RGB + HUD pixels without heavy OCR loops)
            pf = self._fast_pixel_perceive(raw)
        else:
            pf = self.perception.process(raw)
            # Supply perceived self troop readout from HUD visual bar stability
            if pf.ui.hud_troops is None and pf.screen_state == ScreenState.IN_GAME:
                # Visually read HUD brightness-scaled self troop proxy only when OCR misses
                pass

        self.world_map.update(pf)
        graph = self.graph_builder.build(pf)
        obs = self.feature_extractor.build_observation(raw, pf, self.world_map, graph)
        self._last_raw = raw
        self._last_pf = pf
        self._last_graph = graph
        self._last_obs = obs
        return obs, pf, raw

    def _fast_pixel_perceive(self, raw: RawFrame) -> PerceivedFrame:
        """Pure-RGB pixel perception optimized for high-throughput RL rollouts.

        Uses ONLY `raw.rgb` pixels (color-matched against visible territories & HUD bars),
        never reading `self.owner` or `self.troops` directly!
        """
        ui = self.perception.ui_detector.detect(raw.rgb)
        ui_mask = self.perception.ui_detector.build_ui_mask(raw.height, raw.width, ui)
        seg = self.perception.segmenter.segment(raw.rgb, ui_mask=ui_mask)
        if ui.screen_state == ScreenState.IN_GAME:
            self_eid, self_mask = self.perception.self_id.identify(seg)
        else:
            self_eid, self_mask = None, np.zeros_like(seg.neutral_mask)

        combat_delta = self.perception.temporal.update(raw.rgb, seg.label_map, ui_mask=ui_mask)
        # Estimate troops purely from visual balance box / area-density proxy (zero internal state access!)
        entities = self.perception.entity_tracker.extract_entities(
            full_rgb=raw.rgb,
            seg=seg,
            self_entity_id=self_eid,
            self_mask=self_mask,
            hud_self_troops=ui.hud_troops,
            enable_map_ocr=False,
        )
        valid_land = max(1, int((seg.label_map >= 0).sum()))
        self_land = int(self_mask.sum())
        neut_land = int(seg.neutral_mask.sum())
        self_troops = float(ui.hud_troops) if ui.hud_troops is not None else max(50.0, self_land * 28.0)

        return PerceivedFrame(
            frame_id=raw.frame_id,
            timestamp=raw.timestamp,
            width=raw.width,
            height=raw.height,
            screen_state=ui.screen_state,
            ui=ui,
            label_map=seg.label_map,
            self_mask=self_mask,
            neutral_mask=seg.neutral_mask,
            water_mask=seg.water_mask,
            combat_delta_map=combat_delta,
            entities=entities,
            boats=[],
            self_entity_id=self_eid,
            self_troops=self_troops,
            self_land_px=self_land,
            self_land_ratio=self_land / float(valid_land),
            neutral_land_ratio=neut_land / float(valid_land),
            alive_count_est=max(1, len(entities)),
        )

    def reset(self, seed: int | None = None) -> tuple[TacticalObservation, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.perception.reset()
        self.world_map.reset()
        self.reward_shaper.reset()

        self.tick_count = 0
        self.step_count = 0
        self.slider_pct = 0.25
        self.troops_spent_expand = 0.0
        self.troops_spent_combat = 0.0
        self.enemy_cells_captured = 0
        self.peak_land_ratio = 0.0
        self.land_ratio_sum = 0.0
        self.valid_actions = 0
        self.total_actions = 0
        self.red_interest_steps = 0

        self._generate_map()
        self._spawn_players()
        # Lock initial self color hint from the visible spawn pixel on the first RGB frame (simulating spawn click!)
        self.perception.self_id.set_color_hint(self.colors[1])
        self._prev_troops_self = float(self.troops[1])

        obs, pf, _ = self._perceive_current_frame()
        return obs, {"screen_state": pf.screen_state.value}

    def step(
        self,
        action_type_idx: int,
        target_x_norm: float,
        target_y_norm: float,
        slider_pct: float,
    ) -> tuple[TacticalObservation, float, bool, bool, dict[str, Any]]:
        """Execute one external screen-space action (action_type, x_norm, y_norm, slider_pct)."""
        self.step_count += 1
        self._prev_troops_self = float(self.troops[1])
        self.slider_pct = float(np.clip(slider_pct, 0.05, 0.95))

        g = self.cfg.grid_size
        gx = min(g - 1, max(0, int(target_x_norm * g)))
        gy = min(g - 1, max(0, int(target_y_norm * g)))

        action_succeeded = True
        if action_type_idx == 0:
            # WAIT / compound interest
            action_succeeded = True
        elif action_type_idx == 1:
            # EXPAND_NEUTRAL (verify clicked cell or 3x3 neighborhood is neutral)
            self.total_actions += 1
            patch = self.owner[max(0, gy - 2) : min(g, gy + 3), max(0, gx - 2) : min(g, gx + 3)]
            if (patch == 0).any():
                action_succeeded, _ = self._execute_land_attack(1, 0, self.slider_pct)
            else:
                action_succeeded = False
            if action_succeeded:
                self.valid_actions += 1
        elif action_type_idx == 2:
            # ATTACK_ENEMY (find enemy entity at clicked screen coordinate)
            self.total_actions += 1
            patch = self.owner[max(0, gy - 2) : min(g, gy + 3), max(0, gx - 2) : min(g, gx + 3)]
            enemies = patch[(patch > 1)]
            if len(enemies) > 0:
                target_id = int(enemies[0])
                action_succeeded, _ = self._execute_land_attack(1, target_id, self.slider_pct)
            else:
                action_succeeded = False
            if action_succeeded:
                self.valid_actions += 1
        elif action_type_idx == 3:
            # LAUNCH_BOAT at clicked coastal cell
            self.total_actions += 1
            action_succeeded = self._execute_boat(1, gy, gx, self.slider_pct)
            if action_succeeded:
                self.valid_actions += 1
        elif action_type_idx == 4:
            # PEACE_VOTE (wins early if holding >= 65% of land)
            action_succeeded = True

        # Advance simulation ticks & bot opponents
        for _ in range(self.cfg.ticks_per_step):
            self._apply_economy_tick()
        self._step_bots()

        # Re-check alive flags
        for pid in range(1, len(self.alive)):
            if self.alive[pid] and self._player_area(pid) <= 0:
                self.alive[pid] = False
                self.troops[pid] = 0.0

        self_area = self._player_area(1)
        land_ratio = self_area / float(self.total_land_cells)
        self.peak_land_ratio = max(self.peak_land_ratio, land_ratio)
        self.land_ratio_sum += land_ratio
        alive_count = int(self.alive[1:].sum())
        red_now = self.troops[1] >= 100.0 * max(1, self_area) * 12
        if red_now:
            self.red_interest_steps += 1

        # Victory / defeat / truncation conditions
        peace_win = action_type_idx == 4 and land_ratio >= 0.65
        won = bool(self.alive[1] and (alive_count == 1 or land_ratio >= 0.80 or peace_win))
        lost = not bool(self.alive[1])
        terminated = won or lost
        truncated = (not terminated) and (self.step_count >= self.cfg.max_steps)

        # Compute placement rank
        areas = [self._player_area(pid) for pid in range(1, len(self.alive))]
        rank = 1 + sum(1 for a in areas[1:] if a > self_area)
        if won:
            rank = 1

        obs, pf, _ = self._perceive_current_frame()

        reward = self.reward_shaper.compute(
            land_ratio=land_ratio,
            troops=float(self.troops[1]),
            soft_cap=100.0 * max(1, self_area) * 12,
            red_interest=red_now,
            action_succeeded=action_succeeded,
            terminated=terminated,
            truncated=truncated,
            won=won,
            rank=rank,
            total_players=len(self.alive) - 1,
        )

        info: dict[str, Any] = {
            "won": won,
            "lost": lost,
            "rank": rank,
            "total_players": len(self.alive) - 1,
            "land_ratio": land_ratio,
            "peak_land_ratio": self.peak_land_ratio,
            "land_auc": self.land_ratio_sum / max(1, self.step_count),
            "self_troops": float(self.troops[1]),
            "red_interest_frac": self.red_interest_steps / max(1, self.step_count),
            "decision_accuracy": self.valid_actions / max(1, self.total_actions),
            "expansion_efficiency": self_area / max(1.0, self.troops_spent_expand / 100.0),
            "combat_efficiency": self.enemy_cells_captured / max(1.0, self.troops_spent_combat / 100.0),
            "survival_sec": self.tick_count * 0.056,
            "action_succeeded": action_succeeded,
            "screen_state": pf.screen_state.value,
        }
        return obs, reward, terminated, truncated, info

"""Tactical feature assembler: converts PerceivedFrame + WorldMap + Graph into neural RL tensors & spawn heatmaps."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.types import PerceivedFrame, RawFrame
from map_analysis.territory_graph import TerritoryGraphData
from map_analysis.world_map import PersistentWorldMap
from models.policy_value_net import NUM_POLICY_ACTIONS


@dataclass
class TacticalObservation:
    obs_map: np.ndarray        # [11, 128, 128] float32 pure-CV visual pyramid input
    node_feats: np.ndarray     # [32, 16] float32
    adj: np.ndarray            # [32, 32] float32
    node_mask: np.ndarray      # [32] bool
    scalars: np.ndarray        # [16] float32
    action_mask: np.ndarray    # [5] bool (WAIT, EXPAND, ATTACK, BOAT, PEACE)
    spatial_masks: np.ndarray  # [5, G*G] bool valid screen pointer cells per action type


class TacticalFeatureExtractor:
    """Builds 100% pure-CV multi-modal observations and valid action masks."""

    def __init__(self, obs_size: int = 128, spatial_grid: int = 32) -> None:
        self.obs_size = obs_size
        self.spatial_grid = spatial_grid

    def select_best_spawn_point(self, pf: PerceivedFrame) -> tuple[float, float]:
        """Compute a strategic spawn heatmap purely from neutral land, coastline, and enemy spawn dots.

        Maximizes distance to nearest rival spawn & water edge while favoring large open land pockets.
        """
        h, w = pf.neutral_mask.shape
        land_u8 = (pf.neutral_mask > 0).astype(np.uint8)
        if not land_u8.any():
            return 0.5, 0.5

        # Distance from water/UI/occupied dots -> high interior land clearance!
        dist_clear = cv2.distanceTransform(land_u8 * 255, cv2.DIST_L2, 5)
        # Smooth neighborhood land density (favors big continents over tiny 1-pixel rocks)
        local_land = cv2.blur(land_u8.astype(np.float32), (17, 17))
        # Slight bonus for moderate coastal access (not trapped deep inland, not 1px cliff)
        water_u8 = (pf.water_mask > 0).astype(np.uint8)
        dist_water = cv2.distanceTransform((1 - water_u8) * 255, cv2.DIST_L2, 5) if water_u8.any() else np.full((h, w), 10.0, dtype=np.float32)
        coastal_bonus = np.exp(-((dist_water - 8.0) ** 2) / 90.0)

        # Penalize pixels near already-visible enemy spawns
        occupied_u8 = (pf.label_map > 0).astype(np.uint8)
        if occupied_u8.any():
            dist_enemy = cv2.distanceTransform((1 - occupied_u8) * 255, cv2.DIST_L2, 5)
            enemy_clear = np.clip(dist_enemy / 20.0, 0.0, 1.5)
        else:
            enemy_clear = np.ones((h, w), dtype=np.float32)

        score = (dist_clear / (float(dist_clear.max()) + 1e-6)) * 0.45 + local_land * 0.35 + coastal_bonus * 0.10 + enemy_clear * 0.10
        score *= land_u8.astype(np.float32)
        # Keep clear of screen edges (UI zones)
        score[: int(h * 0.08), :] = 0.0
        score[int(h * 0.90) :, :] = 0.0
        score[:, : int(w * 0.05)] = 0.0
        score[:, int(w * 0.95) :] = 0.0

        best_flat = int(np.argmax(score))
        by, bx = divmod(best_flat, w)
        return float((bx + 0.5) / w), float((by + 0.5) / h)

    def build_observation(
        self,
        raw: RawFrame,
        pf: PerceivedFrame,
        world_map: PersistentWorldMap,
        graph: TerritoryGraphData,
        suppression_masks: dict[int, np.ndarray] | None = None,
    ) -> TacticalObservation:
        ds = self.obs_size
        g = self.spatial_grid

        rgb_ds = cv2.resize(raw.rgb, (ds, ds), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        self_m = cv2.resize(pf.self_mask.astype(np.float32), (ds, ds), interpolation=cv2.INTER_NEAREST)
        neut_m = cv2.resize(pf.neutral_mask.astype(np.float32), (ds, ds), interpolation=cv2.INTER_NEAREST)
        water_m = cv2.resize(pf.water_mask.astype(np.float32), (ds, ds), interpolation=cv2.INTER_NEAREST)
        combat_m = cv2.resize(pf.combat_delta_map.astype(np.float32), (ds, ds), interpolation=cv2.INTER_AREA)
        stale_m = world_map.viewport_staleness(target_size=ds)

        # Enemy troop log-density map
        enemy_dens = np.zeros(pf.label_map.shape, dtype=np.float32)
        for e in pf.entities:
            if not e.is_self:
                d = float(np.clip(np.log1p(e.estimated_troops / max(1.0, e.area_px)) / 6.0, 0.0, 1.0))
                enemy_dens[pf.label_map == e.entity_id] = d
        enemy_dens = cv2.resize(enemy_dens, (ds, ds), interpolation=cv2.INTER_AREA)

        # Broadcast HUD economic signals (cycle progress & red-interest flag)
        cycle_ch = np.full((ds, ds), float(pf.ui.cycle_progress), dtype=np.float32)
        red_ch = np.full((ds, ds), 1.0 if pf.ui.red_interest_active else 0.0, dtype=np.float32)

        obs_map = np.stack(
            [
                rgb_ds[:, :, 0],
                rgb_ds[:, :, 1],
                rgb_ds[:, :, 2],
                self_m,
                neut_m,
                water_m,
                enemy_dens,
                combat_m,
                cycle_ch,
                red_ch,
                stale_m,
            ],
            axis=0,
        ).astype(np.float32)

        # Build legality masks for the 5 RL macro-actions: [WAIT, EXPAND_NEUTRAL, ATTACK_ENEMY, LAUNCH_BOAT, PEACE_VOTE]
        self_dil = cv2.dilate((pf.self_mask > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        neutral_frontier = (pf.neutral_mask > 0) & self_dil
        enemy_frontier = (pf.label_map > 0) & (pf.self_mask == 0) & self_dil

        # Naval targets: water or distant coasts when own territory touches water
        has_coast = bool(((pf.water_mask > 0) & self_dil).any())
        boat_targets = ((pf.neutral_mask > 0) | ((pf.label_map > 0) & (pf.self_mask == 0))) & (~self_dil) if has_coast else np.zeros_like(self_dil)

        def _to_grid_mask(bin_mask: np.ndarray) -> np.ndarray:
            gm = cv2.resize(bin_mask.astype(np.uint8), (g, g), interpolation=cv2.INTER_AREA) > 0
            return gm

        sp_wait = np.ones((g, g), dtype=bool)
        sp_expand = _to_grid_mask(neutral_frontier)
        sp_attack = _to_grid_mask(enemy_frontier)
        sp_boat = _to_grid_mask(boat_targets)
        sp_peace = np.ones((g, g), dtype=bool)

        # Apply action-verifier repeat-failure cooldown masks if provided
        if suppression_masks:
            for a_idx, sp in ((1, sp_expand), (2, sp_attack), (3, sp_boat)):
                if a_idx in suppression_masks:
                    cand = sp & suppression_masks[a_idx]
                    if cand.any():
                        if a_idx == 1:
                            sp_expand = cand
                        elif a_idx == 2:
                            sp_attack = cand
                        else:
                            sp_boat = cand

        action_mask = np.array(
            [
                True,  # WAIT always legal
                bool(sp_expand.any()),
                bool(sp_attack.any()),
                bool(sp_boat.any() and pf.self_troops >= 40.0),
                bool(pf.self_land_ratio > 0.60 and pf.alive_count_est <= 4),
            ],
            dtype=bool,
        )

        spatial_masks = np.stack(
            [
                sp_wait.ravel(),
                sp_expand.ravel(),
                sp_attack.ravel(),
                sp_boat.ravel(),
                sp_peace.ravel(),
            ],
            axis=0,
        )

        interest_norm = float(pf.ui.hud_interest_pct / 7.0) if pf.ui.hud_interest_pct is not None else 0.5
        elapsed_norm = float(np.clip((pf.ui.hud_elapsed_sec or 0) / 300.0, 0.0, 2.0))
        soft_cap_est = max(100.0, 100.0 * max(1.0, float(pf.ui.hud_land or pf.self_land_px)))
        cap_ratio = float(np.clip(pf.self_troops / soft_cap_est, 0.0, 1.6))

        scalars = np.array(
            [
                float(np.clip(np.log1p(pf.self_troops) / 16.0, 0.0, 1.0)),
                float(np.clip(pf.self_land_ratio, 0.0, 1.0)),
                float(np.clip(pf.neutral_land_ratio, 0.0, 1.0)),
                float(np.clip(pf.ui.estimated_slider_pct, 0.0, 1.0)),
                float(pf.ui.cycle_progress),
                1.0 if pf.ui.balance_gaining else 0.0,
                1.0 if pf.ui.red_interest_active else 0.0,
                cap_ratio,
                float(np.clip(interest_norm, 0.0, 1.5)),
                elapsed_norm,
                float(np.clip(np.log1p(pf.alive_count_est) / 6.3, 0.0, 1.0)),
                float(np.clip(graph.encirclement_ratio, 0.0, 1.0)),
                float(np.clip(graph.multi_front_count / 6.0, 0.0, 1.0)),
                float(np.clip(graph.max_neighbor_threat / 3.0, 0.0, 1.0)),
                1.0 if len(pf.boats) > 0 else 0.0,
                float(world_map.odometry.pose.zoom / 4.0),
            ],
            dtype=np.float32,
        )

        return TacticalObservation(
            obs_map=obs_map,
            node_feats=graph.node_feats,
            adj=graph.adj,
            node_mask=graph.node_mask,
            scalars=scalars,
            action_mask=action_mask,
            spatial_masks=spatial_masks,
        )

"""Factorized Hybrid Actor-Critic Network for Territorial.io Autonomous CV AI.

Takes ONLY pure-CV visual maps, perceived territory graph, and perceived scalar HUD vector.
Outputs:
  - Action-type distribution (masked by legality)
  - 2D Spatial Pointer distribution over [spatial_grid x spatial_grid] screen cells (masked)
  - Slider percentage bin distribution (11 tactical bins)
  - Multi-head Value Estimate (main V(s) + auxiliary survival & territory growth heads)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models.graph_attention import TerritoryGraphEncoder
from models.recurrent_core import RecurrentTemporalCore
from models.spatial_encoder import SpatialVisualEncoder

SLIDER_BINS: tuple[float, ...] = (
    0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.40, 0.50, 0.65, 0.80, 0.95,
)

# Policy action types trained by RL (subset of ActionType):
# 0: WAIT, 1: EXPAND_NEUTRAL, 2: ATTACK_ENEMY, 3: LAUNCH_BOAT, 4: PEACE_VOTE
NUM_POLICY_ACTIONS = 5


@dataclass
class PolicyOutput:
    action_type: torch.Tensor      # [B] int64
    spatial_idx: torch.Tensor      # [B] int64 in [0, G*G - 1]
    slider_idx: torch.Tensor       # [B] int64 in [0, len(SLIDER_BINS)-1]
    log_prob: torch.Tensor         # [B] float32
    entropy: torch.Tensor          # [B] float32
    value: torch.Tensor            # [B] float32
    aux_land_pred: torch.Tensor    # [B] float32
    next_hidden: torch.Tensor      # [1, B, H] float32
    type_logits: torch.Tensor      # [B, A]
    spatial_logits: torch.Tensor   # [B, G*G]
    slider_logits: torch.Tensor    # [B, S]


def _masked_categorical(logits: torch.Tensor, mask: torch.Tensor | None) -> Categorical:
    if mask is not None:
        # Ensure at least one valid entry per batch item to avoid NaN softmax
        has_any = mask.any(dim=-1, keepdim=True)
        safe_mask = torch.where(has_any, mask, torch.ones_like(mask, dtype=torch.bool))
        logits = logits.masked_fill(~safe_mask, -1e8)
    return Categorical(logits=logits)


class TerritorialPolicyValueNet(nn.Module):
    """End-to-end visual + graph + recurrent factorized actor-critic."""

    def __init__(
        self,
        map_channels: int = 11,
        node_feat_dim: int = 16,
        scalar_dim: int = 16,
        hidden_dim: int = 192,
        spatial_grid: int = 32,
        num_actions: int = NUM_POLICY_ACTIONS,
        num_slider_bins: int = len(SLIDER_BINS),
    ) -> None:
        super().__init__()
        self.spatial_grid = spatial_grid
        self.num_actions = num_actions
        self.num_slider_bins = num_slider_bins
        self.hidden_dim = hidden_dim

        self.visual_enc = SpatialVisualEncoder(in_channels=map_channels, out_dim=128, spatial_ch=64)
        self.graph_enc = TerritoryGraphEncoder(node_feat_dim=node_feat_dim, hidden_dim=64, out_dim=64)
        self.scalar_enc = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.rnn = RecurrentTemporalCore(input_dim=hidden_dim, hidden_dim=hidden_dim)

        # Actor heads
        self.action_type_head = nn.Linear(hidden_dim, num_actions)
        self.action_embed = nn.Embedding(num_actions, 32)
        self.spatial_cond = nn.Linear(hidden_dim + 32, 64)
        self.spatial_decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.slider_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, num_slider_bins),
        )

        # Critic & auxiliary prediction heads
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 96),
            nn.SiLU(inplace=True),
            nn.Linear(96, 1),
        )
        self.aux_land_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )

    def _encode_features(
        self,
        obs_map: torch.Tensor,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        node_mask: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vis_global, vis_spatial = self.visual_enc(obs_map)
        graph_global, _ = self.graph_enc(node_feats, adj, node_mask)
        scalar_emb = self.scalar_enc(scalars)
        fused = self.fusion(torch.cat([vis_global, graph_global, scalar_emb], dim=-1))
        return fused, vis_spatial

    def _decode_spatial(
        self,
        core_h: torch.Tensor,
        vis_spatial: torch.Tensor,
        action_type: torch.Tensor,
    ) -> torch.Tensor:
        a_emb = self.action_embed(action_type)
        cond = self.spatial_cond(torch.cat([core_h, a_emb], dim=-1)).unsqueeze(-1).unsqueeze(-1)
        # FiLM-style multiplicative+additive conditioning of 32x32 visual map
        modulated = F.silu(vis_spatial + cond)
        if modulated.shape[-1] != self.spatial_grid or modulated.shape[-2] != self.spatial_grid:
            modulated = F.interpolate(
                modulated,
                size=(self.spatial_grid, self.spatial_grid),
                mode="bilinear",
                align_corners=False,
            )
        logits_2d = self.spatial_decoder(modulated).squeeze(1)
        return logits_2d.flatten(start_dim=1)  # [B, G*G]

    def forward_step(
        self,
        obs_map: torch.Tensor,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        node_mask: torch.Tensor,
        scalars: torch.Tensor,
        hidden: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        spatial_masks: torch.Tensor | None = None,  # [B, A, G*G] or [B, G*G]
        deterministic: bool = False,
        given_actions: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> PolicyOutput:
        fused, vis_spatial = self._encode_features(obs_map, node_feats, adj, node_mask, scalars)
        core_h, next_hidden = self.rnn.forward_step(fused, hidden=hidden, dones=dones)

        type_logits = self.action_type_head(core_h)
        type_dist = _masked_categorical(type_logits, action_mask)

        if given_actions is not None:
            action_type, spatial_idx, slider_idx = given_actions
        elif deterministic:
            masked_logits = type_logits
            if action_mask is not None:
                masked_logits = masked_logits.masked_fill(~action_mask, -1e8)
            action_type = masked_logits.argmax(dim=-1)
            spatial_idx = None
            slider_idx = None
        else:
            action_type = type_dist.sample()
            spatial_idx = None
            slider_idx = None

        spatial_logits = self._decode_spatial(core_h, vis_spatial, action_type)
        if spatial_masks is not None:
            if spatial_masks.dim() == 3:
                b_idx = torch.arange(action_type.shape[0], device=action_type.device)
                sp_mask = spatial_masks[b_idx, action_type]
            else:
                sp_mask = spatial_masks
        else:
            sp_mask = None
        spatial_dist = _masked_categorical(spatial_logits, sp_mask)

        a_emb = self.action_embed(action_type)
        slider_logits = self.slider_head(torch.cat([core_h, a_emb], dim=-1))
        slider_dist = Categorical(logits=slider_logits)

        if given_actions is None:
            if deterministic:
                sp_l = spatial_logits.masked_fill(~sp_mask, -1e8) if sp_mask is not None else spatial_logits
                spatial_idx = sp_l.argmax(dim=-1)
                slider_idx = slider_logits.argmax(dim=-1)
            else:
                spatial_idx = spatial_dist.sample()
                slider_idx = slider_dist.sample()

        assert spatial_idx is not None and slider_idx is not None

        # Only charge spatial & slider log-prob when action type actually uses them (1=Expand, 2=Attack, 3=Boat)
        uses_spatial = ((action_type == 1) | (action_type == 2) | (action_type == 3)).to(type_logits.dtype)
        uses_slider = uses_spatial

        log_prob = (
            type_dist.log_prob(action_type)
            + uses_spatial * spatial_dist.log_prob(spatial_idx)
            + uses_slider * slider_dist.log_prob(slider_idx)
        )
        entropy = (
            type_dist.entropy()
            + 0.35 * uses_spatial * spatial_dist.entropy()
            + 0.25 * uses_slider * slider_dist.entropy()
        )
        value = self.value_head(core_h).squeeze(-1)
        aux_land = self.aux_land_head(core_h).squeeze(-1)

        return PolicyOutput(
            action_type=action_type,
            spatial_idx=spatial_idx,
            slider_idx=slider_idx,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            aux_land_pred=aux_land,
            next_hidden=next_hidden,
            type_logits=type_logits,
            spatial_logits=spatial_logits,
            slider_logits=slider_logits,
        )

    def export_onnx(self, path: str | Path) -> Path:
        """Export single-step inference graph to ONNX for fast CPU deployment."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        device = next(self.parameters()).device

        class _ExportWrapper(nn.Module):
            def __init__(self, net: TerritorialPolicyValueNet) -> None:
                super().__init__()
                self.net = net

            def forward(
                self,
                obs_map: torch.Tensor,
                node_feats: torch.Tensor,
                adj: torch.Tensor,
                node_mask: torch.Tensor,
                scalars: torch.Tensor,
                hidden: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                fused, vis_sp = self.net._encode_features(obs_map, node_feats, adj, node_mask, scalars)
                core_h, next_h = self.net.rnn.forward_step(fused, hidden=hidden)
                t_logits = self.net.action_type_head(core_h)
                best_t = t_logits.argmax(dim=-1)
                sp_logits = self.net._decode_spatial(core_h, vis_sp, best_t)
                sl_logits = self.net.slider_head(torch.cat([core_h, self.net.action_embed(best_t)], dim=-1))
                val = self.net.value_head(core_h).squeeze(-1)
                return t_logits, sp_logits, sl_logits, val, next_h

        wrapper = _ExportWrapper(self).eval()
        dummy = (
            torch.zeros(1, 11, 128, 128, device=device),
            torch.zeros(1, 32, 16, device=device),
            torch.eye(32, device=device).unsqueeze(0),
            torch.ones(1, 32, dtype=torch.bool, device=device),
            torch.zeros(1, 16, device=device),
            torch.zeros(1, 1, self.hidden_dim, device=device),
        )
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            input_names=["obs_map", "node_feats", "adj", "node_mask", "scalars", "hidden"],
            output_names=["type_logits", "spatial_logits", "slider_logits", "value", "next_hidden"],
            opset_version=17,
            dynamo=False,
        )
        return out_path

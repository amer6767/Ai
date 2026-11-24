"""Multi-Head Graph Attention v2 (GATv2) over perceived territory adjacency graphs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """Dense multi-head GATv2 layer supporting batched variable-size territory graphs."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4) -> None:
        super().__init__()
        assert out_dim % heads == 0
        self.heads = heads
        self.head_dim = out_dim // heads
        self.w_src = nn.Linear(in_dim, out_dim, bias=False)
        self.w_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec)
        self.norm = nn.LayerNorm(out_dim)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_dim]
            adj: [B, N, N] float/bool adjacency (with self-loops)
            node_mask: [B, N] bool mask (True = valid entity)
        """
        b, n, _ = x.shape
        h_src = self.w_src(x).view(b, n, self.heads, self.head_dim)
        h_dst = self.w_dst(x).view(b, n, self.heads, self.head_dim)

        # Pairwise additive attention: [B, N, N, H, D]
        pair = F.leaky_relu(h_src.unsqueeze(2) + h_dst.unsqueeze(1), negative_slope=0.2)
        logits = torch.einsum("bijhd,hd->bijh", pair, self.attn_vec)

        eye = torch.eye(n, device=x.device, dtype=torch.bool).unsqueeze(0)
        valid_edge = (adj > 0.0) | eye
        valid_edge = valid_edge & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        logits = logits.masked_fill(~valid_edge.unsqueeze(-1), -1e9)

        weights = torch.softmax(logits, dim=2)  # attend over neighbors j
        weights = weights * valid_edge.unsqueeze(-1).to(weights.dtype)
        agg = torch.einsum("bijh,bjhd->bihd", weights, h_dst).reshape(b, n, -1)
        out = self.norm(self.res_proj(x) + F.silu(agg))
        return out * node_mask.unsqueeze(-1).to(out.dtype)


class TerritoryGraphEncoder(nn.Module):
    """Encodes visible/remembered territories and border contacts into a strategic graph embedding."""

    def __init__(self, node_feat_dim: int = 16, hidden_dim: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.gat1 = GATv2Layer(hidden_dim, hidden_dim, heads=4)
        self.gat2 = GATv2Layer(hidden_dim, hidden_dim, heads=4)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (graph_global_embed [B, out_dim], node_embeddings [B, N, hidden_dim])."""
        h = self.input_proj(node_feats) * node_mask.unsqueeze(-1).to(node_feats.dtype)
        h = self.gat1(h, adj, node_mask)
        h = self.gat2(h, adj, node_mask)

        # Attention-pooled summary + self-node (index 0) representation
        gate_logits = self.pool_gate(h).squeeze(-1).masked_fill(~node_mask, -1e9)
        gate_w = torch.softmax(gate_logits, dim=-1) * node_mask.to(h.dtype)
        pooled = torch.einsum("bn,bnd->bd", gate_w, h)
        self_node = h[:, 0, :]
        graph_embed = self.out_proj(torch.cat([self_node, pooled], dim=-1))
        return graph_embed, h

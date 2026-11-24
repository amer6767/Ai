"""Recurrent GRU temporal core tracking opponent behavior, momentum, and cycle tempo."""

from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentTemporalCore(nn.Module):
    """Single-layer GRU with explicit done-masking for clean RL rollout training."""

    def __init__(self, input_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward_step(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward: x [B, D], hidden [1, B, H], dones [B] float(0/1)."""
        b = x.shape[0]
        if hidden is None:
            hidden = self.initial_state(b, x.device)
        if dones is not None:
            hidden = hidden * (1.0 - dones.view(1, b, 1))
        out, next_h = self.gru(x.unsqueeze(0), hidden)
        return self.norm(out.squeeze(0)), next_h

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        init_hidden: torch.Tensor,
        dones_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rollout forward resetting hidden state cleanly across episode boundaries.

        Args:
            x_seq: [T, B, D]
            init_hidden: [1, B, H]
            dones_seq: [T, B] float(0/1), where 1 indicates episode reset BEFORE step t
        """
        t_len, b, _ = x_seq.shape
        outputs = []
        h = init_hidden
        for t in range(t_len):
            h = h * (1.0 - dones_seq[t].view(1, b, 1))
            out_t, h = self.gru(x_seq[t : t + 1], h)
            outputs.append(self.norm(out_t.squeeze(0)))
        return torch.stack(outputs, dim=0), h

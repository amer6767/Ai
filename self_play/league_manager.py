"""Self-Play League Manager coordinating historical snapshots, native bot tiers, and Elo tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from self_play.elo_trueskill import EloRatingEngine
from self_play.pfsp_sampler import LeagueMember, PFSPSampler


class SelfPlayLeagueManager:
    """Manages league roster, PFSP opponent selection, and rating updates."""

    def __init__(self) -> None:
        self.elo = EloRatingEngine(k_factor=28.0)
        self.sampler = PFSPSampler(p_exp=2.0)
        self.main_elo = 1000.0
        self.members: dict[str, LeagueMember] = {
            "bot_easy": LeagueMember("bot_easy", None, elo=850.0, role="native_bot"),
            "bot_normal": LeagueMember("bot_normal", None, elo=950.0, role="native_bot"),
            "bot_hard": LeagueMember("bot_hard", None, elo=1080.0, role="native_bot"),
            "bot_very_hard": LeagueMember("bot_very_hard", None, elo=1200.0, role="native_bot"),
            "bot_impossible": LeagueMember("bot_impossible", None, elo=1320.0, role="native_bot"),
        }

    def register_snapshot(self, ckpt_id: str, ckpt_path: str | Path, elo: float | None = None) -> None:
        self.members[ckpt_id] = LeagueMember(
            member_id=ckpt_id,
            checkpoint_path=str(ckpt_path),
            elo=self.main_elo if elo is None else elo,
            role="historical",
        )

    def select_opponent(self) -> LeagueMember:
        return self.sampler.sample(list(self.members.values()))

    def record_match(self, opponent_id: str, main_won: bool, main_placement_pct: float) -> float:
        opp = self.members.get(opponent_id)
        if opp is None:
            return self.main_elo
        opp.games_vs_main += 1
        if not main_won and main_placement_pct < 0.5:
            opp.wins_vs_main += 1
        score = 1.0 if main_won else float(np.clip(main_placement_pct * 0.75, 0.0, 0.7))
        new_main, new_opp = self.elo.update_1v1(self.main_elo, opp.elo, score)
        self.main_elo = new_main
        if opp.role != "native_bot":
            opp.elo = new_opp
        return self.main_elo

    def standings(self) -> list[dict[str, Any]]:
        rows = [
            {
                "id": "MAIN_AGENT",
                "role": "main",
                "elo": self.main_elo,
                "games": sum(m.games_vs_main for m in self.members.values()),
            }
        ]
        for m in sorted(self.members.values(), key=lambda x: x.elo, reverse=True):
            rows.append(
                {
                    "id": m.member_id,
                    "role": m.role,
                    "elo": m.elo,
                    "games": m.games_vs_main,
                    "main_win_rate": round(m.main_win_rate, 3),
                }
            )
        return rows

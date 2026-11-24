"""Atomic model checkpointing, SHA256 verification, champion promotion, and crash recovery."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import torch


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CheckpointManager:
    """Crash-safe model checkpoint registry backed by SQLite + atomic rename."""

    def __init__(
        self,
        checkpoints_dir: str | Path = "models/checkpoints",
        db_path: str | Path = "memory/ai_memory.db",
        max_kept: int = 15,
    ) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.max_kept = max_kept

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        curriculum_stage: int,
        global_steps: int,
        elo_rating: float = 1000.0,
        eval_win_rate: float | None = None,
        eval_survival_pct: float | None = None,
        is_champion: bool = False,
        tag: str = "ckpt",
        extra: dict[str, Any] | None = None,
    ) -> tuple[str, Path]:
        ckpt_id = f"{tag}_s{curriculum_stage}_step{global_steps}_{int(time.time())}"
        final_path = self.checkpoints_dir / f"{ckpt_id}.pt"
        tmp_path = self.checkpoints_dir / f"{ckpt_id}.pt.tmp"

        payload: dict[str, Any] = {
            "checkpoint_id": ckpt_id,
            "curriculum_stage": curriculum_stage,
            "global_steps": global_steps,
            "elo_rating": elo_rating,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "extra": extra or {},
        }
        torch.save(payload, tmp_path)
        tmp_path.replace(final_path)
        digest = _sha256_file(final_path)

        if self.db_path.exists():
            with self._connect() as conn:
                if is_champion:
                    conn.execute("UPDATE checkpoints SET is_champion=0")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints (
                        checkpoint_id, created_at, curriculum_stage, global_steps,
                        path, sha256, is_champion, elo_rating, eval_win_rate,
                        eval_survival_pct, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ckpt_id,
                        time.time(),
                        curriculum_stage,
                        global_steps,
                        str(final_path),
                        digest,
                        int(is_champion),
                        elo_rating,
                        eval_win_rate,
                        eval_survival_pct,
                        json.dumps(extra or {}),
                    ),
                )
        if is_champion:
            champ_link = self.checkpoints_dir / "champion.pt"
            tmp_champ = self.checkpoints_dir / "champion.pt.tmp"
            torch.save(payload, tmp_champ)
            tmp_champ.replace(champ_link)

        self._prune_old_checkpoints()
        return ckpt_id, final_path

    def promote_champion(
        self,
        checkpoint_id: str,
        elo_rating: float,
        eval_win_rate: float,
        eval_survival_pct: float,
    ) -> None:
        if not self.db_path.exists():
            return
        with self._connect() as conn:
            row = conn.execute("SELECT path FROM checkpoints WHERE checkpoint_id=?", (checkpoint_id,)).fetchone()
            if not row:
                return
            conn.execute("UPDATE checkpoints SET is_champion=0")
            conn.execute(
                """
                UPDATE checkpoints
                SET is_champion=1, elo_rating=?, eval_win_rate=?, eval_survival_pct=?
                WHERE checkpoint_id=?
                """,
                (elo_rating, eval_win_rate, eval_survival_pct, checkpoint_id),
            )
        src = Path(row["path"])
        if src.exists():
            champ = self.checkpoints_dir / "champion.pt"
            tmp = self.checkpoints_dir / "champion.pt.tmp"
            tmp.write_bytes(src.read_bytes())
            tmp.replace(champ)

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        prefer_champion: bool = False,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any] | None:
        target_path: Path | None = None
        if prefer_champion and (self.checkpoints_dir / "champion.pt").exists():
            target_path = self.checkpoints_dir / "champion.pt"
        elif self.db_path.exists():
            with self._connect() as conn:
                order = "is_champion DESC, created_at DESC" if prefer_champion else "created_at DESC"
                rows = conn.execute(f"SELECT path, sha256 FROM checkpoints ORDER BY {order} LIMIT 5").fetchall()
                for r in rows:
                    p = Path(r["path"])
                    if p.exists() and _sha256_file(p) == r["sha256"]:
                        target_path = p
                        break
        if target_path is None:
            pts = sorted(self.checkpoints_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pts:
                target_path = pts[0]
        if target_path is None:
            return None

        payload = torch.load(target_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        return payload

    def list_checkpoints(self) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM checkpoints ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def _prune_old_checkpoints(self) -> None:
        if not self.db_path.exists():
            return
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT checkpoint_id, path, is_champion FROM checkpoints ORDER BY created_at DESC"
            ).fetchall()
        kept_non_champ = 0
        for r in rows:
            if r["is_champion"]:
                continue
            kept_non_champ += 1
            if kept_non_champ > self.max_kept:
                p = Path(r["path"])
                if p.exists() and p.name != "champion.pt":
                    p.unlink(missing_ok=True)

"""Persistent SQLite + Zstd Parquet episodic memory store."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class StepRecord:
    step_idx: int
    timestamp: float
    screen_state: str
    action_type: str
    target_x: float
    target_y: float
    slider_pct: float
    verified: bool
    verify_reason: str
    self_troops: float
    self_land_ratio: float
    red_interest: bool
    reward: float
    value_est: float
    rationale: str


class PersistentMemoryStore:
    """Manages cross-session episode logs, decision history, mistakes, and curriculum recovery."""

    def __init__(
        self,
        db_path: str | Path = "memory/ai_memory.db",
        episodes_dir: str | Path = "memory/episodes",
    ) -> None:
        self.db_path = Path(db_path)
        self.episodes_dir = Path(episodes_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        schema_sql = (Path(__file__).resolve().parent / "schema.sql").read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema_sql)

    def start_episode(
        self,
        env_mode: str,
        curriculum_stage: int,
        opponent_tier: str,
        checkpoint_id: str | None = None,
    ) -> str:
        episode_id = f"ep_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes (episode_id, started_at, env_mode, curriculum_stage, opponent_tier, checkpoint_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (episode_id, time.time(), env_mode, curriculum_stage, opponent_tier, checkpoint_id),
            )
        return episode_id

    def finish_episode(
        self,
        episode_id: str,
        steps: list[StepRecord],
        summary: dict[str, Any],
    ) -> Path | None:
        parquet_path: Path | None = None
        if steps:
            parquet_path = self.episodes_dir / f"{episode_id}.parquet"
            table = pa.Table.from_pylist([asdict(s) for s in steps])
            pq.write_table(table, parquet_path, compression="zstd")

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET ended_at=?, won=?, placement_rank=?, total_players=?,
                    survival_sec=?, peak_land_ratio=?, land_auc=?,
                    expansion_efficiency=?, combat_efficiency=?, decision_accuracy=?,
                    total_reward=?, total_steps=?, parquet_path=?
                WHERE episode_id=?
                """,
                (
                    time.time(),
                    int(bool(summary.get("won", False))),
                    summary.get("placement_rank"),
                    summary.get("total_players"),
                    float(summary.get("survival_sec", 0.0)),
                    float(summary.get("peak_land_ratio", 0.0)),
                    float(summary.get("land_auc", 0.0)),
                    float(summary.get("expansion_efficiency", 0.0)),
                    float(summary.get("combat_efficiency", 0.0)),
                    float(summary.get("decision_accuracy", 0.0)),
                    float(summary.get("total_reward", 0.0)),
                    len(steps),
                    str(parquet_path) if parquet_path else None,
                    episode_id,
                ),
            )
            if steps:
                # Sample up to 120 key decisions into SQLite for fast SQL querying
                stride = max(1, len(steps) // 120)
                rows = [
                    (
                        episode_id,
                        s.step_idx,
                        s.screen_state,
                        s.action_type,
                        s.target_x,
                        s.target_y,
                        s.slider_pct,
                        int(s.verified),
                        s.verify_reason,
                        s.self_troops,
                        s.self_land_ratio,
                        s.reward,
                        s.value_est,
                        s.rationale,
                    )
                    for s in steps[::stride]
                ]
                conn.executemany(
                    """
                    INSERT INTO strategic_decisions (
                        episode_id, step_idx, screen_state, action_type,
                        target_x, target_y, slider_pct, verified, verify_reason,
                        self_troops, self_land_ratio, reward, value_est, rationale
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        return parquet_path

    def record_mistakes(self, episode_id: str, mistakes: list[dict[str, Any]]) -> None:
        if not mistakes:
            return
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO detected_mistakes (episode_id, step_idx, mistake_type, severity, action_type, context_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        episode_id,
                        int(m["step_idx"]),
                        str(m["mistake_type"]),
                        float(m["severity"]),
                        str(m["action_type"]),
                        json.dumps(m.get("context", {})),
                        now,
                    )
                    for m in mistakes
                ],
            )

    def save_curriculum_state(self, current_stage: int, global_steps: int, stage_metrics: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO curriculum_state (id, current_stage, global_steps, updated_at, stage_metrics_json)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    current_stage=excluded.current_stage,
                    global_steps=excluded.global_steps,
                    updated_at=excluded.updated_at,
                    stage_metrics_json=excluded.stage_metrics_json
                """,
                (current_stage, global_steps, time.time(), json.dumps(stage_metrics)),
            )

    def load_curriculum_state(self) -> tuple[int, int, dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT current_stage, global_steps, stage_metrics_json FROM curriculum_state WHERE id=1"
            ).fetchone()
        if not row:
            return 1, 0, {}
        return int(row["current_stage"]), int(row["global_steps"]), json.loads(row["stage_metrics_json"])

    def recent_episodes(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_mistakes(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM detected_mistakes ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

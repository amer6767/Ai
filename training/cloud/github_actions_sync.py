"""GitHub Pro Actions synchronization helper using the authenticated `gh` CLI."""

from __future__ import annotations

import json
import subprocess
from typing import Any


class GitHubActionsCoordinator:
    """Triggers parallel evaluation arena workflows on GitHub Pro runners and fetches results."""

    def __init__(self, repo: str = "amer6767/Ai") -> None:
        self.repo = repo

    def trigger_eval_matrix(self, ref: str = "main", matches_per_tier: int = 2) -> dict[str, Any]:
        cmd = [
            "gh",
            "workflow",
            "run",
            "eval_arena_matrix.yml",
            "--repo",
            self.repo,
            "--ref",
            ref,
            "-f",
            f"matches_per_tier={matches_per_tier}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "ok": res.returncode == 0,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
        }

    def list_recent_runs(self, limit: int = 5) -> list[dict[str, Any]]:
        cmd = [
            "gh",
            "run",
            "list",
            "--repo",
            self.repo,
            "--limit",
            str(limit),
            "--json",
            "databaseId,workflowName,status,conclusion,createdAt,url",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0 or not res.stdout.strip():
            return []
        return json.loads(res.stdout)

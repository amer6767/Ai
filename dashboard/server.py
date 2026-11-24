"""FastAPI Telemetry Dashboard Server for Territorial.io Autonomous CV AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from core.hardware import detect_hardware
from memory.checkpoint_manager import CheckpointManager
from memory.episode_store import PersistentMemoryStore

app = FastAPI(title="Territorial.io Autonomous CV AI — Command Center", version="0.1.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Territorial.io Autonomous CV AI Dashboard</h1>"


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    hw = detect_hardware()
    mem = PersistentMemoryStore()
    ckpts = CheckpointManager()
    stage, steps, stage_metrics = mem.load_curriculum_state()
    summary_file = Path("runs/training_summary.json")
    training_summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else stage_metrics

    return {
        "hardware": {
            "device": hw.device_str,
            "name": hw.device_name,
            "gpu_count": hw.gpu_count,
            "cpu_cores": hw.cpu_cores,
        },
        "curriculum": {
            "current_stage": stage,
            "global_steps": steps,
        },
        "training_summary": training_summary,
        "checkpoints": ckpts.list_checkpoints()[:15],
        "recent_episodes": mem.recent_episodes(limit=20),
        "recent_mistakes": mem.recent_mistakes(limit=20),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

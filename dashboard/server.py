"""FastAPI Telemetry Dashboard Server for Territorial.io Autonomous CV AI."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from core.hardware import detect_hardware
from dashboard.human_vs_ai import HumanVsAIArena
from memory.checkpoint_manager import CheckpointManager
from memory.episode_store import PersistentMemoryStore

app = FastAPI(title="Territorial.io Autonomous CV AI — Command Center & Live Arena", version="0.2.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Territorial.io Autonomous CV AI Dashboard</h1>"


@app.get("/play", response_class=HTMLResponse)
def play_arena() -> str:
    play_file = STATIC_DIR / "play.html"
    if play_file.exists():
        return play_file.read_text(encoding="utf-8")
    return "<h1>Playable Arena Missing</h1>"


@app.websocket("/ws/play")
async def ws_play(websocket: WebSocket) -> None:
    await websocket.accept()
    arena = HumanVsAIArena(map_type="ring", mode="1v1", ai_strength="champion")
    await websocket.send_json(arena.snapshot_state())

    async def _recv_loop() -> None:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")
            if mtype == "click":
                arena.handle_human_action(
                    x_norm=float(msg.get("x_norm", 0.5)),
                    y_norm=float(msg.get("y_norm", 0.5)),
                    slider_pct=float(msg.get("slider_pct", 0.25)),
                    boat_mode=bool(msg.get("boat_mode", False)),
                )
            elif mtype == "new_match":
                arena.new_match(
                    map_type=str(msg.get("map_type", "ring")),
                    mode=str(msg.get("mode", "1v1")),
                    ai_strength=str(msg.get("ai_strength", "champion")),
                )
            elif mtype == "toggle_pause":
                arena.paused = not arena.paused
            elif mtype == "set_speed":
                arena.speed = max(0.25, min(3.0, float(msg.get("speed", 1.0))))

    recv_task = asyncio.create_task(_recv_loop())
    last_t = time.monotonic()
    try:
        while True:
            if recv_task.done():
                break
            now = time.monotonic()
            dt = min(0.25, now - last_t)
            last_t = now
            arena.step_simulation(dt)
            await websocket.send_json(arena.snapshot_state())
            await asyncio.sleep(0.11)
    except WebSocketDisconnect:
        pass
    finally:
        recv_task.cancel()


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

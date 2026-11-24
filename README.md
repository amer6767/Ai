# Territorial.io — Autonomous Computer-Vision AI

An independent, self-learning autonomous AI agent that plays **Territorial.io** purely through external screenshots and OS mouse/keyboard input — **zero game-engine hooks, zero CDP/Playwright injection, zero memory reading**.

## Core Principle: Strict Engine-Isolation Firewall
- `text.html` is studied offline solely as a textbook (`research/` → `docs/GAME_MECHANICS_TEXTBOOK.md`).
- At runtime, Territorial.io runs unmodified in a plain Google Chrome `--app` window on an isolated X11 display (`Xvfb`).
- The AI runs as a separate OS process observing only RGB screen pixels (`mss`) and actuating only human-like Bezier X11 mouse trajectories, clicks, drags, scroll wheel zooms, and legitimate game shortcuts (`python-xlib` XTEST).
- Automated AST firewall tests (`tests/firewall/test_zero_engine_coupling.py`) enforce zero engine coupling on every test run and CI push.

## Architecture Overview
- `vision/` — Scale-invariant white-border UI detector, screen classifier, dynamic terrain color calibration, hybrid LAB-ridge + `VisionUNetMobile` territory segmentation, sub-2ms `CanvasOCRNet` digit reader, naval boat & crown detector.
- `perception/` — Spawn-locked + color/IoU self-territory tracker, light/dark-text entity-to-troop matcher, camera-shift-filtered temporal combat flux tracker.
- `map_analysis/` — Coastline phase-correlation camera odometry, persistent `256×256` allocentric world map with fog staleness, territory adjacency graph `G=(V,E)`, 11-channel pure-CV tensor & spawn heatmap extractor.
- `strategic_brain/` — Compound-interest & red-cap `EconomyEngine`, rolling `OpponentModeler` (aggression & full-send vulture tracking), multi-action 2:1 defender-advantage `ForwardStrategicPlanner`, and RL-blended `StrategicBrain`.
- `models/` — `CanvasOCRNet`, `VisionUNetMobile`, `SpatialVisualEncoder` (IMPALA-ResNet + CoordConv), `TerritoryGraphEncoder` (multi-head `GATv2`), `RecurrentTemporalCore` (`GRU`), and factorized hybrid `TerritorialPolicyValueNet` (ActionType + 2D Spatial Pointer + Slider + Multi-Head Critic + ONNX export).
- `reinforcement_learning/` & `self_play/` — Potential-based reward shaping, recurrent GAE clipped PPO (CPU & CUDA AMP), visual behavioral cloning warm-start, 9-stage curriculum controller, Prioritized Fictitious Self-Play (PFSP) league, and Elo ratings.
- `memory/` — Persistent SQLite + Zstd Parquet episode store, automated TD/overextension/click-failure `MistakeAnalyzer`, SHA256 atomic `CheckpointManager`.
- `training/cloud/` — Automated Kaggle GPU (`T4`/`P100`) kernel orchestrator & GitHub Pro parallel evaluation matrix workflows.
- `dashboard/` — Real-time FastAPI + HTML telemetry UI.

## Quickstart

```bash
# 1. Verify Engine-Isolation Firewall & Unit Tests
make firewall
make test

# 2. Run Full 9-Stage Autonomous Training (auto-detects CPU/GPU)
make train

# 3. Play Live Unmodified Territorial.io via External X11 Controls
make play

# 4. Benchmark Champion Checkpoint (Visual Arena or Live Chrome)
make eval
make eval-live

# 5. Launch Telemetry Dashboard (http://localhost:8500)
make dashboard

# 6. Generate / Push Kaggle GPU Kernel (requires KAGGLE_API_TOKEN or ~/.kaggle/kaggle.json)
PYTHONPATH=. python3 cli.py kaggle --username <your-kaggle-username> --push
```

## Documentation
- [docs/GAME_MECHANICS_TEXTBOOK.md](docs/GAME_MECHANICS_TEXTBOOK.md) — Exact equations, tick rates, 2:1 combat math, bot tiers, visual grammar, and hotkeys.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Complete system design and module contracts.
- [docs/TRAINING_LOGBOOK.md](docs/TRAINING_LOGBOOK.md) — Genuine measured training metrics, evaluations, and Elo history.
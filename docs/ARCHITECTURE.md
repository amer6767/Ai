# Territorial.io Autonomous Computer-Vision AI — Architecture

## Core Design Principle: Strict Engine-Isolation Firewall

The AI operates 100% externally as a human sitting in front of a computer:
- **Sensory input**: Raw RGB screen pixels captured from an external X11 display (`mss` / `XShm`).
- **Actuation**: External OS pointer trajectories, clicks, drags, scroll wheel events, and legitimate keyboard shortcuts (`python-xlib` XTEST).
- **Zero engine coupling**: Plain `google-chrome --app=...` is launched without CDP/Playwright/Selenium/extensions. An automated AST test gate (`tests/firewall/test_zero_engine_coupling.py`) runs in pytest and GitHub Actions to guarantee no runtime module ever imports browser automation libraries or reads `text.html`.

---

## Module Architecture

| Package | Responsibility |
| :--- | :--- |
| `research/` | **Zone A offline textbook only** (never imported at runtime). Decodes string table, equations, bot tiers, and UI ratios from `text.html` into `docs/GAME_MECHANICS_TEXTBOOK.md`. |
| `core/` | Hardware auto-detection (`CPU/AVX2` vs `CUDA/AMP` vs `MPS`), Pydantic configs, immutable visual/action data contracts (`RawFrame`, `PerceivedFrame`, `ActionDecision`, `VerificationResult`), and the master `ClosedLoopAgent`. |
| `computer_control/` | Isolated `Xvfb` display + static HTTP server + plain Chrome launcher (`GameDisplaySession`), dynamic X11 window locator (`WindowLocator`), `mss` frame capturer (`ScreenCapturer`), smooth Bezier human-like mouse/keyboard (`ExternalInputController`), and mandatory visual post-action consequence verifier (`ActionVerifier`). |
| `vision/` | Scale-invariant white-border UI detector (`ScaleInvariantUIDetector`), match screen classifier (`ScreenStateClassifier`), dynamic color/biome terrain calibration (`DynamicColorCalibrator`), hybrid LAB-border + `VisionUNetMobile` segmenter (`TerritorySegmenter`), sub-2ms `CanvasOCRNet` reader (`FastCanvasOCR`), and naval boat/crown detector (`BoatAndIconDetector`). |
| `perception/` | Spawn-locked + color/IoU self-territory tracker (`SelfTerritoryIdentifier`), light/dark-text polygon-to-troop entity matcher (`TerritoryEntityTracker`), camera-shift-filtered temporal combat flux (`TemporalChangeTracker`), and unified `VisualPerceptionPipeline`. |
| `map_analysis/` | Coastline phase-correlation camera odometry (`VisualCameraOdometry`), persistent `256×256` allocentric world map with fog-of-vision staleness (`PersistentWorldMap`), 4-connected + naval adjacency graph `G=(V,E)` (`TerritoryGraphBuilder`), and 11-channel pure-CV neural observation + spawn heatmap builder (`TacticalFeatureExtractor`). |
| `strategic_brain/` | Compound-interest & red-cap analyzer (`EconomyEngine`), rolling aggression/vulture opponent profiler (`OpponentModeler`), legal macro-action generator (`CandidateActionGenerator`), 2:1 defender-advantage multi-action forward planner (`ForwardStrategicPlanner`), and RL-blended decision arbiter (`StrategicBrain`). |
| `models/` | PyTorch `CanvasOCRNet`, `VisionUNetMobile`, `SpatialVisualEncoder` (IMPALA-ResNet + CoordConv), `TerritoryGraphEncoder` (multi-head `GATv2`), `RecurrentTemporalCore` (`GRU`), and factorized hybrid `TerritorialPolicyValueNet` (ActionType + 2D Spatial Pointer + Slider + Multi-Head Critic + ONNX export). |
| `reinforcement_learning/` | Potential-based `RewardShaper`, recurrent GAE `RolloutBuffer` + `PrioritizedAuxiliaryReplay`, clipped `RecurrentPPOTrainer` (with AMP), visual planner Behavioral Cloning (`imitation.py`), and 9-Stage `CurriculumController`. |
| `training/` | Clean-room RGB-pixel-rendered simulator (`VisualPixelSimEnv` — pixel screenshots out, screen clicks in), live unmodified Chrome X11 environment (`LiveX11BrowserEnv`), `VectorizedPixelEnv`, full 9-stage `train_pipeline.py`, and cloud orchestrators (`training/cloud/kaggle_runner.py`, `github_actions_sync.py`). |
| `self_play/` | `EloRatingEngine`, Prioritized Fictitious Self-Play (`PFSPSampler`), and `SelfPlayLeagueManager`. |
| `memory/` | Persistent SQLite (`schema.sql`) + Zstd Parquet `PersistentMemoryStore`, TD-error/overextension/click-failure `MistakeAnalyzer`, and SHA256-verified atomic `CheckpointManager` with auto-recovery. |
| `evaluation/` | 7-metric benchmark suite (`Win Rate + 95% Wilson CI`, `Survival`, `Peak Land & AUC`, `Expansion Efficiency`, `Combat Efficiency`, `Decision Accuracy`, `Strategic Consistency`) and statistical `should_promote_champion` gating. |
| `dashboard/` | FastAPI server + dark-mode live telemetry web UI (`dashboard/server.py`, `dashboard/static/index.html`). |

---

## Closed-Loop Dataflow

```text
Xvfb X11 Screen -> mss RGB [H,W,3] -> Vision (UI/Seg/OCR/Boats) -> Perception (Self/Entities/Flux)
   -> Map Analysis (Odometry + 256x256 WorldMap + Graph G=(V,E) + 11ch Tensor)
   -> Strategic Brain (IMPALA-CNN + GATv2 + GRU Policy/Value + Forward Planner)
   -> External X11 Bezier Mouse/Keys -> Post-Action Screenshot -> ActionVerifier (Cooldown Mask)
   -> Persistent SQLite + Parquet Memory + Mistake Analyzer + PPO/Self-Play Learning
```

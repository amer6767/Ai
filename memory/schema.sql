-- Persistent SQLite Schema for Territorial.io Autonomous CV AI Memory

CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at REAL,
    env_mode TEXT NOT NULL,            -- 'pixel_sim' or 'live_browser'
    curriculum_stage INTEGER NOT NULL,
    opponent_tier TEXT NOT NULL,
    checkpoint_id TEXT,
    won INTEGER NOT NULL DEFAULT 0,
    placement_rank INTEGER,
    total_players INTEGER,
    survival_sec REAL DEFAULT 0.0,
    peak_land_ratio REAL DEFAULT 0.0,
    land_auc REAL DEFAULT 0.0,
    expansion_efficiency REAL DEFAULT 0.0,
    combat_efficiency REAL DEFAULT 0.0,
    decision_accuracy REAL DEFAULT 0.0,
    total_reward REAL DEFAULT 0.0,
    total_steps INTEGER DEFAULT 0,
    parquet_path TEXT
);

CREATE TABLE IF NOT EXISTS strategic_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    step_idx INTEGER NOT NULL,
    screen_state TEXT NOT NULL,
    action_type TEXT NOT NULL,
    target_x REAL,
    target_y REAL,
    slider_pct REAL,
    verified INTEGER NOT NULL,
    verify_reason TEXT,
    self_troops REAL,
    self_land_ratio REAL,
    reward REAL,
    value_est REAL,
    rationale TEXT,
    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS detected_mistakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    step_idx INTEGER NOT NULL,
    mistake_type TEXT NOT NULL,        -- 'unverified_action', 'red_interest_waste', 'overextension_collapse', 'blunder_td_drop'
    severity REAL NOT NULL,
    action_type TEXT NOT NULL,
    context_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    curriculum_stage INTEGER NOT NULL,
    global_steps INTEGER NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    is_champion INTEGER NOT NULL DEFAULT 0,
    elo_rating REAL NOT NULL DEFAULT 1000.0,
    eval_win_rate REAL,
    eval_survival_pct REAL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS curriculum_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    current_stage INTEGER NOT NULL DEFAULT 1,
    global_steps INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    stage_metrics_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_ep ON strategic_decisions(episode_id);
CREATE INDEX IF NOT EXISTS idx_mistakes_type ON detected_mistakes(mistake_type, severity DESC);
CREATE INDEX IF NOT EXISTS idx_checkpoints_champ ON checkpoints(is_champion, elo_rating DESC);

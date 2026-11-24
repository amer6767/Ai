"""Automated Kaggle GPU Kernel & Dataset Runner (supports Dual-T4 / P100 GPU acceleration).

Reads credentials strictly from environment (`KAGGLE_API_TOKEN`) or `~/.kaggle/kaggle.json`
/ `~/.kaggle/access_token`. Never hardcodes or logs secrets.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


class KaggleGPUOrchestrator:
    """Packages repository code, pushes GPU training kernels to Kaggle, polls, and pulls checkpoints."""

    def __init__(self, username: str, kernel_slug: str = "territorial-vision-ai-train") -> None:
        self.username = username
        self.kernel_slug = kernel_slug

    @staticmethod
    def has_credentials() -> bool:
        if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            return True
        home = Path.home()
        return (home / ".kaggle" / "kaggle.json").exists() or (home / ".kaggle" / "access_token").exists()

    def build_kernel_bundle(
        self,
        out_dir: str | Path,
        repo_url: str = "https://github.com/amer6767/Ai.git",
        ppo_updates_per_stage: int = 28,
        rollout_steps: int = 64,
        num_envs: int = 8,
        enable_gpu: bool = True,
    ) -> Path:
        bundle = Path(out_dir)
        bundle.mkdir(parents=True, exist_ok=True)

        metadata = {
            "id": f"{self.username}/{self.kernel_slug}",
            "title": "Territorial.io Autonomous CV AI GPU Train",
            "code_file": "run_kaggle_gpu.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if enable_gpu else "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (bundle / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        script = f'''"""Kaggle Dual-T4 GPU 9-Stage Autonomous Training Runner for Territorial.io CV AI."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

t0 = time.time()
REPO = Path("/kaggle/working/Ai")
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call(["git", "clone", "--depth", "1", "{repo_url}", str(REPO)])

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "opencv-python-headless", "mss", "python-xlib", "pydantic",
    "pyarrow", "zstandard", "gymnasium", "onnx", "onnxruntime"
])

sys.path.insert(0, str(REPO))
os.chdir(str(REPO))

import torch
print("CUDA available:", torch.cuda.is_available(), "Device count:", torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {{i}}:", torch.cuda.get_device_name(i), flush=True)

from training.train_pipeline import run_full_9_stage_training
summary = run_full_9_stage_training(
    ppo_updates_per_stage={ppo_updates_per_stage},
    rollout_steps={rollout_steps},
    num_envs={num_envs},
    run_live_stage8=False,
    resume=False,
)

# Export all trained artifacts to /kaggle/working/ so kaggle kernels output pulls them directly!
out_dir = Path("/kaggle/working")
(out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
for fname in ("ocr_net.pt", "vision_unet.pt", "champion.pt", "champion.onnx"):
    src = REPO / "models" / "checkpoints" / fname
    if src.exists():
        shutil.copy2(src, out_dir / fname)
db_src = REPO / "memory" / "ai_memory.db"
if db_src.exists():
    shutil.copy2(db_src, out_dir / "ai_memory.db")

# Remove cloned repo directory from /kaggle/working so kernel output stays clean and small
shutil.rmtree(REPO, ignore_errors=True)
print(f"KAGGLE_GPU_TRAINING_COMPLETE in {{time.time()-t0:.1f}}s", flush=True)
print(json.dumps(summary.get("final_benchmark", {{}}), indent=2), flush=True)
'''
        (bundle / "run_kaggle_gpu.py").write_text(script, encoding="utf-8")
        return bundle

    def push_and_monitor(
        self,
        bundle_dir: str | Path,
        accelerator: str = "NvidiaTeslaT4",
        poll_interval_sec: int = 20,
        timeout_sec: int = 7200,
        download_dir: str | Path = "runs/kaggle_artifacts",
    ) -> dict[str, Any]:
        if not self.has_credentials():
            raise RuntimeError(
                "Kaggle API credentials not found. Set KAGGLE_API_TOKEN or place ~/.kaggle/access_token (chmod 600)."
            )
        env = os.environ.copy()
        tok_file = Path.home() / ".kaggle" / "access_token"
        if "KAGGLE_API_TOKEN" not in env and tok_file.exists():
            env["KAGGLE_API_TOKEN"] = tok_file.read_text(encoding="utf-8").strip()

        cmd = ["kaggle", "kernels", "push", "-p", str(bundle_dir)]
        if accelerator:
            cmd.extend(["--accelerator", accelerator])
        subprocess.check_call(cmd, env=env)
        kernel_ref = f"{self.username}/{self.kernel_slug}"
        deadline = time.monotonic() + timeout_sec
        status_text = "running"

        while time.monotonic() < deadline:
            res = subprocess.run(
                ["kaggle", "kernels", "status", kernel_ref],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            status_text = (res.stdout + res.stderr).strip()
            low = status_text.lower()
            if "complete" in low:
                out = Path(download_dir)
                out.mkdir(parents=True, exist_ok=True)
                subprocess.check_call(["kaggle", "kernels", "output", kernel_ref, "-p", str(out)], env=env)
                return {"status": "complete", "output_dir": str(out), "raw": status_text}
            if "error" in low or "cancel" in low:
                return {"status": "failed", "raw": status_text}
            time.sleep(poll_interval_sec)

        return {"status": "timeout", "raw": status_text}

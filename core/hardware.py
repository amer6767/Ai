"""Automatic CPU/GPU hardware detection and compute tuning."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HardwareProfile:
    device_str: str
    device_name: str
    gpu_count: int
    total_vram_gb: float
    cpu_cores: int
    supports_amp: bool
    recommended_num_envs: int
    recommended_batch_size: int
    recommended_minibatch_size: int
    onnx_providers: tuple[str, ...]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device_str)


def detect_hardware(prefer_cpu: bool = False) -> HardwareProfile:
    cpu_cores = os.cpu_count() or 2
    cuda_ok = (not prefer_cpu) and torch.cuda.is_available()
    mps_ok = (
        (not prefer_cpu)
        and (not cuda_ok)
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

    if cuda_ok:
        gpu_count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024**3), 2)
        return HardwareProfile(
            device_str="cuda:0",
            device_name=f"{props.name} (x{gpu_count})",
            gpu_count=gpu_count,
            total_vram_gb=vram_gb * gpu_count,
            cpu_cores=cpu_cores,
            supports_amp=True,
            recommended_num_envs=min(16, max(4, cpu_cores * 2)),
            recommended_batch_size=1024,
            recommended_minibatch_size=256,
            onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        )

    if mps_ok:
        return HardwareProfile(
            device_str="mps",
            device_name="Apple Silicon MPS",
            gpu_count=1,
            total_vram_gb=0.0,
            cpu_cores=cpu_cores,
            supports_amp=False,
            recommended_num_envs=max(2, min(8, cpu_cores)),
            recommended_batch_size=512,
            recommended_minibatch_size=128,
            onnx_providers=("CPUExecutionProvider",),
        )

    # Optimize PyTorch thread count for CPU-only execution
    torch.set_num_threads(max(1, cpu_cores))
    return HardwareProfile(
        device_str="cpu",
        device_name=f"{platform.processor() or platform.machine()} ({cpu_cores} cores)",
        gpu_count=0,
        total_vram_gb=0.0,
        cpu_cores=cpu_cores,
        supports_amp=False,
        recommended_num_envs=max(2, min(4, cpu_cores)),
        recommended_batch_size=256,
        recommended_minibatch_size=64,
        onnx_providers=("CPUExecutionProvider",),
    )

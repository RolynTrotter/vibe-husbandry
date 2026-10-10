"""``husbandry doctor``: hardware, package versions, and a quick throughput check.

Paste its output into an issue or chat so experiments can be sized for the machine.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
import time

PACKAGES = (
    "torch", "numpy", "safetensors", "tokenizers", "transformers", "datasets",
    "huggingface-hub", "mlx", "mlx-lm", "peft", "trl", "lm-eval",
)  # fmt: skip


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def hardware() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": str(os.cpu_count()),
        "disk_free_gb": f"{shutil.disk_usage('.').free / 1024**3:.0f}",
    }
    if sys.platform == "darwin":
        info["chip"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["memory_gb"] = f"{int(mem) / 1024**3:.0f}" if mem.isdigit() else "?"
        info["perf_cores"] = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        info["eff_cores"] = _run(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
        gpu = _run(["system_profiler", "SPDisplaysDataType"])
        for line in gpu.splitlines():
            if "Total Number of Cores" in line:
                info["gpu_cores"] = line.split(":")[-1].strip()
        info["macos"] = platform.mac_ver()[0]
        info["wired_limit_mb"] = _run(["sysctl", "-n", "iogpu.wired_limit_mb"]) or "default"
    elif sys.platform == "linux":
        try:
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])
            info["memory_gb"] = f"{kb / 1024**2:.0f}"
        except OSError:
            pass
    return info


def versions() -> dict[str, str]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "-"
    return out


def matmul_tflops(device: str, dtype_name: str, n: int = 2048, iters: int = 20) -> float:
    import torch

    dtype = getattr(torch, dtype_name)
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    for _ in range(3):
        a @ b
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync()
    return 2 * n**3 * iters / (time.perf_counter() - t) / 1e12


def main(bench: bool = True) -> None:
    print("## hardware")
    for k, v in hardware().items():
        print(f"{k:>16}: {v}")
    print("## packages")
    for k, v in versions().items():
        print(f"{k:>16}: {v}")
    try:
        import torch
    except ImportError:
        print("torch not installed")
        return
    print("## torch backends")
    print(f"{'cuda':>16}: {torch.cuda.is_available()}")
    print(f"{'mps':>16}: {torch.backends.mps.is_available()}")
    try:
        import mlx.core as mx

        print(f"{'mlx device':>16}: {mx.default_device()}")
    except ImportError:
        pass
    if not bench:
        return
    print("## matmul throughput")
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    devices += ["cuda"] if torch.cuda.is_available() else []
    for dev in devices:
        # CPU half precision is slow and irrelevant for training here; skip it
        for dt in ("float32",) if dev == "cpu" else ("float32", "bfloat16", "float16"):
            n, iters = (1024, 3) if dev == "cpu" else (2048, 20)
            try:
                print(f"{dev + ' ' + dt:>16}: {matmul_tflops(dev, dt, n, iters):.2f} TFLOPS")
            except (RuntimeError, TypeError) as e:
                print(f"{dev + ' ' + dt:>16}: unsupported ({str(e).splitlines()[0][:60]})")

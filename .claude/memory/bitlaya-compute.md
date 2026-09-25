---
name: bitlaya-compute
description: Compute environments for BitLaya; the original laptop was CPU-only and power-limited, the work moved to a GPU PC (i9-10900K + RTX 2080 SUPER) on 2026-09-24
metadata:
  type: project
---

Until 2026-09-24 BitLaya was developed on an Intel i7-1165G7 laptop (4 cores / 8 threads, 16 GB), Windows 11, PyTorch 2.14.0+cpu, no CUDA. On 2026-09-24 the user moved the project to a GPU PC via https://github.com/dell5463/BitComp. Do not reuse laptop timings as GPU-PC numbers.

GPU PC (measured 2026-09-24):
- CPU Intel Core i9-10900K (10 cores / 20 threads, 3.7 GHz base), 32 GB RAM (2 x 16 GB DDR4-3200), Windows 11 Pro.
- GPU NVIDIA GeForce RTX 2080 SUPER, 8 GB, compute capability 7.5 (sm_75), driver 610.60 (CUDA 13.3 UMD), WDDM; the desktop uses ~1.1 GB of GPU memory.
- Python 3.11 venv at `.venv` (C:\Python311 base; the default `python` on PATH is Anaconda 3.13, do not use it). PyTorch 2.14.0+cu130 (CUDA 13.0 runtime), torchvision 0.29.0+cu130, NumPy 2.4.6. Same PyTorch version as the laptop (2.14.0), which matters for v1 streams that pin the runtime.
- Throughput (measured 2026-09-24, `results/throughput_gpupc/SUMMARY.md`): teacher forcing batch 32 x chunk 256 = 6.8 ms/step CUDA vs 257 ms/step CPU (1 thread; slower per job than the laptop's ~200 ms). Rollout/scheduled ~190-260 ms/step on CUDA, flat in batch size. GPU saturates with one teacher-forcing job (8 concurrent -> 1.27x aggregate; batch 128 concurrent is worse than serial); rollout -> ~1.4x aggregate at >= 2 jobs. CPU training scales to ~8 concurrent jobs (6x aggregate). RD sweep 200 val images, qgru-v2 = 957 s single job; sweeps scale poorly (8 concurrent -> 3.3x aggregate), so sweeps, not training, dominate teacher-forcing ablation wall time.

Laptop measurements (historical baseline only):
- Training was latency-bound on CPU: ~40 kbit/s per process regardless of thread count; ~0.2 s/step (batch 32 x 256-bit chunk) idle, ~0.5 s/step with 4 concurrent jobs.
- Going from 4 to 6 concurrent CPU jobs made every job ~3.5x slower (power/thermal limited), so total throughput dropped.

**Why:** these limits drove the design choices (fixed optimizer-step budgets of 6400 steps, parallel single-threaded ablations, 200-image validation sweeps).

**How to apply:** use `.venv/Scripts/python`. The codec itself (exact engine, NumPy) is CPU-only regardless of GPU; only training uses CUDA. Controlled timing benchmarks need an otherwise idle machine. See [[powershell-utf8-edits]].

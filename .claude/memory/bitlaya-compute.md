---
name: bitlaya-compute
description: Compute environments for BitLaya; the original laptop was CPU-only and power-limited, the work moved to a GPU PC on 2026-09-24
metadata:
  type: project
---

Until 2026-09-24 BitLaya was developed on an Intel i7-1165G7 laptop (4 cores / 8 threads, 16 GB), Windows 11, PyTorch 2.14.0+cpu, no CUDA. On 2026-09-24 the user moved the project to a GPU PC (target GPU: RTX 2080 Super) via https://github.com/dell5463/BitComp. Re-measure everything below on the new machine; do not reuse laptop timings as GPU-PC numbers.

Laptop measurements (still valid as a historical baseline):
- Training was latency-bound on CPU: ~40 kbit/s per process regardless of thread count; ~0.2 s/step (batch 32 x 256-bit chunk) idle, ~0.5 s/step with 4 concurrent jobs.
- Going from 4 to 6 concurrent CPU jobs made every job ~3.5x slower (clock pinned at 2.8 GHz base; power/thermal limited), so total throughput dropped. Keep concurrent CPU jobs <= physical cores.

**Why:** these limits drove the design choices (fixed optimizer-step budgets of 6400 steps, parallel single-threaded ablations, 200-image validation sweeps).

**How to apply:** on the GPU PC, check `torch.cuda.is_available()`, verify the GPU with `nvidia-smi`, and benchmark CPU vs CUDA before choosing budgets. The codec itself (exact engine, NumPy) is CPU-only regardless of GPU. Controlled timing benchmarks need an otherwise idle machine. See [[powershell-utf8-edits]].

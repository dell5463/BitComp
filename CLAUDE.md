# BitLaya — project context for Claude Code

Research codec: a causal GRU predicts each bit of a 32x32 grayscale CIFAR-10 image
(raster order, MSB first). Confident predictions are omitted (lossy); others are
transmitted. Encoder and decoder both feed back the *reconstructed* bit, so they
stay synchronized with no omission mask. **Objective: complete-file rate-distortion
(bpp vs PSNR/MSE/SSIM) at practical speed with deterministic sync — not BCE/accuracy.**

Read first: `README.md` (commands), `docs/FORMAT.md` (v1/v2/container formats),
`docs/ENGINE.md` (exact fixed-point engine and why), `.claude/memory/` (lessons).

## Rules the user set (keep following them)

- Change one factor per experiment; every comparison needs a controlled baseline.
- Decide on the **validation** split; use **test** only for held-out reporting.
- An improvement must show up in complete-file rate-distortion (or equal RD with
  much better speed/size), never in BCE/accuracy/omission rate alone.
- Never report estimated numbers as measured; label missing ones NOT YET MEASURED.
- Label sample scale: < 100 images smoke, < 1,000 development, >= 1,000 research.
- Do not remove synchronization/determinism safeguards for speed.
- Do not infer RTX 2080 Super performance until actually measured on CUDA.

## Architecture (src/bitlaya)

| Module | Role |
| --- | --- |
| `model.py` | GRU predictor; optional bit-plane/row/col embeddings (plane 0 = MSB, features describe the bit being predicted). Legacy configs keep their fingerprints. |
| `engine.py` | Exact fixed-point inference `qgru-v1` (F=18; integer-valued float64, proven < 2^52 so batch/thread/platform invariant). `step_reference` (spec) and `Stepper` (batched) are bit-identical. |
| `protocol.py` | The omit-or-transmit rule, model-independent (omit iff abs(L) >= bound; tie predicts 1). |
| `blaya.py` | v2 binary format (34-byte overhead), `BitLayaCodec` (reusable; raw/zlib/range payloads; threshold-1 fast path), container (6 B/image), reference encoder/decoder. |
| `rangecoder.py`, `lossless.py` | Integer LZMA-style range coder; lossless neural coding = threshold 1.0 + range payload. |
| `codec.py` | **Preserved** v1 float reference codec + JSON framing (do not change). |
| `training.py` | Modes teacher_forcing / scheduled / rollout; bce / weighted_bce; LR schedules; step budgets; accumulation. |
| `research.py`, `research_plots.py` | Fixed-subset RD sweeps (engines `qgru-v2`, `reference-v1`), resumable journal, bootstrap CIs, JPEG/WebP/PNG/zlib baselines, matched-PSNR, zlib and range-coded payload measurement, plots. |
| `analysis.py` | `diagnose` (calibration/reliability) and `analyze-errors` (cascade analysis with exact teacher-forced counterfactual). |
| `ablation.py` | `bitlaya ablate --config configs/*.json`; compares variants by operational RD envelope. Partial training runs are moved aside and restarted. |
| `profiling.py`, `overhead.py` | Controlled latency + cProfile; measured framing overhead. |
| `scripts/summarize.py` | Renders Markdown tables from result artifacts (use it; don't hand-copy numbers). |

## Status on the GPU PC (2026-09-24)

Machine: i9-10900K (10C/20T), 32 GB, RTX 2080 SUPER 8 GB (sm_75), Windows 11. Use the
`.venv` (Python 3.11, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130), not the Anaconda
`python` on PATH. `data/cifar10` prepared; content hashes equal the laptop's
(train `1505ba16...`, test `e11116a8...`). `pytest -q`: 105 passed, 0 failed, 0 skipped.
CUDA training verified run-to-run deterministic (same checkpoint fingerprint twice).

Step 2 throughput measured (`results/throughput_gpupc/SUMMARY.md`, made by
`scripts/bench_gpupc.sh` + `python scripts/summarize.py throughput results/throughput_gpupc`):
teacher forcing batch 32 = 6.8 ms/step CUDA vs 257.5 ms/step CPU (1 thread; 4 threads barely
helps); rollout/scheduled ~190-260 ms/step on CUDA regardless of batch (sequential per-bit
launches). One teacher-forcing job saturates the GPU (8 concurrent -> 1.27x aggregate);
rollout plateaus at ~1.4x aggregate from 2 jobs. RD sweep (CPU, exact engine) of 200 val
images = 957 s single job; 8 concurrent sweeps -> 3.3x aggregate. **Awaiting the user's
budget/parallelism decision before Step 3 long runs.** Laptop results not yet moved aside.

## Status at handoff (2026-09-24, from the CPU-only laptop)

First milestone **implemented and unit-tested** (see "Tests" below): 1,000-image
held-out harness, reproducible artifacts, dataset scaling support, bit-plane and
row/col embeddings, teacher-forcing / scheduled / rollout training, weighted BCE,
error-propagation instrumentation, compact binary format + container, reusable codec,
threshold-1 fast path, profiling tooling. Also done early: range-coded payloads and
lossless neural mode (Phase 9).

Measured so far (smoke/development scale unless stated; all in `results/`):
- Framing overhead: v1 JSON **418 B/image** -> v2 **34 B** (-91.9%); container 6.6 B/image
  with per-image CRCs, 2.6 without, at N = 50 (`results/smoke_overhead`).
- Float32 GRU logits for the same image were bit-identical in only **12.3%** of positions
  between batch 1 and batch 32 -> motivates the exact engine (`docs/ENGINE.md`).
- Exact engine at F=18: teacher-forced BCE equal to float32 to 6 digits (32 val images).
- Lossless smoke (20 test images, weak 32-image legacy model): BitLaya 7.09 bpp file /
  6.83 bpp payload vs PNG-9 6.55, zlib-9 7.61; range-coder overhead ~27 bits/image.
- Early teacher-forced signal only (single seed, not RD): bit-plane model val BCE 0.496 vs
  bit-only ~0.566 at step 800. **Must be confirmed by RD sweeps.**

**Interrupted / incomplete (laptop runs; rerun on the GPU PC):**
All laptop jobs were stopped deliberately before the move (not crashes).

- `results/baseline_reference_v1_test100` — preserved v1 float codec on 100 test images,
  partial (1,016/1,400 rows: thresholds 0.5-0.925 complete, 0.95 has 16/100 images,
  0.975-1.0 missing; no summary.json was produced). Started with the pre-v2 research config format, so
  `--resume` with current code will refuse; either rerun
  (`bitlaya sweep runs/cifar_10epochs/best.pt --engine reference-v1 --split test --size 100 --output ...`,
  ~2 h single core) or report the completed thresholds as partial.
- `results/baseline_qgru_v2_test1000` — legacy checkpoint, exact engine, 1,000 test images,
  partial (7,040/14,000 rows); started before range-payload measurement existed. Rerun fresh.
- `results/data_scaling` — n00032 and n00256 **completed** 6,400 steps on CPU (best val BCE
  0.5276 at step 3,200 for n32, overfitting afterwards; 0.5065 at step 6,400 for n256 — teacher-forced
  diagnostics only, no RD sweep yet). n01024 / n05000 had just started (no checkpoint); n45000 not
  started. `bitlaya ablate --config configs/data_scaling.json` moves incomplete runs aside and
  restarts them. Configs now use `"device": "auto"`: CPU- and CUDA-trained variants are not
  bitwise comparable, so for a clean comparison rerun all sizes on the same device (move the
  n00032/n00256 dirs aside first).
- Position ablation: not run (a speculative CPU launch was cancelled).
- Controlled profiling of v2 vs reference: NOT YET MEASURED (needs an idle machine;
  older reference-only profile: `results/stage1_reference_profile`, ~2.75 s/image median encode).
- CUDA/RTX 2080 Super: NOT YET MEASURED. `docs/MILESTONE1_REPORT.md` not yet written.

## Suggested next steps on the GPU PC

1. `python -m pip install -e ".[dev]"` with a CUDA PyTorch build; `bitlaya prepare --data data/cifar10`;
   `python -m pytest -q` (expect all passing); `nvidia-smi`.
2. Benchmark CPU vs CUDA training throughput (batch 32/64/128) before choosing budgets.
3. Stage 2 in order: `bitlaya ablate --config configs/data_scaling.json`, then
   `configs/position_ablation.json` (set its base from the scaling winner), each swept on the same
   200 validation images. Then `training_modes.json`, `objectives.json`, then `lr_schedule.json`,
   `tbptt.json`, `architecture.json` — each with the previous winner as base.
4. For the chosen model: `sweep --split test --size 1000`, `diagnose`, `analyze-errors`,
   `benchmark-lossless`, `overhead`, and `profile` for both engines on an idle machine.
5. Write `docs/MILESTONE1_REPORT.md` from `scripts/summarize.py` output.

## Tests

`python -m pytest -q` (offline, synthetic data). Covers serialization, causality,
legacy fingerprints, reference-vs-optimized byte equality, batch invariance, 3,000-case
sync fuzzing, confident-wrong feedback, `>=` boundary, fast path, payload exhaustion/
excess, header corruption, fresh-process decode, range coder (incl. worst-case carries),
lossless mode, containers, training modes, harness, and ablation bookkeeping.

## Environment notes

- Windows; prefer the Edit/Write tools over PowerShell text replacement (see memory).
- CLI sets BLAS pools to 1 thread by default; the codec engine is NumPy/CPU only.
- To restore Claude Code memory on a new machine, copy `.claude/memory/*.md` into
  `~/.claude/projects/<this-project-key>/memory/` (this file already carries the essentials).

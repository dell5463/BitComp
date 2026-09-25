# Laptop (CPU-only) partial results — do not mix into GPU-PC comparisons

Moved here on 2026-09-24 when the project moved to the GPU PC. All were stopped
deliberately before the move and are incomplete. They were produced on an
i7-1165G7 laptop with PyTorch 2.14.0+cpu, with a 6,400-step budget.

| Directory | State |
| --- | --- |
| `baseline_reference_v1_test100` | v1 float codec, 100 test images; 1,016/1,400 rows (thresholds 0.5-0.925 complete, 0.95 partial, 0.975-1.0 missing); no summary.json; pre-v2 research config format. |
| `baseline_qgru_v2_test1000` | legacy checkpoint, exact engine, 1,000 test images; 7,040/14,000 rows; no range-payload measurement. |
| `data_scaling` | n00032 and n00256 trained 6,400 CPU steps (no RD sweep); n01024/n05000 just started; n45000 not started. |
| `position_ablation` | config only; not run. |

Reruns on the GPU PC live in `results/baseline_*_gpupc`, `results/data_scaling`,
`results/position_ablation`, etc. (CUDA, 64,000-step budget).

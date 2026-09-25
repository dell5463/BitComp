GPU NVIDIA GeForce RTX 2080 SUPER, torch 2.14.0+cu130 (CUDA 13.0), chunk 256 bits; one optimizer step = batch x one chunk; 32 steps = one pass over a batch of images.

| model | mode | device | threads | batch | timed steps | ms/step | steps/s | kbit/s | image passes/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| bit only (default) | teacher_forcing | cuda | - | 32 | 400 | 6.8 | 147.71 | 1210 | 147.71 |
| bit only (default) | teacher_forcing | cpu | 1 | 32 | 40 | 257.5 | 3.88 | 32 | 3.88 |
| bit only (default) | teacher_forcing | cpu | 4 | 32 | 40 | 250.5 | 3.99 | 33 | 3.99 |
| bit only (default) | teacher_forcing | cuda | - | 64 | 400 | 9.8 | 102.13 | 1673 | 204.27 |
| bit only (default) | teacher_forcing | cpu | 1 | 64 | 40 | 376.5 | 2.66 | 44 | 5.31 |
| bit only (default) | teacher_forcing | cpu | 4 | 64 | 40 | 345.4 | 2.89 | 47 | 5.79 |
| bit only (default) | teacher_forcing | cuda | - | 128 | 400 | 14.5 | 69.17 | 2267 | 276.68 |
| bit only (default) | teacher_forcing | cpu | 1 | 128 | 40 | 637.7 | 1.57 | 51 | 6.27 |
| bit only (default) | teacher_forcing | cpu | 4 | 128 | 40 | 567.0 | 1.76 | 58 | 7.05 |
| bit only (default) | rollout | cuda | - | 32 | 20 | 188.6 | 5.30 | 43 | 5.30 |
| bit only (default) | rollout | cpu | 1 | 32 | 8 | 380.4 | 2.63 | 22 | 2.63 |
| bit only (default) | rollout | cuda | - | 64 | 20 | 186.7 | 5.36 | 88 | 10.71 |
| bit only (default) | rollout | cpu | 1 | 64 | 8 | 518.1 | 1.93 | 32 | 3.86 |
| bit only (default) | rollout | cuda | - | 128 | 20 | 187.4 | 5.34 | 175 | 21.35 |
| bit only (default) | rollout | cpu | 1 | 128 | 8 | 1065.3 | 0.94 | 31 | 3.75 |
| bit only (default) | scheduled | cuda | - | 32 | 20 | 227.1 | 4.40 | 36 | 4.40 |
| bit only (default) | scheduled | cpu | 1 | 32 | 8 | 450.1 | 2.22 | 18 | 2.22 |
| bit only (default) | scheduled | cuda | - | 64 | 20 | 238.9 | 4.19 | 69 | 8.37 |
| bit only (default) | scheduled | cpu | 1 | 64 | 8 | 641.5 | 1.56 | 26 | 3.12 |
| bit only (default) | scheduled | cuda | - | 128 | 20 | 263.9 | 3.79 | 124 | 15.16 |
| bit only (default) | scheduled | cpu | 1 | 128 | 8 | 1037.0 | 0.96 | 32 | 3.86 |
| bit+plane+row/col | teacher_forcing | cuda | - | 32 | 400 | 7.5 | 132.51 | 1085 | 132.51 |
| bit+plane+row/col | teacher_forcing | cuda | - | 64 | 400 | 10.6 | 94.58 | 1550 | 189.16 |
| bit+plane+row/col | teacher_forcing | cuda | - | 128 | 400 | 16.6 | 60.11 | 1970 | 240.46 |

Concurrency: N identical jobs started together. Per-job and aggregate throughput (aggregate = sum over jobs; windows overlap but are not perfectly aligned).

| job | N | per-job ms/step (mean) | slowest job ms/step | per-job slowdown vs N=1 | aggregate steps/s | aggregate vs N=1 |
|---|---:|---:|---:|---:|---:|---:|
| cpu_tf_b32 | 1 | 272.0 | 272.0 | 1.00x | 3.68 | 1.00x |
| cpu_tf_b32 | 4 | 332.6 | 334.3 | 1.22x | 12.03 | 3.27x |
| cpu_tf_b32 | 8 | 364.2 | 368.4 | 1.34x | 21.97 | 5.98x |
| cpu_tf_b32 | 10 | 407.3 | 411.9 | 1.50x | 24.55 | 6.68x |
| cpu_tf_b32 | 12 | 443.3 | 449.3 | 1.63x | 27.07 | 7.36x |
| cuda_rollout_b32 | 1 | 186.5 | 186.5 | 1.00x | 5.36 | 1.00x |
| cuda_rollout_b32 | 2 | 265.0 | 265.7 | 1.42x | 7.55 | 1.41x |
| cuda_rollout_b32 | 4 | 525.0 | 530.0 | 2.81x | 7.62 | 1.42x |
| cuda_rollout_b32 | 6 | 798.2 | 809.0 | 4.28x | 7.52 | 1.40x |
| cuda_rollout_b32 | 8 | 1102.0 | 1113.7 | 5.91x | 7.26 | 1.35x |
| cuda_tf_b128 | 1 | 13.5 | 13.5 | 1.00x | 74.18 | 1.00x |
| cuda_tf_b128 | 2 | 34.3 | 34.3 | 2.55x | 58.27 | 0.79x |
| cuda_tf_b128 | 4 | 64.9 | 65.3 | 4.81x | 61.65 | 0.83x |
| cuda_tf_b32 | 1 | 7.1 | 7.1 | 1.00x | 140.72 | 1.00x |
| cuda_tf_b32 | 2 | 13.2 | 13.2 | 1.86x | 151.61 | 1.08x |
| cuda_tf_b32 | 4 | 24.1 | 24.2 | 3.39x | 165.97 | 1.18x |
| cuda_tf_b32 | 6 | 33.9 | 33.9 | 4.77x | 177.16 | 1.26x |
| cuda_tf_b32 | 8 | 44.8 | 45.0 | 6.30x | 178.78 | 1.27x |

RD sweep cost (CPU, NumPy exact engine, 1 thread per job; all thresholds per image). Aggregate = jobs x (single-job s/image / per-job s/image).

| images/job | concurrent jobs | codec s/image (mean over jobs) | per-job slowdown | aggregate speedup | wall s/job (mean) | decodes verified | mismatches |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1 | 3.02 | 1.00x | 1.00x | 48 | 140 | 0 |
| 10 | 4 | 5.10 | 1.69x | 2.37x | 80 | 560 | 0 |
| 10 | 8 | 7.35 | 2.43x | 3.29x | 114 | 1120 | 0 |
| 200 | 1 | 3.04 | 1.00x | 1.00x | 957 | 2800 | 0 |

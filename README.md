# BitLaya

A research platform for **lossy omission of predictable image bits**. A small
causal GRU predicts the next bit. When its confidence reaches a threshold, both
encoder and decoder use the prediction and nothing is transmitted; otherwise the
encoder transmits the true bit. Both sides feed the *reconstructed* bit back into
the model, so they stay synchronized without an omission mask. BitLaya is an
independent implementation: no Laya dependency, no pretrained weights.

The question it is built to answer: *can a small shared causal predictor reduce
the information that must actually be transmitted, and under what representation,
training objective and distortion budget does that become worthwhile?* The primary
metric is complete-file rate-distortion (bits per pixel vs PSNR/MSE/SSIM), not
prediction accuracy.

Target data: CIFAR-10 converted to **32x32 uint8 grayscale** (8,192 bits/image).

## Install

Python 3.11+. Install a PyTorch/torchvision pair for your hardware from the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m bitlaya --help
```

`bitlaya` and `python -m bitlaya` are equivalent. Commands default to one CPU
thread (`--threads N` before the subcommand); BLAS pools default to one thread so
parallel experiments do not oversubscribe cores.

## Codec in one paragraph

At bit t the model produces a logit L from the reconstructed history. If
`confidence = sigmoid(|L|) >= threshold` (comparison `>=`; a tie `L = 0` predicts
1), the predicted bit is used and nothing is sent -- even when it is wrong.
Otherwise the next explicit payload bit is used. Threshold 0.5 omits everything;
threshold 1.0 is lossless and never runs the model (fast path). Serialization is
row-major pixels, MSB first. See [docs/FORMAT.md](docs/FORMAT.md).

## Formats and engines

| | v2 `.blaya` (default) | v1 `.blay` (legacy, preserved) |
| --- | --- | --- |
| Header | 34 bytes, fixed binary fields | ~415-420 bytes, JSON + SHA-256 |
| Inference | exact fixed-point engine `qgru-v1` | float32 PyTorch GRU |
| Determinism | independent of batch size, threads, BLAS, CPU | only within one PyTorch build/platform |
| Payload | raw bits, zlib, or range-coded with model probabilities | raw bits |
| Commands | `compress`, `decompress` | `encode`, `decode` |

The exact engine and why it exists: [docs/ENGINE.md](docs/ENGINE.md). A container
format stores many images with 6 bytes of per-image overhead (2 without per-image
CRCs). **The model is never embedded**; checkpoint cost is reported separately
and amortized over N images in experiment summaries.

```sh
bitlaya compress image.png image.blaya --checkpoint runs/x/best.pt --threshold 0.9
bitlaya inspect image.blaya
bitlaya decompress image.blaya decoded.png --checkpoint runs/x/best.pt
```

## Research workflow

```sh
bitlaya prepare --data data/cifar10            # CIFAR-10 -> cached grayscale arrays

# Train (45k/5k train/val split; --train-size/--val-size take nested subsets).
bitlaya train --output runs/pilot --train-size 1024 --val-size 256 --max-steps 6400 --val-every 800
bitlaya train --config configs/position_ablation.json --output runs/x   # JSON config; flags override

# Rate-distortion sweep on a FIXED held-out subset (val for decisions, test for reporting).
bitlaya sweep runs/pilot/best.pt --output results/pilot_val --split val --size 200
bitlaya sweep runs/pilot/best.pt --output results/pilot_test --split test --size 1000

bitlaya diagnose runs/pilot/best.pt --output results/pilot_diag         # BCE, calibration, reliability
bitlaya analyze-errors runs/pilot/best.pt --output results/pilot_errors  # error propagation
bitlaya benchmark-lossless runs/pilot/best.pt --output results/pilot_lossless
bitlaya profile runs/pilot/best.pt --output results/pilot_profile        # controlled latency + cProfile
bitlaya ablate --config configs/data_scaling.json --parallel 2           # controlled ablations
```

Training options: `--mode teacher_forcing|scheduled|rollout` (reconstructed-context
training), `--objective bce|weighted_bce --weights sqrt|linear|uniform|w0,...,w7`
(bit-plane weights, MSB first, normalized to mean 1), `--lr-schedule
constant|plateau|cosine`, `--chunk-length`, `--accumulate`, position features
`--plane-dim N --spatial-dim N` (bit plane index 0 = MSB; row/column of the pixel
being predicted). With all feature dimensions 0 the model is the original
161,505-parameter predictor and old checkpoints/v1 streams keep their fingerprints.

Every sweep writes `config.json` (seed, checkpoint hashes, dataset indices and
content hash, environment, source hashes), `measurements.jsonl` (resumable),
`metrics.csv`, `summary.json` (mean/median/std, bootstrap 95% CIs, pooled PSNR,
payload zlib and range-coding results, matched-PSNR JPEG/WebP bytes, model
amortization for N = 1..10,000), `timing.json`, baselines, RD plots and sample
reconstructions. Every stream is decoded independently from its bytes and must
equal the encoder's reconstruction.

Sampling labels: < 100 images "smoke", < 1,000 "development", >= 1,000 "research".
Do not select models or thresholds on the test split.

## Tests

```sh
python -m pytest -q
```

Offline; no CIFAR download needed. They cover exact serialization; causality;
position features; legacy fingerprint compatibility; reference vs optimized
encoder/decoder byte-for-byte equality; batch-size invariance of the exact engine;
3,000-case random synchronization fuzzing; confidently wrong predictions fed on both
sides; threshold boundary (`>=`) behavior; threshold-1 fast path; payload
exhaustion and excess; header corruption (magic, version, flags, dimensions,
lengths, model id, checksums, truncation); fresh-process decoding; range-coder
roundtrips including worst-case carries; lossless neural coding; containers;
training modes/objectives/schedules; and the research harness.

## Results and limitations

Current status, measurements so far, and incomplete runs are summarized in
[CLAUDE.md](CLAUDE.md); the milestone report (`docs/MILESTONE1_REPORT.md`) will be
written once the research-scale runs finish. The original v1 validation record is
in [docs/VALIDATION.md](docs/VALIDATION.md).

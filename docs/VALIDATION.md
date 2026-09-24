# Local validation record

Verified on 2026-09-22 with Python 3.11.9, Windows, PyTorch 2.14.0+cpu,
torchvision 0.29.0+cpu, NumPy 2.4.6, Pillow 12.3.0, matplotlib 3.11.2,
and pytest 9.1.1. CUDA was unavailable. No RTX 2080 Super benchmark was run.

## Automated checks

- `python -m pytest -q`: **49 passed**.
- `python -m pip check`: no broken requirements.
- `python -m compileall -q src tests`: passed.
- An actual CIFAR-10 PNG encoded at threshold 1.0 and decoded through separate
  CLI processes reproduced all 1,024 grayscale pixels exactly.

Tests include an encoder deliberately making wrong omissions followed by explicit
bits, proving that the decoder stays synchronized with reconstructed context.
They also cover fresh-process decoding without access to the original image.

## CIFAR-10 smoke experiment

The complete official CIFAR-10 archive was obtained from a mirror after the
original host was slow, and verified against torchvision's archive MD5
`c58f30108f718f92721af3b95e74349a` before extraction/loading. All 50,000 training
and 10,000 test images were converted to grayscale. Cache hashes and conversion
versions are in `data/cifar10/preprocessing.json`.

```sh
python -m bitlaya train --output runs/cifar_smoke --epochs 2 --train-limit 32 --val-limit 8 --batch-size 8 --device cpu
python -m bitlaya evaluate --checkpoint runs/cifar_smoke/best.pt --split val --limit 2 --thresholds 0.5 0.6 0.75 0.9 0.95 1.0 --examples 2 --output runs/cifar_smoke/evaluation
```

This uses the default 32-dimensional embedding, 2-layer 128-unit GRU, and 161,505
parameters. Split and initialization seed are 42. Training and validation subsets
are disjoint; the test set was not used. Validation BCE decreased from 0.682646
after epoch 1 to 0.654428 after epoch 2. Final teacher-forced bit accuracy was
60.88%. Each CPU epoch took about 55 seconds in this environment.

**These are correctness smoke results from only two evaluation images, not a
trained-model benchmark or evidence of competitive compression.**

| Threshold | Omitted bits | Payload bits/pixel | Complete-file bits/pixel | Pooled PSNR |
| --- | ---: | ---: | ---: | ---: |
| 0.50 | 100.00% | 0.000 | 3.242 | 4.77 dB |
| 0.60 | 87.35% | 1.012 | 4.281 | 6.88 dB |
| 0.75 | 10.58% | 7.153 | 10.430 | 21.20 dB |
| 0.90 | 0.00% | 8.000 | 11.266 | infinity (exact) |
| 0.95 | 0.00% | 8.000 | 11.273 | infinity (exact) |
| 1.00 | 0.00% | 8.000 | 11.266 | infinity (exact) |

Every encoded stream was independently decoded and matched the encoder's
reconstruction. Thresholds 0.9 and 0.95 happened to transmit everything here;
only the threshold-1 sentinel guarantees this behavior for all finite models.

The raw images are 1,024 bytes each. Lossless baselines averaged 1,014.5 bytes
for zlib and 867.5 bytes for PNG. The checkpoint is 651,650 bytes and is excluded
from the complete-file column above; the JSON report also includes its amortized
cost. Aggressive omission sacrifices substantial image quality. At 0.75, payload
savings are overwhelmed by the research header. There is no demonstrated useful
compression advantage from this short smoke run.

Local artifacts:

- `runs/cifar_smoke/best.pt`, `config.json`, `history.json`
- `runs/cifar_smoke/evaluation/results.json`, `per_image.csv`
- `runs/cifar_smoke/evaluation/rate_distortion.png`, `reconstructions.png`
- `runs/cifar_smoke/evaluation/example_*`: twelve streams and reconstructed PNGs
- `runs/cifar_smoke/source.png`, `lossless.blay`, `lossless.png`: CLI round trip
- `runs/setup/environment.txt`: installed package versions

Generated data and runs are intentionally ignored by Git; these artifacts exist
in the local workspace. The source and tests can reproduce the workflow.

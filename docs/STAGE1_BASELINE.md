# Stage 1: preserved baseline and evaluation infrastructure

This pass adds measurement infrastructure. Model architecture, optimizer,
training objective, reference codec decisions, and the v1 JSON format remain
unchanged. Later stages will use these results as a controlled baseline.

## Existing repository

- `bits.py`: deterministic uint8 raster/MSB-first serialization.
- `model.py`: bit embedding, two-layer GRU, checkpoints, model fingerprints.
- `data.py`: CIFAR-10 grayscale preparation and disjoint training/validation splits.
- `training.py`: teacher-forced BCE, AdamW, truncated backpropagation.
- `codec.py`: independent sequential encode/decode, reconstructed-bit feedback,
  model/runtime identity checks, corruption detection, and readable v1 framing.
- `evaluation.py`: original small-sample evaluator, retained for compatibility.
- `cli.py`: original command interface, extended with research commands.
- `tests/`: 49 pre-existing tests, all passing before this pass.

The original implementation and measurements remain in `runs/cifar_smoke/` and
`runs/cifar_10epochs/`. A source snapshot was also saved locally under
`results/pre_milestone_source/` before any model or codec changes (none in this pass).

## Baseline limitations

The best available model was trained on 32 images, selected on eight validation
images, and previously codec-evaluated on two validation images. Those results
are smoke measurements. Increasing the codec evaluation sample does not change
the model's training history or establish competitive compression performance.

The development machine has CPU-only PyTorch; RTX 2080 Super performance is
**NOT YET MEASURED**. All configurations record actual environment and thread
settings. Profiler timings include instrumentation overhead; separate warmed
latency runs are used for performance reporting.

"""Teacher-forced BCE training with truncated backpropagation through time."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import BitImageDataset
from .model import BitPredictor, ModelConfig, save_checkpoint, teacher_inputs


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    chunk_length: int = 256
    learning_rate: float = 0.001
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    workers: int = 0

    def __post_init__(self):
        for name in ("epochs", "batch_size", "chunk_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.workers < 0 or not 0 <= self.seed < 2**32:
            raise ValueError("workers must be nonnegative and seed must fit uint32")
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.grad_clip) or self.grad_clip <= 0):
            raise ValueError("learning rate and gradient clip must be finite and positive")


def seed_everything(seed: int):
    # Must be set before creating CUDA contexts for deterministic cuBLAS.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("training supports CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return device


def _epoch(model, loader, device, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    # Same confidence bins used for teacher-forced validation calibration.
    bin_count = np.zeros(10, dtype=np.int64)
    bin_correct = np.zeros(10, dtype=np.float64)
    bin_confidence = np.zeros(10, dtype=np.float64)
    with torch.set_grad_enabled(training):
        for batch in loader:
            targets = batch.to(device)
            inputs = teacher_inputs(targets)
            hidden = None  # Reset between independent image batches.
            for start in range(0, targets.shape[1], config.chunk_length):
                stop = start + config.chunk_length
                chunk = targets[:, start:stop]
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits, hidden = model(inputs[:, start:stop], hidden)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, chunk.float())
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                             error_if_nonfinite=True)
                    optimizer.step()
                hidden = hidden.detach()  # Carry context, truncate gradient graph.
                total_loss += loss.item() * chunk.numel()
                matches = ((logits.detach() >= 0) == chunk.bool())
                correct += matches.sum().item()
                count += chunk.numel()
                if not training:
                    confidence = torch.sigmoid(logits.abs()).detach()
                    # Weighted CUDA bincount is nondeterministic; aggregate on CPU.
                    confidence_np = confidence.cpu().numpy().reshape(-1)
                    bins = np.clip(((confidence_np - 0.5) * 20).astype(int), 0, 9)
                    bin_count += np.bincount(bins, minlength=10)
                    bin_correct += np.bincount(bins, weights=matches.cpu().numpy().reshape(-1),
                                               minlength=10)
                    bin_confidence += np.bincount(bins, weights=confidence_np, minlength=10)
    metrics = {"bce_nats": total_loss / count,
               "cross_entropy_bits": total_loss / count / math.log(2),
               "bit_accuracy": correct / count}
    if not training:
        calibration = []
        ece = 0.0
        for index, n in enumerate(bin_count):
            accuracy = float(bin_correct[index] / n) if n else None
            confidence = float(bin_confidence[index] / n) if n else None
            if n:
                ece += int(n) / count * abs(accuracy - confidence)
            calibration.append({"lower": 0.5 + index * 0.05, "count": int(n),
                                "accuracy": accuracy, "confidence": confidence})
        metrics.update(ece=ece, calibration=calibration)
    return metrics


def train(train_images: np.ndarray, validation_images: np.ndarray, output: str | Path,
          model_config: ModelConfig = ModelConfig(), config: TrainConfig = TrainConfig(),
          provenance: dict | None = None) -> tuple[BitPredictor, list[dict]]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    model = BitPredictor(model_config).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(BitImageDataset(train_images), batch_size=config.batch_size,
                        shuffle=True, generator=generator, num_workers=config.workers)
    validation_loader = DataLoader(BitImageDataset(validation_images),
                                   batch_size=config.batch_size, num_workers=config.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("best.pt", "last.pt", "history.json")):
        raise ValueError("training output already contains a run; choose a new --output directory")
    run = {"model": asdict(model_config), "training": asdict(config),
           "parameters": model.parameter_count, "device": str(device),
           "torch": str(torch.__version__), "numpy": np.__version__,
           "train_images": len(train_images), "validation_images": len(validation_images),
           "train_sha256": hashlib.sha256(train_images.tobytes()).hexdigest(),
           "validation_sha256": hashlib.sha256(validation_images.tobytes()).hexdigest(),
           "provenance": provenance or {}}
    (output / "config.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({"parameters": model.parameter_count, "device": str(device)}), flush=True)
    history = []
    best = math.inf
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        training = _epoch(model, loader, device, config, optimizer)
        validation = _epoch(model, validation_loader, device, config)
        record = {"epoch": epoch, "train": training, "validation": validation,
                  "seconds": time.perf_counter() - started}
        history.append(record)
        metadata = {"run": run, "epoch": epoch, "validation": validation}
        save_checkpoint(output / "last.pt", model, metadata)
        if validation["bce_nats"] < best:
            best = validation["bce_nats"]
            save_checkpoint(output / "best.pt", model, metadata)
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"epoch": epoch, "train_bce": training["bce_nats"],
                          "validation_bce": validation["bce_nats"],
                          "validation_accuracy": validation["bit_accuracy"],
                          "seconds": record["seconds"]}), flush=True)
    return model.eval(), history

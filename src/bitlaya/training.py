"""BCE training with truncated backpropagation through time (TBPTT).

Training modes (context fed to the GRU; the loss always targets SOURCE bits):

* ``teacher_forcing`` -- inputs are the original previous bits (baseline).
* ``scheduled`` -- a no-grad sequential rollout per chunk replaces each previous
  bit, with probability rho(epoch), by the model's own argmax prediction made
  from the (possibly already replaced) history. ``schedule`` maps epochs to rho.
* ``rollout`` -- the rollout applies the actual codec rule: omit (feed the
  prediction) iff confidence >= threshold, else feed the source bit. The
  threshold is drawn per batch uniformly from ``rollout_thresholds``.

For the reconstructed-context modes the gradient pass re-runs the chunk in
parallel on the rollout's inputs, so the hidden trajectory is the rollout's own;
discrete decisions receive no gradient. Hidden state carries across chunks and
is detached; it resets between batches (independent images).

Objectives: ``bce`` (uniform) or ``weighted_bce`` with per-bit-plane weights
(index 0 = MSB), normalized to mean 1 over the 8 planes so loss magnitude stays
comparable. Validation always reports unweighted BCE as well.
"""

from dataclasses import asdict, dataclass, field
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
from .model import BOS, BitPredictor, ModelConfig, save_checkpoint, teacher_inputs

MODES = ("teacher_forcing", "scheduled", "rollout")
OBJECTIVES = ("bce", "weighted_bce")
LR_SCHEDULES = ("constant", "plateau", "cosine")
HIGH_CONFIDENCE_BINS = ((0.90, 0.95), (0.95, 0.99), (0.99, 1.0000001))


def plane_weights(strategy) -> list[float]:
    """Per-plane loss weights, plane 0 = MSB, normalized to mean 1."""
    significance = np.array([2.0 ** (7 - k) for k in range(8)])
    if isinstance(strategy, (list, tuple)):
        weights = np.asarray(strategy, dtype=float)
        if weights.shape != (8,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("custom plane weights must be 8 positive finite numbers (MSB first)")
    elif strategy == "uniform":
        weights = np.ones(8)
    elif strategy == "sqrt":
        weights = np.sqrt(significance)
    elif strategy == "linear":
        weights = significance
    else:
        raise ValueError("plane weights must be uniform, sqrt, linear or a list of 8 numbers")
    return (weights / weights.mean()).tolist()


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
    mode: str = "teacher_forcing"
    schedule: tuple = ((1, 0.0), (3, 0.1), (5, 0.25), (7, 0.5))
    rollout_thresholds: tuple = (0.6, 0.95)
    objective: str = "bce"
    weights: object = "sqrt"
    lr_schedule: str = "constant"
    max_steps: int | None = None
    val_every: int | None = None
    accumulate: int = 1
    width: int = 32
    schedule_unit: str = "epoch"  # "epoch": first_epoch >= 1; "progress": fraction of planned steps in [0, 1)

    def __post_init__(self):
        for name in ("epochs", "batch_size", "chunk_length", "accumulate", "width"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_steps", "val_every"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.workers < 0 or not 0 <= self.seed < 2**32:
            raise ValueError("workers must be nonnegative and seed must fit uint32")
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.grad_clip) or self.grad_clip <= 0):
            raise ValueError("learning rate and gradient clip must be finite and positive")
        if self.mode not in MODES or self.objective not in OBJECTIVES or self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"mode in {MODES}, objective in {OBJECTIVES}, lr_schedule in {LR_SCHEDULES}")
        object.__setattr__(self, "schedule", tuple(tuple(x) for x in self.schedule))
        object.__setattr__(self, "rollout_thresholds", tuple(self.rollout_thresholds))
        if isinstance(self.weights, list):
            object.__setattr__(self, "weights", tuple(self.weights))
        if self.schedule_unit not in ("epoch", "progress"):
            raise ValueError("schedule_unit must be epoch or progress")
        start = 1 if self.schedule_unit == "epoch" else 0
        if not self.schedule or any(len(x) != 2 or x[0] < start or not 0 <= x[1] <= 1 for x in self.schedule):
            raise ValueError("schedule entries are (first_epoch >= 1 | first_progress >= 0, fraction in [0, 1])")
        low, high = (self.rollout_thresholds * 2)[:2]
        if len(self.rollout_thresholds) not in (1, 2) or not 0.5 <= low <= high <= 1.0:
            raise ValueError("rollout_thresholds is (t,) or (low, high) within [0.5, 1]")
        plane_weights(list(self.weights) if isinstance(self.weights, tuple) else self.weights)

    def replacement_fraction(self, epoch: int, progress: float = 0.0) -> float:
        """Scheduled-mode replacement probability at an epoch (or training progress)."""
        position = epoch if self.schedule_unit == "epoch" else progress
        fraction = 0.0
        for first, value in sorted(self.schedule):
            if position >= first:
                fraction = value
        return fraction

    def loss_weights(self) -> list[float]:
        if self.objective == "bce":
            return [1.0] * 8
        return plane_weights(list(self.weights) if isinstance(self.weights, tuple) else self.weights)


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


@torch.no_grad()
def rollout_inputs(model, targets, first_input, hidden, start, width, *, threshold=None, fraction=None,
                   generator=None):
    """Sequentially build reconstructed-context inputs for one chunk (no gradient).

    Exactly one of ``threshold`` (codec rule) or ``fraction`` (scheduled sampling).
    Returns (inputs, replaced mask, last reconstructed bit per row).
    """
    batch, length = targets.shape
    inputs = torch.empty_like(targets)
    replaced = torch.zeros(batch, length, dtype=torch.bool, device=targets.device)
    symbol = first_input.clone()
    for j in range(length):
        inputs[:, j] = symbol
        logits, hidden = model(symbol[:, None], hidden, start=start + j, width=width)
        logits = logits[:, 0]
        predicted = (logits >= 0).long()
        if threshold is not None:
            use = torch.sigmoid(logits.abs()) >= threshold
        else:
            use = torch.rand(batch, generator=generator, device="cpu").to(targets.device) < fraction
        replaced[:, j] = use
        symbol = torch.where(use, predicted, targets[:, j])
    return inputs, replaced, symbol


class Calibration:
    """Confidence-bucket reliability, including high-confidence buckets."""

    def __init__(self):
        self.edges = np.linspace(0.5, 1.0, 11)
        self.count = np.zeros(10, dtype=np.int64)
        self.correct = np.zeros(10)
        self.confidence = np.zeros(10)
        self.high = np.zeros((len(HIGH_CONFIDENCE_BINS), 2))  # count, correct

    def update(self, confidence: np.ndarray, correct: np.ndarray):
        bins = np.clip(((confidence - 0.5) * 20).astype(int), 0, 9)
        self.count += np.bincount(bins, minlength=10)
        self.correct += np.bincount(bins, weights=correct, minlength=10)
        self.confidence += np.bincount(bins, weights=confidence, minlength=10)
        for index, (low, high) in enumerate(HIGH_CONFIDENCE_BINS):
            mask = (confidence >= low) & (confidence < high)
            self.high[index] += (mask.sum(), correct[mask].sum())

    def report(self) -> dict:
        total = int(self.count.sum())
        calibration, ece = [], 0.0
        for index, n in enumerate(self.count):
            accuracy = float(self.correct[index] / n) if n else None
            confidence = float(self.confidence[index] / n) if n else None
            if n:
                ece += int(n) / total * abs(accuracy - confidence)
            calibration.append({"lower": 0.5 + index * 0.05, "count": int(n),
                                "accuracy": accuracy, "confidence": confidence})
        high = [{"range": [low, min(high, 1.0)], "count": int(n), "accuracy": float(c / n) if n else None,
                 "errors": int(n - c)} for (low, high), (n, c) in zip(HIGH_CONFIDENCE_BINS, self.high)]
        return {"ece": ece, "calibration": calibration, "high_confidence": high}


@torch.no_grad()
def evaluate_teacher_forced(model, loader, device, config: TrainConfig) -> dict:
    """Teacher-forced validation diagnostics (not codec rates)."""
    model.eval()
    weights = torch.tensor(config.loss_weights(), device=device)
    total, weighted, correct, count = 0.0, 0.0, 0, 0
    plane_loss, plane_correct, plane_count = np.zeros(8), np.zeros(8), np.zeros(8)
    calibration = Calibration()
    for batch in loader:
        targets = batch.to(device)
        inputs = teacher_inputs(targets)
        hidden = None
        for start in range(0, targets.shape[1], config.chunk_length):
            stop = start + config.chunk_length
            chunk = targets[:, start:stop]
            logits, hidden = model(inputs[:, start:stop], hidden, start=start, width=config.width)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, chunk.float(), reduction="none")
            plane = torch.arange(start, start + chunk.shape[1], device=device) % 8
            total += loss.sum().item()
            weighted += (loss * weights[plane]).sum().item()
            matches = (logits >= 0) == chunk.bool()
            correct += matches.sum().item()
            count += chunk.numel()
            plane_np = plane.cpu().numpy()
            plane_loss += np.bincount(plane_np, weights=loss.sum(0).cpu().numpy(), minlength=8)
            plane_correct += np.bincount(plane_np, weights=matches.sum(0).cpu().numpy(), minlength=8)
            plane_count += np.bincount(plane_np, minlength=8) * chunk.shape[0]
            # Weighted CUDA bincount is nondeterministic; aggregate on CPU.
            calibration.update(torch.sigmoid(logits.abs()).cpu().numpy().reshape(-1),
                               matches.cpu().numpy().reshape(-1).astype(float))
    metrics = {"bce_nats": total / count, "cross_entropy_bits": total / count / math.log(2),
               "objective": weighted / count, "bit_accuracy": correct / count,
               "plane_bce_nats_msb_first": (plane_loss / plane_count).tolist(),
               "plane_accuracy_msb_first": (plane_correct / plane_count).tolist()}
    metrics.update(calibration.report())
    return metrics


def _make_scheduler(optimizer, config: TrainConfig, total_steps: int):
    if config.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    if config.lr_schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    return None


def train(train_images: np.ndarray, validation_images: np.ndarray, output: str | Path,
          model_config: ModelConfig = ModelConfig(), config: TrainConfig = TrainConfig(),
          provenance: dict | None = None) -> tuple[BitPredictor, list[dict]]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    model = BitPredictor(model_config).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    sampler = torch.Generator().manual_seed(config.seed + 1)  # rollout thresholds / replacement draws
    loader = DataLoader(BitImageDataset(train_images), batch_size=config.batch_size,
                        shuffle=True, generator=generator, num_workers=config.workers)
    validation_loader = DataLoader(BitImageDataset(validation_images),
                                   batch_size=config.batch_size, num_workers=config.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    length = train_images.shape[1] * train_images.shape[2] * 8
    chunks_per_batch = math.ceil(length / config.chunk_length)
    steps_per_epoch = len(loader) * math.ceil(chunks_per_batch / config.accumulate)
    total_steps = min(config.max_steps or math.inf, steps_per_epoch * config.epochs)
    scheduler = _make_scheduler(optimizer, config, total_steps)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("best.pt", "last.pt", "history.json")):
        raise ValueError("training output already contains a run; choose a new --output directory")
    config_document = asdict(config)
    run = {"model": model_config.identity(), "training": config_document,
           "loss_weights_msb_first": config.loss_weights(),
           "parameters": model.parameter_count, "device": str(device),
           "torch": str(torch.__version__), "numpy": np.__version__, "threads": torch.get_num_threads(),
           "train_images": len(train_images), "validation_images": len(validation_images),
           "train_sha256": hashlib.sha256(train_images.tobytes()).hexdigest(),
           "validation_sha256": hashlib.sha256(validation_images.tobytes()).hexdigest(),
           "planned_optimizer_steps": total_steps, "provenance": provenance or {}}
    (output / "config.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({"parameters": model.parameter_count, "device": str(device),
                      "planned_optimizer_steps": total_steps}), flush=True)
    weights = torch.tensor(config.loss_weights(), device=device)
    history, best, step, started = [], math.inf, 0, time.perf_counter()
    running = {"loss": 0.0, "bce": 0.0, "bits": 0, "replaced": 0}

    def validate(epoch):
        nonlocal best, running
        validation = evaluate_teacher_forced(model, validation_loader, device, config)
        model.train()
        seen = max(running["bits"], 1)
        record = {"epoch": epoch, "step": step, "images_seen": step * config.batch_size * config.accumulate
                  * config.chunk_length / length,
                  "train": {"objective": running["loss"] / seen, "bce_nats": running["bce"] / seen,
                            "replaced_input_fraction": running["replaced"] / seen},
                  "validation": validation, "learning_rate": optimizer.param_groups[0]["lr"],
                  "seconds": time.perf_counter() - started}
        running = {"loss": 0.0, "bce": 0.0, "bits": 0, "replaced": 0}
        history.append(record)
        metadata = {"run": run, "epoch": epoch, "step": step, "validation": validation}
        save_checkpoint(output / "last.pt", model, metadata)
        if validation["objective"] < best:
            best = validation["objective"]
            save_checkpoint(output / "best.pt", model, metadata)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(validation["objective"])
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"epoch": epoch, "step": step, "train_objective": record["train"]["objective"],
                          "validation_bce": validation["bce_nats"], "validation_objective": validation["objective"],
                          "validation_accuracy": validation["bit_accuracy"], "ece": validation["ece"],
                          "lr": record["learning_rate"], "seconds": record["seconds"]}), flush=True)

    done = False
    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in loader:
            fraction = config.replacement_fraction(epoch, step / total_steps)
            targets = batch.to(device)
            teacher = teacher_inputs(targets)
            hidden, carry = None, torch.full((targets.shape[0],), BOS, dtype=torch.long, device=device)
            threshold = None
            if config.mode == "rollout":
                low, high = (config.rollout_thresholds * 2)[:2]
                threshold = low + (high - low) * torch.rand((), generator=sampler).item()
            pending = 0
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, targets.shape[1], config.chunk_length):
                stop = start + config.chunk_length
                chunk = targets[:, start:stop]
                if config.mode == "teacher_forcing" or (config.mode == "scheduled" and fraction == 0):
                    inputs, replaced = teacher[:, start:stop], None
                    carry = chunk[:, -1]
                else:
                    was_training = model.training
                    model.eval()
                    inputs, replaced, carry = rollout_inputs(
                        model, chunk, carry, None if hidden is None else hidden.detach(), start, config.width,
                        threshold=threshold, fraction=None if config.mode == "rollout" else fraction,
                        generator=sampler)
                    model.train(was_training)
                logits, hidden = model(inputs, hidden, start=start, width=config.width)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, chunk.float(), reduction="none")
                plane = torch.arange(start, start + chunk.shape[1], device=device) % 8
                loss = (bce * weights[plane]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                (loss / config.accumulate).backward()
                pending += 1
                hidden = hidden.detach()  # carry context, truncate the gradient graph
                running["loss"] += loss.item() * chunk.numel()
                running["bce"] += bce.detach().sum().item()
                running["bits"] += chunk.numel()
                if replaced is not None:
                    running["replaced"] += int(replaced.sum())
                if pending == config.accumulate or stop >= targets.shape[1]:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    step += 1
                    if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                        scheduler.step()
                    if config.val_every and step % config.val_every == 0:
                        validate(epoch)
                    if config.max_steps and step >= config.max_steps:
                        done = True
                        break
            if done:
                break
        final = done or epoch == config.epochs
        # Per-epoch validation by default; with val_every, also validate the final state once.
        if not config.val_every or (final and (not history or history[-1]["step"] != step)):
            validate(epoch)
        if done:
            break
    return model.eval(), history

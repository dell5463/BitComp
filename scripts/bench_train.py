"""Measure steady-state training throughput of the real training loop.

Runs ``bitlaya.training.train`` (unchanged) for ``--warmup`` + ``--steps`` optimizer
steps with ``val_every = --warmup`` (32 validation images). The timed window is the
``--steps`` optimizer steps after the first validation; every validation and
checkpoint write is timed separately and subtracted. CUDA is synchronized at the
window edges by the ``.item()`` calls inside the loop and the wrappers.

    python scripts/bench_train.py --device cuda --batch-size 64 --mode teacher_forcing --json out.json

One optimizer step = batch_size images x one chunk_length-bit chunk.
"""
import argparse
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch

from bitlaya import training
from bitlaya.data import load_images
from bitlaya.model import ModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--mode", choices=training.MODES, default="teacher_forcing")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--plane-dim", type=int, default=0)
    parser.add_argument("--row-dim", type=int, default=0)
    parser.add_argument("--col-dim", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument("--json", type=Path, help="append the result as one JSON line")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    images = load_images(args.data, "train", limit=args.train_size)
    validation = load_images(args.data, "val", limit=args.batch_size)
    extra = {}
    if args.mode == "scheduled":
        # Replacement active from the first step so every timed step pays the rollout.
        extra = {"schedule": ((0.0, 0.25),), "schedule_unit": "progress"}
    config = training.TrainConfig(epochs=100000, batch_size=args.batch_size, chunk_length=args.chunk_length,
                                  device=args.device, mode=args.mode, max_steps=args.warmup + args.steps,
                                  val_every=args.warmup, **extra)
    model_config = ModelConfig(plane_dim=args.plane_dim, row_dim=args.row_dim, col_dim=args.col_dim)

    events = []  # (kind, start, end)
    evaluate, save = training.evaluate_teacher_forced, training.save_checkpoint

    def timed(kind, function):
        def wrapper(*a, **k):
            start = time.perf_counter()
            result = function(*a, **k)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            events.append((kind, start, time.perf_counter()))
            return result
        return wrapper

    training.evaluate_teacher_forced = timed("validate", evaluate)
    training.save_checkpoint = timed("save", save)
    with tempfile.TemporaryDirectory() as directory:
        started = time.perf_counter()
        training.train(images, validation, Path(directory) / "run", model_config, config)
        total = time.perf_counter() - started
    # train() validates every val_every (= warmup) steps and once more at max_steps. Window: end of the
    # warmup validate/save block -> start of the final validation, minus every validate/save inside it.
    validations = [e for e in events if e[0] == "validate"]
    first_end = max(e[2] for e in events if e[2] <= validations[1][1])
    last_start = validations[-1][1]
    inside = sum(end - start for _, start, end in events if start >= first_end and end <= last_start)
    seconds = last_start - first_end - inside
    bits = args.steps * args.batch_size * args.chunk_length
    result = {"label": args.label, "device": args.device, "threads": args.threads, "mode": args.mode,
              "batch_size": args.batch_size, "chunk_length": args.chunk_length,
              "model": model_config.identity(), "warmup_steps": args.warmup, "timed_steps": args.steps,
              "seconds": seconds, "seconds_per_step": seconds / args.steps,
              "steps_per_second": args.steps / seconds, "kbit_per_second": bits / seconds / 1000,
              "image_passes_per_second": bits / 8192 / seconds, "wall_seconds_total": total,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "cpu": platform.processor(), "finished_unix": time.time()}
    print(json.dumps(result))
    if args.json:
        with args.json.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

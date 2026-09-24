"""Command-line entry points. Run `python -m bitlaya --help`.

Legacy v1 commands (unchanged): prepare, train, evaluate, encode, decode, inspect.
v2 / research commands: compress, decompress, sweep, diagnose, profile,
analyze-errors, ablate.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .bits import grayscale
from .codec import decode, encode, inspect_stream
from .data import load_images, prepare_cifar10
from .metrics import distortion
from .model import ModelConfig, load_checkpoint
from .training import TrainConfig, train

TRAIN_FIELDS = set(TrainConfig.__dataclass_fields__)
MODEL_FIELDS = set(ModelConfig.__dataclass_fields__)


def _dataset_args(parser):
    parser.add_argument("--data", type=Path, default=Path("data/cifar10"))


def _floats(text):
    return [float(x) for x in text.replace(",", " ").split()]


def build_parser():
    parser = argparse.ArgumentParser(description="BitLaya: small-image causal bit omission research")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads (codec always uses one)")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="download CIFAR-10 and cache native grayscale images")
    _dataset_args(prepare)
    prepare.add_argument("--no-download", action="store_true")

    training = commands.add_parser("train", help="train a GRU predictor from scratch")
    _dataset_args(training)
    training.add_argument("--config", type=Path, help="JSON with optional data/model/training sections; "
                                                        "explicit flags override it")
    training.add_argument("--output", type=Path, default=Path("runs/baseline"))
    training.add_argument("--validation-size", type=int, default=5000, help="validation pool carved from train")
    training.add_argument("--train-limit", "--train-size", dest="train_limit", type=int)
    training.add_argument("--val-limit", "--val-size", dest="val_limit", type=int)
    for name, kind in (("epochs", int), ("batch-size", int), ("chunk-length", int), ("learning-rate", float),
                       ("grad-clip", float), ("seed", int), ("workers", int), ("max-steps", int),
                       ("val-every", int), ("accumulate", int)):
        training.add_argument(f"--{name}", type=kind)
    training.add_argument("--device")
    training.add_argument("--mode", choices=("teacher_forcing", "scheduled", "rollout"))
    training.add_argument("--schedule", type=json.loads, help='e.g. "[[1,0],[3,0.1],[5,0.25],[7,0.5]]"')
    training.add_argument("--schedule-unit", choices=("epoch", "progress"),
                          help="scheduled-mode schedule keys: epochs, or fraction of the step budget")
    training.add_argument("--rollout-thresholds", type=_floats, help='"0.6 0.95" (uniform range) or "0.8"')
    training.add_argument("--objective", choices=("bce", "weighted_bce"))
    training.add_argument("--weights", help="uniform | sqrt | linear | 8 comma-separated numbers (MSB first)")
    training.add_argument("--lr-schedule", choices=("constant", "plateau", "cosine"))
    for name in ("embedding-dim", "hidden-size", "num-layers", "plane-dim", "row-dim", "col-dim"):
        training.add_argument(f"--{name}", type=int)
    training.add_argument("--spatial-dim", type=int, help="shorthand for --row-dim N --col-dim N")

    evaluation = commands.add_parser("evaluate", help="legacy v1 small-sample evaluator (see `sweep`)")
    _dataset_args(evaluation)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    evaluation.add_argument("--split", choices=("val", "test"), default="val")
    evaluation.add_argument("--limit", type=int, default=16, help="sequential reference decoding is slow")
    evaluation.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    evaluation.add_argument("--examples", type=int, default=4)
    evaluation.add_argument("--seed", type=int, help="split seed; defaults to checkpoint provenance")
    evaluation.add_argument("--validation-size", type=int, help="defaults to checkpoint provenance")

    sweep = commands.add_parser("sweep", help="rate-distortion threshold sweep on a fixed held-out subset")
    _dataset_args(sweep)
    sweep.add_argument("checkpoint", type=Path)
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument("--split", choices=("val", "test"), default="val",
                       help="val for decisions (default), test for held-out reporting")
    sweep.add_argument("--size", "--test-size", "--val-size", dest="size", type=int, default=1000)
    sweep.add_argument("--seed", type=int, default=42)
    sweep.add_argument("--thresholds", type=float, nargs="+")
    sweep.add_argument("--engine", choices=("qgru-v2", "reference-v1"), default="qgru-v2")
    sweep.add_argument("--batch-rows", type=int, default=128)
    sweep.add_argument("--bootstrap", type=int, default=1000)
    sweep.add_argument("--examples", type=int, default=4)
    sweep.add_argument("--resume", action="store_true")

    diagnose = commands.add_parser("diagnose", help="teacher-forced BCE/calibration/reliability on held-out data")
    _dataset_args(diagnose)
    diagnose.add_argument("checkpoint", type=Path)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--split", choices=("val", "test"), default="val")
    diagnose.add_argument("--size", type=int, default=1000)
    diagnose.add_argument("--seed", type=int, default=42)

    for name, text in (("encode", "v1 JSON stream (legacy float reference codec)"),
                       ("compress", "v2 binary .blaya stream (exact fixed-point engine)")):
        command = commands.add_parser(name, help=f"encode an image, converted to grayscale, as a {text}")
        command.add_argument("input", type=Path)
        command.add_argument("output", type=Path)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--threshold", type=float, default=0.95)
        if name == "compress":
            command.add_argument("--zlib-payload", action="store_true", help="zlib the explicit payload")
    for name in ("decode", "decompress"):
        command = commands.add_parser(name, help=f"reconstruct a {'v1 .blay' if name == 'decode' else 'v2 .blaya'}"
                                                 " file using its exact checkpoint")
        command.add_argument("input", type=Path)
        command.add_argument("output", type=Path, help="output .png (lossless grayscale)")
        command.add_argument("--checkpoint", type=Path, required=True)
    inspection = commands.add_parser("inspect", help="validate framing/checksum and show v1 or v2 stream header")
    inspection.add_argument("input", type=Path)

    profile = commands.add_parser("profile", help="controlled latency benchmark + cProfile/torch.profiler")
    _dataset_args(profile)
    profile.add_argument("checkpoint", type=Path)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--engine", choices=("qgru-v2", "reference-v1"), default="qgru-v2")
    profile.add_argument("--size", type=int, default=100)
    profile.add_argument("--threshold", type=float, default=0.9)
    profile.add_argument("--warmup", type=int, default=5)
    profile.add_argument("--repeats", type=int, default=3)
    profile.add_argument("--seed", type=int, default=42)

    analysis = commands.add_parser("analyze-errors", help="error-propagation analysis (cascades after wrong omissions)")
    _dataset_args(analysis)
    analysis.add_argument("checkpoint", type=Path)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--split", choices=("val", "test"), default="val")
    analysis.add_argument("--size", type=int, default=200)
    analysis.add_argument("--seed", type=int, default=42)
    analysis.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    analysis.add_argument("--max-distance", type=int, default=512)

    lossless = commands.add_parser("benchmark-lossless", help="range-coded lossless mode vs raw/zlib/PNG")
    _dataset_args(lossless)
    lossless.add_argument("checkpoint", type=Path)
    lossless.add_argument("--output", type=Path, required=True)
    lossless.add_argument("--split", choices=("val", "test"), default="test")
    lossless.add_argument("--size", type=int, default=1000)
    lossless.add_argument("--seed", type=int, default=42)

    overhead = commands.add_parser("overhead", help="measure v1 vs v2 vs container framing overhead")
    _dataset_args(overhead)
    overhead.add_argument("checkpoint", type=Path)
    overhead.add_argument("--output", type=Path, required=True)
    overhead.add_argument("--split", choices=("val", "test"), default="test")
    overhead.add_argument("--size", type=int, default=1000)
    overhead.add_argument("--seed", type=int, default=42)
    overhead.add_argument("--thresholds", type=float, nargs="+", default=[0.7, 0.9, 1.0])

    ablate = commands.add_parser("ablate", help="train and sweep several variants from one JSON config")
    ablate.add_argument("--config", type=Path, required=True)
    ablate.add_argument("--output", type=Path)
    ablate.add_argument("--parallel", type=int, default=1, help="variants trained concurrently (1 thread each)")
    ablate.add_argument("--only", nargs="+", help="run only these variant names")
    return parser


def _write_new(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def training_configs(args) -> tuple[dict, ModelConfig, TrainConfig]:
    """Merge an optional JSON config file with explicit CLI flags (flags win)."""
    document = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    data = dict(document.get("data", {}))
    for alias, key in (("train_size", "train_limit"), ("val_size", "val_limit")):
        if alias in data:
            data[key] = data.pop(alias)
    model = dict(document.get("model", {}))
    training = dict(document.get("training", {}))
    for key in ("train_limit", "val_limit", "validation_size"):
        if getattr(args, key) is not None and (key != "validation_size" or "validation_size" not in data
                                               or args.validation_size != 5000):
            data[key] = getattr(args, key)
    data.setdefault("validation_size", 5000)
    for key in MODEL_FIELDS:
        if getattr(args, key, None) is not None:
            model[key] = getattr(args, key)
    if getattr(args, "spatial_dim", None) is not None:
        model["row_dim"] = model["col_dim"] = args.spatial_dim
    for key in TRAIN_FIELDS:
        if getattr(args, key, None) is not None:
            training[key] = getattr(args, key)
    if isinstance(training.get("weights"), str) and "," in training["weights"]:
        training["weights"] = [float(x) for x in training["weights"].split(",")]
    for key in ("schedule", "rollout_thresholds"):
        if key in training:
            training[key] = tuple(tuple(x) if isinstance(x, list) else x for x in training[key])
    if "seed" not in training:
        training["seed"] = 42
    return data, ModelConfig(**model), TrainConfig(**training)


def _read_image(path):
    with Image.open(path) as source:
        return grayscale(source)


def run(args):
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "prepare":
        print(json.dumps(prepare_cifar10(args.data, download=not args.no_download), indent=2))
    elif args.command == "train":
        data, model, config = training_configs(args)
        root = Path(data.get("root", args.data))
        images = load_images(root, "train", validation_size=data["validation_size"],
                             seed=config.seed, limit=data.get("train_limit"))
        validation = load_images(root, "val", validation_size=data["validation_size"],
                                 seed=config.seed, limit=data.get("val_limit"))
        provenance = {"dataset": "CIFAR-10 grayscale", "data": str(root),
                      "validation_size": data["validation_size"], "split_seed": config.seed,
                      "train_limit": data.get("train_limit"), "val_limit": data.get("val_limit"),
                      "config_file": str(args.config) if args.config else None}
        manifest = root / "preprocessing.json"
        if manifest.exists():
            provenance["preprocessing"] = json.loads(manifest.read_text(encoding="utf-8"))
        train(images, validation, args.output, model, config, provenance)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        model, metadata = load_checkpoint(args.checkpoint)
        source = metadata.get("run", {}).get("provenance", {})
        seed = args.seed if args.seed is not None else source.get("split_seed", 42)
        validation_size = args.validation_size if args.validation_size is not None else source.get("validation_size", 5000)
        if args.split == "val" and source and (seed != source.get("split_seed", 42)
                or validation_size != source.get("validation_size", 5000)):
            raise ValueError("validation split must match training to avoid data leakage")
        images = load_images(args.data, args.split, validation_size=validation_size, seed=seed, limit=args.limit)
        evaluate(images, model, args.thresholds, args.output,
                 checkpoint_bytes=args.checkpoint.stat().st_size, examples=args.examples,
                 provenance={"split": args.split, "split_seed": seed, "validation_size": validation_size,
                             "checkpoint": str(args.checkpoint), "checkpoint_metadata": metadata})
    elif args.command == "sweep":
        from .research import DEFAULT_THRESHOLDS, EvaluationConfig, run_evaluation
        config = EvaluationConfig(checkpoint=str(args.checkpoint), output=str(args.output), data=str(args.data),
                                  split=args.split, size=args.size, seed=args.seed,
                                  thresholds=args.thresholds or DEFAULT_THRESHOLDS.copy(),
                                  bootstrap=args.bootstrap, examples=args.examples, engine=args.engine,
                                  batch_rows=args.batch_rows)
        summary = run_evaluation(config, resume=args.resume)
        for row in summary["thresholds"]:
            print(json.dumps({"threshold": row["threshold"], "file_bpp": round(row["file_bpp"], 4),
                              "psnr_db": row["pooled_psnr_db"], "omitted": row["statistics"]["omitted_fraction"]["mean"]}))
    elif args.command == "diagnose":
        from .analysis import diagnose
        diagnose(args.checkpoint, args.data, args.split, args.size, args.seed, args.output)
    elif args.command == "encode":
        model, _ = load_checkpoint(args.checkpoint)
        image = _read_image(args.input)
        result = encode(image, model, args.threshold)
        _write_new(args.output, result.data)
        print(json.dumps({"file_bytes": len(result.data), "explicit_bits": result.metadata["explicit_bits"],
                          **distortion(image, result.reconstruction)}, indent=2, allow_nan=False))
    elif args.command == "compress":
        from .blaya import BitLayaCodec
        model, _ = load_checkpoint(args.checkpoint)
        image = _read_image(args.input)
        result = BitLayaCodec(model).compress(image, args.threshold, zlib_payload=args.zlib_payload)
        _write_new(args.output, result.data)
        print(json.dumps({"file_bytes": len(result.data), "header_bytes": result.header_bytes,
                          "explicit_bits": result.payload_bits, **distortion(image, result.reconstruction)},
                         indent=2, allow_nan=False))
    elif args.command in ("decode", "decompress"):
        if args.output.suffix.lower() != ".png":
            raise ValueError("decode output must use .png to preserve exact reconstructed pixels")
        model, _ = load_checkpoint(args.checkpoint)
        data = args.input.read_bytes()
        if args.command == "decode":
            image = decode(data, model)
        else:
            from .blaya import BitLayaCodec
            image = BitLayaCodec(model).decompress(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            Image.fromarray(image).save(stream, format="PNG")
        print(f"Decoded {image.shape[1]}x{image.shape[0]} grayscale image to {args.output}")
    elif args.command == "inspect":
        data = args.input.read_bytes()
        if data[:4] == b"BLYA":
            from .blaya import parse_stream
            header, _ = parse_stream(data)
            print(json.dumps({"format": "blaya-v2", "width": header.width, "height": header.height,
                              "flags": header.flags, "decision_threshold": header.decision,
                              "approx_confidence_threshold": header.threshold,
                              "payload_bits": header.payload_bits, "model_id": header.model_id.hex(),
                              "reconstruction_crc32": f"{header.reconstruction_crc:08x}",
                              "file_bytes": len(data)}, indent=2))
        else:
            metadata, _ = inspect_stream(data)
            print(json.dumps(metadata, indent=2))
    elif args.command == "profile":
        from .profiling import profile_command
        profile_command(args)
    elif args.command == "analyze-errors":
        from .analysis import analyze_errors
        analyze_errors(args.checkpoint, args.data, args.split, args.size, args.seed, args.thresholds,
                       args.output, args.max_distance)
    elif args.command == "benchmark-lossless":
        from .lossless import benchmark
        benchmark(args.checkpoint, args.data, args.split, args.size, args.seed, args.output)
    elif args.command == "overhead":
        from .overhead import measure_overhead
        measure_overhead(args.checkpoint, args.data, args.split, args.size, args.seed, args.thresholds, args.output)
    elif args.command == "ablate":
        from .ablation import run_ablation
        run_ablation(args.config, args.output, parallel=args.parallel, only=args.only)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"bitlaya: error: {exc}\n")


if __name__ == "__main__":
    main()

"""Command-line entry points. Run `python -m bitlaya --help`."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .bits import grayscale
from .codec import decode, encode, inspect_stream
from .data import load_images, prepare_cifar10
from .evaluation import evaluate
from .metrics import distortion
from .model import ModelConfig, load_checkpoint
from .training import TrainConfig, train


def _dataset_args(parser):
    parser.add_argument("--data", type=Path, default=Path("data/cifar10"))


def build_parser():
    parser = argparse.ArgumentParser(description="BitLaya: small-image causal bit omission research")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads (codec always uses one)")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="download CIFAR-10 and cache native grayscale images")
    _dataset_args(prepare)
    prepare.add_argument("--no-download", action="store_true")
    training = commands.add_parser("train", help="train a GRU from scratch using teacher-forced BCE")
    _dataset_args(training)
    training.add_argument("--output", type=Path, default=Path("runs/baseline"))
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--chunk-length", type=int, default=256)
    training.add_argument("--embedding-dim", type=int, default=32)
    training.add_argument("--hidden-size", type=int, default=128)
    training.add_argument("--num-layers", type=int, default=2)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default="auto")
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--validation-size", type=int, default=5000)
    training.add_argument("--train-limit", type=int)
    training.add_argument("--val-limit", type=int)
    evaluation = commands.add_parser("evaluate", help="independently decode a threshold sweep and plot rate/distortion")
    _dataset_args(evaluation)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    evaluation.add_argument("--split", choices=("val", "test"), default="val")
    evaluation.add_argument("--limit", type=int, default=16, help="sequential reference decoding is slow")
    evaluation.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    evaluation.add_argument("--examples", type=int, default=4)
    evaluation.add_argument("--seed", type=int, help="split seed; defaults to checkpoint provenance")
    evaluation.add_argument("--validation-size", type=int, help="defaults to checkpoint provenance")
    encoding = commands.add_parser("encode", help="encode an image of at most 1024 pixels, converted to grayscale")
    encoding.add_argument("input", type=Path)
    encoding.add_argument("output", type=Path)
    encoding.add_argument("--checkpoint", type=Path, required=True)
    encoding.add_argument("--threshold", type=float, default=0.95)
    decoding = commands.add_parser("decode", help="reconstruct a .blay file using its exact checkpoint")
    decoding.add_argument("input", type=Path)
    decoding.add_argument("output", type=Path, help="output .png (lossless grayscale)")
    decoding.add_argument("--checkpoint", type=Path, required=True)
    inspection = commands.add_parser("inspect", help="validate framing/checksum and show stream metadata")
    inspection.add_argument("input", type=Path)
    return parser


def _write_new(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def run(args):
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "prepare":
        print(json.dumps(prepare_cifar10(args.data, download=not args.no_download), indent=2))
    elif args.command == "train":
        images = load_images(args.data, "train", validation_size=args.validation_size,
                             seed=args.seed, limit=args.train_limit)
        validation = load_images(args.data, "val", validation_size=args.validation_size,
                                 seed=args.seed, limit=args.val_limit)
        config = TrainConfig(**{key: getattr(args, key) for key in TrainConfig.__dataclass_fields__})
        model = ModelConfig(args.embedding_dim, args.hidden_size, args.num_layers)
        provenance = {"dataset": "CIFAR-10 grayscale", "data": str(args.data),
                      "validation_size": args.validation_size, "split_seed": args.seed}
        manifest = args.data / "preprocessing.json"
        if manifest.exists():
            provenance["preprocessing"] = json.loads(manifest.read_text(encoding="utf-8"))
        train(images, validation, args.output, model, config, provenance)
    elif args.command == "evaluate":
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
    elif args.command == "encode":
        model, _ = load_checkpoint(args.checkpoint)
        with Image.open(args.input) as source:
            image = grayscale(source)
        result = encode(image, model, args.threshold)
        _write_new(args.output, result.data)
        print(json.dumps({"file_bytes": len(result.data), "explicit_bits": result.metadata["explicit_bits"],
                          **distortion(image, result.reconstruction)}, indent=2, allow_nan=False))
    elif args.command == "decode":
        if args.output.suffix.lower() != ".png":
            raise ValueError("decode output must use .png to preserve exact reconstructed pixels")
        model, _ = load_checkpoint(args.checkpoint)
        image = decode(args.input.read_bytes(), model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            Image.fromarray(image).save(stream, format="PNG")
        print(f"Decoded {image.shape[1]}x{image.shape[0]} grayscale image to {args.output}")
    elif args.command == "inspect":
        metadata, _ = inspect_stream(args.input.read_bytes())
        print(json.dumps(metadata, indent=2))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"bitlaya: error: {exc}\n")


if __name__ == "__main__":
    main()

"""Controlled latency benchmarks, cProfile, and sampled PyTorch operator profiles.

Protocol: fixed held-out images, warmup calls excluded, several timed repeats,
single-image calls (latency) plus, for the exact engine, batched throughput.
Profiler runs are separate from timed runs (instrumentation inflates time).
"""
import argparse
import copy
import cProfile
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import platform
import pstats
import subprocess
import time

import numpy as np
import torch

from . import codec as reference
from .data import benchmark_images
from .model import BOS, load_checkpoint, model_fingerprint


def environment():
    cpu = platform.processor()
    if platform.system() == "Windows":
        try:
            cpu = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor).Name"], text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    import os
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": cpu, "logical_cpus": os.cpu_count(), "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "threads": torch.get_num_threads(),
            "blas_thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                              "MKL_NUM_THREADS")},
            "runtime": reference.runtime_signature()}


class ReferenceCodec:
    """Adapter: the preserved v1 codec (deep-copies and verifies the model per call)."""

    def __init__(self, model):
        self.model = model

    def compress(self, image, threshold):
        return reference.encode(image, self.model, threshold)

    def decompress(self, data):
        return reference.decode(data, self.model)


def timing_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean_seconds": float(values.mean()),
            "median_seconds": float(np.median(values)), "p95_seconds": float(np.quantile(values, .95)),
            "std_seconds": float(values.std()), "images_per_second": float(1 / np.median(values))}


def _cprofile(output, name, function):
    profiler = cProfile.Profile()
    profiler.enable()
    result = function()
    profiler.disable()
    profiler.dump_stats(str(output / f"{name}.prof"))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
    stats.print_stats(30)
    (output / f"{name}_cprofile.txt").write_text(stream.getvalue(), encoding="utf-8")
    functions = [{"function": f"{Path(file).name}:{line}({fn})", "calls": calls, "self_seconds": self_time,
                  "cumulative_seconds": cumulative}
                 for (file, line, fn), (_, calls, self_time, cumulative, _) in stats.stats.items()]
    functions.sort(key=lambda x: x["self_seconds"], reverse=True)
    total = sum(f["self_seconds"] for f in functions)
    top = [{**f, "self_fraction": f["self_seconds"] / total} for f in functions[:25]]
    (output / f"{name}_top_functions.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    return result, top


def _latency(codec, images, threshold, warmup, repeats, output, name):
    for index in range(warmup):
        result = codec.compress(images[index % len(images)], threshold)
        codec.decompress(result.data)
    rows = []
    for repeat in range(repeats):
        for index, image in enumerate(images):
            started = time.perf_counter()
            encoded = codec.compress(image, threshold)
            encoding = time.perf_counter() - started
            started = time.perf_counter()
            decoded = codec.decompress(encoded.data)
            decoding = time.perf_counter() - started
            if not np.array_equal(decoded, encoded.reconstruction):
                raise RuntimeError("decoder diverged during benchmark")
            rows.append({"run": repeat, "image": index, "encode_seconds": encoding, "decode_seconds": decoding})
        print(f"{name}: timed run {repeat + 1}/{repeats} done", flush=True)
    with (output / f"{name}_latency.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"encode": timing_stats([r["encode_seconds"] for r in rows]),
            "decode": timing_stats([r["decode_seconds"] for r in rows]), "independent_decodes_verified": len(rows)}


def profile_checkpoint(checkpoint, images, manifest, output, *, engine="qgru-v2", threshold=.9, warmup=5,
                       repeats=3, torch_steps=128, latency=True):
    if warmup < 0 or repeats < 1 or not 0 <= torch_steps <= 8192:
        raise ValueError("invalid profiling configuration")
    if engine not in ("qgru-v2", "reference-v1"):
        raise ValueError("engine must be qgru-v2 or reference-v1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    model, metadata = load_checkpoint(checkpoint)
    setup = {"checkpoint_load_including_validation_seconds": time.perf_counter() - started}
    config = {"checkpoint": str(checkpoint), "model": model.config.identity(), "dataset": manifest,
              "engine": engine, "threshold": threshold, "warmup": warmup, "repeats": repeats,
              "environment": environment(),
              "controlled_protocol": len(images) >= 100 and warmup >= 5 and repeats >= 3 and latency}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"config": config, "setup": setup}
    if engine == "reference-v1":
        # Per-call setup inside reference.encode/decode, measured in isolation.
        started = time.perf_counter()
        cloned = copy.deepcopy(model)
        setup["per_call_deepcopy_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        cloned = cloned.cpu().float().eval()
        setup["per_call_cpu_float_eval_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        assert all(torch.isfinite(p).all() for p in cloned.parameters())
        setup["per_call_finite_verification_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        model_fingerprint(cloned)
        setup["per_call_fingerprint_seconds"] = time.perf_counter() - started
        codec = ReferenceCodec(model)
    else:
        from .blaya import BitLayaCodec
        started = time.perf_counter()
        codec = BitLayaCodec(model)
        setup["one_time_codec_init_seconds_quantize_tables_exactness_fingerprint"] = time.perf_counter() - started
    started = time.perf_counter()
    first = codec.compress(images[0], threshold)
    codec.decompress(first.data)
    setup["cold_first_encode_decode_seconds"] = time.perf_counter() - started
    _, top = _cprofile(output, "encode_decode", lambda: codec.decompress(codec.compress(images[0], threshold).data))
    report["cprofile_top_self_time"] = top[:12]
    if engine == "qgru-v2":
        started = time.perf_counter()
        for image in images[:20]:
            lossless = codec.compress(image, 1.0)
            if not np.array_equal(codec.decompress(lossless.data), image):
                raise RuntimeError("threshold-1 fast path is not lossless")
        report["threshold_1_fast_path_seconds_per_image_encode_plus_decode"] = (time.perf_counter() - started) / 20
        rows = min(len(images), 128)
        started = time.perf_counter()
        batch = codec.compress_many(list(images[:rows]), [threshold] * rows)
        encode_time = time.perf_counter() - started
        started = time.perf_counter()
        decoded = codec.decompress_many([b.data for b in batch])
        decode_time = time.perf_counter() - started
        if any(not np.array_equal(d, b.reconstruction) for d, b in zip(decoded, batch)):
            raise RuntimeError("batched decoder diverged")
        report["batched_throughput"] = {"rows": rows, "encode_images_per_second": rows / encode_time,
                                        "decode_images_per_second": rows / decode_time}
    if torch_steps and engine == "reference-v1":
        from torch.profiler import ProfilerActivity, profile, record_function
        cloned = copy.deepcopy(model).eval()
        with reference.reference_runtime():
            hidden, previous = None, torch.tensor([[BOS]], dtype=torch.long)
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                for _ in range(torch_steps):
                    with record_function("bitlaya_step"):
                        logits, hidden = cloned(previous, hidden)
                    with record_function("decision_and_update"):
                        _, prediction = reference._decision(logits.item(), threshold)
                        previous.fill_(prediction)
        (output / "torch_operators.txt").write_text(
            prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30), encoding="utf-8")
    if latency:
        report["latency"] = _latency(codec, images, threshold, warmup, repeats, output, engine)
    else:
        report["latency"] = "NOT YET MEASURED"
    report["notes"] = ("Timed calls exclude warmups/profilers and disk I/O; they include framing, checksums "
                       "and (reference-v1) the per-call model deep copy and verification.")
    (output / "timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "setup_timing.json").write_text(json.dumps(setup, indent=2), encoding="utf-8")
    return report


def profile_command(args):
    images, manifest = benchmark_images(args.data, "test", args.size, args.seed)
    report = profile_checkpoint(args.checkpoint, images, manifest, args.output, engine=args.engine,
                                threshold=args.threshold, warmup=args.warmup, repeats=args.repeats)
    latency = report["latency"]
    print(json.dumps({"setup": report["setup"], "encode": latency["encode"], "decode": latency["decode"],
                      "batched": report.get("batched_throughput")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", default="data/cifar10")
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=.9)
    parser.add_argument("--engine", choices=["reference-v1", "qgru-v2"], default="reference-v1")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    profile_command(args)


if __name__ == "__main__":
    main()

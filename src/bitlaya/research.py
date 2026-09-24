"""Reproducible held-out rate/distortion sweeps on a fixed image subset.

Engines:

* ``reference-v1`` -- the preserved float reference codec with JSON framing
  (codec.py), one image at a time. This is the pre-improvement baseline.
* ``qgru-v2`` -- the integer-exact engine with compact binary framing
  (blaya.py). Rows (image, threshold) are encoded in batches; every stream is
  then decoded by the independent batched decoder from its bytes alone, and the
  decoded image must equal the encoder's reconstruction.

Images come from ``data.benchmark_images``: the first N of a fixed seeded
permutation of the chosen held-out split, so every checkpoint compared with the
same (split, size, seed) sees exactly the same images (the manifest records
dataset indices and a content hash). Use ``val`` for decisions, ``test`` for
held-out reporting.
"""
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zlib

import numpy as np
from PIL import Image, features
import torch

from .bits import image_to_bits, pack_bits
from .blaya import OVERHEAD, BitLayaCodec, parse_stream
from .codec import decode, encode, inspect_stream, validate_threshold
from .data import benchmark_images
from .metrics import conventional_baselines, distortion, structural_similarity
from .model import load_checkpoint, model_fingerprint
from .profiling import environment
from .statistics import bootstrap_means, describe, psnr

DEFAULT_THRESHOLDS = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .925, .95, .975, .99, 1.]
ENGINES = ("reference-v1", "qgru-v2")


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint: str
    output: str
    data: str = "data/cifar10"
    split: str = "test"
    size: int = 1000
    seed: int = 42
    validation_pool_size: int = 5000
    thresholds: list[float] = field(default_factory=lambda: DEFAULT_THRESHOLDS.copy())
    bootstrap: int = 1000
    examples: int = 4
    warmup: int = 5
    # Wide enough to bracket BitLaya's PSNR range for matched-quality comparison.
    lossy_qualities: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 35, 50, 75, 90, 95])
    engine: str = "reference-v1"
    batch_rows: int = 128

    def __post_init__(self):
        if self.split not in {"val", "test"} or self.size <= 0:
            raise ValueError("codec evaluation must use a positive held-out val/test subset")
        if not 0 <= self.seed < 2**32 or self.validation_pool_size <= 0:
            raise ValueError("invalid seed or validation pool size")
        if (not self.thresholds or len(self.thresholds) != len(set(self.thresholds))
                or any(validate_threshold(t) != t for t in self.thresholds)):
            raise ValueError("provide nonempty unique thresholds in [0.5, 1.0]")
        if min(self.bootstrap, self.examples, self.warmup) < 0 or self.batch_rows <= 0:
            raise ValueError("bootstrap, examples and warmup must be nonnegative; batch_rows positive")
        if any(type(q) is not int or not 1 <= q <= 100 for q in self.lossy_qualities):
            raise ValueError("lossy baseline qualities must be integers in [1,100]")
        if len(self.lossy_qualities) != len(set(self.lossy_qualities)):
            raise ValueError("lossy baseline qualities must be unique")
        if self.engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")


def sampling_scale(size: int) -> str:
    return "smoke" if size < 100 else "development" if size < 1000 else "research"


def write_json(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_identity():
    files = sorted(Path(__file__).parent.glob("*.py"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    git = shutil.which("git")
    commit, dirty = None, None
    if git:
        try:
            commit = subprocess.check_output([git, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                              text=True, timeout=5).strip()
            dirty = bool(subprocess.check_output([git, "status", "--porcelain"], text=True, timeout=5).strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return {"git_commit": commit, "git_dirty": dirty, "source_sha256": hashes}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


_write_csv = write_csv


def _load_journal(path):
    """Only recover a partially written final line; never ignore malformed rows."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1]
        path.write_bytes(data)
    return [json.loads(line) for line in data.splitlines()]


def error_statistics(original: np.ndarray, reconstruction: np.ndarray) -> dict:
    """Per-image error-propagation descriptors from source vs reconstructed bits.

    Every bit error is an incorrect omitted prediction (transmitted bits are exact).
    """
    errors = image_to_bits(original) != image_to_bits(reconstruction)
    positions = np.flatnonzero(errors)
    result = {"first_incorrect_omission": int(positions[0]) if len(positions) else None,
              "mean_gap_between_errors": float(np.diff(positions).mean()) if len(positions) > 1 else None,
              "max_error_run": 0, "error_runs": 0, "mse_before_first_error_pixel": None,
              "mse_after_first_error_pixel": None}
    if len(positions):
        breaks = np.flatnonzero(np.diff(positions) != 1)
        runs = np.diff(np.concatenate(([0], breaks + 1, [len(positions)])))
        result.update(max_error_run=int(runs.max()), error_runs=int(len(runs)))
        first_pixel = int(positions[0]) // 8
        difference = (original.astype(float) - reconstruction.astype(float)).ravel() ** 2
        result["mse_before_first_error_pixel"] = float(difference[:first_pixel].mean()) if first_pixel else None
        result["mse_after_first_error_pixel"] = float(difference[first_pixel:].mean())
    return result


def aggregate(rows: list[dict], bootstrap=1000, seed=42):
    keys = ["file_bytes", "file_bpp", "payload_bits", "payload_bytes", "payload_bpp",
            "header_bytes", "original_bytes", "omitted_fraction", "predicted_bits",
            "incorrect_omitted_predictions", "omitted_error_rate", "mse", "mae", "psnr_db",
            "ssim", "bit_error_rate", "encode_seconds", "decode_seconds", "zlib_payload_bytes",
            "range_payload_bytes", "range_file_bytes", "range_file_bpp", "range_decode_seconds",
            "first_incorrect_omission", "max_error_run", "mean_gap_between_errors"]
    stats = {key: describe([row[key] for row in rows]) for key in keys if key in rows[0]}
    cis = bootstrap_means({key: [row[key] for row in rows] for key in ("file_bpp", "file_bytes", "mse", "ssim",
                                                                        "range_file_bpp") if key in rows[0]},
                          bootstrap, seed)
    count = len(rows)
    pixels = sum(r["original_bytes"] for r in rows) if "original_bytes" in rows[0] else count * 1024
    # The harness uses fixed-shape images: image bootstrap and pooled MSE align.
    mse = float(np.mean([r["mse"] for r in rows]))
    result = {"images": count, "statistics": stats, "bootstrap_95_ci": cis,
              "pooled_mse": mse, "pooled_psnr_db": psnr(mse),
              "total_file_bytes": sum(r["file_bytes"] for r in rows),
              "file_bpp": 8 * sum(r["file_bytes"] for r in rows) / pixels,
              "lossless_images": sum(r["mse"] == 0 for r in rows)}
    if "predicted_bits" in rows[0]:
        predicted = sum(r["predicted_bits"] for r in rows)
        errors = sum(r["incorrect_omitted_predictions"] for r in rows)
        payload = sum(r["payload_bytes"] for r in rows)
        compressed = sum(r["zlib_payload_bytes"] for r in rows)
        result.update(total_original_bytes=pixels,
                      total_payload_bits=sum(r["payload_bits"] for r in rows),
                      total_payload_bytes=payload,
                      total_header_bytes=sum(r["header_bytes"] for r in rows),
                      total_predicted_bits=predicted, total_incorrect_omitted_predictions=errors,
                      pooled_omitted_error_rate=errors / predicted if predicted else None,
                      payload_bpp=8 * payload / pixels,
                      zlib_payload={"raw_bytes": payload, "zlib_bytes": compressed,
                                    "ratio": compressed / payload if payload else None,
                                    "images_where_zlib_smaller": sum(r["zlib_payload_bytes"] < r["payload_bytes"]
                                                                     for r in rows)})
    if "range_file_bytes" in rows[0]:
        coded = sum(r["range_payload_bytes"] for r in rows)
        result["range_payload"] = {"raw_bytes": result["total_payload_bytes"], "range_bytes": coded,
                                   "ratio": coded / result["total_payload_bytes"] if result["total_payload_bytes"] else None,
                                   "file_bpp": 8 * sum(r["range_file_bytes"] for r in rows) / pixels,
                                   "images_where_range_smaller": sum(r["range_payload_bytes"] < r["payload_bytes"]
                                                                     for r in rows)}
    return result


def matched_quality(points, baseline_curves):
    """Baseline bytes interpolated at each BitLaya point's PSNR (log-bytes vs PSNR).

    Returns None where the BitLaya PSNR lies outside the baseline's measured range
    (never extrapolated).
    """
    comparisons = []
    for point in points:
        target = point["pooled_psnr_db"]
        entry = {"threshold": point["threshold"], "bitlaya_psnr_db": target,
                 "bitlaya_file_bytes": point["statistics"]["file_bytes"]["mean"]}
        for codec, curve in baseline_curves.items():
            finite = sorted((c["pooled_psnr_db"], c["statistics"]["file_bytes"]["mean"]) for c in curve
                            if c["pooled_psnr_db"] is not None)
            value = None
            if target is not None and len(finite) >= 2 and finite[0][0] <= target <= finite[-1][0]:
                value = float(np.exp(np.interp(target, [p for p, _ in finite], [np.log(b) for _, b in finite])))
            entry[f"{codec}_bytes_at_matched_psnr"] = value
        comparisons.append(entry)
    return comparisons


def _row(threshold, index, dataset_index, original, reconstructed, payload_bits, file_bytes, header_bytes,
         payload, encode_seconds, decode_seconds, **extra):
    errors = image_to_bits(original) != image_to_bits(reconstructed)
    wrong = int(errors.sum())
    predicted = original.size * 8 - payload_bits
    if wrong > predicted:
        raise RuntimeError("more errors than omitted bits")
    quality = distortion(original, reconstructed)
    compressed = len(zlib.compress(payload, 9))
    return {"threshold": threshold, "image_index": index, "dataset_index": dataset_index,
            "original_bytes": original.nbytes, "payload_bits": payload_bits,
            "payload_bytes": len(payload), "header_bytes": header_bytes,
            "file_bytes": file_bytes, "file_bpp": file_bytes * 8 / original.size,
            "payload_bpp": payload_bits / original.size,
            "predicted_bits": predicted, "omitted_fraction": predicted / (original.size * 8),
            "incorrect_omitted_predictions": wrong,
            "omitted_error_rate": wrong / predicted if predicted else None,
            "mse": quality["mse"], "mae": quality["mae"], "psnr_db": quality["psnr_db"],
            "ssim": structural_similarity(original, reconstructed), "bit_error_rate": quality["bit_error_rate"],
            "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
            "raw_payload_bytes": len(payload), "zlib_payload_bytes": compressed,
            "payload_zlib_to_raw_ratio": compressed / len(payload) if payload else None,
            **error_statistics(original, reconstructed), **extra, "encoder_decoder_equal": True}


def run_evaluation(config: EvaluationConfig, *, resume=False, progress=True):
    torch.set_num_threads(1)
    model, checkpoint_metadata = load_checkpoint(config.checkpoint)
    provenance = checkpoint_metadata.get("run", {}).get("provenance", {})
    if config.split == "val" and provenance:
        if (config.seed != provenance.get("split_seed", 42)
                or config.validation_pool_size != provenance.get("validation_size", 5000)):
            raise ValueError("validation selection must match the training split to prevent leakage")
    images, manifest = benchmark_images(config.data, config.split, config.size,
                                        config.seed, config.validation_pool_size)
    checkpoint_path = Path(config.checkpoint)
    codec = BitLayaCodec(model) if config.engine == "qgru-v2" else None
    resolved = {"format": "bitlaya-research-evaluation-v2", "evaluation": asdict(config),
                "dataset": manifest, "model": model.config.identity(), "parameters": model.parameter_count,
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "model_sha256": model_fingerprint(model), "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_metadata": checkpoint_metadata, "environment": environment(),
                "source": source_identity(), "codec": config.engine,
                "codec_model_id": codec.model_id.hex() if codec else None,
                "codec_fingerprint": codec.predictor.fingerprint if codec else None,
                "webp_available": features.check("webp"),
                "timing_protocol": ("reference-v1: warmups, then one timed encode and decode per image/threshold"
                                    if codec is None else
                                    "qgru-v2: batched rows; per-image time = batch wall time / rows "
                                    "(throughput, not single-image latency; see `bitlaya profile`)")}
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        if not resume:
            raise ValueError("experiment exists; use --resume with the identical configuration")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("evaluation", "dataset", "checkpoint_sha256", "codec", "codec_fingerprint"):
            if previous.get(key) != resolved.get(key):
                raise ValueError(f"resume configuration differs in {key}")
    elif resume:
        raise ValueError("cannot resume: no saved configuration")
    else:
        write_json(config_path, resolved)
    samples = output / "sample_reconstructions"
    samples.mkdir(exist_ok=True)
    baseline_path = output / "baselines.json"
    if baseline_path.exists():
        baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baselines = []
        for index, original in enumerate(images):
            for row in conventional_baselines(original, config.lossy_qualities):
                baselines.append({"image_index": index, "dataset_index": manifest["indices"][index], **row})
        write_json(baseline_path, baselines)
        write_csv(output / "baselines.csv", baselines)
    rows = _load_journal(output / "measurements.jsonl")
    done = set()
    for row in rows:
        key = (row["threshold"], row["image_index"])
        if key in done or key[0] not in config.thresholds or not 0 <= key[1] < len(images):
            raise ValueError("journal contains duplicate or invalid records")
        if row["dataset_index"] != manifest["indices"][key[1]]:
            raise ValueError("journal image identity mismatch")
        done.add(key)
    pending = [(t, i) for t in config.thresholds for i in range(len(images)) if (t, i) not in done]
    with (output / "measurements.jsonl").open("a", encoding="utf-8") as journal:
        def record(row, data, reconstructed, original):
            index, threshold = row["image_index"], row["threshold"]
            if index < config.examples:
                stem = f"image_{index:04d}_t_{threshold:g}"
                (samples / f"{stem}.{'blaya' if codec else 'blay'}").write_bytes(data)
                Image.fromarray(reconstructed).save(samples / f"{stem}.png")
                Image.fromarray(original).save(samples / f"image_{index:04d}_original.png")
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            rows.append(row)

        if codec is None:
            for index in range(min(config.warmup, len(pending))):
                encoded = encode(images[index % len(images)], model, config.thresholds[0])
                np.testing.assert_array_equal(decode(encoded.data, model), encoded.reconstruction)
            for count, (threshold, index) in enumerate(pending, 1):
                original = images[index]
                started = time.perf_counter()
                encoded = encode(original, model, threshold)
                encode_seconds = time.perf_counter() - started
                started = time.perf_counter()
                reconstructed = decode(encoded.data, model)
                decode_seconds = time.perf_counter() - started
                if not np.array_equal(reconstructed, encoded.reconstruction):
                    raise RuntimeError("independent decoder diverged from encoder reconstruction")
                _, explicit = inspect_stream(encoded.data)
                payload = pack_bits(explicit)
                record(_row(threshold, index, manifest["indices"][index], original, reconstructed, len(explicit),
                            len(encoded.data), len(encoded.data) - len(payload), payload, encode_seconds,
                            decode_seconds), encoded.data, reconstructed, original)
                if progress and (count % 10 == 0 or count == len(pending)):
                    print(f"reference-v1: independently decoded {count}/{len(pending)}", flush=True)
        else:
            for start in range(0, len(pending), config.batch_rows):
                batch = pending[start:start + config.batch_rows]
                originals = [images[i] for _, i in batch]
                started = time.perf_counter()
                # One encoding; the trace (exact logits, model run on every row) also yields the
                # range-coded variant of the same stream: identical decisions and reconstruction.
                encoded = codec.compress_many(originals, [t for t, _ in batch], trace=True, run_model=True)
                encode_seconds = (time.perf_counter() - started) / len(batch)
                ranged = [codec.as_range_coded(e) for e in encoded]
                started = time.perf_counter()
                decoded = codec.decompress_many([e.data for e in encoded])  # bytes only
                decode_seconds = (time.perf_counter() - started) / len(batch)
                started = time.perf_counter()
                decoded_range = codec.decompress_many(ranged)  # bytes only
                range_seconds = (time.perf_counter() - started) / len(batch)
                for (threshold, index), result, reconstructed, from_range, range_data in zip(
                        batch, encoded, decoded, decoded_range, ranged):
                    if not (np.array_equal(reconstructed, result.reconstruction)
                            and np.array_equal(from_range, result.reconstruction)):
                        raise RuntimeError("independent decoder diverged from encoder reconstruction")
                    _, explicit = parse_stream(result.data)
                    record(_row(threshold, index, manifest["indices"][index], images[index], reconstructed,
                                result.payload_bits, len(result.data), OVERHEAD, pack_bits(explicit),
                                encode_seconds, decode_seconds, range_payload_bytes=len(range_data) - OVERHEAD,
                                range_file_bytes=len(range_data), range_file_bpp=len(range_data) * 8 / images[index].size,
                                range_decode_seconds=range_seconds), result.data, reconstructed, images[index])
                if progress:
                    print(f"qgru-v2: independently decoded {start + len(batch)}/{len(pending)} rows", flush=True)
    rows.sort(key=lambda r: (config.thresholds.index(r["threshold"]), r["image_index"]))
    write_csv(output / "metrics.csv", rows)
    summary = summarize_experiment(rows, baselines, resolved, config)
    write_json(output / "summary.json", summary)
    write_json(output / "timing.json", {"protocol": resolved["timing_protocol"],
               "controlled_repeated_benchmark": False,
               "environment": resolved["environment"], "thresholds": [
                   {"threshold": s["threshold"], "encode": s["statistics"]["encode_seconds"],
                    "decode": s["statistics"]["decode_seconds"]} for s in summary["thresholds"]]})
    from .research_plots import plot_report
    plot_report(summary, output)
    return summary


def summarize_experiment(rows, baselines, resolved, config):
    summaries = []
    for threshold in config.thresholds:
        group = [r for r in rows if r["threshold"] == threshold]
        summary = {"threshold": threshold, **aggregate(group, config.bootstrap, config.seed)}
        file_bytes = summary["statistics"]["file_bytes"]["mean"]
        summary["model_amortization"] = [{"images_sharing_model": n,
            "mean_bytes_per_image_including_model": file_bytes + resolved["checkpoint_bytes"] / n,
            "bpp_including_model": 8 * (file_bytes + resolved["checkpoint_bytes"] / n)
                                   / summary["statistics"]["original_bytes"]["mean"]}
            for n in (1, 10, 100, 1000, 10000)]
        summaries.append(summary)
    baseline_summaries = []
    for codec, quality in dict.fromkeys((b["codec"], b["quality"]) for b in baselines):
        group = [b for b in baselines if b["codec"] == codec and b["quality"] == quality]
        baseline_summaries.append({"codec": codec, "quality": quality,
                                   **aggregate(group, config.bootstrap, config.seed)})
    curves = {}
    for b in baseline_summaries:
        if b["quality"] is not None:
            curves.setdefault(b["codec"], []).append(b)
    return {"format": "bitlaya-research-summary-v2", "config": resolved,
            "sampling_scale": sampling_scale(config.size),
            "independent_decodes_verified": len(rows),
            "encoder_decoder_mismatches": sum(not r["encoder_decoder_equal"] for r in rows),
            "thresholds": summaries, "baselines": baseline_summaries,
            "matched_psnr": matched_quality(summaries, curves),
            "notes": ["Checkpoint cost excluded from file rates; model_amortization charges it once per N images.",
                      "PSNR null denotes +infinity for zero MSE; undefined metrics also use null with counts.",
                      "Per-image PSNR descriptive stats use only finite values; primary PSNR is from pooled MSE.",
                      "Bootstrap resamples images jointly across metrics, not bits; it excludes training-seed variance.",
                      "SSIM uses an 11x11 Gaussian sigma=1.5 window, population covariance, valid centers, range=255.",
                      "Header bytes include framing and checksums; payload padding is included in payload bytes.",
                      "Matched-PSNR baseline bytes are interpolated in log-bytes, never extrapolated (null outside).",
                      "Use validation for model/threshold selection and test for held-out reporting."]}

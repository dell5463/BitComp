"""Real encode/decode threshold sweeps, byte accounting, and shareable plots."""

import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .codec import decode, encode, runtime_signature, validate_threshold
from .metrics import distortion, lossless_baselines
from .model import BitPredictor, model_fingerprint


def evaluate(images: np.ndarray, model: BitPredictor, thresholds: list[float],
             output: str | Path, *, checkpoint_bytes: int = 0,
             provenance: dict | None = None, examples: int = 4) -> dict:
    thresholds = [validate_threshold(t) for t in thresholds]
    if len(images) == 0 or not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("provide images and unique thresholds")
    if checkpoint_bytes < 0 or examples < 0:
        raise ValueError("checkpoint_bytes and examples must be nonnegative")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("evaluation output already contains results; choose a new directory")
    examples = min(examples, len(images))
    baselines = [lossless_baselines(image) for image in images]
    rows, summaries, previews = [], [], []
    for threshold in thresholds:
        group, reconstructed_examples = [], []
        for index, original in enumerate(images):
            started = time.perf_counter()
            encoded = encode(original, model, threshold)
            encode_seconds = time.perf_counter() - started
            started = time.perf_counter()
            reconstruction = decode(encoded.data, model)
            decode_seconds = time.perf_counter() - started
            if not np.array_equal(reconstruction, encoded.reconstruction):
                raise RuntimeError("independent decoder disagrees with encoder reconstruction")
            quality = distortion(original, reconstruction)
            transmitted = encoded.metadata["explicit_bits"]
            row = {"threshold": threshold, "image_index": index, **baselines[index],
                   "explicit_bits": transmitted, "omitted_fraction": 1 - transmitted / (original.size * 8),
                   "payload_bytes": (transmitted + 7) // 8, "file_bytes": len(encoded.data),
                   "header_and_checksum_bytes": len(encoded.data) - (transmitted + 7) // 8,
                   "payload_bpp": transmitted / original.size,
                   "file_bpp": len(encoded.data) * 8 / original.size,
                   "raw_to_file_ratio": original.nbytes / len(encoded.data),
                   "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
                   **quality}
            group.append(row)
            rows.append(row)
            if index < examples:
                reconstructed_examples.append(reconstruction)
                prefix = output / f"example_{index:03d}_threshold_{threshold:g}"
                Path(str(prefix) + ".blay").write_bytes(encoded.data)
                from PIL import Image
                Image.fromarray(reconstruction).save(str(prefix) + ".png")
            if (index + 1) % 10 == 0:
                print(f"threshold={threshold:g}: decoded {index + 1}/{len(images)} images", flush=True)
        pixels = sum(image.size for image in images)
        file_bytes = sum(row["file_bytes"] for row in group)
        total_mse = sum(row["mse"] * image.size for row, image in zip(group, images)) / pixels
        summary = {"threshold": threshold, "images": len(images),
                   "payload_bpp": sum(r["explicit_bits"] for r in group) / pixels,
                   "file_bpp": 8 * file_bytes / pixels,
                   "file_plus_amortized_checkpoint_bpp": 8 * (file_bytes + checkpoint_bytes) / pixels,
                   "raw_to_file_ratio": pixels / file_bytes,
                   "mse": total_mse, "psnr_db": None if total_mse == 0 else 10 * math.log10(255**2 / total_mse),
                   "lossless_images": sum(r["lossless"] for r in group),
                   "bit_error_rate": sum(r["bit_error_rate"] * im.size for r, im in zip(group, images)) / pixels,
                   "omitted_fraction": 1 - sum(r["explicit_bits"] for r in group) / (pixels * 8),
                   "encode_seconds": sum(r["encode_seconds"] for r in group),
                   "decode_seconds": sum(r["decode_seconds"] for r in group)}
        summaries.append(summary)
        previews.append(reconstructed_examples)
        print(json.dumps(summary, allow_nan=False), flush=True)
    report = {"format": "bitlaya-evaluation-v1", "model_sha256": model_fingerprint(model),
              "runtime": runtime_signature(), "checkpoint_bytes": checkpoint_bytes,
              "image_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
              "provenance": provenance or {},
              "psnr_note": "null means positive infinity (zero MSE); aggregate PSNR uses pooled MSE",
              "baselines": {k: float(np.mean([b[k] for b in baselines])) for k in baselines[0]},
              "summary": summaries, "images": rows}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    fields = [k for k in rows[0] if k != "bitplane_error_rate_msb_first"]
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _plots(output, images[:examples], thresholds, previews, summaries, report["baselines"])
    return report


def _plots(output, originals, thresholds, previews, summaries, baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([s["payload_bpp"] for s in summaries], [s["mse"] for s in summaries], "o--", label="Explicit bits only")
    ax.plot([s["file_bpp"] for s in summaries], [s["mse"] for s in summaries], "o-", label="Full BitLaya file")
    # Group visually coincident endpoints so lossless thresholds remain readable.
    labels = {}
    for s in summaries:
        key = (round(s["file_bpp"], 1), round(s["mse"], 6))
        labels.setdefault(key, []).append(s)
    for group in labels.values():
        anchor = group[0]
        label = "t=" + ", ".join(f"{s['threshold']:g}" for s in group)
        ax.annotate(label, (anchor["file_bpp"], anchor["mse"]),
                    xytext=(-6, 28 if len(group) > 1 else 6),
                    textcoords="offset points", fontsize=8, ha="right",
                    arrowprops={"arrowstyle": "-", "color": "0.6"} if len(group) > 1 else None)
    pixels = baselines["raw_bytes"]
    for label, key in (("Raw grayscale", "raw_bytes"), ("zlib", "zlib_bytes"), ("PNG", "png_bytes")):
        ax.scatter([8 * baselines[key] / pixels], [0], marker="x", label=label)
    ax.set(xlabel="Bits per grayscale pixel (lower is smaller)", ylabel="Mean squared error (lower is better)",
           title="BitLaya rate–distortion; shared model excluded")
    ax.legend(fontsize=8)
    ax.margins(x=0.12, y=0.12)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "rate_distortion.png", dpi=180)
    plt.close(fig)
    if len(originals):
        fig, axes = plt.subplots(len(originals), len(thresholds) + 1,
                                  figsize=(2 * (len(thresholds) + 1), 2 * len(originals)), squeeze=False)
        for row, original in enumerate(originals):
            for col in range(len(thresholds) + 1):
                image = original if col == 0 else previews[col - 1][row]
                axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title("Original" if col == 0 else f"t={thresholds[col - 1]:g}")
        fig.tight_layout()
        fig.savefig(output / "reconstructions.png", dpi=180)
        plt.close(fig)

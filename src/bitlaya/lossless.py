"""True lossless neural compression: exact-engine probabilities + range coding.

Separate from predict-or-omit: EVERY bit is range-coded with the model's
probability, so decoded bits are exact and encoder/decoder contexts are always
the true history. Streams use the v2 ``BLYA`` layout with flag bit 1 (``FLAG_RANGE``),
decision threshold ``0xFFFFFFFF`` (lossless), payload bit count = 8 x pixels,
and the payload holding range-coder bytes. This is the ``range`` payload coding of
``BitLayaCodec`` at threshold 1.0 (the same code also range-codes the explicit bits
of lossy streams).

Probability quantization (documented, deterministic): the exact integer logit L
(scale 2**-S) is rounded to 1/256 logit units and clamped to [-16, 16]:
``i = clip(floor(L / 2**(S-8) + 1/2), -4096, 4096)``. Then
``p0 = clip(round(65536 / (1 + exp(i / 256))), 1, 65535)`` = P(bit = 0) in units of
2**-16, from a 8,193-entry table built once in float64. P is never 0 or 1: the
worst-case cost of a bit is 16 bits. The table is hashed into the stream's model
id, so decoders with a different table are rejected.
"""
import math

import numpy as np

from .bits import image_to_bits
from .blaya import OVERHEAD, BitLayaCodec, probability_table  # noqa: F401  (re-exported)
from .engine import teacher_forced_logits
from .rangecoder import PROB_ONE


class LosslessCodec:
    """Threshold 1.0 + range-coded payload of a ``BitLayaCodec``: every bit coded."""

    def __init__(self, codec: BitLayaCodec):
        self.codec = codec
        self.predictor = codec.predictor
        self.model_id = codec.range_model_id

    def compress_many(self, images) -> list:
        return [e.data for e in self.codec.compress_many(list(images), [1.0] * len(images), payload="range")]

    def decompress_many(self, streams) -> list:
        from .blaya import FLAG_RANGE, parse_stream
        for data in streams:
            header, _ = parse_stream(data)
            if not header.flags & FLAG_RANGE:
                raise ValueError("not a range-coded lossless stream")
        return self.codec.decompress_many(streams)

    def ideal_bits(self, images) -> dict:
        """Theoretical code lengths: sum -log2 P(actual bit) with table and raw probabilities."""
        images = np.stack(images)
        bits = np.stack([image_to_bits(image) for image in images])
        logits = teacher_forced_logits(self.predictor, bits, images.shape[2])
        p0 = self.codec.p0(logits) / PROB_ONE
        quantized = -np.log2(np.where(bits == 0, p0, 1 - p0)).sum(1)
        x = logits * 2.0 ** -self.predictor.S
        exact = (np.logaddexp(0, np.where(bits == 1, -x, x)) / math.log(2)).sum(1)
        return {"table_probabilities": quantized, "engine_probabilities": exact}

def benchmark(checkpoint, data, split, size, seed, output, batch=128, latency_images=5):
    """Lossless BitLaya vs raw / zlib-9 / PNG-9 on a fixed held-out subset."""
    import json
    from pathlib import Path
    import time

    import torch

    from .data import benchmark_images
    from .metrics import lossless_baselines
    from .model import load_checkpoint
    from .profiling import environment
    from .research import sampling_scale, source_identity, write_csv, write_json
    from .research_plots import SERIES, plt, style
    from .statistics import bootstrap_means, describe

    torch.set_num_threads(1)
    model, _ = load_checkpoint(checkpoint)
    images, manifest = benchmark_images(data, split, size, seed)
    codec = LosslessCodec(BitLayaCodec(model))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, encode_time, decode_time = [], 0.0, 0.0
    for start in range(0, len(images), batch):
        chunk = list(images[start:start + batch])
        began = time.perf_counter()
        streams = codec.compress_many(chunk)
        encode_time += time.perf_counter() - began
        began = time.perf_counter()
        decoded = codec.decompress_many(streams)
        decode_time += time.perf_counter() - began
        ideal = codec.ideal_bits(chunk)
        for offset, (image, stream, result) in enumerate(zip(chunk, streams, decoded)):
            if not np.array_equal(result, image):
                raise RuntimeError("lossless roundtrip failed")
            index = start + offset
            baselines = lossless_baselines(image)
            rows.append({"image_index": index, "dataset_index": manifest["indices"][index],
                         "raw_bytes": baselines["raw_bytes"], "zlib9_bytes": baselines["zlib_bytes"],
                         "png9_bytes": baselines["png_bytes"], "bitlaya_file_bytes": len(stream),
                         "bitlaya_payload_bytes": len(stream) - OVERHEAD,
                         "ideal_bits_table": float(ideal["table_probabilities"][offset]),
                         "ideal_bits_engine": float(ideal["engine_probabilities"][offset]),
                         "coder_overhead_bits": (len(stream) - OVERHEAD) * 8 - float(ideal["table_probabilities"][offset]),
                         "roundtrip_exact": True})
        print(f"lossless: {len(rows)}/{len(images)} images verified", flush=True)
    single = []
    for image in images[:latency_images]:
        began = time.perf_counter()
        stream = codec.compress_many([image])[0]
        middle = time.perf_counter()
        codec.decompress_many([stream])
        single.append({"encode_seconds": middle - began, "decode_seconds": time.perf_counter() - middle})
    pixels = images[0].size
    keys = ("raw_bytes", "zlib9_bytes", "png9_bytes", "bitlaya_file_bytes", "bitlaya_payload_bytes",
            "ideal_bits_table", "ideal_bits_engine", "coder_overhead_bits")
    stats = {key: describe([r[key] for r in rows]) for key in keys}
    bpp = {key: stats[key]["mean"] * 8 / pixels for key in keys if key.endswith("bytes")}
    bpp["ideal_table"] = stats["ideal_bits_table"]["mean"] / pixels
    bpp["ideal_engine"] = stats["ideal_bits_engine"]["mean"] / pixels
    summary = {"format": "bitlaya-lossless-benchmark-v1", "checkpoint": str(checkpoint), "dataset": manifest,
               "sampling_scale": sampling_scale(len(images)), "model_id": codec.model_id.hex(),
               "images": len(rows), "all_roundtrips_exact": all(r["roundtrip_exact"] for r in rows),
               "statistics": stats, "mean_bits_per_pixel": bpp,
               "bootstrap_95_ci_bytes": bootstrap_means({k: [r[k] for r in rows] for k in
                                                         ("zlib9_bytes", "png9_bytes", "bitlaya_file_bytes")}),
               "fraction_images_bitlaya_smaller_than_png": float(np.mean([r["bitlaya_file_bytes"] < r["png9_bytes"]
                                                                          for r in rows])),
               "timing": {"batched_encode_seconds_per_image": encode_time / len(rows),
                          "batched_decode_seconds_per_image": decode_time / len(rows), "batch_rows": batch,
                          "single_image": single, "environment": environment(),
                          "note": "uncontrolled wall-clock; see `bitlaya profile` for controlled latency"},
               "container_note": "bitlaya_payload_bytes approximates container mode (header shared; +2-6 B/image).",
               "checkpoint_bytes": Path(checkpoint).stat().st_size, "source": source_identity()}
    write_json(output / "summary.json", summary)
    write_csv(output / "per_image.csv", rows)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = ["raw", "zlib-9", "PNG-9", "BitLaya lossless\n(file)", "BitLaya\n(payload only)", "ideal\n(-log2 p)"]
    values = [bpp["raw_bytes"], bpp["zlib9_bytes"], bpp["png9_bytes"], bpp["bitlaya_file_bytes"],
              bpp["bitlaya_payload_bytes"], bpp["ideal_table"]]
    ax.barh(labels[::-1], values[::-1], color=[SERIES[0] if "BitLaya" in l or "ideal" in l else "#898781"
                                               for l in labels[::-1]], height=0.6)
    for y, value in enumerate(values[::-1]):
        ax.annotate(f"{value:.3f}", (value, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    style(ax, f"Lossless bits per pixel, {len(rows)} {split} images (model excluded)", "Mean bits per pixel", "")
    fig.tight_layout()
    fig.savefig(output / "lossless_bpp.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"mean_bits_per_pixel": bpp, "timing": summary["timing"]["batched_encode_seconds_per_image"]},
                     indent=2))
    return summary

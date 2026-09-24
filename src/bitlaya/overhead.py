"""Measured stream overhead: v1 JSON framing vs v2 binary vs v2 container."""
import json
from pathlib import Path

import numpy as np
import torch

from .bits import pack_bits
from .blaya import OVERHEAD, BitLayaCodec, parse_stream
from .codec import encode as encode_v1, inspect_stream
from .data import benchmark_images
from .model import load_checkpoint
from .research import write_json


def measure_overhead(checkpoint, data, split, size, seed, thresholds, output, v1_images=3):
    torch.set_num_threads(1)
    model, _ = load_checkpoint(checkpoint)
    images, manifest = benchmark_images(data, split, size, seed)
    codec = BitLayaCodec(model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(checkpoint), "dataset": manifest, "thresholds": []}
    for threshold in thresholds:
        singles = codec.compress_many(list(images), [threshold] * len(images))
        payload_bytes = [len(pack_bits(parse_stream(s.data)[1])) for s in singles]
        single_total = sum(len(s.data) for s in singles)
        entry = {"threshold": threshold, "images": len(images),
                 "v2_single_file_overhead_bytes": OVERHEAD,
                 "v2_single_total_bytes": single_total,
                 "v2_single_payload_bytes": sum(payload_bytes)}
        for per_image_crc in (True, False):
            container, reconstructions = codec.compress_container(list(images), threshold,
                                                                  per_image_crc=per_image_crc)
            decoded = codec.decompress_container(container)
            if any(not np.array_equal(a, b) for a, b in zip(decoded, reconstructions)):
                raise RuntimeError("container roundtrip failed")
            if any(not np.array_equal(a, s.reconstruction) for a, s in zip(reconstructions, singles)):
                raise RuntimeError("container and single-file reconstructions differ")
            bits = sum(s.payload_bits for s in singles)
            key = "with_crc" if per_image_crc else "without_crc"
            entry[f"container_{key}_total_bytes"] = len(container)
            entry[f"container_{key}_overhead_bytes_per_image"] = (len(container) - (bits + 7) // 8) / len(images)
        v1 = []
        for image in images[:v1_images]:  # the float reference codec is slow; a few suffice for framing size
            stream = encode_v1(image, model, threshold).data
            _, explicit = inspect_stream(stream)
            v1.append(len(stream) - len(pack_bits(explicit)))
        entry["v1_json_overhead_bytes_measured"] = v1
        entry["v1_json_overhead_bytes_mean"] = float(np.mean(v1))
        entry["single_file_overhead_reduction_percent"] = 100 * (1 - OVERHEAD / np.mean(v1))
        report["thresholds"].append(entry)
        print(json.dumps(entry), flush=True)
    write_json(output / "overhead.json", report)
    return report

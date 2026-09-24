"""Reference CPU codec: decisions depend exclusively on reconstructed history.

No omission mask is transmitted. An omitted, confidently wrong prediction is
intentionally accepted by BOTH encoder and decoder, keeping their states equal.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import platform
import struct

import numpy as np
import torch

from .bits import bits_to_image, image_to_bits, pack_bits, unpack_bits
from .model import BOS, BitPredictor, model_fingerprint

MAGIC = b"BLAY\x01\r\n\x00"
MAX_HEADER = 16_384
MAX_PIXELS = 1024  # MVP: small grayscale images only, including 32x32 CIFAR-10.


@dataclass
class EncodeResult:
    data: bytes
    reconstruction: np.ndarray
    metadata: dict


def runtime_signature() -> dict:
    return {
        "torch": str(torch.__version__),
        "machine": platform.machine(),
        "system": platform.system(),
        "backend": "cpu-float32-threads1-no-mkldnn-v1",
    }


@contextmanager
def reference_runtime():
    """Process-global PyTorch settings are restored; do not call concurrently."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    mkldnn = torch.backends.mkldnn.enabled
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        with torch.inference_mode():
            yield
    finally:
        torch.backends.mkldnn.enabled = mkldnn
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.5 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0.5, 1.0]")
    return threshold


def _decision(logit: float, threshold: float) -> tuple[bool, int]:
    if not math.isfinite(logit):
        raise ValueError("model produced a non-finite logit")
    predicted = int(logit >= 0)  # Explicit tie rule: p=0.5 predicts 1.
    if threshold == 1.0:
        return False, predicted  # Guaranteed lossless sentinel, even at saturation.
    confidence = 1.0 / (1.0 + math.exp(-abs(logit)))
    return confidence >= threshold, predicted


def _reference_model(model: BitPredictor) -> BitPredictor:
    # Preserve the caller's training mode, device, precision, and hidden state.
    result = copy.deepcopy(model).cpu().float().eval()
    if not all(torch.isfinite(p).all() for p in result.parameters()):
        raise ValueError("model contains non-finite weights")
    return result


def _serialize(metadata: dict, payload: bytes) -> bytes:
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                        allow_nan=False).encode("utf-8")
    if len(header) > MAX_HEADER:
        raise ValueError("header too large")
    body = MAGIC + struct.pack(">I", len(header)) + header + payload
    return body + hashlib.sha256(body).digest()


def inspect_stream(data: bytes) -> tuple[dict, np.ndarray]:
    """Parse and validate framing, checksums, counts, and zero padding."""
    if len(data) < len(MAGIC) + 4 + 32 or data[:len(MAGIC)] != MAGIC:
        raise ValueError("not a BitLaya v1 stream")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("stream checksum mismatch (corrupt or truncated data)")
    length = struct.unpack(">I", data[len(MAGIC):len(MAGIC) + 4])[0]
    start = len(MAGIC) + 4
    end = start + length
    if length > MAX_HEADER or end > len(data) - 32:
        raise ValueError("invalid header length")
    try:
        metadata = json.loads(data[start:end])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid stream metadata") from exc
    required = {"shape", "bit_order", "threshold", "explicit_bits", "model_sha256",
                "reconstruction_sha256", "runtime"}
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("invalid stream metadata fields")
    shape = metadata["shape"]
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(x) is not int or x <= 0 for x in shape)
            or shape[0] * shape[1] > MAX_PIXELS):
        raise ValueError("invalid image dimensions")
    if metadata["bit_order"] != "raster-msb-first":
        raise ValueError("unsupported bit order")
    if type(metadata["threshold"]) not in (int, float):
        raise ValueError("invalid threshold type")
    validate_threshold(metadata["threshold"])
    count = metadata["explicit_bits"]
    if type(count) is not int or not 0 <= count <= shape[0] * shape[1] * 8:
        raise ValueError("invalid explicit bit count")
    for field in ("model_sha256", "reconstruction_sha256"):
        value = metadata[field]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("invalid fingerprint")
    if not isinstance(metadata["runtime"], dict):
        raise ValueError("invalid runtime signature")
    return metadata, unpack_bits(data[end:-32], count)


def encode(image: np.ndarray, model: BitPredictor, threshold: float = 0.95) -> EncodeResult:
    threshold = validate_threshold(threshold)
    original = image_to_bits(image)
    if image.size > MAX_PIXELS:
        raise ValueError("MVP supports at most 1024 grayscale pixels per image")
    reference = _reference_model(model)
    reconstructed = np.empty_like(original)
    explicit = []
    with reference_runtime():
        hidden = None
        previous = torch.tensor([[BOS]], dtype=torch.long)
        for index, true_bit in enumerate(original):
            logits, hidden = reference(previous, hidden)
            omit, predicted = _decision(logits.item(), threshold)
            bit = predicted if omit else int(true_bit)
            if not omit:
                explicit.append(bit)
            reconstructed[index] = bit
            previous.fill_(bit)  # Never feed the original after a wrong omission.
    reconstruction = bits_to_image(reconstructed, image.shape)
    metadata = {
        "shape": list(image.shape),
        "bit_order": "raster-msb-first",
        "threshold": threshold,
        "explicit_bits": len(explicit),
        "model_sha256": model_fingerprint(reference),
        "reconstruction_sha256": hashlib.sha256(reconstruction.tobytes()).hexdigest(),
        "runtime": runtime_signature(),
    }
    return EncodeResult(_serialize(metadata, pack_bits(np.array(explicit))),
                        reconstruction, metadata)


def decode(data: bytes, model: BitPredictor) -> np.ndarray:
    metadata, explicit = inspect_stream(data)
    reference = _reference_model(model)
    if metadata["model_sha256"] != model_fingerprint(reference):
        raise ValueError("stream requires a different model checkpoint")
    if metadata["runtime"] != runtime_signature():
        raise ValueError("codec runtime mismatch; use the encoder's PyTorch build/platform")
    shape = tuple(metadata["shape"])
    bits = np.empty(shape[0] * shape[1] * 8, dtype=np.uint8)
    cursor = 0
    with reference_runtime():
        hidden = None
        previous = torch.tensor([[BOS]], dtype=torch.long)
        for index in range(bits.size):
            logits, hidden = reference(previous, hidden)
            omit, predicted = _decision(logits.item(), metadata["threshold"])
            if omit:
                bit = predicted
            else:
                if cursor >= explicit.size:
                    raise ValueError("explicit payload exhausted; decoder diverged")
                bit = int(explicit[cursor])
                cursor += 1
            bits[index] = bit
            previous.fill_(bit)
    if cursor != explicit.size:
        raise ValueError("unused payload bits; decoder diverged")
    image = bits_to_image(bits, shape)
    if hashlib.sha256(image.tobytes()).hexdigest() != metadata["reconstruction_sha256"]:
        raise ValueError("reconstruction checksum mismatch; numerical runtime divergence")
    return image

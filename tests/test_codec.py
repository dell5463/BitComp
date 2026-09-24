import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from bitlaya.codec import _serialize, decode, encode, inspect_stream, runtime_signature
from bitlaya.model import BitPredictor, ModelConfig, save_checkpoint


def constant_model(logit):
    model = BitPredictor(ModelConfig(4, 8, 1))
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.head.bias.fill_(logit)
    return model


@pytest.mark.parametrize("threshold", [0.5, 0.6, 0.95, 1.0])
def test_independent_reconstruction_and_repeatability(threshold):
    torch.manual_seed(27)
    model = BitPredictor(ModelConfig(4, 8, 1))
    image = np.random.default_rng(4).integers(0, 256, (4, 4), dtype=np.uint8)
    first = encode(image, model, threshold)
    second = encode(image, model, threshold)
    assert first.data == second.data
    np.testing.assert_array_equal(decode(first.data, model), first.reconstruction)
    assert model.training  # Codec must preserve caller's state.
    if threshold == 1.0:
        np.testing.assert_array_equal(first.reconstruction, image)


def test_native_cifar_size_lossless_at_threshold_one_despite_saturation():
    model = constant_model(1000)
    original = np.random.default_rng(7).integers(0, 256, (32, 32), dtype=np.uint8)
    result = encode(original, model, 1.0)
    assert result.metadata["explicit_bits"] == 8192
    np.testing.assert_array_equal(decode(result.data, model), original)


def test_confidently_wrong_omissions_are_accepted_by_both_sides():
    model = constant_model(-8)
    original = np.full((4, 4), 255, dtype=np.uint8)
    result = encode(original, model, 0.99)
    assert result.metadata["explicit_bits"] == 0
    assert not result.reconstruction.any()
    np.testing.assert_array_equal(decode(result.data, model), result.reconstruction)


class ContextSensitivePredictor(BitPredictor):
    """After a reconstructed 0 demand a payload bit; otherwise omit a 0."""
    def forward(self, previous_bits, hidden=None):
        logits = torch.where(previous_bits == 0, 0.0, -8.0)
        return logits, hidden


def test_mixed_decisions_use_reconstructed_context_including_wrong_bits():
    model = ContextSensitivePredictor(ModelConfig(4, 8, 1))
    original = np.full((2, 2), 255, dtype=np.uint8)
    result = encode(original, model, 0.9)
    # BOS -> omit 0 -> explicit 1 -> omit 0 -> explicit 1 ...
    assert result.metadata["explicit_bits"] == 16
    np.testing.assert_array_equal(result.reconstruction, np.full((2, 2), 85, dtype=np.uint8))
    np.testing.assert_array_equal(decode(result.data, model), result.reconstruction)


def test_probability_tie_rule():
    result = encode(np.zeros((1, 1), dtype=np.uint8), constant_model(0), 0.5)
    assert result.metadata["explicit_bits"] == 0
    assert result.reconstruction.item() == 255


@pytest.mark.parametrize("threshold", [0.49, 1.01, float("nan"), float("inf")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        encode(np.zeros((1, 1), dtype=np.uint8), constant_model(0), threshold)


def test_corruption_truncation_trailing_bytes_and_wrong_model():
    model = constant_model(0)
    result = encode(np.zeros((1, 1), dtype=np.uint8), model, 1)
    mutated = bytearray(result.data)
    mutated[-33] ^= 1
    for data in [bytes(mutated), result.data[:-1], result.data + b"\x00", b"nope"]:
        with pytest.raises(ValueError):
            decode(data, model)
    with pytest.raises(ValueError, match="different model"):
        decode(result.data, constant_model(1))


def test_metadata_runtime_and_reconstruction_checks():
    model = constant_model(0)
    result = encode(np.zeros((1, 1), dtype=np.uint8), model, 1)
    metadata = result.metadata.copy()
    metadata["runtime"] = {**runtime_signature(), "torch": "wrong"}
    with pytest.raises(ValueError, match="runtime mismatch"):
        decode(_serialize(metadata, b"\x00"), model)
    metadata = result.metadata.copy()
    metadata["reconstruction_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reconstruction checksum"):
        decode(_serialize(metadata, b"\x00"), model)
    metadata = result.metadata.copy()
    metadata["explicit_bits"] = 0
    with pytest.raises(ValueError, match="exhausted"):
        decode(_serialize(metadata, b""), model)
    metadata = result.metadata.copy()
    metadata["shape"] = [99999999, 99999999]
    with pytest.raises(ValueError, match="dimensions"):
        inspect_stream(_serialize(metadata, b"\x00"))


def test_decode_in_fresh_process_without_original(tmp_path):
    model = constant_model(-8)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model)
    encoded = encode(np.full((2, 2), 255, dtype=np.uint8), model, 0.95)
    stream = tmp_path / "image.blay"
    stream.write_bytes(encoded.data)
    output = tmp_path / "reconstructed.png"
    completed = subprocess.run([sys.executable, "-m", "bitlaya", "decode", str(stream),
                                str(output), "--checkpoint", str(checkpoint)],
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    np.testing.assert_array_equal(np.array(Image.open(output)), encoded.reconstruction)

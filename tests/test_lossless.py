"""Range coder and lossless neural mode: exact roundtrips and overhead bounds."""
import math

import numpy as np
import pytest
import torch

from bitlaya.blaya import FLAG_RANGE, HEADER, OVERHEAD, BitLayaCodec
from bitlaya.lossless import LosslessCodec, probability_table
from bitlaya.model import BitPredictor, ModelConfig
from bitlaya.rangecoder import PROB_ONE, RangeDecoder, RangeEncoder


def roundtrip(bits, probabilities):
    encoder = RangeEncoder()
    for bit, p0 in zip(bits, probabilities):
        encoder.encode(bit, p0)
    data = encoder.finish()
    decoder = RangeDecoder(data)
    decoded = [decoder.decode(p0) for p0 in probabilities]
    decoder.finish()
    return data, decoded


def test_random_sequences_and_probabilities_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(1500):
        length = int(rng.integers(0, 300))
        bits = rng.integers(0, 2, length).tolist()
        kind = rng.integers(0, 3)
        if kind == 0:
            probabilities = rng.integers(1, PROB_ONE, length).tolist()
        elif kind == 1:  # extreme, frequently wrong probabilities (forces long carries)
            probabilities = rng.choice([1, 2, PROB_ONE - 2, PROB_ONE - 1], length).tolist()
        else:  # probabilities consistent with the bits (realistic, highly compressible)
            probabilities = [PROB_ONE - 50 if b == 0 else 50 for b in bits]
        data, decoded = roundtrip(bits, probabilities)
        assert decoded == bits


def test_carry_propagation_worst_case():
    bits = [1] * 5000  # always the unlikely symbol at p0 = 65535/65536
    data, decoded = roundtrip(bits, [PROB_ONE - 1] * 5000)
    assert decoded == bits
    ideal = 5000 * -math.log2(1 / PROB_ONE)
    assert len(data) * 8 <= ideal + 64
    zeros = [0] * 20000
    data, decoded = roundtrip(zeros, [PROB_ONE - 1] * 20000)
    assert decoded == zeros and len(data) <= 8


def test_code_length_close_to_ideal():
    rng = np.random.default_rng(1)
    probabilities = rng.integers(2000, PROB_ONE - 2000, 20000)
    bits = (rng.random(20000) * PROB_ONE >= probabilities).astype(int)
    data, _ = roundtrip(bits.tolist(), probabilities.tolist())
    ideal = -np.log2(np.where(bits == 0, probabilities, PROB_ONE - probabilities) / PROB_ONE).sum()
    assert ideal <= len(data) * 8 <= ideal + 40


def test_truncated_and_excess_payloads_rejected():
    bits = [0, 1] * 400
    data, _ = roundtrip(bits, [30000] * 800)
    with pytest.raises(ValueError, match="exhausted"):
        decoder = RangeDecoder(data[:-2])
        [decoder.decode(30000) for _ in range(800)]
    decoder = RangeDecoder(data + b"\x00")
    [decoder.decode(30000) for _ in range(800)]
    with pytest.raises(ValueError, match="excess"):
        decoder.finish()
    with pytest.raises(ValueError, match="truncated"):
        RangeDecoder(b"\x00\x00")


def test_probability_table_is_clamped_and_monotone():
    table = probability_table()
    assert table.min() == 1 and table.max() == PROB_ONE - 1
    assert np.all(np.diff(table) <= 0)  # larger logit => more likely 1 => smaller P(0)
    assert table[len(table) // 2] == PROB_ONE // 2


def test_lossless_neural_codec_roundtrip_and_ideal_length():
    torch.manual_seed(3)
    model = BitPredictor(ModelConfig(8, 16, 2, plane_dim=2))
    codec = LosslessCodec(BitLayaCodec(model))
    rng = np.random.default_rng(4)
    images = [np.clip(np.cumsum(rng.integers(-9, 10, (6, 5)), 1) + 128, 0, 255).astype(np.uint8)
              for _ in range(7)] + [np.zeros((3, 3), np.uint8)]
    streams = codec.compress_many(images)
    for image, data in zip(images, streams):
        assert data[5] == FLAG_RANGE
    for image, decoded in zip(images, codec.decompress_many(streams)):
        np.testing.assert_array_equal(decoded, image)
    ideal = codec.ideal_bits(images[:7])
    for data, bits in zip(streams[:7], ideal["table_probabilities"]):
        realized = (len(data) - OVERHEAD) * 8
        assert bits <= realized <= bits + 40
    assert np.allclose(ideal["table_probabilities"], ideal["engine_probabilities"], rtol=0.01)
    np.testing.assert_array_equal(BitLayaCodec(model).decompress(streams[0]), images[0])  # unified codec
    corrupt = bytearray(streams[0])
    corrupt[HEADER.size + 1] ^= 0x10
    with pytest.raises(ValueError, match="checksum"):
        codec.decompress_many([bytes(corrupt)])
    other = LosslessCodec(BitLayaCodec(BitPredictor(ModelConfig(8, 16, 2, plane_dim=2))))
    with pytest.raises(ValueError, match="different model"):
        other.decompress_many(streams[:1])

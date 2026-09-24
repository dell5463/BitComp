"""Correctness of the v2 binary codec, fixed-point engine and synchronization rule."""
import struct
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from bitlaya import protocol
from bitlaya.bits import image_to_bits
from bitlaya.blaya import (FLAG_RANGE, FLAG_ZLIB, HEADER, MAX_PIXELS, OVERHEAD, BitLayaCodec, StreamHeader, crc32,
                           pack_stream, parse_stream, reference_decode, reference_encode)
from bitlaya.engine import (TRANSMIT_ALL, QuantizedPredictor, Stepper, decision_threshold,
                            teacher_forced_logits)
from bitlaya.model import BOS, BitPredictor, ModelConfig, save_checkpoint

THRESHOLDS = [0.5, 0.55, 0.7, 0.9, 0.99, 1.0]


def random_model(seed=0, **features):
    torch.manual_seed(seed)
    model = BitPredictor(ModelConfig(8, 16, 2, **features))
    with torch.no_grad():  # sharpen predictions so thresholds produce mixed decisions
        model.head.weight.mul_(8)
    return model


def constant_model(logit):
    model = BitPredictor(ModelConfig(4, 8, 1))
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.head.bias.fill_(logit)
    return model


def images(seed, count, shape=(4, 5)):
    rng = np.random.default_rng(seed)
    smooth = np.cumsum(rng.integers(-20, 21, (count,) + shape), axis=-1) + 128
    return list(np.clip(smooth, 0, 255).astype(np.uint8))


def rebuild(data, payload=None, **changes):
    """Re-pack a stream with modified header fields and a valid checksum."""
    fields = dict(zip(("magic", "version", "flags", "width", "height", "decision", "payload_bits",
                       "model_id", "recon_crc"), HEADER.unpack_from(data)))
    fields.update(changes)
    body = HEADER.pack(*fields.values()) + (data[HEADER.size:-4] if payload is None else payload)
    return body + struct.pack(">I", crc32(body))


@pytest.mark.parametrize("features", [{}, {"plane_dim": 3}, {"plane_dim": 3, "row_dim": 2, "col_dim": 2}])
def test_stepper_matches_reference_step_bit_for_bit(features):
    predictor = QuantizedPredictor(random_model(1, **features))
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, (3, 200))
    fast = teacher_forced_logits(predictor, bits, width=5)
    for row in range(3):
        state, symbol = predictor.initial_state(), BOS
        for t in range(200):
            logit, state = predictor.step_reference(symbol, t, 5, state)
            assert logit == fast[row, t]
            symbol = int(bits[row, t])


def test_batch_size_invariance_of_exact_engine():
    predictor = QuantizedPredictor(random_model(3, plane_dim=4))
    bits = np.random.default_rng(4).integers(0, 2, (7, 160))
    together = teacher_forced_logits(predictor, bits, 5)
    alone = np.concatenate([teacher_forced_logits(predictor, bits[i:i + 1], 5) for i in range(7)])
    np.testing.assert_array_equal(together, alone)


def test_quantized_engine_tracks_float_model():
    model = random_model(5, plane_dim=4, row_dim=2, col_dim=2)
    bits = torch.randint(0, 2, (2, 160), generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        expected, _ = model(torch.cat((torch.full((2, 1), BOS), bits[:, :-1]), 1), width=5)
    predictor = QuantizedPredictor(model)
    actual = teacher_forced_logits(predictor, bits.numpy(), 5) * 2.0 ** -predictor.S
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-3)


@pytest.mark.parametrize("features", [{}, {"plane_dim": 3, "row_dim": 2, "col_dim": 2}])
@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_optimized_codec_equals_reference_codec(features, threshold):
    predictor = QuantizedPredictor(random_model(7, **features))
    codec = BitLayaCodec(predictor)
    for image in images(8, 3):
        reference = reference_encode(predictor, image, threshold)
        optimized = codec.compress(image, threshold)
        assert optimized.data == reference.data  # payload, counts, framing identical
        np.testing.assert_array_equal(optimized.reconstruction, reference.reconstruction)
        np.testing.assert_array_equal(codec.decompress(reference.data), reference.reconstruction)
        np.testing.assert_array_equal(reference_decode(predictor, optimized.data), optimized.reconstruction)


def test_batched_compress_and_decompress_equal_single_calls():
    codec = BitLayaCodec(random_model(9, plane_dim=4))
    batch = images(10, 4) + images(11, 2, (3, 3))
    thresholds = [0.6, 0.8, 0.95, 1.0, 0.7, 0.9]
    together = codec.compress_many(batch, thresholds)
    for image, threshold, encoded in zip(batch, thresholds, together):
        assert codec.compress(image, threshold).data == encoded.data
    decoded = codec.decompress_many([e.data for e in together])
    for encoded, image in zip(together, decoded):
        np.testing.assert_array_equal(image, encoded.reconstruction)
    assert any(0 < e.payload_bits < e.reconstruction.size * 8 for e in together)  # mixed decisions exercised


class HistoryPredictor:
    """Random but deterministic function of (t, reconstructed history)."""

    def __init__(self, table, window=3):
        self.table, self.window, self.history = table, window, 0

    def __call__(self, t, symbol):
        if symbol != BOS:
            self.history = ((self.history << 1) | symbol) & ((1 << self.window) - 1)
        return int(self.table[t, self.history])


def test_random_synchronization_fuzzing():
    rng = np.random.default_rng(11)
    wrong = 0
    for _ in range(3000):
        length = int(rng.integers(1, 48))
        source = rng.integers(0, 2, length)
        table = rng.integers(-1000, 1001, (length, 8))
        bound = None if rng.random() < 0.05 else int(rng.integers(0, 1100))
        explicit, reconstructed, omitted = protocol.encode(source, HistoryPredictor(table), bound)
        decoded = protocol.decode(explicit, length, HistoryPredictor(table), bound)
        np.testing.assert_array_equal(decoded, reconstructed)
        np.testing.assert_array_equal(reconstructed[~omitted], source[~omitted])
        assert len(explicit) == (~omitted).sum()
        wrong += int((reconstructed != source).sum())
    assert wrong > 0  # confidently wrong omissions actually occurred


class RecordingPredictor:
    def __init__(self, logits):
        self.logits, self.fed = logits, []

    def __call__(self, t, symbol):
        self.fed.append(symbol)
        return self.logits[t]


def test_confident_wrong_prediction_feeds_prediction_on_both_sides():
    source = np.array([0, 0, 1])
    logits = [10_000, -5, 10_000]  # bit 0: predicted 1 with high confidence, but source is 0
    encoder = RecordingPredictor(logits)
    explicit, reconstructed, omitted = protocol.encode(source, encoder, bound=9_000)
    decoder = RecordingPredictor(logits)
    decoded = protocol.decode(explicit, 3, decoder, bound=9_000)
    assert reconstructed.tolist() == [1, 0, 1] and omitted.tolist() == [True, False, True]
    assert encoder.fed == decoder.fed == [BOS, 1, 0]  # the wrong 1 was fed, not the source 0
    np.testing.assert_array_equal(decoded, reconstructed)


def test_confident_wrong_prediction_with_real_codec():
    codec = BitLayaCodec(constant_model(8.0))  # p(1) = 0.99966 everywhere
    image = np.zeros((2, 2), dtype=np.uint8)
    encoded = codec.compress(image, 0.999)
    assert encoded.payload_bits == 0
    assert (encoded.reconstruction == 255).all()
    np.testing.assert_array_equal(codec.decompress(encoded.data), encoded.reconstruction)


def test_threshold_boundary_comparison_is_greater_or_equal():
    for logit, bound, omit in [(500, 500, True), (-500, 500, True), (499, 500, False),
                               (501, 500, True), (-499, 500, False), (0, 0, True)]:
        assert protocol.decide(logit, bound)[0] is omit
    assert protocol.decide(0, 0)[1] == 1  # tie predicts 1
    # Same boundary through the engine: a constant logit exactly on the bound.
    codec = BitLayaCodec(constant_model(0.75))
    predictor = codec.predictor
    logit = teacher_forced_logits(predictor, np.zeros((1, 1)), 1)[0, 0]
    decision = int(logit / predictor.decision_scale)
    assert decision * predictor.decision_scale == logit  # exactly representable
    source = np.zeros((1, 64), dtype=np.uint8)
    for d, expect in [(decision, True), (decision + 1, False), (decision - 1, True)]:
        omitted = codec.encode_bits(source, (1, 8), [d])["omitted"]
        assert omitted.all() if expect else not omitted.any()


def test_decision_threshold_mapping():
    assert decision_threshold(0.5) == 0
    assert decision_threshold(1.0) == TRANSMIT_ALL
    assert decision_threshold(0.9) == int(np.ceil(np.log(9) * 2**24))
    for bad in (0.49, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="threshold"):
            decision_threshold(bad)


def test_threshold_one_is_exact_lossless_fast_path(monkeypatch):
    codec = BitLayaCodec(constant_model(1000))  # saturated model must not matter
    original = np.random.default_rng(12).integers(0, 256, (32, 32), dtype=np.uint8)
    encoded = codec.compress(original, 1.0)
    assert encoded.data[HEADER.size:-4] == original.tobytes()  # payload is the raw raster
    assert len(encoded.data) == 1024 + OVERHEAD

    def forbidden(*args, **kwargs):
        raise AssertionError("model evaluated on the threshold-1 fast path")

    monkeypatch.setattr(Stepper, "step", forbidden)
    np.testing.assert_array_equal(codec.decompress(encoded.data), original)
    assert codec.compress(original, 1.0).data == encoded.data
    np.testing.assert_array_equal(reference_decode(codec.predictor, encoded.data), original)


def transmitting_stream():
    codec = BitLayaCodec(constant_model(0.0))  # confidence 0.5 < 0.95: everything transmitted
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    return codec, codec.compress(image, 0.95)


def test_payload_exhaustion_and_excess_payload_are_rejected():
    codec, encoded = transmitting_stream()
    assert encoded.payload_bits == 32
    short = rebuild(encoded.data, payload_bits=24, payload=encoded.data[HEADER.size:HEADER.size + 3])
    with pytest.raises(ValueError, match="exhausted"):
        codec.decompress(short)
    with pytest.raises(ValueError, match="exhausted"):
        reference_decode(codec.predictor, short)
    omitting = BitLayaCodec(constant_model(8.0))
    extra = omitting.compress(np.zeros((2, 2), np.uint8), 0.99)
    padded = rebuild(extra.data, payload_bits=8, payload=b"\xff")
    with pytest.raises(ValueError, match="unused payload"):
        omitting.decompress(padded)
    with pytest.raises(ValueError, match="excess payload"):
        codec.decompress(rebuild(encoded.data, payload=encoded.data[HEADER.size:-4] + b"\x00"))
    # The vectorized multi-row decoder applies the same checks.
    with pytest.raises(ValueError, match="exhausted"):
        codec.decompress_many([encoded.data, short])
    with pytest.raises(ValueError, match="unused payload"):
        omitting.decompress_many([padded, extra.data])


def test_header_corruption_is_rejected():
    codec, encoded = transmitting_stream()
    data = encoded.data
    cases = {
        "bad magic": b"XLYA" + data[4:],
        "unsupported stream version": rebuild(data, version=3),
        "unsupported stream flags": rebuild(data, flags=0x80),
        "impossible image dimensions": rebuild(data, width=0),
        "invalid payload length": rebuild(data, payload_bits=33),
        "truncated header": data[:20],
        "truncated payload": data[:-5] + data[-4:],
        "checksum mismatch": data[:-5] + bytes([data[-5] ^ 1]) + data[-4:],
        "reconstruction checksum": rebuild(data, recon_crc=0),
    }
    cases["impossible image dimensions (area)"] = rebuild(data, width=300, height=300)
    for message, stream in cases.items():
        with pytest.raises(ValueError, match=message.split(" (")[0]):
            codec.decompress(stream)
    with pytest.raises(ValueError, match="different model"):
        BitLayaCodec(constant_model(0.5)).decompress(data)
    assert MAX_PIXELS < 300 * 300


def test_zlib_payload_flag_roundtrip():
    codec = BitLayaCodec(random_model(13))
    for image in images(14, 2, (8, 8)):
        plain = codec.compress(image, 0.7)
        packed = codec.compress(image, 0.7, zlib_payload=True)
        assert packed.data[5] == FLAG_ZLIB
        np.testing.assert_array_equal(codec.decompress(packed.data), plain.reconstruction)


@pytest.mark.parametrize("features", [{}, {"plane_dim": 3, "row_dim": 2, "col_dim": 2}])
def test_range_coded_payload_changes_only_the_payload(features):
    codec = BitLayaCodec(random_model(20, **features))
    batch = images(21, 5)
    thresholds = [0.55, 0.7, 0.85, 0.99, 1.0]
    raw = codec.compress_many(batch, thresholds)
    ranged = codec.compress_many(batch, thresholds, payload="range")
    for plain, coded, image, threshold in zip(raw, ranged, batch, thresholds):
        assert coded.data[5] == FLAG_RANGE and coded.data[18:26] == codec.range_model_id
        np.testing.assert_array_equal(coded.reconstruction, plain.reconstruction)  # identical decisions
        assert coded.payload_bits == plain.payload_bits
        single = codec.compress(image, threshold, payload="range")
        assert single.data == coded.data  # scalar and batched paths agree
        np.testing.assert_array_equal(codec.decompress(coded.data), plain.reconstruction)
    for decoded, plain in zip(codec.decompress_many([c.data for c in ranged] + [r.data for r in raw]), raw + raw):
        np.testing.assert_array_equal(decoded, plain.reconstruction)  # mixed codings in one batch
    with pytest.raises(ValueError, match="different model"):
        codec.decompress(rebuild(ranged[1].data, model_id=codec.model_id))
    with pytest.raises(ValueError, match="range-coded"):
        reference_decode(codec.predictor, ranged[1].data)
    truncated = rebuild(ranged[0].data, payload=ranged[0].data[HEADER.size:-6])
    with pytest.raises(ValueError, match="exhausted|truncated|checksum"):
        codec.decompress(truncated)


def test_container_roundtrip_overhead_and_corruption():
    codec = BitLayaCodec(random_model(15, plane_dim=2))
    batch = images(16, 12)
    for threshold in (0.6, 0.9, 1.0):
        data, reconstructions = codec.compress_container(batch, threshold)
        singles = codec.compress_many(batch, [threshold] * len(batch))
        for single, reconstruction in zip(singles, reconstructions):
            np.testing.assert_array_equal(single.reconstruction, reconstruction)
        decoded = codec.decompress_container(data)
        for image, expected in zip(decoded, reconstructions):
            np.testing.assert_array_equal(image, expected)
        payload_bits = sum(s.payload_bits for s in singles)
        overhead = len(data) - (payload_bits + 7) // 8
        assert overhead == 30 + 6 * len(batch)
        bare, _ = codec.compress_container(batch, threshold, per_image_crc=False)
        assert len(bare) == len(data) - 4 * len(batch)
        np.testing.assert_array_equal(codec.decompress_container(bare)[3], reconstructions[3])
    with pytest.raises(ValueError, match="checksum"):
        codec.decompress_container(data[:40] + bytes([data[40] ^ 1]) + data[41:])
    with pytest.raises(ValueError, match="different model"):
        BitLayaCodec(random_model(99, plane_dim=2)).decompress_container(data)


def test_spatial_model_rejects_images_beyond_its_embeddings():
    codec = BitLayaCodec(random_model(17, row_dim=2, col_dim=2))
    with pytest.raises(ValueError, match="spatial"):
        codec.compress(np.zeros((2, 33), np.uint8), 0.9)


def test_exactness_guard_rejects_huge_weights():
    model = constant_model(0.0)
    with torch.no_grad():
        model.gru.weight_hh_l0.fill_(2.0 ** 30)
    with pytest.raises(ValueError, match="too large"):
        QuantizedPredictor(model)


def test_fresh_process_compress_decompress(tmp_path):
    model = random_model(18, plane_dim=2)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model)
    source = tmp_path / "source.png"
    image = images(19, 1, (6, 7))[0]
    Image.fromarray(image).save(source)
    stream, output = tmp_path / "image.blaya", tmp_path / "decoded.png"
    run = lambda *args: subprocess.run([sys.executable, "-m", "bitlaya", *map(str, args)],
                                       capture_output=True, text=True, timeout=120)
    first = run("compress", source, stream, "--checkpoint", checkpoint, "--threshold", 0.8)
    assert first.returncode == 0, first.stderr
    second = run("decompress", stream, output, "--checkpoint", checkpoint)
    assert second.returncode == 0, second.stderr
    expected = BitLayaCodec(model).compress(image, 0.8)
    assert stream.read_bytes() == expected.data
    np.testing.assert_array_equal(np.array(Image.open(output)), expected.reconstruction)


def test_pack_parse_header_roundtrip_is_34_bytes_of_overhead():
    header = StreamHeader(0, 32, 32, decision_threshold(0.9), 13, b"12345678", 42)
    data = pack_stream(header, np.ones(13, np.uint8))
    assert len(data) == OVERHEAD + 2 and OVERHEAD == 34
    parsed, bits = parse_stream(data)
    assert parsed == header and bits.tolist() == [1] * 13
    assert image_to_bits(np.zeros((1, 1), np.uint8)).size == 8

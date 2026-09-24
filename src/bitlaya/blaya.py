"""Compact binary ``.blaya`` streams (format v2) and a reusable codec object.

Decisions come from the integer-exact ``qgru-v1`` engine (see engine.py), so the
encoder, the single-image decoder and the batched decoder make identical choices
regardless of batch size, thread count or platform. The v1 JSON format and its
float reference codec (codec.py) are unchanged and still decodable.

Synchronization rule (same as v1): at bit t both sides compute the integer logit
L from the RECONSTRUCTED history. If ``|L| >= D * 2**(S-24)`` the predicted bit
``1 if L >= 0 else 0`` is used and nothing is transmitted -- even when it is
wrong; otherwise the next explicit payload bit is used. The reconstructed bit,
never the source bit, is fed to the next step. There is no omission mask.

Single-image stream layout (big-endian), 34 bytes of overhead::

    0   4  magic  b"BLYA"
    4   1  version = 2
    5   1  flags  (bit 0: payload is zlib-compressed; bit 1: explicit bits are
                   range-coded with model probabilities; at most one; others 0)
    6   2  width  (1..65535)
    8   2  height (1..65535), width*height <= MAX_PIXELS
    10  4  decision threshold D, logit units of 2**-24; 0xFFFFFFFF = transmit all
    14  4  explicit payload bit count
    18  8  model id: SHA-256 prefix of engine id, precision, quantized weights, tables
    26  4  CRC-32 of the reconstructed raster-order pixels
    30  n  payload: explicit bits MSB-first, zero-padded to a byte (or zlib of
           that, or range-coder bytes)
    30+n 4 CRC-32 of bytes [0, 30+n)

Container layout (many same-shape images, one threshold and model)::

    0   4  magic  b"BLYC"
    4   1  version = 1
    5   1  flags  (bit 0: per-image reconstruction CRC-32s present)
    6   2  width, 8 2 height, 10 4 D, 14 8 model id, 22 4 image count N
    26     N counts (uint16 if 8*width*height <= 65535, else uint32)
           N CRC-32s (if flag bit 0)
           one bitstream: all images' explicit bits concatenated, zero-padded
    end 4  CRC-32 of everything before

Per-image container overhead is 2 bytes (+4 with per-image CRCs) plus a 30-byte
header/trailer shared by the whole container. The shared model is never embedded.
"""
from dataclasses import dataclass, field
import hashlib
import math
import struct
import zlib

import numpy as np

from . import protocol
from .bits import bits_to_image, image_to_bits, pack_bits, unpack_bits, validate_image
from .engine import (BOS, QuantizedPredictor, Stepper, TRANSMIT_ALL, decision_threshold,
                     threshold_from_decision)
from .model import BitPredictor
from .rangecoder import PROB_ONE, RangeDecoder, RangeEncoder

MAGIC = b"BLYA"
VERSION = 2
FLAG_ZLIB = 0x01
FLAG_RANGE = 0x02
HEADER = struct.Struct(">4sBBHHII8sI")
TRAILER = 4
OVERHEAD = HEADER.size + TRAILER
MAX_PIXELS = 65536
CONTAINER_MAGIC = b"BLYC"
CONTAINER_VERSION = 1
CONTAINER_FLAG_CRC = 0x01
CONTAINER_HEADER = struct.Struct(">4sBBHHI8sI")


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


@dataclass
class StreamHeader:
    flags: int
    width: int
    height: int
    decision: int
    payload_bits: int
    model_id: bytes
    reconstruction_crc: int

    @property
    def shape(self):
        return self.height, self.width

    @property
    def threshold(self) -> float:
        return threshold_from_decision(self.decision)


@dataclass
class Encoded:
    data: bytes
    reconstruction: np.ndarray
    payload_bits: int
    decision: int
    trace: dict | None = field(default=None, repr=False)

    @property
    def header_bytes(self) -> int:
        return OVERHEAD

    @property
    def payload_bytes(self) -> int:
        return len(self.data) - OVERHEAD


def _check_shape(shape):
    height, width = shape
    if not (0 < width <= 0xFFFF and 0 < height <= 0xFFFF and width * height <= MAX_PIXELS):
        raise ValueError("impossible image dimensions")


def pack_stream(header: StreamHeader, explicit: np.ndarray | None, raw_payload: bytes | None = None) -> bytes:
    payload = pack_bits(explicit) if raw_payload is None else raw_payload
    flags = header.flags
    if flags & FLAG_ZLIB:
        payload = zlib.compress(payload, 9)
    body = HEADER.pack(MAGIC, VERSION, flags, header.width, header.height, header.decision,
                       header.payload_bits, header.model_id, header.reconstruction_crc) + payload
    return body + struct.pack(">I", crc32(body))


def parse_stream(data: bytes) -> tuple[StreamHeader, np.ndarray | bytes]:
    """Validate framing, lengths, padding and checksum; return header and explicit bits.

    For range-coded (lossless) streams the second value is the raw coder payload.
    """
    data = bytes(data)
    if len(data) < 4 or data[:4] != MAGIC:
        raise ValueError("bad magic: not a BitLaya v2 (.blaya) stream")
    if len(data) < OVERHEAD:
        raise ValueError("truncated header")
    magic, version, flags, width, height, decision, count, model_id, recon_crc = HEADER.unpack_from(data)
    if version != VERSION:
        raise ValueError(f"unsupported stream version {version}")
    if flags & ~(FLAG_ZLIB | FLAG_RANGE) or flags == FLAG_ZLIB | FLAG_RANGE:
        raise ValueError("unsupported stream flags")
    _check_shape((height, width))
    total = width * height * 8
    if count > total or (decision == TRANSMIT_ALL and count != total):
        raise ValueError("invalid payload length")
    payload = data[HEADER.size:-TRAILER]
    if not flags & (FLAG_ZLIB | FLAG_RANGE):
        expected = (count + 7) // 8
        if len(payload) < expected:
            raise ValueError("truncated payload")
        if len(payload) > expected:
            raise ValueError("excess payload bytes")
    if crc32(data[:-TRAILER]) != struct.unpack(">I", data[-TRAILER:])[0]:
        raise ValueError("stream checksum mismatch (corrupt or truncated data)")
    header = StreamHeader(flags, width, height, decision, count, model_id, recon_crc)
    if flags & FLAG_RANGE:
        return header, payload
    if flags & FLAG_ZLIB:
        try:
            payload = zlib.decompress(payload)
        except zlib.error as exc:
            raise ValueError("invalid compressed payload") from exc
        if len(payload) != (count + 7) // 8:
            raise ValueError("invalid payload length")
    return header, unpack_bits(payload, count)  # also rejects nonzero padding


PAYLOAD_CODINGS = ("raw", "zlib", "range")
LOGIT_STEPS, LOGIT_RANGE = 256, 16


def probability_table() -> np.ndarray:
    """P(bit = 0) in units of 2**-16 for logits -16..16 in steps of 1/256 (8,193 entries).

    ``p0 = clip(round(65536 / (1 + exp(i / 256))), 1, 65535)``: probabilities are
    never 0 or 1, so a wrongly confident bit costs at most 16 bits.
    """
    index = np.arange(-LOGIT_RANGE * LOGIT_STEPS, LOGIT_RANGE * LOGIT_STEPS + 1, dtype=np.float64)
    p0 = np.rint(PROB_ONE / (1.0 + np.exp(index / LOGIT_STEPS)))
    return np.clip(p0, 1, PROB_ONE - 1).astype(np.int64)


class BitLayaCodec:
    """Prepared once, reused for many images: no per-call copies or verification.

    Payload codings (the omission decisions and reconstruction are identical for all):

    * ``raw``   -- explicit bits packed MSB-first (default);
    * ``zlib``  -- zlib level 9 of the packed explicit bits (flag bit 0);
    * ``range`` -- each explicit bit range-coded with the model's probability at that
      step (flag bit 1). At threshold 1.0 this is full lossless neural coding.

    Range probabilities: the exact integer logit L is rounded to 1/256 logit and
    clamped to [-16, 16]: ``i = clip(floor(L / 2**(S-8) + 1/2), -4096, 4096)``, then
    ``p0 = TABLE[i]`` (see ``probability_table``). Range-coded streams carry
    ``range_model_id`` (engine fingerprint + table hash) instead of ``model_id``.
    """

    def __init__(self, model: BitPredictor | QuantizedPredictor, precision: dict | None = None):
        self.predictor = model if isinstance(model, QuantizedPredictor) else QuantizedPredictor(model, precision)
        self.model_id = self.predictor.model_id
        self.table = probability_table()
        self.table_offset = LOGIT_RANGE * LOGIT_STEPS
        self.table_shift = 2.0 ** -(self.predictor.S - 8)
        digest = hashlib.sha256(bytes.fromhex(self.predictor.fingerprint))
        digest.update(b"range-v1")
        digest.update(self.table.astype("<i8").tobytes())
        self.range_model_id = digest.digest()[:8]

    def p0(self, logits) -> np.ndarray:
        index = np.floor(np.asarray(logits, dtype=np.float64) * self.table_shift + 0.5)
        return self.table[np.clip(index, -self.table_offset, self.table_offset).astype(np.int64) + self.table_offset]

    def _p0_scalar(self, logit: float) -> int:
        index = math.floor(logit * self.table_shift + 0.5)
        return int(self.table[min(max(index, -self.table_offset), self.table_offset) + self.table_offset])

    # ------------------------------------------------------------------ core rows
    def encode_bits(self, sources: np.ndarray, shape, decisions, trace: bool = False, *,
                    model_rows=None) -> dict:
        """Encode rows of source bits (B, L) sharing one shape; decisions per row.

        Returns reconstructed bits, the omission mask and, with ``trace``, the exact
        integer-valued logits (scale 2**-S) of rows where the model ran. Transmit-all
        rows take the threshold-1 fast path (no model) unless listed in ``model_rows``
        (range coding needs probabilities even when nothing is omitted).
        """
        sources = np.asarray(sources, dtype=np.uint8)
        rows, length = sources.shape
        decisions = np.asarray(decisions, dtype=np.int64)
        forced = np.zeros(rows, dtype=bool) if model_rows is None else np.asarray(model_rows, dtype=bool)
        reconstructed = sources.copy()
        omitted = np.zeros((rows, length), dtype=bool)
        logits = np.full((rows, length), np.nan) if trace else None
        active = np.flatnonzero((decisions != TRANSMIT_ALL) | forced)
        if not len(active):
            return {"reconstructed": reconstructed, "omitted": omitted, "logits": logits}
        self.predictor.check_shape(shape)
        bounds = np.where(decisions[active] == TRANSMIT_ALL, np.inf,
                          decisions[active].astype(np.float64) * self.predictor.decision_scale)
        if len(active) == 1:
            # Single image: identical decisions with Python scalars (profiling showed per-bit
            # NumPy bookkeeping on 1-element arrays cost ~10-15% of single-image time).
            row, bound = int(active[0]), float(bounds[0])
            stepper = Stepper(self.predictor, 1, shape[1])
            symbol = np.array([BOS], dtype=np.intp)
            source, recon, omit_row = sources[row].tolist(), reconstructed[row], omitted[row]
            record = logits[row] if trace else None
            for t in range(length):
                logit = stepper.step(symbol, t)[0]
                if record is not None:
                    record[t] = logit
                if abs(logit) >= bound:
                    bit = 1 if logit >= 0 else 0
                    omit_row[t] = True
                else:
                    bit = source[t]
                recon[t] = bit
                symbol[0] = bit  # reconstructed bit, never the source after a wrong omission
            return {"reconstructed": reconstructed, "omitted": omitted, "logits": logits}
        source = sources[active]
        stepper = Stepper(self.predictor, len(active), shape[1])
        symbols = np.full(len(active), BOS, dtype=np.intp)
        recon = np.empty((len(active), length), dtype=np.uint8)
        omit_mask = np.empty((len(active), length), dtype=bool)
        record = np.empty((len(active), length)) if trace else None
        for t in range(length):
            logit = stepper.step(symbols, t)
            omit = np.abs(logit) >= bounds
            bit = np.where(omit, logit >= 0, source[:, t]).astype(np.uint8)
            recon[:, t] = bit
            omit_mask[:, t] = omit
            symbols[:] = bit  # reconstructed bit, never the source after a wrong omission
            if record is not None:
                record[:, t] = logit
        reconstructed[active] = recon
        omitted[active] = omit_mask
        if trace:
            logits[active] = record
        return {"reconstructed": reconstructed, "omitted": omitted, "logits": logits}

    def decode_bits(self, payloads: list, shape, decisions, *, counts=None, coders=None) -> np.ndarray:
        """Independent decoder for rows sharing one shape.

        Row r's explicit bits come from ``payloads[r]`` (a bit array) or, if
        ``coders[r]`` is set, from that range decoder using the model probability at
        each explicit step; ``counts[r]`` is then the header's explicit-bit count.
        """
        rows, length = len(payloads), shape[0] * shape[1] * 8
        decisions = np.asarray(decisions, dtype=np.int64)
        coders = coders or [None] * rows
        counts = np.array([len(p) if counts is None or coders[i] is None else counts[i]
                           for i, p in enumerate(payloads)], dtype=np.int64)
        coded = np.array([c is not None for c in coders], dtype=bool)
        output = np.empty((rows, length), dtype=np.uint8)
        for row in np.flatnonzero((decisions == TRANSMIT_ALL) & ~coded):
            if counts[row] != length:
                raise ValueError("invalid payload length")
            output[row] = payloads[row]
        active = np.flatnonzero((decisions != TRANSMIT_ALL) | coded)
        if not len(active):
            return output
        self.predictor.check_shape(shape)
        bounds = np.where(decisions[active] == TRANSMIT_ALL, np.inf,
                          decisions[active].astype(np.float64) * self.predictor.decision_scale)
        exhausted = "explicit payload exhausted; stream is corrupt or decoder diverged"
        if len(active) == 1:
            row, bound = int(active[0]), float(bounds[0])
            coder, limit, cursor = coders[row], int(counts[row]), 0
            explicit = None if coder is not None else payloads[row].tolist()
            stepper = Stepper(self.predictor, 1, shape[1])
            symbol = np.array([BOS], dtype=np.intp)
            bits = output[row]
            for t in range(length):
                logit = stepper.step(symbol, t)[0]
                if abs(logit) >= bound:
                    bit = 1 if logit >= 0 else 0
                else:
                    if cursor >= limit:
                        raise ValueError(exhausted)
                    bit = explicit[cursor] if coder is None else coder.decode(self._p0_scalar(logit))
                    cursor += 1
                bits[t] = bit
                symbol[0] = bit
            if cursor != limit:
                raise ValueError("unused payload bits; stream is corrupt or decoder diverged")
            if coder is not None:
                coder.finish()
            return output
        width = int(counts[active].max()) + 1
        matrix = np.zeros((len(active), width), dtype=np.uint8)
        for index, row in enumerate(active):
            if coders[row] is None:
                matrix[index, :counts[row]] = payloads[row]
        active_coders = [(index, coders[row]) for index, row in enumerate(active) if coders[row] is not None]
        limit = counts[active]
        cursor = np.zeros(len(active), dtype=np.int64)
        positions = np.arange(len(active))
        stepper = Stepper(self.predictor, len(active), shape[1])
        symbols = np.full(len(active), BOS, dtype=np.intp)
        for t in range(length):
            logit = stepper.step(symbols, t)
            omit = np.abs(logit) >= bounds
            need = ~omit
            if np.any(need & (cursor >= limit)):
                raise ValueError(exhausted)
            explicit = matrix[positions, np.minimum(cursor, width - 1)]
            for index, coder in active_coders:
                if need[index]:
                    explicit[index] = coder.decode(self._p0_scalar(logit[index]))
            bit = np.where(omit, logit >= 0, explicit).astype(np.uint8)
            cursor += need
            output[active, t] = bit
            symbols[:] = bit
        if np.any(cursor != limit):
            raise ValueError("unused payload bits; stream is corrupt or decoder diverged")
        for _, coder in active_coders:
            coder.finish()
        return output

    # --------------------------------------------------------------- single files
    def compress(self, image: np.ndarray, threshold: float = 0.95, *, payload: str = "raw",
                 zlib_payload: bool = False, trace: bool = False) -> Encoded:
        return self.compress_many([image], [threshold], payload=payload, zlib_payload=zlib_payload,
                                  trace=trace)[0]

    def range_payload(self, explicit: np.ndarray, logits: np.ndarray) -> bytes:
        """Range-code explicit bits with the probabilities of their (integer) logits."""
        encoder = RangeEncoder()
        for bit, p0 in zip(np.asarray(explicit).tolist(), self.p0(logits).tolist()):
            encoder.encode(bit, p0)
        return encoder.finish()

    def as_range_coded(self, encoded: Encoded) -> bytes:
        """The ``range`` stream for an encoding traced with the model run on every row.

        Same decisions and reconstruction; only the payload coding differs. Equals
        ``compress(..., payload="range").data`` (tested).
        """
        trace = encoded.trace
        if trace is None or np.isnan(trace["integer_logits"]).any():
            raise ValueError("needs compress_many(..., trace=True, run_model=True)")
        omitted = trace["omitted"]
        explicit = trace["source_bits"][~omitted]
        header, _ = parse_stream(encoded.data)
        header.flags, header.model_id = FLAG_RANGE, self.range_model_id
        return pack_stream(header, explicit, self.range_payload(explicit, trace["integer_logits"][~omitted]))

    def compress_many(self, images, thresholds, *, payload: str = "raw", zlib_payload: bool = False,
                      trace: bool = False, run_model: bool = False) -> list:
        """Encode (image, threshold) rows; same-shape rows are stepped together.

        ``run_model`` evaluates the model even on threshold-1 rows (for traces).
        """
        if zlib_payload:
            payload = "zlib"
        if payload not in PAYLOAD_CODINGS:
            raise ValueError(f"payload coding must be one of {PAYLOAD_CODINGS}")
        if len(images) != len(thresholds):
            raise ValueError("images and thresholds must have equal length")
        images = [validate_image(image) for image in images]
        for image in images:
            _check_shape(image.shape)
        decisions = [decision_threshold(t) for t in thresholds]
        ranged = payload == "range"
        flags = {"raw": 0, "zlib": FLAG_ZLIB, "range": FLAG_RANGE}[payload]
        model_id = self.range_model_id if ranged else self.model_id
        results = [None] * len(images)
        by_shape = {}
        for index, image in enumerate(images):
            by_shape.setdefault(image.shape, []).append(index)
        for shape, indices in by_shape.items():
            sources = np.stack([image_to_bits(images[i]) for i in indices])
            coded = self.encode_bits(sources, shape, [decisions[i] for i in indices], trace or ranged,
                                     model_rows=[ranged or run_model] * len(indices))
            for row, index in enumerate(indices):
                recon_bits = coded["reconstructed"][row]
                omitted = coded["omitted"][row]
                explicit = sources[row][~omitted]
                reconstruction = bits_to_image(recon_bits, shape)
                header = StreamHeader(flags, shape[1], shape[0], decisions[index], len(explicit), model_id,
                                      crc32(reconstruction.tobytes()))
                raw_payload = self.range_payload(explicit, coded["logits"][row][~omitted]) if ranged else None
                row_trace = None
                if trace:
                    row_trace = {"omitted": omitted, "logits": coded["logits"][row] * 2.0 ** -self.predictor.S,
                                 "integer_logits": coded["logits"][row],
                                 "source_bits": sources[row], "reconstructed_bits": recon_bits}
                results[index] = Encoded(pack_stream(header, explicit, raw_payload), reconstruction, len(explicit),
                                         decisions[index], row_trace)
        return results

    def decompress(self, data: bytes) -> np.ndarray:
        return self.decompress_many([data])[0]

    def decompress_many(self, streams) -> list:
        parsed = [parse_stream(data) for data in streams]
        for header, _ in parsed:
            expected = self.range_model_id if header.flags & FLAG_RANGE else self.model_id
            if header.model_id != expected:
                raise ValueError("stream requires a different model (model id mismatch)")
        results = [None] * len(parsed)
        by_shape = {}
        for index, (header, _) in enumerate(parsed):
            by_shape.setdefault(header.shape, []).append(index)
        for shape, indices in by_shape.items():
            coders = [RangeDecoder(parsed[i][1]) if parsed[i][0].flags & FLAG_RANGE else None for i in indices]
            bits = self.decode_bits([None if coders[k] else parsed[i][1] for k, i in enumerate(indices)], shape,
                                    [parsed[i][0].decision for i in indices],
                                    counts=[parsed[i][0].payload_bits for i in indices], coders=coders)
            for row, index in enumerate(indices):
                image = bits_to_image(bits[row], shape)
                if crc32(image.tobytes()) != parsed[index][0].reconstruction_crc:
                    raise ValueError("reconstruction checksum mismatch; decoder diverged")
                results[index] = image
        return results

    # ----------------------------------------------------------------- containers
    def compress_container(self, images, threshold: float, *, per_image_crc: bool = True) -> tuple[bytes, list]:
        images = [validate_image(image) for image in images]
        if not images or len({image.shape for image in images}) != 1:
            raise ValueError("a container holds one or more images of a single shape")
        shape = images[0].shape
        _check_shape(shape)
        decision = decision_threshold(threshold)
        sources = np.stack([image_to_bits(image) for image in images])
        coded = self.encode_bits(sources, shape, [decision] * len(images))
        explicit = [sources[row][~coded["omitted"][row]] for row in range(len(images))]
        reconstructions = [bits_to_image(bits, shape) for bits in coded["reconstructed"]]
        count_format = ">H" if shape[0] * shape[1] * 8 <= 0xFFFF else ">I"
        flags = CONTAINER_FLAG_CRC if per_image_crc else 0
        parts = [CONTAINER_HEADER.pack(CONTAINER_MAGIC, CONTAINER_VERSION, flags, shape[1], shape[0],
                                       decision, self.model_id, len(images))]
        parts += [struct.pack(count_format, len(bits)) for bits in explicit]
        if per_image_crc:
            parts += [struct.pack(">I", crc32(image.tobytes())) for image in reconstructions]
        parts.append(pack_bits(np.concatenate(explicit) if explicit else np.zeros(0, np.uint8)))
        body = b"".join(parts)
        return body + struct.pack(">I", crc32(body)), reconstructions

    def decompress_container(self, data: bytes) -> list:
        data = bytes(data)
        if len(data) < 4 or data[:4] != CONTAINER_MAGIC:
            raise ValueError("bad magic: not a BitLaya container")
        if len(data) < CONTAINER_HEADER.size + TRAILER:
            raise ValueError("truncated header")
        _, version, flags, width, height, decision, model_id, count = CONTAINER_HEADER.unpack_from(data)
        if version != CONTAINER_VERSION:
            raise ValueError(f"unsupported container version {version}")
        if flags & ~CONTAINER_FLAG_CRC:
            raise ValueError("unsupported container flags")
        shape = (height, width)
        _check_shape(shape)
        if crc32(data[:-TRAILER]) != struct.unpack(">I", data[-TRAILER:])[0]:
            raise ValueError("container checksum mismatch (corrupt or truncated data)")
        if model_id != self.model_id:
            raise ValueError("container requires a different model (model id mismatch)")
        total = width * height * 8
        size = 2 if total <= 0xFFFF else 4
        offset = CONTAINER_HEADER.size
        table_end = offset + count * (size + (4 if flags & CONTAINER_FLAG_CRC else 0))
        if table_end > len(data) - TRAILER:
            raise ValueError("truncated container index")
        counts = [int.from_bytes(data[offset + i * size: offset + (i + 1) * size], "big") for i in range(count)]
        offset += count * size
        crcs = None
        if flags & CONTAINER_FLAG_CRC:
            crcs = [struct.unpack_from(">I", data, offset + 4 * i)[0] for i in range(count)]
            offset += 4 * count
        if any(c > total for c in counts) or (decision == TRANSMIT_ALL and any(c != total for c in counts)):
            raise ValueError("invalid payload length")
        bits = unpack_bits(data[offset:-TRAILER], sum(counts))
        payloads = np.split(bits, np.cumsum(counts)[:-1]) if count else []
        decoded = self.decode_bits(payloads, shape, [decision] * count)
        images = [bits_to_image(row, shape) for row in decoded]
        if crcs is not None:
            for image, expected in zip(images, crcs):
                if crc32(image.tobytes()) != expected:
                    raise ValueError("reconstruction checksum mismatch; decoder diverged")
        return images


# ---------------------------------------------------------------------- reference
class _ReferencePredict:
    """Adapts the allocating reference step to the protocol's predict callback."""

    def __init__(self, predictor: QuantizedPredictor, width: int):
        self.predictor, self.width, self.state = predictor, width, predictor.initial_state()

    def __call__(self, t, symbol):
        logit, self.state = self.predictor.step_reference(symbol, t, self.width, self.state)
        return logit


def _bound(predictor, decision):
    return None if decision == TRANSMIT_ALL else decision * predictor.decision_scale


def reference_encode(predictor: QuantizedPredictor, image: np.ndarray, threshold: float) -> Encoded:
    """Straightforward scalar encoder: the specification the optimized paths must match."""
    image = validate_image(image)
    _check_shape(image.shape)
    decision = decision_threshold(threshold)
    if decision != TRANSMIT_ALL:
        predictor.check_shape(image.shape)
    explicit, reconstructed, _ = protocol.encode(image_to_bits(image), _ReferencePredict(predictor, image.shape[1]),
                                                 _bound(predictor, decision))
    reconstruction = bits_to_image(reconstructed, image.shape)
    header = StreamHeader(0, image.shape[1], image.shape[0], decision, len(explicit), predictor.model_id,
                          crc32(reconstruction.tobytes()))
    return Encoded(pack_stream(header, explicit), reconstruction, len(explicit), decision)


def reference_decode(predictor: QuantizedPredictor, data: bytes) -> np.ndarray:
    header, explicit = parse_stream(data)
    if header.flags & FLAG_RANGE:
        raise ValueError("range-coded lossless stream: decode with lossless.LosslessCodec")
    if header.model_id != predictor.model_id:
        raise ValueError("stream requires a different model (model id mismatch)")
    shape = header.shape
    bits = protocol.decode(explicit, shape[0] * shape[1] * 8, _ReferencePredict(predictor, shape[1]),
                           _bound(predictor, header.decision))
    image = bits_to_image(bits, shape)
    if crc32(image.tobytes()) != header.reconstruction_crc:
        raise ValueError("reconstruction checksum mismatch; decoder diverged")
    return image
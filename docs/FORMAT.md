# BitLaya stream formats

Two formats exist. **v2 (`.blaya`, binary)** is the production/research format.
**v1 (`.blay`, JSON header)** is preserved unchanged for backward compatibility and
as the pre-improvement reference; v1 files remain decodable with `bitlaya decode`.

Neither format embeds the model. The checkpoint is a shared, out-of-band cost;
experiment summaries report it separately as an amortized cost over N images.

## v2 single-image stream (`BLYA`)

Big-endian, fixed-size fields. 34 bytes of overhead per image (v1: ~415-420).

| Offset | Size | Field | Notes |
| ---: | ---: | --- | --- |
| 0 | 4 | magic | `42 4c 59 41` (`BLYA`) |
| 4 | 1 | version | `2` |
| 5 | 1 | flags | bit 0: payload is zlib-compressed. Other bits must be 0. |
| 6 | 2 | width | 1..65535 |
| 8 | 2 | height | 1..65535; width x height <= 65,536 |
| 10 | 4 | decision threshold D | `ceil(log(t/(1-t)) * 2**24)`; `0xFFFFFFFF` = transmit all (t = 1.0) |
| 14 | 4 | explicit payload bit count | <= 8 x width x height; must equal it when D = `0xFFFFFFFF` |
| 18 | 8 | model id | first 8 bytes of the exact engine's SHA-256 fingerprint |
| 26 | 4 | reconstruction CRC-32 | of the decoded raster-order pixels |
| 30 | n | payload | explicit bits, MSB first, zero-padded to a byte (or zlib of that) |
| 30+n | 4 | stream CRC-32 | of bytes `[0, 30+n)` |

**Decision rule** (encoder and decoder): at bit t compute the exact integer logit
`L` (scale `2**-S`, S = 36 for the default engine precision) from the
*reconstructed* history. Omit iff `|L| >= D * 2**(S-24)`; predicted bit is
`1 if L >= 0 else 0` (a tie predicts 1). Otherwise consume the next payload bit.
Feed the resulting reconstructed bit (never the source bit) to the next step.
Storing D (an integer) rather than the float threshold means the decoder never
repeats a floating-point log. `bitlaya inspect` shows the approximate threshold.

**Model id.** SHA-256 over the engine id (`qgru-v1`), its precision parameters,
the model's architecture identity, all quantized weights, and the sigmoid/tanh
lookup tables; the first 8 bytes are stored. A different checkpoint, precision,
or a platform whose math library would build different tables yields a
different id and is rejected ("different model"), never silently desynchronized.

**Validation order and errors.** bad magic -> truncated header -> unsupported
version -> unsupported flags -> impossible dimensions -> invalid payload length
-> truncated payload / excess payload bytes -> stream checksum mismatch ->
(zlib) invalid compressed payload -> nonzero padding -> model id mismatch ->
payload exhausted during decoding -> unused payload bits after decoding ->
reconstruction checksum mismatch. **Excess payload is invalid**: the payload
must be exactly `ceil(count/8)` bytes (uncompressed), padding must be zero, and
every counted bit must be consumed.

**Threshold 1.0 fast path.** D = `0xFFFFFFFF` means every bit is explicit, so the
payload is exactly the raw raster bytes and neither side evaluates the model
(the model id is still checked). This is bit-identical to running the model.

CRC-32 detects accidental corruption; it is not authentication.

## v2 container (`BLYC`)

For many same-shape images sharing one threshold and model.

| Field | Size |
| --- | --- |
| magic `BLYC`, version `1`, flags (bit 0: per-image reconstruction CRCs) | 6 |
| width, height (uint16 each) | 4 |
| decision threshold D | 4 |
| model id | 8 |
| image count N | 4 |
| N payload bit counts (uint16 if 8·w·h <= 65535, else uint32) | 2N (or 4N) |
| N reconstruction CRC-32s (if flag) | 4N |
| one bitstream: all explicit bits concatenated, zero-padded once | ceil(total/8) |
| container CRC-32 | 4 |

Per-image overhead: **6 bytes** with per-image CRCs, **2 bytes** without, plus a
30-byte header/trailer shared by the container. Bits are concatenated without
per-image byte padding (saves ~0.44 bytes/image on average).

## v1 stream (legacy, `.blay`)

| Field | Size | Encoding |
| --- | --- | --- |
| Magic/version | 8 bytes | `42 4c 41 59 01 0d 0a 00` |
| Header length | 4 bytes | Unsigned big-endian integer |
| Header | Header length bytes | Canonical UTF-8 JSON |
| Explicit payload | `ceil(explicit_bits / 8)` bytes | MSB first, zero low-bit padding |
| Stream digest | 32 bytes | SHA-256 of every preceding byte |

JSON fields: `shape`, `bit_order`, `threshold`, `explicit_bits`, `model_sha256`,
`reconstruction_sha256`, `runtime` (PyTorch version/build, OS, architecture,
backend id). The v1 decoder clones the model to CPU float32, one thread, MKLDNN
off, and runs PyTorch's GRU one step per bit. Float inference is only
deterministic within an identical runtime, which is why v1 pins the runtime
signature. Decisions: predict 1 iff logit >= 0; confidence = `1/(1+exp(-|z|))`;
omit iff threshold < 1 and confidence >= threshold.

v1 checkpoints keep their fingerprints: the model identity omits the optional
position-feature fields when they are disabled.

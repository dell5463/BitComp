# Exact fixed-point inference engine (`qgru-v1`)

The v2 codec's predictions come from an integer-exact re-implementation of the
trained float GRU (`src/bitlaya/engine.py`). Training stays in float32.

## Why not float inference

Omission decisions compare a confidence with a threshold. If encoder and decoder
compute even slightly different logits, a decision can flip, the decoder consumes
the wrong payload bit, and every later bit is wrong. Measured on this machine
(PyTorch 2.14 CPU, 32 validation images, legacy best checkpoint):

| Comparison | Result |
| --- | --- |
| float32 vs float64, same model | max logit difference 0.0029 (model is not chaotic) |
| float32 batch of 32 vs the same images one at a time | only **12.3%** of logits bit-identical; max diff 0.0092 |

So a float encoder that batches images (as a fast evaluator must) cannot be
guaranteed to agree with a float decoder that decodes one image. The v1 format
avoids this only by pinning one PyTorch build, platform, thread count and a
batch size of one.

## How exactness is achieved

Weights, embeddings and activations are integers at scale `2**-F`; accumulators
are integers at `2**-2F`. At build time the engine computes, from the actual
quantized weights, a bound on every intermediate magnitude and refuses the model
unless all stay below `2**52`. Every float64 operation on such integers - sums,
products, BLAS matmuls in any blocking order, with or without FMA - is then
exact, so results cannot depend on batch size, thread count, BLAS library or CPU.
Sigmoid and tanh are lookup tables indexed by an exact integer; the tables are
hashed into the model id. See the step protocol in the `engine.py` docstring.

Two implementations share the protocol and are tested to agree bit-for-bit:
`QuantizedPredictor.step_reference` (plain, allocating, one image) and `Stepper`
(preallocated, batched). The codec's reference encoder/decoder
(`blaya.reference_encode/decode`, built on `protocol.py`) must produce byte-
identical streams to the optimized `BitLayaCodec`.

## Precision choice (measured)

Teacher-forced logits vs the float32 model, 32 validation images x 8,192 bits,
legacy best checkpoint (float32 BCE 0.593739 nats/bit):

| F | sigmoid table (T, range) | tanh table (T, range) | BCE | max abs logit diff | diffs > 0.1 | sign disagreements |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 14 | 2^-10, +-16 | 2^-12, +-8 | 0.593762 | 2.90 | 143 | 88 |
| 16 | 2^-10, +-16 | 2^-12, +-8 | 0.593733 | 0.75 | 28 | 32 |
| 16 | 2^-12, +-16 | 2^-14, +-8 | 0.593734 | 0.59 | 15 | 13 |
| **18** | **2^-14, +-16** | **2^-15, +-8** | **0.593740** | **0.038** | **0** | **2** |
| 20 | 2^-16, +-12 | 2^-16, +-8 | 0.593738 | 0.23 | 9 | 3 |

Hidden-state precision (F) dominates; beyond F = 18 the residual differences
come from one sensitive trajectory region and are not monotone in precision.
BCE is equal to float32 at every setting. Default: F = 18 (tables 8.4 MB).

The quantized model is a slightly different predictor from the float model. That
is harmless for the codec (encoder and decoder both use it) and the measured
train/inference gap is ~1e-6 nats/bit. `bitlaya diagnose` reports both.

## Cost

Per bit (2-layer, 128 hidden): three 128x384 integer-valued float64 matvecs,
one 128-element dot product, table lookups instead of the layer-0 input matmul,
and ~40 small NumPy operations. Measured speeds are in the profiling results
(`results/profile_*`). It is faster than the float reference because the
reference pays PyTorch module dispatch and GRU-kernel overhead per bit.

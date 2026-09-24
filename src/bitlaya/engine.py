"""Deterministic fixed-point GRU inference: engine ``qgru-v1``.

Why: float GRU results change with batch size, thread count, BLAS kernel and
platform (measured: float32 logits for the same image are bit-identical in only
~12% of positions between batch 1 and batch 32). A float encoder/decoder pair is
therefore only synchronized within one exact runtime and one batch layout; the
v1 format pins the PyTorch build for this reason. This engine instead uses
integer-valued arithmetic that is EXACT in IEEE-754 float64:

* weights, embeddings and activations are integers at scale ``2**-F``;
* every accumulator is an integer at scale ``2**-(2F)`` whose magnitude is proven
  at build time, from the actual quantized weights, to stay below ``2**52``. All
  products and sums -- including BLAS matmuls in any order, with or without FMA
  -- are then exact, so results cannot depend on batch size, threads, BLAS or CPU;
* nonlinearities are lookup tables indexed by an exact integer. Table contents
  are hashed into the model identity: a platform whose libm built a different
  table would produce a different model id and be rejected, never desynchronized;
* the output logit is an exact integer ``L`` (logit = L * 2**-2F), so omission
  decisions (and arithmetic-coder probabilities) are integer comparisons/lookups.

Step protocol for one image (``h[l]`` integer vectors at scale 2**-F, zero at start):

    x0 = BIT[sym] + PLANE[t % 8] + ROW[row] + COL[col]      # layer-0 input gates, scale 2**-2F
    for each layer l:
        gi = x0 if l == 0 else h[l-1] @ Wih[l] + bih[l]      # scale 2**-2F
        gh = h[l] @ Whh[l] + bhh[l]                          # scale 2**-2F
        r, z = SIG(gi[:2H] + gh[:2H])                        # scale 2**-F
        n = TANH(gi[2H:] + r * round(gh[2H:] / 2**F))        # scale 2**-F
        h[l] = n + round(z * (h[l] - n) / 2**F)
    L = h[-1] @ w_head + b_head                              # scale 2**-2F

``round(v) = floor(v + 1/2)``. ``SIG``/``TANH`` map an accumulator ``a`` (scale
2**-2F) to ``TABLE[clip(round(a / 2**(2F-T)) + R*2**T, 0, 2*R*2**T)]`` where T is
the table's fractional bits and R its input half-range. Gate order (r, z, n) and
the n-gate formula match PyTorch's GRU.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch

from .model import BOS, BitPredictor, model_fingerprint

ENGINE_ID = "qgru-v1"
# Chosen by measurement (docs/ENGINE.md): F=18 kept teacher-forced BCE equal to
# float32 to 6 digits and max |logit - float logit| at 0.038 on 32 validation images.
DEFAULT_PRECISION = {"F": 18, "sig_t": 14, "sig_r": 16, "tanh_t": 15, "tanh_r": 8}
EXACT_LIMIT = 2.0 ** 52      # margin below float64's 2**53 exact-integer range
TRANSMIT_ALL = 0xFFFFFFFF    # decision-threshold sentinel: threshold 1.0, lossless
MAX_DECISION = 0xFFFFFFFE
DECISION_FRAC = 24           # stored decision threshold D is in units of 2**-24 logit


def _quantize(values, frac) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("model contains non-finite weights")
    return np.rint(array * 2.0 ** frac)  # exact: power-of-two scale, ties to even


def _table(function, frac, span, out_frac):
    offset = span << frac
    x = (np.arange(2 * offset + 1, dtype=np.float64) - offset) / 2.0 ** frac
    return np.rint(function(x) * 2.0 ** out_frac), offset


def decision_threshold(threshold: float) -> int:
    """Stored integer decision threshold ``D = ceil(log(t / (1 - t)) * 2**24)``.

    Omission rule (the codec's single comparison, used by encoder and decoder):
    omit iff ``|L| >= D * 2**(S - 24)`` where L is the exact integer logit at
    scale 2**-S. For thresholds representable this way this is exactly
    ``confidence >= threshold`` up to the 2**-24 quantization of the logit bound
    (``confidence = sigmoid(|logit|) >= t  <=>  |logit| >= log(t / (1 - t))``).
    The encoder computes D once and the stream stores D, so decoders never repeat
    this floating-point computation. 1.0 maps to the transmit-all sentinel.
    """
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.5 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0.5, 1.0]")
    if threshold == 1.0:
        return TRANSMIT_ALL
    value = math.ceil(math.log(threshold / (1.0 - threshold)) * 2.0 ** DECISION_FRAC)
    return min(max(value, 0), MAX_DECISION)


def threshold_from_decision(decision: int) -> float:
    """Approximate confidence threshold for display only."""
    if decision == TRANSMIT_ALL:
        return 1.0
    return 1.0 / (1.0 + math.exp(-decision / 2.0 ** DECISION_FRAC))


@dataclass(frozen=True)
class _Layer:
    wih: np.ndarray | None   # (in, 3H); None for layer 0 (uses feature tables)
    bih: np.ndarray | None
    whh: np.ndarray          # (H, 3H)
    bhh: np.ndarray


class QuantizedPredictor:
    """Immutable integer-exact snapshot of a float ``BitPredictor``."""

    def __init__(self, model: BitPredictor, precision: dict | None = None):
        precision = dict(DEFAULT_PRECISION if precision is None else precision)
        if set(precision) != set(DEFAULT_PRECISION):
            raise ValueError("invalid precision specification")
        self.precision = precision
        f = self.F = precision["F"]
        s = self.S = 2 * f
        config = self.config = model.config
        self.hidden = config.hidden_size
        state = {k: v.detach().cpu().to(torch.float32).numpy() for k, v in model.state_dict().items()}
        e, p, r, c = config.embedding_dim, config.plane_dim, config.row_dim, config.col_dim
        wih0 = _quantize(state["gru.weight_ih_l0"], f).astype(np.int64)

        def project(key, lo, hi):  # exact int64 embedding @ weight products
            return (_quantize(state[key], f).astype(np.int64) @ wih0[:, lo:hi].T).astype(np.float64)

        self.bit_table = project("embedding.weight", 0, e) + _quantize(state["gru.bias_ih_l0"], s)
        self.plane_table = project("plane_embedding.weight", e, e + p) if p else None
        self.row_table = project("row_embedding.weight", e + p, e + p + r) if r else None
        self.col_table = project("col_embedding.weight", e + p + r, e + p + r + c) if c else None
        self.layers = []
        for index in range(config.num_layers):
            whh = np.ascontiguousarray(_quantize(state[f"gru.weight_hh_l{index}"], f).T)
            bhh = _quantize(state[f"gru.bias_hh_l{index}"], s)
            wih = bih = None
            if index:
                wih = np.ascontiguousarray(_quantize(state[f"gru.weight_ih_l{index}"], f).T)
                bih = _quantize(state[f"gru.bias_ih_l{index}"], s)
            self.layers.append(_Layer(wih, bih, whh, bhh))
        self.head_w = _quantize(state["head.weight"][0], f)
        self.head_b = float(_quantize(state["head.bias"][0], s))
        self.sig, self.sig_offset = _table(lambda x: 1.0 / (1.0 + np.exp(-x)), precision["sig_t"],
                                           precision["sig_r"], f)
        self.tanh, self.tanh_offset = _table(np.tanh, precision["tanh_t"], precision["tanh_r"], f)
        if s < DECISION_FRAC:
            raise ValueError("precision too low for the stored decision threshold")
        self.decision_scale = 2.0 ** (s - DECISION_FRAC)
        self.sig_shift = 2.0 ** -(s - precision["sig_t"])
        self.tanh_shift = 2.0 ** -(s - precision["tanh_t"])
        self._check_exactness()
        digest = hashlib.sha256(json.dumps({"engine": ENGINE_ID, "precision": precision,
                                            "model": config.identity()}, sort_keys=True).encode())
        for array in self._arrays():
            digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        self.fingerprint = digest.hexdigest()
        self.model_id = digest.digest()[:8]
        self.float_fingerprint = model_fingerprint(model)

    def _arrays(self):
        yield from (self.bit_table, self.sig, self.tanh, self.head_w, np.array([self.head_b]))
        for table in (self.plane_table, self.row_table, self.col_table):
            if table is not None:
                yield table
        for layer in self.layers:
            yield from (a for a in (layer.wih, layer.bih, layer.whh, layer.bhh) if a is not None)

    def _check_exactness(self):
        """Prove every intermediate integer stays below 2**52 in magnitude."""
        one, h = 2.0 ** self.F, self.hidden  # |h|, |r|, |z|, |n| <= 1 at scale 2**-F
        bound = np.abs(self.bit_table).max(axis=0)
        for table in (self.plane_table, self.row_table, self.col_table):
            if table is not None:
                bound = bound + np.abs(table).max(axis=0)
        worst = [bound.max()]
        for index, layer in enumerate(self.layers):
            gi = bound if index == 0 else np.abs(layer.wih).sum(axis=0) * one + np.abs(layer.bih)
            gh = np.abs(layer.whh).sum(axis=0) * one + np.abs(layer.bhh)
            # r * round(gh/2**F) <= gh + 2**F; (h - n) * z <= 2**(2F+1)
            worst += [gi.max(), gh.max(), (gi[:2 * h] + gh[:2 * h]).max(),
                      (gi[2 * h:] + gh[2 * h:] + one).max(), 2 * one * one]
        worst.append(np.abs(self.head_w).sum() * one + abs(self.head_b))
        if max(worst) >= EXACT_LIMIT:
            raise ValueError("weights too large for exact qgru-v1 fixed-point inference")

    def check_shape(self, shape):
        height, width = shape
        if self.config.spatial and (height > self.config.max_height or width > self.config.max_width):
            raise ValueError("image dimensions exceed the model's spatial embeddings")

    def initial_state(self) -> list:
        return [np.zeros(self.hidden) for _ in self.layers]

    # ---------------------------------------------------------------- reference
    def input_gates(self, symbol: int, t: int, width: int) -> np.ndarray:
        gates = self.bit_table[symbol]
        pixel = t // 8
        if self.plane_table is not None:
            gates = gates + self.plane_table[t % 8]
        if self.row_table is not None:
            gates = gates + self.row_table[pixel // width]
        if self.col_table is not None:
            gates = gates + self.col_table[pixel % width]
        return gates

    def _lookup(self, table, offset, shift, accumulator):
        index = np.floor(accumulator * shift + 0.5) + offset
        return table[np.clip(index, 0, 2 * offset).astype(np.intp)]

    def step_reference(self, symbol: int, t: int, width: int, state: list) -> tuple[int, list]:
        """Plain, allocating implementation of the protocol (one image)."""
        h, f = self.hidden, self.F
        new_state, x = [], None
        for index, layer in enumerate(self.layers):
            gi = self.input_gates(symbol, t, width) if index == 0 else x @ layer.wih + layer.bih
            gh = state[index] @ layer.whh + layer.bhh
            rz = self._lookup(self.sig, self.sig_offset, self.sig_shift, gi[:2 * h] + gh[:2 * h])
            r, z = rz[:h], rz[h:]
            ghn = np.floor(gh[2 * h:] * 2.0 ** -f + 0.5)
            n = self._lookup(self.tanh, self.tanh_offset, self.tanh_shift, gi[2 * h:] + r * ghn)
            x = n + np.floor(z * (state[index] - n) * 2.0 ** -f + 0.5)
            new_state.append(x)
        return int(x @ self.head_w + self.head_b), new_state


class Stepper:
    """Optimized batched implementation of the same protocol (bit-identical).

    Rows are independent sequences sharing the step index ``t`` and image width.
    Buffers are preallocated; profiling showed ``np.clip`` and ``take(out=)`` cost
    4-6x more than ``maximum``/``minimum`` and fancy indexing on small arrays.
    """

    def __init__(self, predictor: QuantizedPredictor, batch: int, width: int):
        self.p = predictor
        self.batch, self.width = batch, width
        h = predictor.hidden
        self.h = [np.zeros((batch, h)) for _ in predictor.layers]
        self.gi = np.empty((batch, 3 * h))
        self.gh = np.empty((batch, 3 * h))
        self.rz = np.empty((batch, 2 * h))
        self.idx2 = np.empty((batch, 2 * h), dtype=np.intp)
        self.idx1 = np.empty((batch, h), dtype=np.intp)
        self.logits = np.empty(batch)
        self.inv = 2.0 ** -predictor.F
        self.sig_add = predictor.sig_offset + 0.5
        self.tanh_add = predictor.tanh_offset + 0.5
        self.sig_max = float(2 * predictor.sig_offset)
        self.tanh_max = float(2 * predictor.tanh_offset)

    def reset(self, rows=None):
        for state in self.h:
            if rows is None:
                state.fill(0.0)
            else:
                state[rows] = 0.0

    def step(self, symbols: np.ndarray, t: int) -> np.ndarray:
        """Advance every row one bit; return integer-valued logits (float64 view, reused)."""
        p, h = self.p, self.p.hidden
        gi, gh, rz = self.gi, self.gh, self.rz
        np.take(p.bit_table, symbols, axis=0, out=gi)
        pixel = t // 8
        if p.plane_table is not None:
            gi += p.plane_table[t % 8]
        if p.row_table is not None:
            gi += p.row_table[pixel // self.width]
        if p.col_table is not None:
            gi += p.col_table[pixel % self.width]
        for index, layer in enumerate(p.layers):
            state = self.h[index]
            if index:
                np.matmul(self.h[index - 1], layer.wih, out=gi)
                gi += layer.bih
            np.matmul(state, layer.whh, out=gh)
            gh += layer.bhh
            np.add(gi[:, :2 * h], gh[:, :2 * h], out=rz)
            rz *= p.sig_shift
            rz += self.sig_add
            np.maximum(rz, 0.0, out=rz)          # clip before truncation == floor for v >= 0
            np.minimum(rz, self.sig_max, out=rz)
            self.idx2[...] = rz
            rz = p.sig[self.idx2]
            r, z = rz[:, :h], rz[:, h:]
            ghn = gh[:, 2 * h:]
            ghn *= self.inv
            ghn += 0.5
            np.floor(ghn, out=ghn)
            ghn *= r
            ghn += gi[:, 2 * h:]
            ghn *= p.tanh_shift
            ghn += self.tanh_add
            np.maximum(ghn, 0.0, out=ghn)
            np.minimum(ghn, self.tanh_max, out=ghn)
            self.idx1[...] = ghn
            n = p.tanh[self.idx1]
            state -= n
            state *= z
            state *= self.inv
            state += 0.5
            np.floor(state, out=state)
            state += n
        np.matmul(self.h[-1], p.head_w, out=self.logits)
        self.logits += p.head_b
        return self.logits


def teacher_forced_logits(predictor: QuantizedPredictor, bits: np.ndarray, width: int) -> np.ndarray:
    """Integer logits (float64) for rows of true bit sequences, shape (B, L)."""
    bits = np.asarray(bits)
    stepper = Stepper(predictor, bits.shape[0], width)
    output = np.empty(bits.shape, dtype=np.float64)
    symbols = np.full(bits.shape[0], BOS, dtype=np.intp)
    for t in range(bits.shape[1]):
        output[:, t] = stepper.step(symbols, t)
        symbols[:] = bits[:, t]
    return output

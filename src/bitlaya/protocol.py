"""The predict-or-transmit rule, independent of any model or file format.

``predict(t, symbol)`` returns the exact integer logit for bit t given that the
previous reconstructed bit was ``symbol`` (BOS at t = 0); the callable may keep
its own recurrent state. ``bound`` is the integer omission bound; ``None`` means
transmit everything (threshold 1.0).

Comparison operator (documented, tested at the boundary): omit iff |L| >= bound.
Tie rule: L == 0 predicts 1.
"""
import numpy as np

from .model import BOS


def decide(logit, bound) -> tuple[bool, int]:
    if bound is None:
        return False, 0
    return abs(logit) >= bound, int(logit >= 0)


def encode(source_bits, predict, bound):
    """Return (explicit bits, reconstructed bits, omitted mask)."""
    source_bits = np.asarray(source_bits, dtype=np.uint8)
    reconstructed = np.empty_like(source_bits)
    omitted = np.zeros(source_bits.shape, dtype=bool)
    explicit, symbol = [], BOS
    for t, true_bit in enumerate(source_bits):
        omit, predicted = decide(predict(t, symbol) if bound is not None else 0, bound)
        bit = predicted if omit else int(true_bit)
        if not omit:
            explicit.append(bit)
        reconstructed[t], omitted[t] = bit, omit
        symbol = bit  # the reconstructed bit -- never the source after a wrong omission
    return np.array(explicit, dtype=np.uint8), reconstructed, omitted


def decode(explicit, length, predict, bound):
    """Rebuild the reconstruction from explicit bits alone; reject malformed payloads."""
    explicit = np.asarray(explicit, dtype=np.uint8)
    bits = np.empty(length, dtype=np.uint8)
    cursor, symbol = 0, BOS
    for t in range(length):
        omit, predicted = decide(predict(t, symbol) if bound is not None else 0, bound)
        if omit:
            bit = predicted
        else:
            if cursor >= explicit.size:
                raise ValueError("explicit payload exhausted; stream is corrupt or decoder diverged")
            bit = int(explicit[cursor])
            cursor += 1
        bits[t] = symbol = bit
    if cursor != explicit.size:
        raise ValueError("unused payload bits; stream is corrupt or decoder diverged")
    return bits

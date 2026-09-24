"""Canonical serialization: uint8 grayscale, row-major pixels, MSB-first bits."""

import numpy as np
from PIL import Image


def validate_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 2 or 0 in image.shape:
        raise ValueError("image must be a nonempty 2D uint8 grayscale array")
    return image


def grayscale(image: Image.Image) -> np.ndarray:
    """Pillow RGB -> L conversion; no resizing, normalization, or dithering."""
    return np.array(image.convert("L"), dtype=np.uint8, copy=True)


def validate_bits(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits)
    if bits.ndim != 1 or not np.isin(bits, (0, 1)).all():
        raise ValueError("bits must be a one-dimensional array containing only 0 and 1")
    return bits.astype(np.uint8, copy=False)


def image_to_bits(image: np.ndarray) -> np.ndarray:
    image = validate_image(image)
    return np.unpackbits(image.ravel(order="C"), bitorder="big")


def bits_to_image(bits: np.ndarray, shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    bits = validate_bits(bits)
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("shape must contain two positive integers")
    if bits.size != shape[0] * shape[1] * 8:
        raise ValueError("bit count does not match the image shape")
    return np.packbits(bits, bitorder="big").reshape(shape)


def pack_bits(bits: np.ndarray) -> bytes:
    """Pad the final byte with zero low bits; the count is stored separately."""
    return np.packbits(validate_bits(bits), bitorder="big").tobytes()


def unpack_bits(payload: bytes, count: int) -> np.ndarray:
    if type(count) is not int or count < 0 or len(payload) != (count + 7) // 8:
        raise ValueError("invalid packed bit length")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    if np.any(bits[count:]):
        raise ValueError("nonzero padding bits")
    return bits[:count].copy()

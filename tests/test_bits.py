import numpy as np
from PIL import Image
import pytest

from bitlaya.bits import bits_to_image, grayscale, image_to_bits, pack_bits, unpack_bits


@pytest.mark.parametrize("kind", ["random", "zero", "white", "ramp", "noncontiguous"])
def test_grayscale_roundtrip_exact(kind):
    image = np.random.default_rng(42).integers(0, 256, (32, 32), dtype=np.uint8)
    if kind == "zero":
        image.fill(0)
    elif kind == "white":
        image.fill(255)
    elif kind == "ramp":
        image = np.tile(np.arange(256, dtype=np.uint8), 4).reshape(32, 32)
    elif kind == "noncontiguous":
        image = image.T[:, ::-1]
    bits = image_to_bits(image)
    assert bits.shape == (8192,)
    assert bits.dtype == np.uint8
    np.testing.assert_array_equal(bits_to_image(bits), image)


def test_raster_msb_first_known_vector():
    image = np.array([[128, 1], [170, 255]], dtype=np.uint8)
    assert image_to_bits(image).tolist() == [int(b) for b in "10000000000000011010101011111111"]


def test_grayscale_pillow_roundtrip():
    rgb = np.random.default_rng(9).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    image = grayscale(Image.fromarray(rgb))
    assert image.shape == (32, 32)
    np.testing.assert_array_equal(image, np.asarray(Image.fromarray(rgb).convert("L")))
    np.testing.assert_array_equal(bits_to_image(image_to_bits(image)), image)


@pytest.mark.parametrize("count", range(18))
def test_bit_packing_padding(count):
    bits = np.random.default_rng(count).integers(0, 2, count, dtype=np.uint8)
    assert len(pack_bits(bits)) == (count + 7) // 8
    np.testing.assert_array_equal(unpack_bits(pack_bits(bits), count), bits)


def test_invalid_serialization_inputs():
    with pytest.raises(ValueError):
        image_to_bits(np.zeros((32, 32), dtype=np.float32))
    with pytest.raises(ValueError):
        bits_to_image(np.array([2] * 8192))
    with pytest.raises(ValueError):
        bits_to_image(np.zeros(8191))
    with pytest.raises(ValueError):
        unpack_bits(b"\x01", 1)
    with pytest.raises(ValueError):
        unpack_bits(b"\x00\x00", 1)

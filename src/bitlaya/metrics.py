"""Distortion and honest byte accounting, relative to raw grayscale pixels."""

from io import BytesIO
import math
import zlib

import numpy as np
from PIL import Image

from .bits import image_to_bits, validate_image


def structural_similarity(original: np.ndarray, reconstruction: np.ndarray) -> float | None:
    """Wang-style SSIM: 11x11 Gaussian, sigma=1.5, population covariance.

    Valid window centers only, L=255, K1=.01, K2=.03. This matches the common
    gaussian_weights=True/use_sample_covariance=False convention. Return None
    for images smaller than the window; do not substitute another SSIM variant.
    """
    original, reconstruction = validate_image(original), validate_image(reconstruction)
    if original.shape != reconstruction.shape:
        raise ValueError("image shapes differ")
    if min(original.shape) < 11:
        return None
    coordinates = np.arange(-5, 6, dtype=float)
    weights = np.exp(-(coordinates**2) / (2 * 1.5**2))
    weights /= weights.sum()
    kernel = np.outer(weights, weights)
    x, y = original.astype(float), reconstruction.astype(float)

    def average(array):
        windows = np.lib.stride_tricks.sliding_window_view(array, (11, 11))
        return np.einsum("ijkl,kl->ij", windows, kernel)

    mean_x, mean_y = average(x), average(y)
    variance_x = np.maximum(average(x*x) - mean_x**2, 0)
    variance_y = np.maximum(average(y*y) - mean_y**2, 0)
    covariance = average(x*y) - mean_x*mean_y
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    result = ((2*mean_x*mean_y + c1) * (2*covariance + c2)
              / ((mean_x**2 + mean_y**2 + c1) * (variance_x + variance_y + c2)))
    return float(result.mean())


def conventional_baselines(image: np.ndarray, qualities=(25, 50, 75, 95)) -> list[dict]:
    """Complete conventional files; lossy metrics use independently read pixels."""
    from PIL import features
    rows = []
    for codec, size in lossless_baselines(image).items():
        rows.append({"codec": codec.removesuffix("_bytes"), "quality": None,
                     "file_bytes": size, "file_bpp": size * 8 / image.size,
                     "mse": 0.0, "psnr_db": None, "ssim": structural_similarity(image, image),
                     "lossless": True})
    for codec in ("JPEG", "WEBP"):
        if codec == "WEBP" and not features.check("webp"):
            continue
        for quality in qualities:
            buffer = BytesIO()
            Image.fromarray(image).save(buffer, format=codec, quality=quality)
            size = buffer.tell()
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                reconstruction = np.asarray(decoded.convert("L"))
            quality_metrics = distortion(image, reconstruction)
            rows.append({"codec": codec.lower(), "quality": quality, "file_bytes": size,
                         "file_bpp": size * 8 / image.size,
                         "mse": quality_metrics["mse"], "psnr_db": quality_metrics["psnr_db"],
                         "ssim": structural_similarity(image, reconstruction),
                         "lossless": quality_metrics["lossless"]})
    return rows


def distortion(original: np.ndarray, reconstruction: np.ndarray) -> dict:
    original, reconstruction = validate_image(original), validate_image(reconstruction)
    if original.shape != reconstruction.shape:
        raise ValueError("image shapes differ")
    difference = original.astype(np.float64) - reconstruction.astype(np.float64)
    mse = float(np.mean(difference**2))
    errors = image_to_bits(original) != image_to_bits(reconstruction)
    return {"mse": mse, "mae": float(np.mean(np.abs(difference))),
            "psnr_db": None if mse == 0 else 10 * math.log10(255**2 / mse),
            "lossless": mse == 0, "bit_error_rate": float(np.mean(errors)),
            "bitplane_error_rate_msb_first": errors.reshape(-1, 8).mean(axis=0).tolist()}


def lossless_baselines(image: np.ndarray) -> dict:
    image = validate_image(image)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", compress_level=9)
    return {"raw_bytes": image.nbytes,
            "zlib_bytes": len(zlib.compress(image.tobytes(order="C"), level=9)),
            "png_bytes": len(buffer.getvalue())}

"""Image-level summaries and seeded bootstrap intervals (not independent bits)."""
import math
import numpy as np


def psnr(mse: float) -> float | None:
    return None if mse == 0 else 10 * math.log10(255**2 / mse)


def describe(values) -> dict:
    """Null values mean undefined (or +infinity for a PSNR column)."""
    finite = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    result = {"count": len(values), "finite_count": len(finite),
              "undefined_or_infinite_count": len(values) - len(finite)}
    for name, function in (("mean", np.mean), ("median", np.median), ("std", np.std)):
        result[name] = float(function(finite)) if len(finite) else None
    return result


def bootstrap_means(columns: dict, repeats: int = 1000, seed: int = 42) -> dict:
    """Paired image resampling across columns; CI describes image sampling only.

    Does not estimate training-seed variance or account for validation selection.
    Missing-valued columns are omitted, never silently imputed. A one-image
    sample has no meaningful bootstrap uncertainty, so intervals are withheld.
    """
    if repeats < 0:
        raise ValueError("bootstrap repeats must be nonnegative")
    if not columns:
        return {}
    count = len(next(iter(columns.values())))
    if any(len(v) != count for v in columns.values()):
        raise ValueError("bootstrap columns must have equal length")
    if count < 2 or repeats == 0:
        return {}
    valid = {k: np.asarray(v, dtype=float) for k, v in columns.items()
             if all(x is not None and math.isfinite(x) for x in v)}
    sampled = {k: [] for k in valid}
    rng = np.random.default_rng(seed)
    # Bounded memory even for 10,000-image evaluation sets.
    for start in range(0, repeats, 64):
        indices = rng.integers(0, count, (min(64, repeats - start), count))
        for name, values in valid.items():
            sampled[name].extend(values[indices].mean(axis=1).tolist())
    intervals = {k: [float(x) for x in np.quantile(v, [.025, .975])]
                 for k, v in sampled.items()}
    if "mse" in intervals:
        low, high = intervals["mse"]
        intervals["pooled_psnr_db"] = [psnr(high), psnr(low)]
    return intervals

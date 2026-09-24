"""Teacher-forced diagnostics and error-propagation analysis.

Error propagation uses the exact engine, which gives a clean counterfactual:
until the first incorrect omission the codec's reconstructed history equals the
source history, so codec (rollout) logits equal teacher-forced logits EXACTLY
(asserted). After it, every difference is caused by the wrong omission(s). Each
later incorrect omission is classified as

* ``induced``   -- the teacher-forced model, given the TRUE history, would have
                   predicted this bit correctly: a cascade error;
* ``intrinsic`` -- it would have predicted it wrongly anyway.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .bits import image_to_bits
from .blaya import BitLayaCodec
from .data import BitImageDataset, benchmark_images
from .engine import decision_threshold, teacher_forced_logits
from .model import load_checkpoint
from .research import write_csv, write_json
from .research_plots import INK2, MUTED, SERIES, plt, style
from .training import Calibration, TrainConfig, evaluate_teacher_forced


def _train_config(metadata) -> TrainConfig:
    stored = dict(metadata.get("run", {}).get("training", {}))
    fields = {k: v for k, v in stored.items() if k in TrainConfig.__dataclass_fields__}
    for key in ("schedule", "rollout_thresholds"):
        if key in fields:
            fields[key] = tuple(tuple(x) if isinstance(x, list) else x for x in fields[key])
    if isinstance(fields.get("weights"), list):
        fields["weights"] = tuple(fields["weights"])
    return replace(TrainConfig(**fields), device="cpu", batch_size=64)


def reliability_plot(report: dict, path: Path, title: str):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 6), height_ratios=[3, 1.3], sharex=True)
    bins = [b for b in report["calibration"] if b["count"]]
    centers = [b["lower"] + 0.025 for b in bins]
    top.plot([0.5, 1], [0.5, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect calibration")
    top.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "-o", color=SERIES[0],
             linewidth=2, markersize=4, label="model")
    style(top, title, "", "Accuracy")
    top.legend(fontsize=7, frameon=False, loc="upper left")
    bottom.bar(centers, [b["count"] for b in bins], width=0.04, color=SERIES[0])
    bottom.set_yscale("log")
    style(bottom, "", "Confidence  max(p, 1-p)", "Predictions")
    for b, x in zip(bins, centers):
        bottom.annotate(f"{b['count']:,}", (x, b["count"]), ha="center", va="bottom", fontsize=6, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def diagnose(checkpoint, data, split, size, seed, output):
    """Float and exact-engine teacher-forced BCE/accuracy/calibration on held-out images."""
    torch.set_num_threads(1)
    model, metadata = load_checkpoint(checkpoint)
    images, manifest = benchmark_images(data, split, size, seed)
    config = _train_config(metadata)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(BitImageDataset(images), batch_size=config.batch_size)
    float_report = evaluate_teacher_forced(model, loader, torch.device("cpu"), config)
    codec = BitLayaCodec(model)
    calibration, total, correct, count = Calibration(), 0.0, 0, 0
    for start in range(0, len(images), 128):
        bits = np.stack([image_to_bits(im) for im in images[start:start + 128]])
        logits = teacher_forced_logits(codec.predictor, bits, images.shape[2]) * 2.0 ** -codec.predictor.S
        total += float(np.sum(np.logaddexp(0, logits) - bits * logits))
        matches = (logits >= 0) == bits.astype(bool)
        correct += int(matches.sum())
        count += bits.size
        calibration.update(1 / (1 + np.exp(-np.abs(logits))).ravel(), matches.ravel().astype(float))
    exact = {"bce_nats": total / count, "cross_entropy_bits": total / count / np.log(2),
             "bit_accuracy": correct / count, **calibration.report()}
    report = {"checkpoint": str(checkpoint), "dataset": manifest, "float32": float_report,
              "qgru_v2_exact_engine": exact, "engine_model_id": codec.model_id.hex(),
              "note": "Teacher-forced diagnostics: predictions from TRUE history; not codec rates."}
    write_json(output / "diagnostics.json", report)
    reliability_plot(float_report, output / "reliability.png",
                     f"Reliability (teacher-forced, float32), {len(images)} {split} images")
    reliability_plot(exact, output / "reliability_exact_engine.png",
                     f"Reliability (teacher-forced, exact engine), {len(images)} {split} images")
    print(json.dumps({"float_bce": float_report["bce_nats"], "exact_bce": exact["bce_nats"],
                      "accuracy": float_report["bit_accuracy"], "ece": float_report["ece"],
                      "high_confidence": float_report["high_confidence"]}, indent=2))
    return report


def _clean(values):
    """NaN (undefined, e.g. no images reach that distance) -> None for strict JSON."""
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=float)]


def _nanmean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def _runs(errors_row):
    positions = np.flatnonzero(errors_row)
    if not len(positions):
        return []
    breaks = np.flatnonzero(np.diff(positions) != 1)
    return np.diff(np.concatenate(([0], breaks + 1, [len(positions)]))).tolist()


def analyze_errors(checkpoint, data, split, size, seed, thresholds, output, max_distance=512):
    torch.set_num_threads(1)
    model, _ = load_checkpoint(checkpoint)
    images, manifest = benchmark_images(data, split, size, seed)
    codec = BitLayaCodec(model)
    predictor = codec.predictor
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shape = images.shape[1:]
    source = np.stack([image_to_bits(im) for im in images])
    rows, length = source.shape
    teacher = teacher_forced_logits(predictor, source, shape[1])
    teacher_correct = (teacher >= 0) == source.astype(bool)
    distances = np.arange(1, max_distance + 1)
    summary, per_image, curves = [], [], {}
    for threshold in thresholds:
        coded = codec.encode_bits(source, shape, [decision_threshold(threshold)] * rows, trace=True)
        omitted, recon = coded["omitted"], coded["reconstructed"]
        errors = omitted & (recon != source)
        first = np.where(errors.any(1), errors.argmax(1), length)
        # Exact counterfactual check: identical history => identical integer logits before the first error.
        for row in range(rows):
            f = first[row] + 1  # logit at the first-error position itself is still from true history
            if not np.array_equal(coded["logits"][row, :f], teacher[row, :f]):
                raise RuntimeError("rollout diverged from teacher forcing before any wrong omission")
        logits = coded["logits"] * 2.0 ** -predictor.S
        teacher_scaled = teacher * 2.0 ** -predictor.S
        after = np.arange(length)[None, :] > first[:, None]
        induced = errors & after & teacher_correct
        intrinsic = errors & after & ~teacher_correct
        omitted_after = omitted & after
        # (a) Conditional error rate at distance d after ANY error, among omitted bits.
        cond_err = np.zeros(max_distance)
        cond_omit = np.zeros(max_distance)
        e, o = errors.astype(np.float64), omitted.astype(np.float64)
        for d in distances:
            cond_err[d - 1] = np.sum(e[:, :-d] * e[:, d:])
            cond_omit[d - 1] = np.sum(e[:, :-d] * o[:, d:])
        base_rate = errors.sum() / max(omitted.sum(), 1)
        # (b) Relative to the FIRST error: logit divergence, decision disagreement, induced errors.
        divergence = np.full(max_distance, np.nan)
        disagreement = np.full(max_distance, np.nan)
        induced_rate = np.full(max_distance, np.nan)
        for d in distances:
            index = first + d
            valid = index < length
            if not valid.any():
                continue
            r, c = np.flatnonzero(valid), index[valid]
            divergence[d - 1] = np.mean(np.abs(logits[r, c] - teacher_scaled[r, c]))
            disagreement[d - 1] = np.mean((logits[r, c] >= 0) != (teacher[r, c] >= 0))
            om = omitted[r, c]
            induced_rate[d - 1] = induced[r, c][om].mean() if om.any() else np.nan
        curves[threshold] = {"conditional_error_rate": _clean(np.where(cond_omit > 0, cond_err / np.maximum(cond_omit, 1), np.nan)),
                             "logit_divergence": _clean(divergence), "decision_disagreement": _clean(disagreement),
                             "induced_error_rate": _clean(induced_rate), "base_omitted_error_rate": float(base_rate)}
        run_lengths = [run for row in errors for run in _runs(row)]
        pixel_after = []
        for row in range(rows):
            original = images[row].astype(float).ravel()
            reconstruction = np.packbits(recon[row]).astype(float)
            diff = np.abs(original - reconstruction)
            if first[row] < length:
                p = first[row] // 8
                pixel_after.append(diff[p:p + 64])
            per_image.append({"threshold": threshold, "image_index": row, "dataset_index": manifest["indices"][row],
                              "first_incorrect_omission": int(first[row]) if first[row] < length else None,
                              "incorrect_omissions": int(errors[row].sum()),
                              "induced_errors": int(induced[row].sum()), "intrinsic_errors": int(intrinsic[row].sum()),
                              "omitted_bits": int(omitted[row].sum()),
                              "mean_gap_between_errors": float(np.diff(np.flatnonzero(errors[row])).mean())
                              if errors[row].sum() > 1 else None,
                              "max_error_run": max(_runs(errors[row]), default=0),
                              "mse_before_first_error_pixel": float(np.mean((images[row].astype(float).ravel()
                                  - np.packbits(recon[row]).astype(float))[:first[row] // 8] ** 2))
                              if 0 < first[row] < length and first[row] // 8 else None,
                              "mse_after_first_error_pixel": float(np.mean((images[row].astype(float).ravel()
                                  - np.packbits(recon[row]).astype(float))[first[row] // 8:] ** 2))
                              if first[row] < length else None})
        after_errors = induced.sum() + intrinsic.sum()
        padded = np.array([np.pad(x, (0, 64 - len(x)), constant_values=np.nan) for x in pixel_after]) \
            if pixel_after else np.full((1, 64), np.nan)
        counts = np.isfinite(padded).sum(0)
        pixel_curve = np.where(counts > 0, np.nansum(padded, 0) / np.maximum(counts, 1), np.nan)
        curves[threshold]["mean_abs_pixel_error_after_first_error_pixel"] = _clean(pixel_curve)
        summary.append({"threshold": threshold, "images": rows,
                        "images_with_errors": int((first < length).sum()),
                        "median_first_incorrect_omission": float(np.median(first[first < length]))
                        if (first < length).any() else None,
                        "incorrect_omissions": int(errors.sum()), "omitted_bits": int(omitted.sum()),
                        "base_omitted_error_rate": float(base_rate),
                        "errors_after_first": int(after_errors),
                        "induced_fraction_of_later_errors": float(induced.sum() / after_errors) if after_errors else None,
                        "teacher_forced_error_rate_on_same_omitted_bits":
                            float((~teacher_correct[omitted_after]).mean()) if omitted_after.any() else None,
                        "codec_error_rate_on_omitted_bits_after_first_error":
                            float(errors[omitted_after].mean()) if omitted_after.any() else None,
                        "conditional_error_rate_d1_to_8": float(cond_err[:8].sum() / max(cond_omit[:8].sum(), 1)),
                        "conditional_error_rate_d64_to_512": float(cond_err[63:].sum() / max(cond_omit[63:].sum(), 1)),
                        "run_length_histogram": {str(k): int(v) for k, v in
                                                 zip(*np.unique(run_lengths, return_counts=True))} if run_lengths else {},
                        "mean_abs_logit_divergence_d1_to_64": _nanmean(divergence[:64])})
        print(json.dumps(summary[-1]), flush=True)
    write_json(output / "summary.json", {"checkpoint": str(checkpoint), "dataset": manifest,
                                         "engine_model_id": codec.model_id.hex(), "thresholds": summary,
                                         "definitions": __doc__})
    write_json(output / "curves.json", {str(k): v for k, v in curves.items()})
    write_csv(output / "per_image.csv", per_image)
    distance_rows = [{"threshold": t, "distance": int(d), **{k: v[d - 1] for k, v in c.items()
                                                              if isinstance(v, list) and len(v) == max_distance}}
                     for t, c in curves.items() for d in distances]
    write_csv(output / "distance_curves.csv", distance_rows)
    _plot_errors(curves, summary, output, len(images), split)
    return summary


def _plot_errors(curves, summary, output, n, split):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for index, (threshold, curve) in enumerate(curves.items()):
        color = SERIES[index % 8]
        x = np.arange(1, len(curve["conditional_error_rate"]) + 1)
        axes[0, 0].plot(x, curve["conditional_error_rate"], color=color, linewidth=1.2, label=f"t={threshold:g}")
        axes[0, 0].axhline(curve["base_omitted_error_rate"], color=color, linewidth=0.8, linestyle=":")
        axes[0, 1].plot(x, curve["logit_divergence"], color=color, linewidth=1.2, label=f"t={threshold:g}")
        axes[1, 0].plot(x, curve["induced_error_rate"], color=color, linewidth=1.2, label=f"t={threshold:g}")
        pixels = curve["mean_abs_pixel_error_after_first_error_pixel"]
        axes[1, 1].plot(np.arange(len(pixels)), pixels, color=color, linewidth=1.2, label=f"t={threshold:g}")
    for ax in axes.ravel():
        ax.set_xscale("symlog", linthresh=8)
        ax.legend(fontsize=7, frameon=False)
    style(axes[0, 0], "P(error at i+d | error at i), omitted bits\n(dotted: overall omitted-bit error rate)",
          "Bit distance d", "Error rate")
    style(axes[0, 1], "Mean |codec logit − teacher-forced logit|\nafter the first wrong omission",
          "Bits after first wrong omission", "Logit divergence")
    style(axes[1, 0], "Induced (cascade) error rate among omitted bits\nafter the first wrong omission",
          "Bits after first wrong omission", "Induced error rate")
    style(axes[1, 1], "Mean |pixel error| from the first wrong pixel", "Pixels after first wrong pixel",
          "Mean absolute error")
    fig.suptitle(f"Error propagation, {n} {split} images (exact engine)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(output / "error_propagation.png", dpi=150)
    plt.close(fig)

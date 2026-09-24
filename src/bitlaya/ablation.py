"""Controlled ablations: train variants from one base config, sweep, compare by RD.

Config (JSON)::

    {"name": "...", "output": "results/...",
     "data": {"train_size": 1024, "val_size": 256, "validation_size": 5000},
     "model": {...}, "training": {...},                       # shared base
     "sweep": {"split": "val", "size": 200, "thresholds": [...], "engine": "qgru-v2"},
     "variants": [{"name": "baseline"}, {"name": "x", "model": {...}, "training": {...}}]}

Each variant overrides only the keys it lists. The first variant is the
reference for comparisons. Comparisons are rate-distortion comparisons:

* ``operational`` -- for each complete-file bpp budget b on a grid spanning
  the overlap of both curves, the best pooled PSNR over thresholds whose mean
  file bpp <= b (what a user choosing the best threshold for a budget gets);
  report the mean and min/max PSNR difference in dB across the grid;
* per-threshold file bpp / PSNR tables.

Teacher-forced BCE/accuracy is reported alongside, as a diagnostic only.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .research import write_csv, write_json
from .research_plots import plot_comparison


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _run_pool(jobs, parallel):
    """jobs: list of (name, argv, log_path). Runs up to ``parallel`` at once."""
    pending, running, failures = list(jobs), [], []
    while pending or running:
        while pending and len(running) < parallel:
            name, argv, log = pending.pop(0)
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            print(f"[ablate] start {name}: {' '.join(map(str, argv[1:]))}", flush=True)
            running.append((name, subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT), handle, log))
        time.sleep(2)
        for item in list(running):
            name, process, handle, log = item
            if process.poll() is not None:
                handle.close()
                running.remove(item)
                status = "ok" if process.returncode == 0 else f"FAILED ({process.returncode}), see {log}"
                print(f"[ablate] done {name}: {status}", flush=True)
                if process.returncode:
                    failures.append(name)
    if failures:
        raise RuntimeError(f"ablation jobs failed: {failures}")


def envelope(points):
    """Operational RD: best PSNR (inf capped later) available at <= each bpp."""
    ordered = sorted((p["file_bpp"], np.inf if p["psnr"] is None else p["psnr"]) for p in points)
    bpp = np.array([x for x, _ in ordered])
    best = np.maximum.accumulate(np.array([y for _, y in ordered]))
    return bpp, best


def rd_difference(reference, candidate, grid_points=200):
    """Mean PSNR gain of candidate over reference on the overlapping bpp range."""
    rb, rp = envelope(reference)
    cb, cp = envelope(candidate)
    low, high = max(rb[0], cb[0]), min(rb[-1], cb[-1])
    if high <= low:
        return None
    grid = np.linspace(low, high, grid_points)
    ref = rp[np.searchsorted(rb, grid, side="right") - 1]
    cand = cp[np.searchsorted(cb, grid, side="right") - 1]
    finite = np.isfinite(ref) & np.isfinite(cand)
    if not finite.any():
        return None
    diff = cand[finite] - ref[finite]
    return {"bpp_range": [float(low), float(high)], "grid_points_compared": int(finite.sum()),
            "mean_psnr_gain_db": float(diff.mean()), "min_psnr_gain_db": float(diff.min()),
            "max_psnr_gain_db": float(diff.max()),
            "note": "operational envelope; lossless (infinite PSNR) grid points excluded"}


def points_from_summary(summary, range_coded=False):
    return [{"threshold": s["threshold"], "psnr": s["pooled_psnr_db"],
             "file_bpp": s["range_payload"]["file_bpp"] if range_coded else s["file_bpp"],
             "file_bytes": s["statistics"]["file_bytes"]["mean"],
             "payload_bpp": s["statistics"]["payload_bpp"]["mean"],
             "ssim": s["statistics"].get("ssim", {}).get("mean")} for s in summary["thresholds"]]


def compare_runs(runs: dict, output: Path, title: str):
    """runs: name -> directory containing train/ and sweep/. First entry is the reference."""
    names = list(runs)
    summaries, rows = {}, []
    for name in names:
        summary = json.loads((runs[name] / "sweep" / "summary.json").read_text(encoding="utf-8"))
        summaries[name] = summary
        history_path = runs[name] / "train" / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        best = min(history, key=lambda h: h["validation"]["objective"]) if history else None
        config_path = runs[name] / "train" / "config.json"
        train_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        for s in summary["thresholds"]:
            rows.append({"variant": name, "threshold": s["threshold"], "file_bpp": s["file_bpp"],
                         "psnr_db": s["pooled_psnr_db"], "mse": s["pooled_mse"],
                         "ssim": s["statistics"].get("ssim", {}).get("mean"),
                         "omitted_fraction": s["statistics"]["omitted_fraction"]["mean"],
                         "omitted_error_rate": s.get("pooled_omitted_error_rate"),
                         "range_file_bpp": s.get("range_payload", {}).get("file_bpp"),
                         "file_bpp_ci95": s["bootstrap_95_ci"].get("file_bpp"),
                         "mse_ci95": s["bootstrap_95_ci"].get("mse"),
                         "val_bce_nats": best["validation"]["bce_nats"] if best else None,
                         "val_bit_accuracy": best["validation"]["bit_accuracy"] if best else None,
                         "val_ece": best["validation"]["ece"] if best else None,
                         "parameters": train_config.get("parameters"),
                         "train_images": train_config.get("train_images"),
                         "optimizer_steps": history[-1]["step"] if history else None,
                         "train_seconds": history[-1]["seconds"] if history else None})
    reference = points_from_summary(summaries[names[0]])
    comparisons = {name: rd_difference(reference, points_from_summary(summaries[name])) for name in names[1:]}
    ranged = all("range_payload" in s["thresholds"][0] for s in summaries.values())
    range_comparisons = ({name: rd_difference(points_from_summary(summaries[names[0]], True),
                                              points_from_summary(summaries[name], True)) for name in names[1:]}
                         if ranged else None)
    images = {name: summaries[name]["config"]["dataset"]["image_sha256"] for name in names}
    if len(set(images.values())) != 1:
        raise ValueError("variants were swept on different image subsets; comparison invalid")
    result = {"title": title, "reference_variant": names[0], "sweep_images": summaries[names[0]]["config"]["dataset"],
              "rd_vs_reference": comparisons, "rd_vs_reference_range_coded_payload": range_comparisons,
              "table": rows,
              "sampling_scale": summaries[names[0]]["sampling_scale"]}
    write_json(output / "comparison.json", result)
    write_csv(output / "comparison.csv", rows)
    curves = [(name, points_from_summary(summaries[name])) for name in names]
    from .research_plots import rd_series
    _, baselines, _ = rd_series(summaries[names[0]])
    plot_comparison(curves, output / "comparison_rd.png", f"{title}: PSNR vs complete-file bpp "
                    f"({summaries[names[0]]['thresholds'][0]['images']} {summaries[names[0]]['config']['dataset']['split']} images)",
                    baselines={k: v for k, v in baselines.items() if k in ("jpeg", "webp", "png")})
    return result


def training_complete(train_dir: Path) -> bool:
    """True once the last validation record reached the planned step budget (or final epoch)."""
    history_path, config_path = train_dir / "history.json", train_dir / "config.json"
    if not (history_path.exists() and config_path.exists() and (train_dir / "best.pt").exists()):
        return False
    history = json.loads(history_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not history:
        return False
    planned = config.get("planned_optimizer_steps")
    return (planned is not None and history[-1]["step"] >= planned) or \
        history[-1]["epoch"] >= config["training"]["epochs"]


def run_ablation(config_path, output=None, parallel=1, only=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output or config["output"])
    output.mkdir(parents=True, exist_ok=True)
    base = {key: config.get(key, {}) for key in ("data", "model", "training")}
    sweep = config.get("sweep", {})
    variants = [v for v in config["variants"] if not only or v["name"] in only]
    write_json(output / "ablation_config.json", config)
    jobs, directories = [], {}
    for variant in variants:
        directory = output / variant["name"]
        directories[variant["name"]] = directory
        resolved = deep_merge(base, {k: variant.get(k, {}) for k in ("data", "model", "training")})
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "experiment.json", resolved)
        train_dir = directory / "train"
        if train_dir.exists() and not training_complete(train_dir):
            # Training cannot resume mid-run (no optimizer state); keep the partial run aside.
            aside = directory / f"train.incomplete-{time.strftime('%Y%m%d-%H%M%S')}"
            train_dir.rename(aside)
            print(f"[ablate] {variant['name']}: incomplete training moved to {aside}; restarting", flush=True)
        if not training_complete(train_dir):
            jobs.append((f"train:{variant['name']}", [sys.executable, "-m", "bitlaya", "--threads", "1", "train",
                                                       "--config", str(directory / "experiment.json"),
                                                       "--output", str(directory / "train")],
                         directory / "train.log"))
    _run_pool(jobs, parallel)
    jobs = []
    for name, directory in directories.items():
        if (directory / "sweep" / "summary.json").exists():
            continue
        argv = [sys.executable, "-m", "bitlaya", "--threads", "1", "sweep", str(directory / "train" / "best.pt"),
                "--output", str(directory / "sweep"), "--split", sweep.get("split", "val"),
                "--size", str(sweep.get("size", 200)), "--engine", sweep.get("engine", "qgru-v2")]
        if (directory / "sweep" / "config.json").exists():
            argv.append("--resume")
        if "thresholds" in sweep:
            argv += ["--thresholds", *map(str, sweep["thresholds"])]
        jobs.append((f"sweep:{name}", argv, directory / "sweep.log"))
    _run_pool(jobs, parallel)
    result = compare_runs(directories, output, config.get("name", "ablation"))
    print(json.dumps(result["rd_vs_reference"], indent=2))
    return result

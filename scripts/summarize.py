"""Render Markdown tables from BitLaya result artifacts (no numbers typed by hand).

    python scripts/summarize.py sweep results/baseline_qgru_v2_test1000
    python scripts/summarize.py compare results/data_scaling
    python scripts/summarize.py lossless results/lossless_x
    python scripts/summarize.py errors results/errors_x
    python scripts/summarize.py profile results/profile_x
    python scripts/summarize.py throughput results/throughput_gpupc
    python scripts/summarize.py matched results/training_modes [0.60 0.85]
    python scripts/summarize.py paired results/data_scaling [reference_variant]
    python scripts/summarize.py rate results/position_ablation [reference_variant]
"""
import json
from pathlib import Path
import sys


def fmt(value, digits=3):
    if value is None:
        return "inf" if digits == "psnr" else "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sweep(directory):
    summary = load(Path(directory) / "summary.json")
    dataset = summary["config"]["dataset"]
    print(f"**{summary['thresholds'][0]['images']} {dataset['split']} images ({summary['sampling_scale']}), "
          f"engine {summary['config']['codec']}, image sha256 {dataset['image_sha256'][:12]}; "
          f"independent decodes verified: {summary['independent_decodes_verified']}, "
          f"mismatches: {summary.get('encoder_decoder_mismatches', 0)}**\n")
    ranged = "range_payload" in summary["thresholds"][0]
    head = ("| t | omitted | wrong/omitted | payload bpp | header B | file B | file bpp [95% CI] | PSNR dB | "
            "PSNR 95% CI | SSIM | zlib/raw payload |" + (" range file bpp | range/raw payload |" if ranged else ""))
    print(head)
    print("|" + "---:|" * (head.count("|") - 1))
    for s in summary["thresholds"]:
        st, ci = s["statistics"], s["bootstrap_95_ci"]
        fb = ci.get("file_bpp")
        psnr_ci = ci.get("pooled_psnr_db")
        row = (f"| {s['threshold']:g} | {st['omitted_fraction']['mean']:.3f} | "
               f"{fmt(s.get('pooled_omitted_error_rate'), 3)} | {st['payload_bpp']['mean']:.3f} | "
               f"{st['header_bytes']['mean']:.0f} | {st['file_bytes']['mean']:.1f} | {s['file_bpp']:.3f} "
               f"[{fb[0]:.3f}, {fb[1]:.3f}] | {fmt(s['pooled_psnr_db'], 2) if s['pooled_psnr_db'] else 'inf'} | "
               + (f"[{fmt(psnr_ci[0], 2) if psnr_ci[0] else 'inf'}, {fmt(psnr_ci[1], 2) if psnr_ci[1] else 'inf'}]"
                  if psnr_ci else "n/a")
               + f" | {fmt(st['ssim']['mean'], 4)} | {fmt(s['zlib_payload']['ratio'], 3)} |")
        if ranged:
            row += f" {s['range_payload']['file_bpp']:.3f} | {fmt(s['range_payload']['ratio'], 3)} |"
        print(row)
    print("\nBaselines (same images):\n")
    print("| codec | quality | file bytes | file bpp | PSNR dB | SSIM |")
    print("|---|---:|---:|---:|---:|---:|")
    for b in summary["baselines"]:
        print(f"| {b['codec']} | {b['quality'] if b['quality'] is not None else 'lossless'} | "
              f"{b['statistics']['file_bytes']['mean']:.1f} | {b['file_bpp']:.3f} | "
              f"{fmt(b['pooled_psnr_db'], 2) if b['pooled_psnr_db'] else 'inf'} | {fmt(b['statistics']['ssim']['mean'], 4)} |")
    print("\nMatched PSNR (baseline bytes interpolated at BitLaya's PSNR; n/a = outside measured range):\n")
    print("| t | BitLaya PSNR | BitLaya bytes | JPEG bytes | WebP bytes |")
    print("|---:|---:|---:|---:|---:|")
    for m in summary["matched_psnr"]:
        print(f"| {m['threshold']:g} | {fmt(m['bitlaya_psnr_db'], 2) if m['bitlaya_psnr_db'] else 'inf'} | "
              f"{m['bitlaya_file_bytes']:.1f} | {fmt(m.get('jpeg_bytes_at_matched_psnr'), 1)} | "
              f"{fmt(m.get('webp_bytes_at_matched_psnr'), 1)} |")


def compare(directory):
    result = load(Path(directory) / "comparison.json")
    print(f"**{result['title']}**, reference variant `{result['reference_variant']}`, "
          f"{result['sweep_images']['size']} {result['sweep_images']['split']} images ({result['sampling_scale']})\n")
    print("| variant | params | train images | steps | val BCE (nats/bit) | val acc | val ECE |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    seen = set()
    for row in result["table"]:
        if row["variant"] in seen:
            continue
        seen.add(row["variant"])
        print(f"| {row['variant']} | {row['parameters']} | {row['train_images']} | {row['optimizer_steps']} | "
              f"{fmt(row['val_bce_nats'], 4)} | {fmt(row['val_bit_accuracy'], 4)} | {fmt(row['val_ece'], 4)} |")
    print("\nOperational RD difference vs reference (mean PSNR gain over the overlapping complete-file bpp range):\n")
    print("| variant | raw payload: mean dB [min, max] | bpp range | range-coded payload: mean dB [min, max] |")
    print("|---|---:|---:|---:|")
    ranged = result.get("rd_vs_reference_range_coded_payload") or {}
    for name, value in result["rd_vs_reference"].items():
        r = ranged.get(name)
        cell = lambda v: (f"{v['mean_psnr_gain_db']:+.2f} [{v['min_psnr_gain_db']:+.2f}, {v['max_psnr_gain_db']:+.2f}]"
                          if v else "n/a")
        span = f"{value['bpp_range'][0]:.2f}-{value['bpp_range'][1]:.2f}" if value else "n/a"
        print(f"| {name} | {cell(value)} | {span} | {cell(r)} |")
    thresholds = sorted({row["threshold"] for row in result["table"]})
    variants = list(dict.fromkeys(row["variant"] for row in result["table"]))
    print("\nPer-threshold complete-file bpp / pooled PSNR (dB):\n")
    print("| t | " + " | ".join(variants) + " |")
    print("|---:|" + "---:|" * len(variants))
    for t in thresholds:
        cells = []
        for v in variants:
            row = next(r for r in result["table"] if r["variant"] == v and r["threshold"] == t)
            psnr = fmt(row["psnr_db"], 2) if row["psnr_db"] is not None else "inf"
            cells.append(f"{row['file_bpp']:.3f} / {psnr}")
        print(f"| {t:g} | " + " | ".join(cells) + " |")


def lossless(directory):
    summary = load(Path(directory) / "summary.json")
    bpp = summary["mean_bits_per_pixel"]
    stats = summary["statistics"]
    print(f"**{summary['images']} {summary['dataset']['split']} images ({summary['sampling_scale']}); "
          f"all roundtrips exact: {summary['all_roundtrips_exact']}**\n")
    print("| codec | mean bytes/image | bpp |")
    print("|---|---:|---:|")
    for label, key in (("raw", "raw_bytes"), ("zlib level 9", "zlib9_bytes"), ("PNG level 9", "png9_bytes"),
                       ("BitLaya lossless (file, 34 B header)", "bitlaya_file_bytes"),
                       ("BitLaya lossless (payload only)", "bitlaya_payload_bytes")):
        print(f"| {label} | {stats[key]['mean']:.1f} | {bpp[key]:.3f} |")
    print(f"| theoretical sum -log2 p (table probabilities) | {stats['ideal_bits_table']['mean'] / 8:.1f} | "
          f"{bpp['ideal_table']:.3f} |")
    print(f"| theoretical sum -log2 p (exact-engine probabilities) | {stats['ideal_bits_engine']['mean'] / 8:.1f} | "
          f"{bpp['ideal_engine']:.3f} |")
    print(f"\nRange-coder overhead vs table ideal: mean {stats['coder_overhead_bits']['mean']:.1f} bits/image. "
          f"Images where BitLaya < PNG: {summary['fraction_images_bitlaya_smaller_than_png']:.1%}. "
          f"Batched encode {summary['timing']['batched_encode_seconds_per_image']:.3f} s/image, "
          f"decode {summary['timing']['batched_decode_seconds_per_image']:.3f} s/image (uncontrolled).")


def errors(directory):
    summary = load(Path(directory) / "summary.json")
    print("| t | images w/ errors | median first error bit | omitted-bit error rate | "
          "P(err | err within 1-8 bits) | P(err | err 64-512 bits away) | later errors | induced share | "
          "TF error rate on same bits | codec error rate on same bits |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summary["thresholds"]:
        print(f"| {s['threshold']:g} | {s['images_with_errors']}/{s['images']} | "
              f"{fmt(s['median_first_incorrect_omission'], 0)} | {s['base_omitted_error_rate']:.4f} | "
              f"{s['conditional_error_rate_d1_to_8']:.4f} | {s['conditional_error_rate_d64_to_512']:.4f} | "
              f"{s['errors_after_first']} | {fmt(s['induced_fraction_of_later_errors'], 3)} | "
              f"{fmt(s['teacher_forced_error_rate_on_same_omitted_bits'], 4)} | "
              f"{fmt(s['codec_error_rate_on_omitted_bits_after_first_error'], 4)} |")


def profile(directory):
    timing = load(Path(directory) / "timing.json")
    config = timing["config"]
    env = config["environment"]
    print(f"engine `{config['engine']}`, threshold {config['threshold']}, {config['dataset']['size']} test images, "
          f"{config['warmup']} warmups, {config['repeats']} timed runs, controlled protocol: "
          f"{config['controlled_protocol']}; CPU {env['cpu']}, torch {env['torch']}, threads {env['threads']}, "
          f"CUDA available: {env['cuda_available']}\n")
    latency = timing["latency"]
    print("| direction | median s | p95 s | mean s | images/s (median) |")
    print("|---|---:|---:|---:|---:|")
    for direction in ("encode", "decode"):
        d = latency[direction]
        print(f"| {direction} | {d['median_seconds']:.3f} | {d['p95_seconds']:.3f} | {d['mean_seconds']:.3f} | "
              f"{d['images_per_second']:.2f} |")
    print("\nSetup costs (seconds):\n")
    for key, value in timing["setup"].items():
        print(f"- {key}: {value:.4f}")
    if "batched_throughput" in timing:
        b = timing["batched_throughput"]
        print(f"- batched ({b['rows']} rows): encode {b['encode_images_per_second']:.2f} img/s, "
              f"decode {b['decode_images_per_second']:.2f} img/s")
    if "threshold_1_fast_path_seconds_per_image_encode_plus_decode" in timing:
        print(f"- threshold-1 fast path: {timing['threshold_1_fast_path_seconds_per_image_encode_plus_decode']*1e3:.2f} "
              "ms/image (encode+decode)")
    print("\nTop self-time functions (cProfile, one encode+decode):\n")
    print("| function | calls | self s | share |")
    print("|---|---:|---:|---:|")
    for f in timing["cprofile_top_self_time"][:10]:
        print(f"| `{f['function']}` | {f['calls']} | {f['self_seconds']:.3f} | {f['self_fraction']:.1%} |")


def _interpolate(points, x):
    """Linear interpolation of y at x over points sorted by x; None outside the measured range."""
    points = sorted(points)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def matched(directory, low=0.60, high=0.85):
    """PSNR/MSE/SSIM of every variant interpolated at the REFERENCE variant's complete-file bpp,
    for the reference thresholds in [low, high]. Same images for all variants (checked by ablate)."""
    result = load(Path(directory) / "comparison.json")
    rows = [r for r in result["table"] if r["psnr_db"] is not None]
    variants = list(dict.fromkeys(r["variant"] for r in result["table"]))
    reference = result["reference_variant"]
    targets = [r for r in rows if r["variant"] == reference and low - 1e-9 <= r["threshold"] <= high + 1e-9]
    print(f"Matched complete-file bpp: each variant's pooled PSNR (dB) / MSE / SSIM linearly interpolated between "
          f"its own threshold points at the reference (`{reference}`) bpp for thresholds {low:g}-{high:g}; "
          f"{result['sweep_images']['size']} {result['sweep_images']['split']} images "
          f"({result['sampling_scale']}). n/a = outside the variant's measured bpp range.\n")
    print("| ref t | bpp | " + " | ".join(f"{v} PSNR" for v in variants) + " | "
          + " | ".join(f"{v} dPSNR" for v in variants[1:]) + " |")
    print("|---:|---:|" + "---:|" * (2 * len(variants) - 1))
    gains = {v: [] for v in variants[1:]}
    for target in targets:
        values = {}
        for v in variants:
            own = [r for r in rows if r["variant"] == v]
            values[v] = _interpolate([(r["file_bpp"], r["psnr_db"]) for r in own], target["file_bpp"])
        cells = [fmt(values[v], 2) for v in variants]
        deltas = []
        for v in variants[1:]:
            d = None if values[v] is None or values[reference] is None else values[v] - values[reference]
            if d is not None:
                gains[v].append(d)
            deltas.append(f"{d:+.2f}" if d is not None else "n/a")
        print(f"| {target['threshold']:g} | {target['file_bpp']:.3f} | " + " | ".join(cells) + " | "
              + " | ".join(deltas) + " |")
    print("\n| variant | mean dPSNR over matched points (dB) | points |")
    print("|---|---:|---:|")
    for v, values in gains.items():
        print(f"| {v} | {fmt(sum(values) / len(values) if values else None, 3)} | {len(values)} |")
    print("\nMSE and SSIM at the same matched bpp:\n")
    print("| ref t | bpp | " + " | ".join(f"{v} MSE / SSIM" for v in variants) + " |")
    print("|---:|---:|" + "---:|" * len(variants))
    for target in targets:
        cells = []
        for v in variants:
            own = [r for r in rows if r["variant"] == v]
            mse = _interpolate([(r["file_bpp"], r["mse"]) for r in own], target["file_bpp"])
            ssim = _interpolate([(r["file_bpp"], r["ssim"]) for r in own if r["ssim"] is not None],
                                target["file_bpp"])
            cells.append(f"{fmt(mse, 2)} / {fmt(ssim, 4)}")
        print(f"| {target['threshold']:g} | {target['file_bpp']:.3f} | " + " | ".join(cells) + " |")


def _sweep_arrays(sweep_dir):
    """measurements.jsonl -> (thresholds, dataset indices, {field: [threshold, image] array})."""
    import numpy as np
    rows = _jsonl(Path(sweep_dir) / "measurements.jsonl")
    thresholds = sorted({r["threshold"] for r in rows})
    images = sorted({r["image_index"] for r in rows})
    fields = ("file_bytes", "range_file_bytes", "mse", "original_bytes", "dataset_index")
    arrays = {f: np.full((len(thresholds), len(images)), np.nan) for f in fields}
    t_pos, i_pos = {t: k for k, t in enumerate(thresholds)}, {i: k for k, i in enumerate(images)}
    for r in rows:
        for f in fields:
            if f in r:
                arrays[f][t_pos[r["threshold"]], i_pos[r["image_index"]]] = r[f]
    if np.isnan(arrays["mse"]).any():
        raise ValueError(f"{sweep_dir}: incomplete sweep (missing threshold/image rows)")
    return thresholds, arrays["dataset_index"][0], arrays


def _rd_points(thresholds, arrays, sample, ranged):
    import numpy as np
    from bitlaya.statistics import psnr
    key = "range_file_bytes" if ranged else "file_bytes"
    pixels = arrays["original_bytes"][:, sample].sum(axis=1)
    bpp = 8 * arrays[key][:, sample].sum(axis=1) / pixels
    mse = arrays["mse"][:, sample].mean(axis=1)
    return [{"threshold": t, "file_bpp": float(b), "psnr": psnr(float(m))} for t, b, m in zip(thresholds, bpp, mse)]


def paired(directory, reference=None, repeats=1000):
    """Paired image-bootstrap 95% CI of the operational-envelope mean PSNR gain (ablation.rd_difference)
    of every variant vs ``reference`` (default: the ablation's reference variant). Both variants are
    resampled with the SAME image indices, so image difficulty cancels. Usage:
        python scripts/summarize.py paired results/data_scaling [reference_variant]"""
    import numpy as np
    from bitlaya.ablation import rd_difference
    result = load(Path(directory) / "comparison.json")
    variants = list(dict.fromkeys(r["variant"] for r in result["table"]))
    reference = reference or result["reference_variant"]
    data = {v: _sweep_arrays(Path(directory) / v / "sweep") for v in variants}
    ref_t, ref_idx, ref_arrays = data[reference]
    n = len(ref_idx)
    rng = np.random.default_rng(42)
    samples = [rng.integers(0, n, n) for _ in range(int(repeats))]
    print(f"Paired image bootstrap ({int(repeats)} resamples, seed 42) of the operational-envelope mean PSNR gain "
          f"vs `{reference}` over the overlapping complete-file bpp range; {n} {result['sweep_images']['split']} "
          f"images ({result['sampling_scale']}). Lossless grid points excluded (as in `ablate`).\n")
    print("| variant | payload | observed gain dB | bootstrap mean | 95% CI | P(gain > 0) |")
    print("|---|---|---:|---:|---:|---:|")
    everything = np.arange(n)
    for v in variants:
        if v == reference:
            continue
        thresholds, idx, arrays = data[v]
        if thresholds != ref_t or not np.array_equal(idx, ref_idx):
            raise ValueError(f"{v}: different thresholds or images than {reference}; not paired")
        for ranged in (False, True):
            observed = rd_difference(_rd_points(ref_t, ref_arrays, everything, ranged),
                                     _rd_points(thresholds, arrays, everything, ranged))
            gains = []
            for sample in samples:
                d = rd_difference(_rd_points(ref_t, ref_arrays, sample, ranged),
                                  _rd_points(thresholds, arrays, sample, ranged))
                if d is not None:
                    gains.append(d["mean_psnr_gain_db"])
            gains = np.array(gains)
            low, high = np.percentile(gains, [2.5, 97.5])
            print(f"| {v} | {'range-coded' if ranged else 'raw'} | "
                  f"{fmt(observed['mean_psnr_gain_db'] if observed else None, 2)} | {gains.mean():+.2f} | "
                  f"[{low:+.2f}, {high:+.2f}] | {(gains > 0).mean():.3f} |")


def _rate_at_quality(points, target_db):
    """Complete-file bpp needed to reach pooled PSNR >= target: linear interpolation in (bpp, MSE) between
    consecutive points of the Pareto front (what mixing two thresholds across images achieves, since both
    pooled MSE and bpp average linearly). Lossless points (MSE 0) are included. None if never reached."""
    target = 255.0 ** 2 / 10 ** (target_db / 10)
    front, best = [], float("inf")
    for p in sorted(points, key=lambda p: (p["file_bpp"], p["mse"])):
        if p["mse"] < best:
            front.append(p)
            best = p["mse"]
    for a, b in zip(front, front[1:]):
        if a["mse"] <= target:
            return a["file_bpp"]
        if b["mse"] <= target:
            return a["file_bpp"] + (b["file_bpp"] - a["file_bpp"]) * (a["mse"] - target) / (a["mse"] - b["mse"])
    return front[0]["file_bpp"] if front and front[0]["mse"] <= target else None


def _mse_points(thresholds, arrays, sample, ranged):
    key = "range_file_bytes" if ranged else "file_bytes"
    pixels = arrays["original_bytes"][:, sample].sum(axis=1)
    bpp = 8 * arrays[key][:, sample].sum(axis=1) / pixels
    mse = arrays["mse"][:, sample].mean(axis=1)
    return [{"threshold": t, "file_bpp": float(b), "mse": float(m)} for t, b, m in zip(thresholds, bpp, mse)]


def rate(directory, reference=None, repeats=1000, targets=(20.0, 25.0, 30.0, 35.0)):
    """Complete-file bpp needed to reach each target PSNR, every variant vs ``reference``, with a paired
    image-bootstrap 95% CI of the bpp difference (negative = fewer bits = better). Usage:
        python scripts/summarize.py rate results/position_ablation [reference_variant]"""
    import numpy as np
    result = load(Path(directory) / "comparison.json")
    variants = list(dict.fromkeys(r["variant"] for r in result["table"]))
    reference = reference or result["reference_variant"]
    data = {v: _sweep_arrays(Path(directory) / v / "sweep") for v in variants}
    ref_t, ref_idx, ref_arrays = data[reference]
    n = len(ref_idx)
    rng = np.random.default_rng(42)
    samples = [rng.integers(0, n, n) for _ in range(int(repeats))]
    everything = np.arange(n)
    print(f"Rate at quality: complete-file bpp needed for pooled PSNR >= target (Pareto front of the threshold "
          f"points, interpolated linearly in (bpp, MSE); lossless point included). Paired image bootstrap "
          f"({int(repeats)} resamples, seed 42) of the bpp difference vs `{reference}`; negative = fewer bits. "
          f"{n} {result['sweep_images']['split']} images ({result['sampling_scale']}).\n")
    print("| payload | target dB | " + " | ".join(f"{v} bpp" for v in variants) + " | "
          + " | ".join(f"{v} - {reference} [95% CI]" for v in variants if v != reference) + " |")
    print("|---|---:|" + "---:|" * (2 * len(variants) - 1))
    for ranged in (False, True):
        for target in targets:
            observed = {v: _rate_at_quality(_mse_points(data[v][0], data[v][2], everything, ranged), target)
                        for v in variants}
            cells = []
            for v in variants:
                if v == reference:
                    continue
                if data[v][0] != ref_t or not np.array_equal(data[v][1], ref_idx):
                    raise ValueError(f"{v}: different thresholds or images than {reference}; not paired")
                diffs = []
                for sample in samples:
                    a = _rate_at_quality(_mse_points(ref_t, ref_arrays, sample, ranged), target)
                    b = _rate_at_quality(_mse_points(data[v][0], data[v][2], sample, ranged), target)
                    if a is not None and b is not None:
                        diffs.append(b - a)
                if observed[v] is None or observed[reference] is None or not diffs:
                    cells.append("n/a")
                    continue
                low, high = np.percentile(diffs, [2.5, 97.5])
                cells.append(f"{observed[v] - observed[reference]:+.3f} [{low:+.3f}, {high:+.3f}]")
            print(f"| {'range-coded' if ranged else 'raw'} | {target:g} | "
                  + " | ".join(fmt(observed[v], 3) for v in variants) + " | " + " | ".join(cells) + " |")


def _jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.exists() else []


def throughput(directory):
    """Tables from scripts/bench_gpupc.sh output (single.jsonl, concurrency.jsonl, sweep_timing.jsonl)."""
    rows = _jsonl(Path(directory) / "single.jsonl")
    if rows:
        r = rows[0]
        print(f"GPU {r['gpu']}, torch {r['torch']} (CUDA {r['cuda']}), chunk {r['chunk_length']} bits; "
              "one optimizer step = batch x one chunk; 32 steps = one pass over a batch of images.\n")
        print("| model | mode | device | threads | batch | timed steps | ms/step | steps/s | kbit/s | image passes/s |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            model = "bit+plane+row/col" if r["model"].get("row_dim") else "bit only (default)"
            print(f"| {model} | {r['mode']} | {r['device']} | {r['threads'] if r['device'] == 'cpu' else '-'} | "
                  f"{r['batch_size']} | {r['timed_steps']} | {r['seconds_per_step'] * 1e3:.1f} | "
                  f"{r['steps_per_second']:.2f} | {r['kbit_per_second']:.0f} | {r['image_passes_per_second']:.2f} |")
    rows = _jsonl(Path(directory) / "concurrency.jsonl")
    if rows:
        groups = {}
        for r in rows:
            name, n = r["label"].rsplit("_n", 1)
            groups.setdefault((name, int(n)), []).append(r)
        print("\nConcurrency: N identical jobs started together. Per-job and aggregate throughput "
              "(aggregate = sum over jobs; windows overlap but are not perfectly aligned).\n")
        print("| job | N | per-job ms/step (mean) | slowest job ms/step | per-job slowdown vs N=1 | "
              "aggregate steps/s | aggregate vs N=1 |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for (name, n), jobs in sorted(groups.items()):
            per_job = sum(j["seconds_per_step"] for j in jobs) / len(jobs)
            aggregate = sum(j["steps_per_second"] for j in jobs)
            single = groups.get((name, 1))
            base = sum(j["seconds_per_step"] for j in single) / len(single) if single else None
            base_rate = sum(j["steps_per_second"] for j in single) if single else None
            print(f"| {name} | {n} | {per_job * 1e3:.1f} | {max(j['seconds_per_step'] for j in jobs) * 1e3:.1f} | "
                  f"{fmt(per_job / base if base else None, 2)}x | {aggregate:.2f} | "
                  f"{fmt(aggregate / base_rate if base_rate else None, 2)}x |")
    rows = _jsonl(Path(directory) / "sweep_timing.jsonl")
    if rows:
        print("\nRD sweep cost (CPU, NumPy exact engine, 1 thread per job; all thresholds per image). "
              "Aggregate = jobs x (single-job s/image / per-job s/image).\n")
        print("| images/job | concurrent jobs | codec s/image (mean over jobs) | per-job slowdown | aggregate speedup | "
              "wall s/job (mean) | decodes verified | mismatches |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")
        groups = {}
        for r in rows:
            groups.setdefault((r["images"], r["concurrent_jobs"]), []).append(r)
        for (images, n), jobs in sorted(groups.items()):
            per = sum(j["codec_seconds_per_image_all_thresholds"] for j in jobs) / len(jobs)
            single = groups.get((images, 1))
            base = single[0]["codec_seconds_per_image_all_thresholds"] if single else None
            wall = sum(j["wall_seconds_config_to_summary"] for j in jobs) / len(jobs)
            print(f"| {images} | {n} | {per:.2f} | {fmt(per / base if base else None, 2)}x | "
                  f"{fmt(n * base / per if base else None, 2)}x | {wall:.0f} | "
                  f"{sum(j['independent_decodes_verified'] for j in jobs)} | "
                  f"{sum(j['encoder_decoder_mismatches'] for j in jobs)} |")


if __name__ == "__main__":
    command = {"sweep": sweep, "compare": compare, "lossless": lossless, "errors": errors, "profile": profile,
               "throughput": throughput, "matched": matched, "paired": paired, "rate": rate}[sys.argv[1]]
    command(sys.argv[2], *(sys.argv[3:] if sys.argv[1] in ("paired", "rate") else map(float, sys.argv[3:])))

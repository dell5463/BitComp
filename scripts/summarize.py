"""Render Markdown tables from BitLaya result artifacts (no numbers typed by hand).

    python scripts/summarize.py sweep results/baseline_qgru_v2_test1000
    python scripts/summarize.py compare results/data_scaling
    python scripts/summarize.py lossless results/lossless_x
    python scripts/summarize.py errors results/errors_x
    python scripts/summarize.py profile results/profile_x
    python scripts/summarize.py throughput results/throughput_gpupc
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
    {"sweep": sweep, "compare": compare, "lossless": lossless, "errors": errors, "profile": profile,
     "throughput": throughput}[sys.argv[1]](sys.argv[2])

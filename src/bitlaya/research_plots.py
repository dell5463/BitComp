"""Matplotlib rate-distortion and diagnostic plots (no seaborn).

Colors follow a fixed categorical order; BitLaya is always slot 1 so the same
entity keeps the same color across every figure. Lossless points (PSNR = +inf)
are drawn on a separate marked band above the finite axis, never at a fake value.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BASELINE_COLORS = {"jpeg": SERIES[1], "webp": SERIES[2], "png": SERIES[3], "zlib": SERIES[4],
                   "raw": SERIES[6]}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def _psnr_axis_cap(values):
    finite = [v for v in values if v is not None]
    return (max(finite) if finite else 40.0) + 6.0


def _lossless_band(ax, cap):
    ax.axhline(cap - 2.5, color=MUTED, linewidth=0.6, linestyle=":")
    ax.text(ax.get_xlim()[0], cap - 2.2, " lossless (PSNR = ∞) ↑", color=MUTED, fontsize=7, va="bottom")


def rd_series(summary):
    """Return BitLaya curve points and baseline groups from a research summary."""
    points = [{"threshold": s["threshold"], "file_bpp": s["file_bpp"],
               "file_bytes": s["statistics"]["file_bytes"]["mean"],
               "payload_bpp": s["statistics"]["payload_bpp"]["mean"] if "payload_bpp" in s["statistics"] else None,
               "psnr": s["pooled_psnr_db"],
               "ssim": s["statistics"].get("ssim", {}).get("mean")} for s in summary["thresholds"]]
    ranged = [{**p, "file_bpp": s["range_payload"]["file_bpp"],
               "file_bytes": s["statistics"]["range_file_bytes"]["mean"]}
              for p, s in zip(points, summary["thresholds"]) if "range_payload" in s]
    baselines = {}
    for b in summary.get("baselines", []):
        baselines.setdefault(b["codec"], []).append(
            {"quality": b["quality"], "file_bpp": b["file_bpp"],
             "file_bytes": b["statistics"]["file_bytes"]["mean"], "psnr": b["pooled_psnr_db"],
             "ssim": b["statistics"].get("ssim", {}).get("mean")})
    return points, baselines, ranged


def _plot_metric(ax, points, baselines, x_key, y_key, label="BitLaya (complete file)", color=SERIES[0]):
    cap = _psnr_axis_cap([p[y_key] for p in points] + [p[y_key] for g in baselines.values() for p in g]) \
        if y_key == "psnr" else None

    def y(p):
        return (cap if p[y_key] is None else p[y_key]) if y_key == "psnr" else p[y_key]

    ordered = sorted((p for p in points if p[x_key] is not None and y(p) is not None), key=lambda p: p["threshold"])
    ax.plot([p[x_key] for p in ordered], [y(p) for p in ordered], "-o", color=color, linewidth=2,
            markersize=4, label=label, zorder=3)
    for p in ordered:
        if p["threshold"] in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0):
            ax.annotate(f"{p['threshold']:g}", (p[x_key], y(p)), xytext=(4, -10), textcoords="offset points",
                        fontsize=7, color=INK2)
    for codec, group in baselines.items():
        group = sorted((g for g in group if y(g) is not None), key=lambda g: g[x_key])
        if not group:
            continue
        color_b = BASELINE_COLORS.get(codec, MUTED)
        lossy = [g for g in group if g["quality"] is not None]
        if lossy:
            ax.plot([g[x_key] for g in lossy], [y(g) for g in lossy], "-s", color=color_b, linewidth=1.5,
                    markersize=4, label=codec.upper(), zorder=2)
        for g in group:
            if g["quality"] is None:
                ax.plot([g[x_key]], [y(g)], "D", color=color_b, markersize=6, zorder=2,
                        label=f"{codec.upper()} (lossless)")
    return cap


def plot_report(summary, output):
    output = Path(output)
    points, baselines, ranged = rd_series(summary)
    scale = summary.get("sampling_scale", "")
    n = summary["thresholds"][0]["images"] if summary["thresholds"] else 0
    subtitle = f"{n} held-out images ({summary['config']['dataset']['split']}), {scale}; shared model excluded"
    specs = [("rd_curve.png", "file_bpp", "psnr", "Complete-file bits per pixel", "PSNR (dB, pooled MSE)"),
             ("rd_bytes.png", "file_bytes", "psnr", "Complete file bytes per image (mean)", "PSNR (dB, pooled MSE)"),
             ("rd_ssim.png", "file_bpp", "ssim", "Complete-file bits per pixel", "SSIM (mean)"),
             ("rd_payload.png", "payload_bpp", "psnr", "Explicit payload bits per pixel", "PSNR (dB, pooled MSE)")]
    for name, x_key, y_key, xlabel, ylabel in specs:
        if y_key == "ssim" and all(p["ssim"] is None for p in points):
            continue
        fig, ax = plt.subplots(figsize=(7, 4.4))
        use_baselines = {} if x_key == "payload_bpp" else baselines
        cap = _plot_metric(ax, points, use_baselines, x_key, y_key,
                           label="BitLaya payload only" if x_key == "payload_bpp" else "BitLaya (complete file)")
        if ranged and x_key in ("file_bpp", "file_bytes"):
            _plot_metric(ax, ranged, {}, x_key, y_key, label="BitLaya + range-coded payload (complete file)",
                         color=SERIES[7])
        style(ax, f"BitLaya {ylabel.split(' ')[0]} vs {xlabel.lower()}\n{subtitle}", xlabel, ylabel)
        if cap is not None:
            _lossless_band(ax, cap)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(output / name, dpi=160)
        plt.close(fig)


def plot_comparison(curves, output, title, x_key="file_bpp", y_key="psnr", xlabel="Complete-file bits per pixel",
                    ylabel="PSNR (dB, pooled MSE)", baselines=None):
    """curves: list of (label, points) sharing one axis; colors follow list order."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cap = None
    for index, (label, points) in enumerate(curves):
        cap = _plot_metric(ax, points, baselines if index == 0 and baselines else {}, x_key, y_key, label=label,
                           color=SERIES[[0, 7, 5, 6, 3, 4, 1, 2][index % 8]])
    style(ax, title, xlabel, ylabel)
    if cap is not None and y_key == "psnr":
        _lossless_band(ax, cap)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)

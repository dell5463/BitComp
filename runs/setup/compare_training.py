"""Summarize the controlled two-versus-ten-epoch experiment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

old_dir = Path("runs/cifar_smoke")
new_dir = Path("runs/cifar_10epochs")
old_config = json.loads((old_dir / "config.json").read_text())
new_config = json.loads((new_dir / "config.json").read_text())
for key in ("model", "train_sha256", "validation_sha256"):
    assert old_config[key] == new_config[key], f"comparison mismatch: {key}"
history = json.loads((new_dir / "history.json").read_text())
old_history = json.loads((old_dir / "history.json").read_text())
old_best = min(old_history, key=lambda row: row["validation"]["bce_nats"])
new_best = min(history, key=lambda row: row["validation"]["bce_nats"])
old_eval = json.loads((old_dir / "evaluation/results.json").read_text())
new_eval = json.loads((new_dir / "evaluation/results.json").read_text())
assert old_eval["image_sha256"] == new_eval["image_sha256"]
old_thresholds = {row["threshold"]: row for row in old_eval["summary"]}
comparison = []
for current in new_eval["summary"]:
    previous = old_thresholds[current["threshold"]]
    comparison.append({"threshold": current["threshold"],
                       "old_file_bpp": previous["file_bpp"], "new_file_bpp": current["file_bpp"],
                       "old_mse": previous["mse"], "new_mse": current["mse"],
                       "old_psnr_db": previous["psnr_db"], "new_psnr_db": current["psnr_db"],
                       "old_omitted_fraction": previous["omitted_fraction"],
                       "new_omitted_fraction": current["omitted_fraction"]})
report = {"epochs_completed": len(history), "training_images": new_config["train_images"],
          "validation_images": new_config["validation_images"], "evaluation_images": 2,
          "old_best_epoch": old_best["epoch"], "new_best_epoch": new_best["epoch"],
          "old_validation": old_best["validation"], "new_validation": new_best["validation"],
          "training_seconds": sum(row["seconds"] for row in history),
          "comparison": comparison,
          "limitations": "Small fixed subset; two-image codec comparison is exploratory, not a benchmark."}
(new_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
fig, ax = plt.subplots(figsize=(7, 4))
for split, label in (("train", "Training"), ("validation", "Validation")):
    ax.plot([row["epoch"] for row in history], [row[split]["bce_nats"] for row in history], "o-", label=label)
ax.axvline(2, color="0.5", linestyle="--", label="Previous run ended")
ax.set(xlabel="Epoch", ylabel="Binary cross entropy (nats/bit)", title="BitLaya: longer training on the same data")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(new_dir / "training_progress.png", dpi=180)
plt.close(fig)
print(json.dumps({k: v for k, v in report.items() if k not in ("old_validation", "new_validation")}, indent=2))
print("Best validation BCE:", old_best["validation"]["bce_nats"], "->", new_best["validation"]["bce_nats"])
print("Best validation accuracy:", old_best["validation"]["bit_accuracy"], "->", new_best["validation"]["bit_accuracy"])

import json
from pathlib import Path
import numpy as np
from PIL import Image
from bitlaya.data import load_images
from bitlaya.evaluation import _plots

output = Path("runs/cifar_smoke/evaluation")
report = json.loads((output / "results.json").read_text())
thresholds = [s["threshold"] for s in report["summary"]]
originals = load_images("data/cifar10", "val", limit=2)
previews = [[np.array(Image.open(output / f"example_{i:03d}_threshold_{t:g}.png"))
             for i in range(2)] for t in thresholds]
_plots(output, originals, thresholds, previews, report["summary"], report["baselines"])

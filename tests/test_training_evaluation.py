import json

import numpy as np
import pytest
import torch

from bitlaya.data import load_images, split_indices
from bitlaya.evaluation import evaluate
from bitlaya.metrics import distortion
from bitlaya.model import BitPredictor, ModelConfig, load_checkpoint
from bitlaya.training import TrainConfig, train


def test_splits_are_disjoint_deterministic_and_exhaustive():
    training, validation = split_indices(100, 20, 4)
    assert set(training).isdisjoint(validation)
    assert set(training) | set(validation) == set(range(100))
    repeat, _ = split_indices(100, 20, 4)
    np.testing.assert_array_equal(training, repeat)
    with pytest.raises(ValueError):
        split_indices(20, 20)


def test_cached_native_resolution_and_split_selection(tmp_path):
    images = np.stack([np.full((32, 32), n, dtype=np.uint8) for n in range(12)])
    np.save(tmp_path / "train_gray.npy", images)
    train_images = load_images(tmp_path, "train", validation_size=3)
    val_images = load_images(tmp_path, "val", validation_size=3)
    assert train_images.shape == (9, 32, 32)
    assert set(train_images[:, 0, 0]).isdisjoint(val_images[:, 0, 0])
    with pytest.raises(ValueError):
        load_images(tmp_path, "val", validation_size=3, limit=0)


def test_training_smoke_and_reproducibility(tmp_path):
    images = np.zeros((4, 2, 2), dtype=np.uint8)
    config = TrainConfig(epochs=2, batch_size=2, chunk_length=13,
                         learning_rate=0.01, seed=3, device="cpu")
    first, history = train(images, images[:2], tmp_path / "first", ModelConfig(4, 8, 1), config)
    second, _ = train(images, images[:2], tmp_path / "second", ModelConfig(4, 8, 1), config)
    assert history[1]["validation"]["bce_nats"] < history[0]["validation"]["bce_nats"]
    assert sum(b["count"] for b in history[1]["validation"]["calibration"]) == 64
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    loaded, metadata = load_checkpoint(tmp_path / "first" / "best.pt")
    assert metadata["epoch"] == 2
    assert loaded.config == ModelConfig(4, 8, 1)


def test_distortion_known_values_no_uint8_overflow():
    original = np.zeros((2, 2), dtype=np.uint8)
    opposite = np.full((2, 2), 255, dtype=np.uint8)
    metrics = distortion(original, opposite)
    assert metrics["mse"] == 65025
    assert metrics["psnr_db"] == 0
    assert metrics["bit_error_rate"] == 1
    assert distortion(original, original)["psnr_db"] is None


def test_evaluation_counts_headers_model_cost_and_distinct_threshold_files(tmp_path):
    model = BitPredictor(ModelConfig(4, 8, 1))
    images = np.zeros((2, 2, 2), dtype=np.uint8)
    report = evaluate(images, model, [0.5, 0.75, 1.0], tmp_path, checkpoint_bytes=100, examples=1)
    lossless = report["summary"][-1]
    assert lossless["lossless_images"] == 2
    assert lossless["payload_bpp"] == 8
    assert lossless["file_bpp"] > 8
    assert lossless["file_plus_amortized_checkpoint_bpp"] == lossless["file_bpp"] + 100
    assert len(list(tmp_path.glob("*.blay"))) == 3
    assert (tmp_path / "rate_distortion.png").exists()
    assert (tmp_path / "reconstructions.png").exists()
    assert json.loads((tmp_path / "results.json").read_text())["summary"][-1]["psnr_db"] is None

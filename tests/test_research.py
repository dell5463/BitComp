"""Training modes/objectives and the research harness on tiny synthetic data."""
import json
import shutil

import numpy as np
import pytest
import torch

from bitlaya.ablation import compare_runs, deep_merge, rd_difference
from bitlaya.analysis import analyze_errors, diagnose
from bitlaya.model import BitPredictor, ModelConfig, load_checkpoint, save_checkpoint, teacher_inputs
from bitlaya.research import EvaluationConfig, error_statistics, matched_quality, run_evaluation
from bitlaya.training import TrainConfig, plane_weights, rollout_inputs, train


def smooth_images(count, seed=0, size=32):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.integers(-6, 7, (count, size, size)), axis=2) + rng.integers(60, 190, (count, 1, 1))
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("cifar")
    np.save(root / "train_gray.npy", smooth_images(40, 1))
    np.save(root / "test_gray.npy", smooth_images(12, 2))
    return root


@pytest.fixture(scope="module")
def checkpoint(dataset, tmp_path_factory):
    images = np.load(dataset / "train_gray.npy")
    output = tmp_path_factory.mktemp("run")
    config = TrainConfig(epochs=1, batch_size=8, chunk_length=512, learning_rate=0.01, device="cpu",
                         max_steps=12)
    provenance = {"validation_size": 10, "split_seed": 42}
    train(images[10:], images[:10], output, ModelConfig(4, 12, 1, plane_dim=2), config, provenance)
    return output / "best.pt"


def test_plane_weight_strategies():
    for strategy in ("uniform", "sqrt", "linear", [8, 7, 6, 5, 4, 3, 2, 1]):
        weights = plane_weights(strategy)
        assert len(weights) == 8 and abs(np.mean(weights) - 1) < 1e-12
    sqrt = plane_weights("sqrt")
    assert sqrt[0] / sqrt[7] == pytest.approx(2 ** 3.5)  # MSB (index 0) weighted most
    assert all(a > b for a, b in zip(sqrt, sqrt[1:]))
    with pytest.raises(ValueError):
        plane_weights([1, 2, 3])
    with pytest.raises(ValueError):
        TrainConfig(mode="nonsense")


def test_scheduled_fraction_follows_schedule():
    config = TrainConfig(mode="scheduled", schedule=((1, 0.0), (3, 0.1), (5, 0.25), (7, 0.5)))
    assert [config.replacement_fraction(e) for e in (1, 2, 3, 4, 5, 6, 7, 20)] == [0, 0, .1, .1, .25, .25, .5, .5]
    progress = TrainConfig(mode="scheduled", schedule_unit="progress",
                           schedule=((0, 0.0), (0.2, 0.1), (0.4, 0.25), (0.6, 0.5)))
    assert [progress.replacement_fraction(1, p) for p in (0, .19, .2, .5, .99)] == [0, 0, .1, .25, .5]
    with pytest.raises(ValueError):
        TrainConfig(schedule=((0, 0.1),))  # epoch schedules start at epoch 1


def test_rollout_inputs_follow_codec_rule_and_match_parallel_pass():
    torch.manual_seed(0)
    model = BitPredictor(ModelConfig(4, 8, 1, plane_dim=2)).eval()
    with torch.no_grad():
        model.head.weight.mul_(20)
    targets = torch.randint(0, 2, (3, 40))
    carry = torch.full((3,), 2)
    inputs, replaced, last = rollout_inputs(model, targets, carry, None, 0, 32, threshold=0.7)
    with torch.no_grad():
        logits, _ = model(inputs, None, start=0, width=32)
    confident = torch.sigmoid(logits.abs()) >= 0.7
    predicted = (logits >= 0).long()
    # Next input is the prediction exactly where the rule omitted, else the source bit.
    expected_next = torch.where(confident, predicted, targets)
    assert torch.equal(inputs[:, 1:], expected_next[:, :-1]) and torch.equal(last, expected_next[:, -1])
    assert torch.equal(replaced, confident)
    assert 0 < replaced.float().mean() < 1
    teacher, none, _ = rollout_inputs(model, targets, carry, None, 0, 32, fraction=0.0)
    assert torch.equal(teacher, teacher_inputs(targets)) and not none.any()


@pytest.mark.parametrize("mode,objective,schedule", [("scheduled", "bce", "cosine"),
                                                      ("rollout", "weighted_bce", "plateau"),
                                                      ("teacher_forcing", "weighted_bce", "constant")])
def test_training_modes_run_reproducibly(tmp_path, mode, objective, schedule):
    images = smooth_images(6, 3, size=4)
    config = TrainConfig(epochs=3, batch_size=3, chunk_length=24, learning_rate=0.01, seed=5, device="cpu",
                         mode=mode, schedule=((1, 0.5),), rollout_thresholds=(0.55, 0.7), objective=objective,
                         lr_schedule=schedule, accumulate=2, width=4, val_every=3)
    first, history = train(images, images[:2], tmp_path / "a", ModelConfig(4, 8, 1, plane_dim=2), config)
    second, _ = train(images, images[:2], tmp_path / "b", ModelConfig(4, 8, 1, plane_dim=2), config)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    # 128 bits / 24 = 6 chunks, accumulate 2 -> 3 steps per batch, 2 batches, 3 epochs = 18 steps
    assert [h["step"] for h in history] == [3, 6, 9, 12, 15, 18]
    if mode != "teacher_forcing":
        assert sum(h["train"]["replaced_input_fraction"] for h in history) > 0
    stored = json.loads((tmp_path / "a" / "config.json").read_text())
    assert stored["training"]["mode"] == mode and len(stored["loss_weights_msb_first"]) == 8
    assert "high_confidence" in history[-1]["validation"]


def test_max_steps_budget(tmp_path):
    images = smooth_images(8, 4, size=4)
    config = TrainConfig(epochs=50, batch_size=2, chunk_length=32, device="cpu", max_steps=5, width=4)
    _, history = train(images, images[:2], tmp_path, ModelConfig(4, 8, 1), config)
    assert history[-1]["step"] == 5 and len(history) == 1  # stopped mid-epoch, validated once


def test_error_statistics_and_matched_quality():
    original = np.zeros((1, 4), np.uint8)
    reconstruction = np.array([[0, 192, 0, 1]], np.uint8)
    stats = error_statistics(original, reconstruction)
    assert stats["first_incorrect_omission"] == 8 and stats["max_error_run"] == 2 and stats["error_runs"] == 2
    assert stats["mse_before_first_error_pixel"] == 0
    points = [{"threshold": .9, "pooled_psnr_db": 30.0, "statistics": {"file_bytes": {"mean": 500}}},
              {"threshold": .5, "pooled_psnr_db": 5.0, "statistics": {"file_bytes": {"mean": 34}}}]
    curve = {"jpeg": [{"pooled_psnr_db": 25.0, "statistics": {"file_bytes": {"mean": 100}}},
                      {"pooled_psnr_db": 35.0, "statistics": {"file_bytes": {"mean": 400}}}]}
    matched = matched_quality(points, curve)
    assert matched[0]["jpeg_bytes_at_matched_psnr"] == pytest.approx(200)  # geometric midpoint
    assert matched[1]["jpeg_bytes_at_matched_psnr"] is None  # never extrapolated


def test_rd_difference_uses_operational_envelope():
    reference = [{"file_bpp": 1, "psnr": 10}, {"file_bpp": 3, "psnr": 20}, {"file_bpp": 8, "psnr": None}]
    better = [{"file_bpp": 1, "psnr": 12}, {"file_bpp": 2, "psnr": 25}, {"file_bpp": 8, "psnr": None}]
    result = rd_difference(reference, better)
    assert result["mean_psnr_gain_db"] > 0 and result["min_psnr_gain_db"] == 2
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_research_sweep_diagnose_analysis_and_comparison(dataset, checkpoint, tmp_path):
    config = EvaluationConfig(checkpoint=str(checkpoint), output=str(tmp_path / "sweep"), data=str(dataset),
                              split="test", size=4, thresholds=[0.5, 0.8, 1.0], bootstrap=20, examples=1,
                              engine="qgru-v2", batch_rows=5, lossy_qualities=[10, 50, 90])
    summary = run_evaluation(config, progress=False)
    assert summary["independent_decodes_verified"] == 12 and summary["encoder_decoder_mismatches"] == 0
    lossless = summary["thresholds"][-1]
    assert lossless["lossless_images"] == 4 and lossless["statistics"]["header_bytes"]["mean"] == 34
    assert lossless["file_bpp"] == pytest.approx((1024 + 34) * 8 / 1024)
    assert "zlib_payload" in lossless and summary["matched_psnr"]
    for row in summary["thresholds"]:  # range-coded payloads decoded independently too
        assert row["range_payload"]["file_bpp"] > 0 and row["statistics"]["range_decode_seconds"]["count"] == 4
    for name in ("metrics.csv", "summary.json", "config.json", "rd_curve.png", "timing.json", "baselines.csv"):
        assert (tmp_path / "sweep" / name).exists()
    assert list((tmp_path / "sweep" / "sample_reconstructions").glob("*.blaya"))
    with pytest.raises(ValueError, match="exists"):
        run_evaluation(config, progress=False)
    resumed = run_evaluation(config, resume=True, progress=False)  # nothing left to do
    assert resumed["independent_decodes_verified"] == 12
    # Validation codec subsets must match the training split.
    with pytest.raises(ValueError, match="leakage"):
        run_evaluation(EvaluationConfig(checkpoint=str(checkpoint), output=str(tmp_path / "v"), data=str(dataset),
                                        split="val", size=2, seed=7, validation_pool_size=10, engine="qgru-v2"))
    report = diagnose(checkpoint, dataset, "test", 4, 42, tmp_path / "diag")
    assert abs(report["float32"]["bce_nats"] - report["qgru_v2_exact_engine"]["bce_nats"]) < 1e-3
    errors = analyze_errors(checkpoint, dataset, "test", 4, 42, [0.55, 0.7], tmp_path / "errors", 32)
    assert len(errors) == 2 and (tmp_path / "errors" / "error_propagation.png").exists()
    # A two-variant comparison over identical images (same run twice).
    for name in ("a", "b"):
        shutil.copytree(tmp_path / "sweep", tmp_path / "cmp" / name / "sweep")
    result = compare_runs({"a": tmp_path / "cmp" / "a", "b": tmp_path / "cmp" / "b"}, tmp_path / "cmp", "same")
    assert result["rd_vs_reference"]["b"] is None or result["rd_vs_reference"]["b"]["mean_psnr_gain_db"] == 0
    assert (tmp_path / "cmp" / "comparison_rd.png").exists()


def test_training_complete_detects_partial_runs(tmp_path):
    from bitlaya.ablation import training_complete
    images = smooth_images(4, 5, size=4)
    config = TrainConfig(epochs=50, batch_size=2, chunk_length=32, device="cpu", max_steps=4, width=4)
    train(images, images[:2], tmp_path / "run", ModelConfig(4, 8, 1), config)
    assert training_complete(tmp_path / "run")
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    history[-1]["step"] = 2  # simulate an interrupted run
    (tmp_path / "run" / "history.json").write_text(json.dumps(history))
    assert not training_complete(tmp_path / "run") and not training_complete(tmp_path / "missing")


def test_overhead_measurement(dataset, checkpoint, tmp_path):
    from bitlaya.overhead import measure_overhead
    report = measure_overhead(checkpoint, dataset, "test", 3, 42, [0.8], tmp_path, v1_images=1)
    entry = report["thresholds"][0]
    assert entry["v2_single_file_overhead_bytes"] == 34 and entry["v1_json_overhead_bytes_mean"] > 300
    assert entry["container_with_crc_overhead_bytes_per_image"] == pytest.approx(6 + 30 / 3)
    assert entry["container_without_crc_overhead_bytes_per_image"] == pytest.approx(2 + 30 / 3)


def test_checkpoint_metadata_is_weights_only_loadable(tmp_path):
    model = BitPredictor(ModelConfig(4, 8, 1))
    save_checkpoint(tmp_path / "m.pt", model, {"value": np.float64(0.5), "items": [np.int64(3)]})
    _, metadata = load_checkpoint(tmp_path / "m.pt")
    assert metadata == {"value": 0.5, "items": [3]}

import torch
import pytest

from bitlaya.model import BOS, BitPredictor, ModelConfig, load_checkpoint, model_fingerprint, save_checkpoint, teacher_inputs


def test_causal_prediction_has_no_future_leakage():
    torch.manual_seed(12)
    model = BitPredictor(ModelConfig(8, 12, 2)).eval()
    bits = torch.randint(0, 2, (2, 30))
    alternate = bits.clone()
    alternate[:, 10:] = 1 - alternate[:, 10:]
    first, _ = model(teacher_inputs(bits))
    second, _ = model(teacher_inputs(alternate))
    torch.testing.assert_close(first[:, :11], second[:, :11], rtol=0, atol=0)
    assert teacher_inputs(bits)[:, 0].eq(BOS).all()


def test_full_sequence_and_incremental_gru_agree():
    torch.manual_seed(3)
    model = BitPredictor(ModelConfig(8, 12, 2)).eval()
    inputs = teacher_inputs(torch.randint(0, 2, (1, 64)))
    full, full_hidden = model(inputs)
    hidden, steps = None, []
    for index in range(inputs.shape[1]):
        logits, hidden = model(inputs[:, index:index + 1], hidden)
        steps.append(logits)
    torch.testing.assert_close(full, torch.cat(steps, dim=1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(full_hidden, hidden, atol=1e-6, rtol=1e-5)


POSITION_CONFIG = ModelConfig(8, 12, 2, plane_dim=3, row_dim=2, col_dim=2, max_height=4, max_width=4)


def test_position_features_are_causal_and_chunking_invariant():
    torch.manual_seed(4)
    model = BitPredictor(POSITION_CONFIG).eval()
    bits = torch.randint(0, 2, (2, 128))
    alternate = bits.clone()
    alternate[:, 40:] = 1 - alternate[:, 40:]
    first, _ = model(teacher_inputs(bits), width=4)
    second, _ = model(teacher_inputs(alternate), width=4)
    torch.testing.assert_close(first[:, :41], second[:, :41], rtol=0, atol=0)
    inputs = teacher_inputs(bits)
    hidden, chunks = None, []
    for start in range(0, 128, 24):  # chunk boundaries not aligned to bytes or rows
        logits, hidden = model(inputs[:, start:start + 24], hidden, start=start, width=4)
        chunks.append(logits)
    torch.testing.assert_close(torch.cat(chunks, 1), first, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="height"):
        model(teacher_inputs(torch.zeros(1, 4 * 5 * 8, dtype=torch.long)), width=4)


def test_bit_plane_index_zero_is_msb():
    from bitlaya.model import position_features
    plane, row, col = position_features(0, 24, width=2)
    assert plane.tolist() == list(range(8)) * 3  # bit t of a pixel byte, MSB first
    assert row.tolist() == [0] * 16 + [1] * 8 and col.tolist() == [0] * 8 + [1] * 8 + [0] * 8


def test_legacy_identity_and_parameter_counts():
    assert set(ModelConfig().identity()) == {"embedding_dim", "hidden_size", "num_layers"}
    assert BitPredictor(ModelConfig(plane_dim=8)).parameter_count == 164641
    assert BitPredictor(ModelConfig(plane_dim=8, row_dim=8, col_dim=8)).parameter_count == 171297
    for config in (ModelConfig(plane_dim=8), POSITION_CONFIG):
        assert config.parameter_estimate() == BitPredictor(config).parameter_count


def test_checkpoint_and_parameter_budget(tmp_path):
    model = BitPredictor()
    assert model.parameter_count == 161505
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, {"epoch": 1})
    loaded, metadata = load_checkpoint(path)
    assert model_fingerprint(model) == model_fingerprint(loaded)
    assert metadata == {"epoch": 1}
    with pytest.raises(ValueError, match="10 million"):
        ModelConfig(hidden_size=4096)

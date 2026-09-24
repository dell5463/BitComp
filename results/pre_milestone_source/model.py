"""A small unidirectional GRU. Input at t is bit t-1 (or BOS), never bit t."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

BOS = 2


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 32
    hidden_size: int = 128
    num_layers: int = 2

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) is not int or value <= 0:
                raise ValueError("model dimensions must be positive integers")
        # Check before allocation, including when reading external checkpoints.
        params = (3 * self.embedding_dim + self.hidden_size + 1
                  + 3 * self.hidden_size * (self.embedding_dim + self.hidden_size + 2)
                  + (self.num_layers - 1) * 3 * self.hidden_size * (2 * self.hidden_size + 2))
        if params > 10_000_000:
            raise ValueError("BitLaya limits predictors to 10 million parameters")


class BitPredictor(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(3, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, config.hidden_size,
                          config.num_layers, batch_first=True)
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, previous_bits: torch.Tensor, hidden=None):
        output, hidden = self.gru(self.embedding(previous_bits), hidden)
        return self.head(output).squeeze(-1), hidden

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def teacher_inputs(targets: torch.Tensor) -> torch.Tensor:
    """Shift each image independently, starting with a separate BOS symbol."""
    return torch.cat((torch.full_like(targets[:, :1], BOS), targets[:, :-1]), dim=1)


def model_fingerprint(model: BitPredictor) -> str:
    digest = hashlib.sha256(json.dumps(asdict(model.config), sort_keys=True).encode())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4")
        digest.update(name.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_checkpoint(path: str | Path, model: BitPredictor, metadata: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "format": "bitlaya-checkpoint-v1",
        "config": asdict(model.config),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "fingerprint": model_fingerprint(model),
        "metadata": metadata or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> tuple[BitPredictor, dict]:
    document = torch.load(path, map_location="cpu", weights_only=True)
    if document.get("format") != "bitlaya-checkpoint-v1":
        raise ValueError("unsupported checkpoint format")
    model = BitPredictor(ModelConfig(**document["config"]))
    model.load_state_dict(document["state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("checkpoint contains non-finite weights")
    if model_fingerprint(model) != document["fingerprint"]:
        raise ValueError("checkpoint fingerprint mismatch")
    return model.eval(), document.get("metadata", {})

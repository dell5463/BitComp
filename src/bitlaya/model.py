"""A small unidirectional GRU. Input at t is bit t-1 (or BOS), never bit t.

Optional position features describe the bit being PREDICTED at step t (index t
in raster/MSB-first order), never the input bit t-1:

- bit plane ``k = t % 8``; **k = 0 is the MSB** (bit 7), k = 7 is the LSB.
  This matches serialization, which emits each byte from bit 7 down to bit 0.
- pixel ``t // 8``; row ``pixel // width``; column ``pixel % width``.

Each enabled feature has its own embedding; embeddings are concatenated with the
bit embedding before the GRU. Dimension 0 disables a feature. With all extra
dimensions at 0 the module, state_dict keys, parameter count, and fingerprint are
identical to the original (legacy) BitLaya predictor.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

BOS = 2

# Added after v1 checkpoints existed. Omitted from the identity when "off" so
# legacy fingerprints (and v1 streams that reference them) remain valid.
_OPTIONAL_DEFAULTS = {"plane_dim": 0, "row_dim": 0, "col_dim": 0, "max_height": 32, "max_width": 32}


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 32
    hidden_size: int = 128
    num_layers: int = 2
    plane_dim: int = 0
    row_dim: int = 0
    col_dim: int = 0
    max_height: int = 32
    max_width: int = 32

    def __post_init__(self):
        for name, value in asdict(self).items():
            minimum = 0 if name in ("plane_dim", "row_dim", "col_dim") else 1
            if type(value) is not int or value < minimum:
                raise ValueError("model dimensions must be positive integers")
        if self.parameter_estimate() > 10_000_000:
            raise ValueError("BitLaya limits predictors to 10 million parameters")

    @property
    def input_dim(self) -> int:
        return self.embedding_dim + self.plane_dim + self.row_dim + self.col_dim

    @property
    def spatial(self) -> bool:
        return self.row_dim > 0 or self.col_dim > 0

    def parameter_estimate(self) -> int:
        # Checked before allocation, including when reading external checkpoints.
        h = self.hidden_size
        return (3 * self.embedding_dim + 8 * self.plane_dim + self.max_height * self.row_dim
                + self.max_width * self.col_dim + h + 1
                + 3 * h * (self.input_dim + h + 2)
                + (self.num_layers - 1) * 3 * h * (2 * h + 2))

    def identity(self) -> dict:
        document = asdict(self)
        spatial = self.spatial
        for key, default in _OPTIONAL_DEFAULTS.items():
            if document[key] == default or (key.startswith("max_") and not spatial):
                del document[key]
        return document


def position_features(start: int, length: int, width: int, device=None) -> tuple[torch.Tensor, ...]:
    """(plane, row, col) index tensors for target bits start..start+length-1."""
    positions = torch.arange(start, start + length, device=device)
    pixel = positions // 8
    return positions % 8, pixel // width, pixel % width


class BitPredictor(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(3, config.embedding_dim)
        if config.plane_dim:
            self.plane_embedding = nn.Embedding(8, config.plane_dim)
        if config.row_dim:
            self.row_embedding = nn.Embedding(config.max_height, config.row_dim)
        if config.col_dim:
            self.col_embedding = nn.Embedding(config.max_width, config.col_dim)
        self.gru = nn.GRU(config.input_dim, config.hidden_size, config.num_layers, batch_first=True)
        self.head = nn.Linear(config.hidden_size, 1)

    def features(self, previous_bits: torch.Tensor, start: int = 0, width: int = 32) -> torch.Tensor:
        parts = [self.embedding(previous_bits)]
        config = self.config
        if config.plane_dim or config.spatial:
            if config.spatial and width > config.max_width:
                raise ValueError("image width exceeds the model's column embedding")
            # Last row from Python ints: no host/device copy or sync per call (rollout calls this per bit).
            if config.spatial and (start + previous_bits.shape[1] - 1) // 8 // width >= config.max_height:
                raise ValueError("image height exceeds the model's row embedding")
            plane, row, col = position_features(start, previous_bits.shape[1], width, previous_bits.device)
            batch = previous_bits.shape[0]
            for dim, module, index in ((config.plane_dim, "plane_embedding", plane),
                                       (config.row_dim, "row_embedding", row),
                                       (config.col_dim, "col_embedding", col)):
                if dim:
                    parts.append(getattr(self, module)(index).unsqueeze(0).expand(batch, -1, -1))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def forward(self, previous_bits: torch.Tensor, hidden=None, start: int = 0, width: int = 32):
        """``start`` is the absolute index of the first predicted bit in this chunk."""
        output, hidden = self.gru(self.features(previous_bits, start, width), hidden)
        return self.head(output).squeeze(-1), hidden

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def teacher_inputs(targets: torch.Tensor) -> torch.Tensor:
    """Shift each image independently, starting with a separate BOS symbol."""
    return torch.cat((torch.full_like(targets[:, :1], BOS), targets[:, :-1]), dim=1)


def model_fingerprint(model: BitPredictor) -> str:
    digest = hashlib.sha256(json.dumps(model.config.identity(), sort_keys=True).encode())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4")
        digest.update(name.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _plain(value):
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:  # NumPy/PyTorch scalars
        return value.item()
    raise TypeError(f"checkpoint metadata value of type {type(value).__name__} is not JSON-serializable")


def save_checkpoint(path: str | Path, model: BitPredictor, metadata: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "format": "bitlaya-checkpoint-v1",
        "config": model.config.identity(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "fingerprint": model_fingerprint(model),
        # JSON round trip: only plain types, loadable with torch.load(weights_only=True).
        "metadata": json.loads(json.dumps(metadata or {}, allow_nan=False, default=_plain)),
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

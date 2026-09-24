"""CIFAR-10 preprocessing and disjoint train/validation/test image splits."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
from torch.utils.data import Dataset

from .bits import grayscale, image_to_bits


def prepare_cifar10(root: str | Path, download: bool = True) -> dict:
    from torchvision import __version__ as torchvision_version
    from torchvision.datasets import CIFAR10

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    report = {"dataset": "CIFAR-10", "shape": [32, 32], "dtype": "uint8",
              "grayscale": "Pillow RGB.convert('L')", "pillow": pillow_version,
              "torchvision": torchvision_version, "bit_order": "raster-msb-first"}
    for split in ("train", "test"):
        dataset = CIFAR10(str(root / "raw"), train=(split == "train"), download=download)
        images = np.stack([grayscale(Image.fromarray(rgb)) for rgb in dataset.data])
        path = root / f"{split}_gray.npy"
        np.save(path, images, allow_pickle=False)
        report[split] = {"images": len(images), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "preprocessing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def split_indices(size: int, validation_size: int = 5000, seed: int = 42):
    if not 0 < validation_size < size:
        raise ValueError("validation_size must be positive and smaller than the training pool")
    indices = np.random.default_rng(seed).permutation(size)
    return indices[validation_size:], indices[:validation_size]


def load_images(root: str | Path, split: str, *, validation_size: int = 5000,
                seed: int = 42, limit: int | None = None) -> np.ndarray:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    path = Path(root) / ("test_gray.npy" if split == "test" else "train_gray.npy")
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run 'bitlaya prepare' first")
    images = np.load(path, mmap_mode="r", allow_pickle=False)
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (32, 32):
        raise ValueError("cached dataset must contain uint8 32x32 grayscale images")
    if split == "test":
        selected = np.arange(len(images))
    else:
        train, validation = split_indices(len(images), validation_size, seed)
        selected = train if split == "train" else validation
    if limit is not None:
        selected = selected[:limit]
    if not len(selected):
        raise ValueError("dataset is empty")
    return np.asarray(images[selected])


class BitImageDataset(Dataset):
    def __init__(self, images: np.ndarray):
        if images.ndim != 3 or images.dtype != np.uint8 or len(images) == 0:
            raise ValueError("expected a nonempty batch of uint8 grayscale images")
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Expand on demand: avoid storing the entire corpus as int64 bits.
        return torch.from_numpy(image_to_bits(self.images[index]).astype(np.int64))


def benchmark_images(root: str | Path, split: str = "test", size: int = 1000,
                     seed: int = 42, validation_size: int = 5000):
    """Fixed, nested subsets with original dataset indices and a content digest.

    The test permutation is independent of train/validation selection. Selecting
    N images always gives the first N of the same permutation, for every model.
    """
    if split not in {"train", "val", "test"} or size <= 0:
        raise ValueError("invalid split or subset size")
    path = Path(root) / ("test_gray.npy" if split == "test" else "train_gray.npy")
    images = np.load(path, mmap_mode="r", allow_pickle=False)
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (32, 32):
        raise ValueError("expected cached uint8 CIFAR-10 grayscale images")
    if split == "test":
        indices = np.random.default_rng(seed).permutation(len(images))
    else:
        training, validation = split_indices(len(images), validation_size, seed)
        indices = training if split == "train" else validation
    if size > len(indices):
        raise ValueError(f"requested {size} images but {split} contains {len(indices)}")
    indices = indices[:size]
    selected = np.asarray(images[indices])
    manifest = {"split": split, "size": size, "seed": seed,
                "validation_pool_size": validation_size, "indices": indices.tolist(),
                "image_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                "selection": "numpy-default_rng-permutation-v1"}
    return selected, manifest

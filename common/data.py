from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


def apply_trigger(image: torch.Tensor, size: int, value: float | Sequence[float] = 1.0) -> torch.Tensor:
    """Apply the canonical bottom-right square trigger used by this benchmark."""
    image = image.clone()
    pixel = torch.as_tensor(value, dtype=image.dtype, device=image.device)
    if pixel.ndim == 1:
        pixel = pixel[:, None, None]
    image[:, -size:, -size:] = pixel
    return image


class ClientSubset(Dataset):
    """Subset that can poison a deterministic fraction of one client's samples."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        poison_fraction: float = 0.0,
        trigger_size: int = 4,
        trigger_value: float = 1.0,
        target_label: int = 0,
    ) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.poison_count = int(len(self.indices) * poison_fraction)
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.target_label = target_label

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        image, label = self.dataset[self.indices[position]]
        if position < self.poison_count:
            image = apply_trigger(image, self.trigger_size, self.trigger_value)
            label = self.target_label
        return image, int(label)


class TriggeredSubset(Dataset):
    """Apply the trigger to every sample and force the target label for ASR."""

    def __init__(self, dataset: Dataset, indices: Sequence[int], trigger_size: int, trigger_value: float, target_label: int):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.target_label = target_label

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        image, _ = self.dataset[self.indices[position]]
        image = apply_trigger(image, self.trigger_size, self.trigger_value)
        return image, self.target_label


@dataclass
class Cifar10Benchmark:
    train: Dataset
    test: Dataset
    retain: Dataset
    forget: Dataset
    clean_forget: Dataset
    backdoor_test: Dataset
    client_datasets: list[Dataset]
    client_indices: list[list[int]]


def make_client_indices(num_samples: int, num_clients: int, seed: int) -> list[list[int]]:
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(num_samples)
    chunks = np.array_split(shuffled, num_clients)
    return [[int(index) for index in chunk] for chunk in chunks]


def build_cifar10_benchmark(config) -> Cifar10Benchmark:
    root = Path(config.data_root)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
    train = datasets.CIFAR10(root=root, train=True, transform=transform, download=config.download)
    test = datasets.CIFAR10(root=root, train=False, transform=transform, download=config.download)
    client_indices = make_client_indices(len(train), config.num_clients, config.seed)
    target_indices = client_indices[config.target_client]
    retain_indices = [index for client_id, indices in enumerate(client_indices) if client_id != config.target_client for index in indices]
    # A white pixel is not 1.0 after CIFAR-10 normalization. Keep the trigger
    # defined in raw pixel space and transform it with the same channel stats.
    mean = torch.tensor((0.4914, 0.4822, 0.4465))
    std = torch.tensor((0.2023, 0.1994, 0.2010))
    trigger_value = ((config.trigger_value - mean) / std).tolist()
    client_datasets = [
        ClientSubset(
            train,
            indices,
            poison_fraction=config.poison_fraction if client_id == config.target_client else 0.0,
            trigger_size=config.trigger_size,
            trigger_value=trigger_value,
            target_label=config.target_label,
        )
        for client_id, indices in enumerate(client_indices)
    ]
    # ASR is conditional on the original class being different from the target.
    backdoor_indices = [i for i, label in enumerate(test.targets) if label != config.target_label]
    return Cifar10Benchmark(
        train=train,
        test=test,
        retain=Subset(train, retain_indices),
        forget=ClientSubset(train, target_indices, poison_fraction=config.poison_fraction, trigger_size=config.trigger_size, trigger_value=trigger_value, target_label=config.target_label),
        clean_forget=Subset(train, target_indices),
        backdoor_test=TriggeredSubset(test, backdoor_indices, config.trigger_size, trigger_value, config.target_label),
        client_datasets=client_datasets,
        client_indices=client_indices,
    )

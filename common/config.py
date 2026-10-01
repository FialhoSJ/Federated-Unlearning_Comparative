from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    backbone: str = "small_cnn"
    data_root: str = "./data"
    num_clients: int = 10
    target_client: int = 0
    client_fraction: float = 1.0
    seed: int = 42
    image_size: int = 32
    num_classes: int = 10
    batch_size: int = 128
    local_epochs: int = 1
    rounds: int = 5
    learning_rate: float = 0.01
    momentum: float = 0.9
    poison_fraction: float = 1.0
    trigger_size: int = 4
    trigger_value: float = 1.0
    target_label: int = 0
    download: bool = False
    output_dir: str = "./results/cifar10_client_backdoor"
    validation_fraction: float = 0.1
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    unlearning_rounds: int = 3
    recovery_rounds: int = 3
    unlearning_lr: float = 0.0004
    recovery_lr: float = 0.000001
    fine_tune_rounds: int = 3
    fedup_epochs: int = 50
    fedup_lr: float = 0.05
    fedup_centroids_per_class: int = 4
    fedup_features_per_client: int = 256
    maverick_epochs: int = 1
    maverick_lr: float = 0.001
    maverick_noise_samples: int = 4
    maverick_sigma_min: float = 0.05
    maverick_sigma_max: float = 0.5
    fastfedul_alpha: float = 0.07
    fastfedul_theta: float = 1.0
    ledger_dtype: str = "float16"
    max_train_samples: int | None = None
    max_test_samples: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open() as stream:
            values: dict[str, Any] = json.load(stream)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    def validate(self) -> None:
        if self.dataset != "cifar10":
            raise ValueError("The initial benchmark supports only CIFAR-10.")
        if self.backbone not in {"small_cnn", "resnet18_gn"}:
            raise ValueError("backbone must be small_cnn or resnet18_gn.")
        if self.num_clients < 2 or self.batch_size < 1 or self.local_epochs < 1 or self.rounds < 1:
            raise ValueError("Use at least two clients and positive batch size, epochs, and rounds.")
        if self.learning_rate <= 0 or not 0 <= self.momentum < 1:
            raise ValueError("learning_rate must be positive and momentum must be in [0, 1).")
        if not 0 <= self.target_client < self.num_clients:
            raise ValueError("target_client must be a valid client index.")
        if not 0 < self.client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1].")
        if not 0 <= self.poison_fraction <= 1:
            raise ValueError("poison_fraction must be in [0, 1].")
        if self.trigger_size <= 0 or self.trigger_size > self.image_size:
            raise ValueError("trigger_size must be positive and fit inside the image.")
        if not 0 <= self.trigger_value <= 1:
            raise ValueError("trigger_value is a raw pixel intensity in [0, 1].")
        if not 0 <= self.target_label < self.num_classes:
            raise ValueError("target_label must be a valid class index.")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5).")
        if self.partition not in {"iid", "dirichlet"} or self.dirichlet_alpha <= 0:
            raise ValueError("partition must be iid or dirichlet with positive alpha.")
        if min(self.unlearning_rounds, self.recovery_rounds, self.fine_tune_rounds) < 0:
            raise ValueError("unlearning, recovery, and fine-tune rounds cannot be negative.")
        if min(self.fedup_epochs, self.fedup_centroids_per_class, self.fedup_features_per_client,
               self.maverick_epochs, self.maverick_noise_samples) < 1:
            raise ValueError("method epoch, centroid, feature, and noise counts must be positive.")
        if min(self.unlearning_lr, self.recovery_lr, self.fedup_lr, self.maverick_lr) <= 0:
            raise ValueError("method learning rates must be positive.")
        if not 0 < self.maverick_sigma_min <= self.maverick_sigma_max:
            raise ValueError("Maverick noise bounds must be positive and ordered.")
        if self.max_train_samples is not None and self.max_train_samples < self.num_clients:
            raise ValueError("max_train_samples must permit at least one sample per client.")
        if self.max_test_samples is not None and self.max_test_samples < 2:
            raise ValueError("max_test_samples must be at least two.")
        if self.ledger_dtype not in {"float16", "float32"}:
            raise ValueError("ledger_dtype must be float16 or float32.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

"""Frozen CIFAR-10 protocol shared by every adapted unlearning method."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from .data import TriggeredSubset, apply_trigger

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


class PoisonedClient(Dataset):
    def __init__(self, source, indices, poison_indices, config):
        self.source = source
        self.indices = list(indices)
        self.poison_indices = set(poison_indices)
        self.config = config
        self.trigger = ((config.trigger_value - torch.tensor(MEAN)) / torch.tensor(STD)).tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        image, label = self.source[index]
        if index in self.poison_indices:
            return apply_trigger(image, self.config.trigger_size, self.trigger), self.config.target_label
        return image, label


@dataclass
class ProtocolData:
    train: Dataset
    validation: Dataset
    backdoor_validation: Dataset
    test: Dataset
    retain: Dataset
    forget: Dataset
    clean_forget: Dataset
    backdoor_test: Dataset
    client_datasets: list[Dataset]
    client_indices: list[list[int]]
    manifest: dict


def _fingerprint(config, labels):
    fields = {name: getattr(config, name) for name in (
        "dataset", "backbone", "num_clients", "target_client", "client_fraction", "seed",
        "validation_fraction", "partition", "dirichlet_alpha", "poison_fraction",
        "trigger_size", "trigger_value", "target_label", "rounds",
        "max_train_samples", "max_test_samples", "batch_size", "local_epochs",
        "learning_rate", "momentum", "image_size", "num_classes")}
    payload = json.dumps(fields, sort_keys=True).encode() + np.asarray(labels, dtype=np.uint8).tobytes()
    return hashlib.sha256(payload).hexdigest()


def create_manifest(config, labels):
    labels = np.asarray(labels)
    rng = np.random.default_rng(config.seed)
    validation = []
    train = []
    for klass in range(config.num_classes):
        indices = rng.permutation(np.flatnonzero(labels == klass))
        count = max(1, round(len(indices) * config.validation_fraction))
        validation.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    if config.max_train_samples is not None and len(train) > config.max_train_samples:
        train = rng.choice(train, config.max_train_samples, replace=False).tolist()
    if config.partition == "iid":
        clients = [chunk.tolist() for chunk in np.array_split(rng.permutation(train), config.num_clients)]
    else:
        clients = [[] for _ in range(config.num_clients)]
        for klass in range(config.num_classes):
            klass_indices = rng.permutation([i for i in train if labels[i] == klass])
            counts = rng.multinomial(len(klass_indices), rng.dirichlet(
                np.full(config.num_clients, config.dirichlet_alpha)))
            offset = 0
            for client_id, count in enumerate(counts):
                clients[client_id].extend(klass_indices[offset:offset + count].tolist())
                offset += count
        for client_id in range(config.num_clients):
            if not clients[client_id]:
                donor = max(range(config.num_clients), key=lambda i: len(clients[i]))
                clients[client_id].append(clients[donor].pop())
            rng.shuffle(clients[client_id])
    eligible = [i for i in clients[config.target_client] if labels[i] != config.target_label]
    poison_count = min(len(eligible), round(len(clients[config.target_client]) * config.poison_fraction))
    poison = sorted(rng.choice(eligible, poison_count, replace=False).tolist()) if poison_count else []
    clients_per_round = max(1, int(config.num_clients * config.client_fraction))
    schedule = [sorted(rng.choice(config.num_clients, clients_per_round, replace=False).tolist())
                for _ in range(config.rounds)]
    manifest = {
        "schema_version": 1, "fingerprint": _fingerprint(config, labels),
        "seed": config.seed, "validation_indices": sorted(validation),
        "client_indices": clients, "poison_indices": poison,
        "selected_clients_by_round": schedule,
        "train_count": len(train),
    }
    assert not set(validation).intersection(train)
    assert sorted(i for client in clients for i in client) == sorted(train)
    return manifest


def load_or_create_manifest(config, train, output_dir: Path):
    path = output_dir / "manifest.json"
    fingerprint = _fingerprint(config, train.targets)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError(f"Manifest configuration mismatch: {path}. Use a different output directory.")
    else:
        manifest = create_manifest(config, train.targets)
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return manifest


def build_protocol(config, output_dir: Path):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train = datasets.CIFAR10(root=config.data_root, train=True, transform=transform, download=config.download)
    raw_test = datasets.CIFAR10(root=config.data_root, train=False, transform=transform, download=config.download)
    manifest = load_or_create_manifest(config, train, output_dir)
    if "test_indices" not in manifest:
        test_rng = np.random.default_rng(config.seed + 818)
        test_ids = np.arange(len(raw_test))
        if config.max_test_samples is not None:
            test_ids = np.sort(test_rng.choice(test_ids, config.max_test_samples, replace=False))
        manifest["test_indices"] = test_ids.tolist()
        (output_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    test = Subset(raw_test, manifest["test_indices"])
    clients = manifest["client_indices"]
    target = config.target_client
    poison = manifest["poison_indices"]
    trigger = ((config.trigger_value - torch.tensor(MEAN)) / torch.tensor(STD)).tolist()
    return ProtocolData(
        train=train, validation=Subset(train, manifest["validation_indices"]),
        backdoor_validation=TriggeredSubset(train,
            [i for i in manifest["validation_indices"] if train.targets[i] != config.target_label],
            config.trigger_size, trigger, config.target_label),
        test=test,
        retain=Subset(train, [i for client_id, ids in enumerate(clients) if client_id != target for i in ids]),
        forget=PoisonedClient(train, clients[target], poison, config),
        clean_forget=Subset(train, clients[target]),
        backdoor_test=TriggeredSubset(raw_test, [i for i in manifest["test_indices"] if raw_test.targets[i] != config.target_label],
                                      config.trigger_size, trigger, config.target_label),
        client_datasets=[PoisonedClient(train, ids, poison if client_id == target else [], config)
                         for client_id, ids in enumerate(clients)],
        client_indices=clients, manifest=manifest,
    )

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def accuracy(model: torch.nn.Module, dataset: Dataset, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        logits = model(images.to(device))
        correct += int((logits.argmax(dim=1).cpu() == labels).sum())
        total += len(labels)
    return 100.0 * correct / total if total else float("nan")


@torch.no_grad()
def mean_cross_entropy(model: torch.nn.Module, dataset: Dataset, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        labels = labels.to(device)
        total_loss += float(F.cross_entropy(model(images.to(device)), labels, reduction="sum"))
        total += len(labels)
    return total_loss / total if total else float("nan")


def l2_distance(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for left, right in zip(model_a.parameters(), model_b.parameters()):
            total += float(torch.sum((left.detach().cpu() - right.detach().cpu()) ** 2))
    return total ** 0.5


def evaluate(model: torch.nn.Module, benchmark, batch_size: int, device: torch.device) -> dict[str, float]:
    return {
        "retain_accuracy": accuracy(model, benchmark.retain, batch_size, device),
        "forget_accuracy": accuracy(model, benchmark.forget, batch_size, device),
        "clean_forget_accuracy": accuracy(model, benchmark.clean_forget, batch_size, device),
        "backdoor_asr": accuracy(model, benchmark.backdoor_test, batch_size, device),
        "test_accuracy": accuracy(model, benchmark.test, batch_size, device),
        "test_loss": mean_cross_entropy(model, benchmark.test, batch_size, device),
    }

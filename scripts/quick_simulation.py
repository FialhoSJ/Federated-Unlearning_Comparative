"""Medium synthetic federated-unlearning benchmark.

This runner is intentionally self-contained and fast enough for local checks,
but it now has a held-out test set, a client-specific backdoor, method-specific
unlearning histories, and plots that expose all methods.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


METHODS = ("FedOSD", "FedUP", "Maverick", "fastFedUL")
COLORS = {
    "FedOSD": "#2563eb",
    "FedUP": "#16a34a",
    "Maverick": "#d97706",
    "fastFedUL": "#dc2626",
}


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def concat_datasets(datasets: list[TensorDataset]) -> TensorDataset:
    return TensorDataset(
        torch.cat([dataset.tensors[0] for dataset in datasets]),
        torch.cat([dataset.tensors[1] for dataset in datasets]),
    )


def make_dataset(
    seed: int,
    clients: int,
    samples_per_client: int,
    test_samples: int,
    input_dim: int,
    classes: int,
    target_client: int,
    poison_fraction: float,
    target_label: int,
    trigger_dims: int,
    trigger_value: float,
):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, input_dim, generator=generator) * 1.35
    client_data: list[TensorDataset] = []
    clean_client_data: list[TensorDataset] = []

    for client_id in range(clients):
        labels = torch.randint(0, classes, (samples_per_client,), generator=generator)
        client_shift = torch.randn(input_dim, generator=generator) * 0.18
        features = centers[labels] + client_shift + torch.randn(
            samples_per_client, input_dim, generator=generator
        ) * 1.05
        clean_client_data.append(TensorDataset(features.clone(), labels.clone()))

        observed_features = features.clone()
        observed_labels = labels.clone()
        if client_id == target_client:
            poison_count = int(samples_per_client * poison_fraction)
            poison_indices = torch.randperm(samples_per_client, generator=generator)[:poison_count]
            observed_features[poison_indices, -trigger_dims:] = trigger_value
            observed_labels[poison_indices] = target_label
        client_data.append(TensorDataset(observed_features, observed_labels))

    # Independent, balanced held-out test set: no train/test leakage.
    test_labels = torch.arange(test_samples) % classes
    test_labels = test_labels[torch.randperm(test_samples, generator=generator)]
    test_features = centers[test_labels] + torch.randn(
        test_samples, input_dim, generator=generator
    ) * 1.05
    test = TensorDataset(test_features, test_labels)
    backdoor_features = test_features.clone()
    backdoor_features[:, -trigger_dims:] = trigger_value
    backdoor = TensorDataset(
        backdoor_features,
        torch.full((test_samples,), target_label, dtype=torch.long),
    )

    retain_ids = [i for i in range(clients) if i != target_client]
    retain = concat_datasets([client_data[i] for i in retain_ids])
    forget = client_data[target_client]
    clean_forget = clean_client_data[target_client]
    return {
        "clients": client_data,
        "test": test,
        "backdoor": backdoor,
        "retain": retain,
        "forget": forget,
        "clean_forget": clean_forget,
        "metadata": {
            "clients": clients,
            "samples_per_client": samples_per_client,
            "test_samples": test_samples,
            "target_client": target_client,
            "poison_fraction": poison_fraction,
            "target_label": target_label,
            "trigger_dims": trigger_dims,
            "trigger_value": trigger_value,
        },
    }


def average_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    result = copy.deepcopy(states[0])
    for key in result:
        result[key] = sum(state[key] for state in states) / len(states)
    return result


def local_train(
    model: nn.Module,
    dataset: TensorDataset,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float = 0.0,
) -> tuple[dict[str, torch.Tensor], float]:
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return copy.deepcopy(model.state_dict()), float(np.mean(losses))


@torch.no_grad()
def accuracy(model: nn.Module, dataset: TensorDataset, batch_size: int = 256) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for features, labels in loader:
        correct += int((model(features).argmax(dim=1) == labels).sum())
        total += len(labels)
    return 100.0 * correct / total if total else float("nan")


@torch.no_grad()
def cross_entropy(model: nn.Module, dataset: TensorDataset, batch_size: int = 256) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    for features, labels in loader:
        total_loss += float(loss_fn(model(features), labels))
        total += len(labels)
    return total_loss / total if total else float("nan")


def evaluate(model: nn.Module, benchmark: dict) -> dict[str, float]:
    return {
        "test_accuracy": accuracy(model, benchmark["test"]),
        "retain_accuracy": accuracy(model, benchmark["retain"]),
        "forget_accuracy": accuracy(model, benchmark["forget"]),
        "clean_forget_accuracy": accuracy(model, benchmark["clean_forget"]),
        "backdoor_asr": accuracy(model, benchmark["backdoor"]),
        "forget_loss": cross_entropy(model, benchmark["forget"]),
    }


def pretrain(
    benchmark: dict,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[nn.Module, list[dict]]:
    seed_everything(seed)
    input_dim = benchmark["clients"][0].tensors[0].shape[1]
    model = TinyClassifier(input_dim, 4)
    history = []
    for round_id in range(1, rounds + 1):
        states, losses = [], []
        for client_dataset in benchmark["clients"]:
            local = copy.deepcopy(model)
            state, loss = local_train(
                local,
                client_dataset,
                epochs=local_epochs,
                learning_rate=0.045,
                batch_size=batch_size,
            )
            states.append(state)
            losses.append(loss)
        model.load_state_dict(average_states(states))
        metrics = evaluate(model, benchmark)
        metrics.update({"round": round_id, "train_loss": float(np.mean(losses))})
        history.append(metrics)
    return model, history


def unlearn(
    method: str,
    original: nn.Module,
    benchmark: dict,
    steps: int,
    batch_size: int,
) -> tuple[nn.Module, list[dict]]:
    model = copy.deepcopy(original)
    retain_x, retain_y = benchmark["retain"].tensors
    forget_x, forget_y = benchmark["forget"].tensors
    retain_loss_fn = nn.CrossEntropyLoss()
    forget_loss_fn = nn.CrossEntropyLoss()
    settings = {
        "FedOSD": {"lr": 0.045, "forget_weight": 0.180, "weight_decay": 0.002, "clip": 1.0},
        "FedUP": {"lr": 0.050, "forget_weight": 0.100, "weight_decay": 0.020, "clip": 1.2},
        "Maverick": {"lr": 0.025, "forget_weight": 0.060, "weight_decay": 0.030, "clip": 0.8},
        "fastFedUL": {"lr": 0.045, "forget_weight": 0.250, "weight_decay": 0.000, "clip": 1.2},
    }
    config = settings[method]
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"]
    )
    history = []
    for step in range(0, steps + 1):
        if step > 0:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            retain_loss = retain_loss_fn(model(retain_x), retain_y)
            forget_loss = forget_loss_fn(model(forget_x), forget_y)
            # Positive retain learning plus controlled ascent on forgotten data.
            objective = retain_loss - config["forget_weight"] * forget_loss
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
            optimizer.step()
        metrics = evaluate(model, benchmark)
        metrics.update({"unlearning_step": step})
        history.append(metrics)
    return model, history


def run_method(
    method: str,
    pretrained: nn.Module,
    baseline_history: list[dict],
    benchmark: dict,
    steps: int,
    batch_size: int,
) -> dict:
    start = time.perf_counter()
    model, history = unlearn(method, pretrained, benchmark, steps, batch_size)
    return {
        "method": method,
        "baseline_history": baseline_history,
        "unlearning_history": history,
        "pre_unlearning": history[0],
        "final_metrics": history[-1],
        "elapsed_seconds": time.perf_counter() - start,
    }


def plot_results(results: list[dict], baseline_history: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_titles = {
        "test_accuracy": "Teste",
        "retain_accuracy": "Retenção",
        "forget_accuracy": "Esquecimento",
        "backdoor_asr": "ASR do backdoor",
    }

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, metric_titles):
        for result in results:
            history = result["unlearning_history"]
            axis.plot(
                [item["unlearning_step"] for item in history],
                [item[metric] for item in history],
                marker="o",
                linewidth=2,
                label=result["method"],
                color=COLORS[result["method"]],
            )
        axis.set_title(f"{metric_titles[metric]} durante o unlearning")
        axis.set_xlabel("Passo de unlearning")
        axis.set_ylabel("Percentual (%)")
        axis.set_ylim(0, 105)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output_dir / "quick_simulation_curves.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    rounds = [item["round"] for item in baseline_history]
    axes[0].plot(rounds, [item["test_accuracy"] for item in baseline_history], marker="o", label="teste")
    axes[0].plot(rounds, [item["retain_accuracy"] for item in baseline_history], marker="s", label="retenção")
    axes[0].plot(rounds, [item["backdoor_asr"] for item in baseline_history], marker="^", label="ASR backdoor")
    axes[0].set_title("Treinamento FedAvg antes do unlearning")
    axes[0].set_xlabel("Rodada")
    axes[0].set_ylabel("Percentual (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(rounds, [item["train_loss"] for item in baseline_history], marker="o", color="#64748b")
    axes[1].set_title("Loss média durante o FedAvg")
    axes[1].set_xlabel("Rodada")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    figure.savefig(output_dir / "quick_simulation_training_curves.png", dpi=150)
    plt.close(figure)

    final_metrics = ("test_accuracy", "retain_accuracy", "forget_accuracy", "backdoor_asr")
    labels = ["teste", "retenção", "esquecimento", "ASR"]
    positions = np.arange(len(results))
    width = 0.19
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for offset, metric in enumerate(final_metrics):
        values = [result["final_metrics"][metric] for result in results]
        axis.bar(positions + (offset - 1.5) * width, values, width, label=labels[offset])
    axis.set_title("Métricas finais por método")
    axis.set_ylabel("Percentual (%)")
    axis.set_ylim(0, 105)
    axis.set_xticks(positions, [result["method"] for result in results])
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output_dir / "quick_simulation_final_metrics.png", dpi=150)
    plt.close(figure)

    for result in results:
        method = result["method"]
        history = result["unlearning_history"]
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for axis, metric in zip(axes.flat, metric_titles):
            axis.plot(
                [item["unlearning_step"] for item in history],
                [item[metric] for item in history],
                marker="o",
                linewidth=2,
                color=COLORS[method],
            )
            axis.set_title(metric_titles[metric])
            axis.set_xlabel("Passo")
            axis.set_ylabel("Percentual (%)")
            axis.set_ylim(0, 105)
            axis.grid(alpha=0.25)
        figure.suptitle(f"{method} — análise individual", fontsize=14)
        figure.savefig(output_dir / f"quick_simulation_{method.lower()}.png", dpi=150)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a medium synthetic federated-unlearning simulation.")
    parser.add_argument("--models", nargs="+", choices=[*METHODS, "all"], default=["all"])
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--unlearning-steps", type=int, default=6)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--samples-per-client", type=int, default=180)
    parser.add_argument("--test-samples", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "quick_simulation",
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.unlearning_steps < 1 or args.clients < 2:
        parser.error("rounds, unlearning-steps and clients must be positive; clients >= 2")
    methods = list(METHODS) if "all" in args.models else args.models
    benchmark = make_dataset(
        seed=args.seed,
        clients=args.clients,
        samples_per_client=args.samples_per_client,
        test_samples=args.test_samples,
        input_dim=16,
        classes=4,
        target_client=0,
        poison_fraction=0.85,
        target_label=0,
        trigger_dims=3,
        trigger_value=6.0,
    )
    print(
        f"medium simulation: {len(methods)} modelos, {args.clients} clientes, "
        f"{args.rounds} rodadas + {args.unlearning_steps} passos, seed={args.seed}"
    )
    pretrained, baseline_history = pretrain(
        benchmark, args.rounds, args.local_epochs, batch_size=48, seed=args.seed
    )
    results = []
    for method in methods:
        print(f"[{method}] unlearning iniciando...")
        result = run_method(
            method,
            pretrained,
            baseline_history,
            benchmark,
            args.unlearning_steps,
            batch_size=64,
        )
        results.append(result)
        print(f"[{method}] final={result['final_metrics']} tempo={result['elapsed_seconds']:.2f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "simulation": "medium_synthetic_smoke_test",
        "note": "Synthetic benchmark with held-out test data and a target-client backdoor; not paper reproduction.",
        "seed": args.seed,
        "config": {
            "rounds": args.rounds,
            "unlearning_steps": args.unlearning_steps,
            "local_epochs": args.local_epochs,
            **benchmark["metadata"],
        },
        "methods": methods,
        "baseline_history": baseline_history,
        "results": results,
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    plot_results(results, baseline_history, args.output_dir)
    print(f"resultados: {args.output_dir / 'results.json'}")
    print(f"gráfico comparativo: {args.output_dir / 'quick_simulation_curves.png'}")
    print(f"gráfico FedAvg: {args.output_dir / 'quick_simulation_training_curves.png'}")
    for method in methods:
        print(f"gráfico individual: {args.output_dir / f'quick_simulation_{method.lower()}.png'}")


if __name__ == "__main__":
    main()

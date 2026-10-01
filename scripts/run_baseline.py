from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ExperimentConfig
from common.data import build_cifar10_benchmark
from common.io import save_json
from common.metrics import evaluate
from common.model import SmallCifarCNN
from common.seed import set_seed


def average_models(states, weights):
    if len(states) != len(weights) or not states or sum(weights) <= 0:
        raise ValueError("states and positive weights must have matching lengths")
    result = copy.deepcopy(states[0])
    total_weight = sum(weights)
    for key in result:
        if torch.is_floating_point(result[key]):
            result[key] = sum(state[key] * weight for state, weight in zip(states, weights)) / total_weight
    return result


def train_local(model, dataset, config, device):
    model.train()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    for _ in range(config.local_epochs):
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return model.state_dict(), sum(losses) / len(losses) if losses else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the common CIFAR-10 FedAvg sanity baseline.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cifar10_client_backdoor.json")
    parser.add_argument("--download", action="store_true", help="download CIFAR-10 if it is not cached")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.download:
        config.download = True
    config.validate()
    set_seed(config.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    benchmark = build_cifar10_benchmark(config)
    model = SmallCifarCNN(config.num_classes).to(device)
    history = []
    start = time.perf_counter()
    clients_per_round = max(1, int(config.client_fraction * config.num_clients))
    client_rng = np.random.default_rng(config.seed)
    for round_id in range(1, config.rounds + 1):
        selected = sorted(int(i) for i in client_rng.choice(config.num_clients, clients_per_round, replace=False))
        states, losses, weights = [], [], []
        for client_id in selected:
            local = copy.deepcopy(model)
            state, loss = train_local(local, benchmark.client_datasets[client_id], config, device)
            states.append(state)
            losses.append(loss)
            weights.append(len(benchmark.client_datasets[client_id]))
        model.load_state_dict(average_models(states, weights))
        metrics = evaluate(model, benchmark, config.batch_size, device)
        metrics.update({"round": round_id, "train_loss": sum(losses) / len(losses), "selected_clients": selected})
        history.append(metrics)
        print(metrics)
    payload = {"config": config.to_dict(), "method": "FedAvg_original", "seed": config.seed,
               "device": str(device), "elapsed_seconds": time.perf_counter() - start,
               "history": history, "final_metrics": history[-1]}
    output = Path(config.output_dir) / f"fedavg_seed{config.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output.with_suffix(".pth"))
    save_json(output, payload)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

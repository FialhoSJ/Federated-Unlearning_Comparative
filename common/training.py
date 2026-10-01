"""FedAvg training with a frozen participation schedule and update ledger."""
from __future__ import annotations

import copy
import time

import torch
from torch.utils.data import DataLoader

from .metrics import accuracy
from .model import ResNet18GroupNorm, SmallCifarCNN


def make_model(config, device, state=None):
    model_class = ResNet18GroupNorm if config.backbone == "resnet18_gn" else SmallCifarCNN
    model = model_class(config.num_classes).to(device)
    if state is not None:
        model.load_state_dict(state)
    return model


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def local_train(model, dataset, config, device, epochs=None):
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    model.train()
    total_loss = 0.0
    total_count = 0
    for _ in range(config.local_epochs if epochs is None else epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            total_count += len(y)
    return cpu_state(model), total_loss / total_count if total_count else float("nan")


def train_federated(config, data, device, initial_state, exclude_target=False, store_target_updates=False):
    model = make_model(config, device, initial_state)
    retained_total = sum(len(ds) for i, ds in enumerate(data.client_datasets) if i != config.target_client)
    all_total = sum(len(ds) for ds in data.client_datasets)
    total = retained_total if exclude_target else all_total
    ledger = []
    history = []
    start = time.perf_counter()
    for round_id, scheduled in enumerate(data.manifest["selected_clients_by_round"], 1):
        selected = [i for i in scheduled if not exclude_target or i != config.target_client]
        old = cpu_state(model)
        states = []
        losses = []
        for client_id in selected:
            local = make_model(config, device, old)
            state, loss = local_train(local, data.client_datasets[client_id], config, device)
            states.append(state)
            losses.append(loss)
            if store_target_updates and client_id == config.target_client:
                ledger_type = torch.float16 if config.ledger_dtype == "float16" else torch.float32
                ledger.append({"round": round_id - 1, "weight": len(data.client_datasets[client_id]) / total,
                               "delta": {key: (old[key] - state[key]).to(ledger_type) for key in old}})
        # weighted_com: clients not selected retain their share of the old model.
        weights = [len(data.client_datasets[i]) / total for i in selected]
        remaining_weight = 1.0 - sum(weights)
        updated = {}
        for key in old:
            if torch.is_floating_point(old[key]):
                updated[key] = old[key] * remaining_weight + sum(
                    (state[key] * weight for state, weight in zip(states, weights)),
                    torch.zeros_like(old[key]))
            else:
                updated[key] = old[key]
        model.load_state_dict(updated)
        validation_accuracy = accuracy(model, data.validation, config.batch_size, device)
        validation_asr = accuracy(model, data.backdoor_validation, config.batch_size, device)
        record = {"round": round_id, "selected_clients": selected,
                  "validation_accuracy": validation_accuracy,
                  "validation_asr": validation_asr,
                  "train_loss": sum(losses) / len(losses) if losses else None}
        history.append(record)
        print(f"round {round_id}/{config.rounds}: val={validation_accuracy:.2f}% val_ASR={validation_asr:.2f}% clients={selected}", flush=True)
    return model, history, ledger, time.perf_counter() - start

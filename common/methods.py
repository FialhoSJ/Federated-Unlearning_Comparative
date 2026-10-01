"""Explicit CIFAR-10 adaptations of the four published unlearning mechanisms.

These are not byte-for-byte executions of the authors' heterogeneous frameworks.
The common model, partition and evaluator make their mechanisms comparable.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import SmallCifarCNN
from .training import cpu_state, local_train, make_model


def fine_tune(config, data, device, original_state):
    model = make_model(config, device, original_state)
    history = []
    start = time.perf_counter()
    for round_id in range(config.fine_tune_rounds):
        states = []
        weights = []
        for client_id, dataset in enumerate(data.client_datasets):
            if client_id == config.target_client:
                continue
            local = make_model(config, device, cpu_state(model))
            state, _ = local_train(local, dataset, config, device)
            states.append(state)
            weights.append(len(dataset))
        total = sum(weights)
        model.load_state_dict({key: sum((s[key] * w / total for s, w in zip(states, weights)),
                                        torch.zeros_like(states[0][key])) for key in states[0]})
        history.append({"step": round_id + 1, "retained_clients": len(states)})
    return model, history, {"additional_rounds": config.fine_tune_rounds,
                            "elapsed_seconds": time.perf_counter() - start}


def _dataset_gradient(model, dataset, device, batch_size, criterion):
    model.eval()
    params = list(model.parameters())
    gradient = torch.zeros(sum(p.numel() for p in params), device=device)
    seen = 0
    for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        pieces = torch.autograd.grad(loss, params)
        gradient += torch.cat([piece.reshape(-1) for piece in pieces]) * len(labels)
        seen += len(labels)
    return gradient / max(seen, 1)


def _uce(logits, labels):
    p_true = torch.softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    return -torch.log1p(-p_true / 2).mean()


def _apply_direction(model, direction, lr):
    position = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            parameter -= lr * direction[position:position + size].view_as(parameter)
            position += size


def fedosd(config, data, device, original_state):
    model = make_model(config, device, original_state)
    initial = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    history = []
    start = time.perf_counter()
    retained = [i for i in range(config.num_clients) if i != config.target_client]
    for step in range(config.unlearning_rounds):
        forget_gradient = _dataset_gradient(model, data.forget, device, config.batch_size, _uce)
        retained_gradients = torch.stack([
            _dataset_gradient(model, data.client_datasets[i], device, config.batch_size,
                              torch.nn.functional.cross_entropy) for i in retained])
        gram = retained_gradients @ retained_gradients.T
        direction = forget_gradient - retained_gradients.T @ (
            torch.linalg.pinv(gram) @ (retained_gradients @ forget_gradient))
        norm = direction.norm()
        if norm > 1e-12:
            direction = direction * (forget_gradient.norm() / norm)
        _apply_direction(model, direction, config.unlearning_lr)
        history.append({"step": step + 1, "stage": "unlearn", "gradient_norm": float(forget_gradient.norm()),
                        "projected_norm": float(direction.norm())})
    for step in range(config.recovery_rounds):
        displacement = torch.nn.utils.parameters_to_vector(model.parameters()).detach() - initial
        distance_sq = displacement.square().sum()
        gradients = []
        for client_id in retained:
            gradient = _dataset_gradient(model, data.client_datasets[client_id], device,
                                         config.batch_size, torch.nn.functional.cross_entropy)
            norm = gradient.norm()
            if distance_sq > 1e-12:
                gradient = gradient - (gradient @ displacement) / distance_sq * displacement
            if gradient.norm() > 1e-12:
                gradient = gradient * (norm / gradient.norm())
            gradients.append(gradient)
        _apply_direction(model, torch.stack(gradients).mean(0), config.recovery_lr)
        history.append({"step": config.unlearning_rounds + step + 1, "stage": "recovery",
                        "distance_from_original": float(displacement.norm())})
    return model, history, {"additional_rounds": config.unlearning_rounds + config.recovery_rounds,
                            "elapsed_seconds": time.perf_counter() - start,
                            "source": "FedOSD UCE/projection/recovery port"}


class FedUPFilter(nn.Module):
    def __init__(self, config, base):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        dimension = base.feature_dim
        self.adapter = nn.Sequential(nn.Linear(dimension, 32, bias=False), nn.ReLU(),
                                     nn.Linear(32, dimension, bias=False), nn.ReLU())

    def forward(self, images):
        features = self.base.forward_features(images)
        return self.base.classify_features(self.adapter(features))


def _centroids(points, count, rng):
    if len(points) <= count:
        return points
    centers = points[rng.choice(len(points), count, replace=False)].clone()
    for _ in range(15):
        assignments = torch.cdist(points, centers).argmin(1)
        new = torch.stack([points[assignments == k].mean(0) if (assignments == k).any()
                           else centers[k] for k in range(count)])
        if torch.allclose(new, centers, atol=1e-4):
            break
        centers = new
    return centers


def fedup(config, data, device, original_state):
    base = make_model(config, device, original_state)
    base.eval()
    rng = np.random.default_rng(config.seed)
    vectors = []
    targets = []
    collected_samples = 0
    start = time.perf_counter()
    with torch.no_grad():
        for client_id, dataset in enumerate(data.client_datasets):
            if client_id == config.target_client:
                continue
            by_class = {i: [] for i in range(config.num_classes)}
            positions = rng.choice(len(dataset), min(len(dataset), config.fedup_features_per_client), replace=False)
            selected = torch.utils.data.Subset(dataset, positions.tolist())
            collected_samples += len(selected)
            for images, labels in DataLoader(selected, batch_size=config.batch_size):
                features = base.forward_features(images.to(device)).cpu()
                for feature, label in zip(features, labels):
                    by_class[int(label)].append(feature)
            # Each retained client compresses its own features before upload.
            for klass, features in by_class.items():
                if not features:
                    continue
                class_centroids = _centroids(torch.stack(features), config.fedup_centroids_per_class, rng)
                vectors.append(class_centroids)
                targets.extend([klass] * len(class_centroids))
    vectors = torch.cat(vectors).to(device)
    targets = torch.tensor(targets, device=device)
    model = FedUPFilter(config, base).to(device)
    optimizer = torch.optim.SGD(model.adapter.parameters(), lr=config.fedup_lr)
    history = []
    for epoch in range(config.fedup_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        adapted = model.adapter(vectors)
        logits = model.base.classify_features(adapted)
        loss = 0.5 * torch.nn.functional.cross_entropy(logits, targets) + 0.5 * torch.nn.functional.mse_loss(adapted, vectors)
        loss.backward()
        optimizer.step()
        history.append({"step": epoch + 1, "centroid_loss": float(loss.detach())})
    return model, history, {"additional_rounds": 0, "centroids": len(targets),
                            "centroid_collection_samples": collected_samples,
                            "elapsed_seconds": time.perf_counter() - start,
                            "source": "FedUP centroid/filter port; no formal DP guarantee"}


def maverick(config, data, device, original_state):
    model = make_model(config, device, original_state)
    reference = make_model(config, device, original_state)
    reference.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.maverick_lr, momentum=config.momentum)
    loader = DataLoader(data.forget, batch_size=config.batch_size, shuffle=True)
    history = []
    start = time.perf_counter()
    for epoch in range(config.maverick_epochs):
        total_loss = 0.0
        batches = 0
        model.train()
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            sensitivity = 0
            for _ in range(config.maverick_noise_samples):
                sigma = float(np.random.uniform(config.maverick_sigma_min, config.maverick_sigma_max))
                perturbed = images.clone()
                noise = torch.randn_like(perturbed[:, :, -config.trigger_size:, -config.trigger_size:]) * sigma
                perturbed[:, :, -config.trigger_size:, -config.trigger_size:] += noise
                with torch.no_grad():
                    reference_output = reference(perturbed)
                numerator = torch.linalg.vector_norm(output - reference_output, dim=1)
                denominator = torch.linalg.vector_norm(noise.flatten(1), dim=1).clamp_min(1e-8)
                sensitivity = sensitivity + (numerator / denominator).mean()
            loss = sensitivity / config.maverick_noise_samples
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        history.append({"step": epoch + 1, "sensitivity": total_loss / max(batches, 1)})
    return model, history, {"additional_rounds": 0, "elapsed_seconds": time.perf_counter() - start,
                            "source": "Maverick sensitivity/noise port"}


def fastfedul(config, data, device, original_state, ledger):
    if ledger is None:
        raise ValueError("Fast-FedUL requires the saved pretraining update ledger.")
    start = time.perf_counter()
    updates = {item["round"]: item for item in ledger}
    correction = {key: torch.zeros_like(value) for key, value in original_state.items()}
    all_total = sum(len(dataset) for dataset in data.client_datasets)
    for round_id, selected in enumerate(data.manifest["selected_clients_by_round"]):
        retained_share = sum(len(data.client_datasets[i]) / all_total for i in selected
                             if i != config.target_client)
        beta = 1.0 - config.fastfedul_alpha * retained_share
        for key in correction:
            correction[key] *= beta
            if round_id in updates:
                correction[key] += (updates[round_id]["weight"] *
                                    updates[round_id]["delta"][key].to(correction[key].dtype))
    corrected = {key: original_state[key] + config.fastfedul_theta * correction[key]
                 for key in original_state}
    model = make_model(config, device, corrected)
    return model, [{"step": 1, "corrected_rounds": len(updates)}], {
        "additional_rounds": 0, "stored_target_updates": len(updates),
        "stored_bytes": sum(t.numel() * t.element_size() for update in ledger for t in update["delta"].values()),
        "ledger_dtype": config.ledger_dtype,
        "elapsed_seconds": time.perf_counter() - start,
        "source": "Fast-FedUL update-correction port with complete target history"}

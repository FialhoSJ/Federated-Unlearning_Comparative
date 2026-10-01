"""One evaluator for all methods, with a clearly labelled MIA diagnostic."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def _collect(model, dataset, batch_size, device, keep_scores=False):
    model.eval()
    correct = 0
    loss_sum = 0.0
    count = 0
    features = []
    probabilities = []
    for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        losses = F.cross_entropy(logits, labels, reduction="none")
        probs = torch.softmax(logits, dim=1)
        correct += int((logits.argmax(1) == labels).sum())
        loss_sum += float(losses.sum())
        count += len(labels)
        if keep_scores:
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(1)
            features.append(torch.stack((losses, entropy), dim=1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())
    return {"accuracy": 100 * correct / count if count else float("nan"),
            "loss": loss_sum / count if count else float("nan"), "count": count,
            "features": np.concatenate(features) if features else None,
            "probabilities": np.concatenate(probabilities) if probabilities else None}


def _sample(dataset, count, rng):
    ids = rng.choice(len(dataset), min(count, len(dataset)), replace=False)
    return Subset(dataset, ids.tolist())


def evaluate_all(model, data, config, device, retrain_model=None, mia_sample_size=1000):
    measured = {}
    for name in ("retain", "forget", "clean_forget", "backdoor_test", "test", "validation", "backdoor_validation"):
        measured[name] = _collect(model, getattr(data, name), config.batch_size, device)
    output = {
        "retain_accuracy": measured["retain"]["accuracy"],
        "forget_accuracy": measured["forget"]["accuracy"],
        "clean_forget_accuracy": measured["clean_forget"]["accuracy"],
        "backdoor_asr": measured["backdoor_test"]["accuracy"],
        "test_accuracy": measured["test"]["accuracy"],
        "test_loss": measured["test"]["loss"],
        "validation_accuracy": measured["validation"]["accuracy"],
        "validation_asr": measured["backdoor_validation"]["accuracy"],
        "counts": {name: result["count"] for name, result in measured.items()},
    }
    # Attack calibration uses retained members and held-out validation
    # nonmembers. The clean forget/test AUC is a diagnostic, not a privacy proof.
    rng = np.random.default_rng(config.seed + 1009)
    member = _collect(model, _sample(data.retain, mia_sample_size, rng), config.batch_size, device, True)
    calibration = _collect(model, _sample(data.validation, mia_sample_size, rng), config.batch_size, device, True)
    forgotten = _collect(model, _sample(data.clean_forget, mia_sample_size, rng), config.batch_size, device, True)
    nonmember = _collect(model, _sample(data.test, mia_sample_size, rng), config.batch_size, device, True)
    x_train = np.vstack((member["features"], calibration["features"]))
    y_train = np.r_[np.ones(len(member["features"])), np.zeros(len(calibration["features"]))]
    attack = LogisticRegression(max_iter=300, class_weight="balanced", random_state=config.seed).fit(x_train, y_train)
    member_scores = attack.predict_proba(forgotten["features"])[:, 1]
    nonmember_scores = attack.predict_proba(nonmember["features"])[:, 1]
    attack_truth = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    attack_scores = np.r_[member_scores, nonmember_scores]
    auc = roc_auc_score(attack_truth, attack_scores)
    output["mia_proxy_auc_clean"] = float(auc)
    output["mia_proxy_advantage"] = float(2 * abs(auc - 0.5))
    output["mia_proxy_balanced_accuracy_clean"] = float(
        balanced_accuracy_score(attack_truth, attack_scores >= 0.5))
    output["mia_proxy_member_probability"] = float(member_scores.mean())
    if retrain_model is not None:
        test_subset = _sample(data.test, mia_sample_size, np.random.default_rng(config.seed + 5003))
        current = _collect(model, test_subset, config.batch_size, device, True)["probabilities"]
        reference = _collect(retrain_model, test_subset, config.batch_size, device, True)["probabilities"]
        output["test_probability_l1_to_retrain"] = float(np.abs(current - reference).mean())
        current_state = model.state_dict()
        retrain_state = retrain_model.state_dict()
        if current_state.keys() == retrain_state.keys():
            output["parameter_l2_to_retrain"] = float(sum(
                (current_state[key].detach().cpu() - retrain_state[key].detach().cpu()).square().sum().item()
                for key in current_state) ** 0.5)
    return output

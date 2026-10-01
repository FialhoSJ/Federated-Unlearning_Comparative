"""Reproducible CIFAR-10 client-unlearning comparison (PowerShell friendly)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ExperimentConfig
from common.evaluation import evaluate_all
from common.methods import fedosd, fedup, fastfedul, fine_tune, maverick
from common.protocol import build_protocol
from common.seed import set_seed
from common.training import cpu_state, make_model, train_federated

METHODS = ("original", "retrain", "fine_tune", "fedosd", "fedup", "maverick", "fastfedul")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def source_revision(name):
    source = PROJECT / "repositories" / name
    if (source / ".git").exists():
        result = subprocess.run(["git", "-c", f"safe.directory={source.as_posix()}", "-C", str(source),
                                 "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    revision_file = source / "SOURCE_REVISION"
    return revision_file.read_text(encoding="utf-8").strip() if revision_file.exists() else None


def metadata(config, data):
    archive = Path(config.data_root) / "cifar-10-python.tar.gz"
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                            text=True, check=False)
    source_files = [
        ROOT / "common/protocol.py", ROOT / "common/training.py", ROOT / "common/methods.py",
        ROOT / "common/evaluation.py", ROOT / "common/model.py",
        ROOT / "repositories/FedOSD/tfedplat/algorithm/unlearning/FedOSD.py",
        ROOT / "repositories/FedUP-code/algs/my_forget.py",
        ROOT / "repositories/fastFedUL/algorithm/fast_fedUL.py",
    ]
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__,
        "sklearn": sklearn.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pip_freeze": freeze.stdout.splitlines() if freeze.returncode == 0 else None,
        "manifest_sha256": hashlib.sha256(json.dumps(data.manifest, sort_keys=True).encode()).hexdigest(),
        "dataset_archive_sha256": digest(archive) if archive.exists() else None,
        "source_commits": {name: source_revision(name) for name in
                           ("FedOSD", "FedUP-code", "fastFedUL")},
        "source_file_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
                               for path in source_files},
    }


def load_state(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run --phase pretrain first.")
    return torch.load(path, map_location="cpu", weights_only=True)


def model_bytes(model):
    return sum(value.numel() * value.element_size() for value in model.state_dict().values())


def result_payload(method, config, data, model, history, costs, device, retrain_model=None):
    costs = dict(costs)
    costs["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters()
                                         if parameter.requires_grad)
    costs["checkpoint_model_bytes"] = model_bytes(model)
    parameter_bytes = model_bytes(model)
    if method == "fedosd":
        costs["unlearning_communication_bytes_estimated"] = parameter_bytes * 2 * (
            config.unlearning_rounds * config.num_clients +
            config.recovery_rounds * (config.num_clients - 1))
    elif method == "fedup":
        costs["unlearning_communication_bytes_estimated"] = (
            costs.get("centroids", 0) * model.base.feature_dim * 4 +
            sum(p.numel() * p.element_size() for p in model.adapter.parameters()))
    elif method == "maverick":
        costs["unlearning_communication_bytes_estimated"] = parameter_bytes * 2
    elif method == "fastfedul":
        costs["unlearning_communication_bytes_estimated"] = 0
    elif method == "fine_tune":
        costs["unlearning_communication_bytes_estimated"] = (
            parameter_bytes * 2 * (config.num_clients - 1) * config.fine_tune_rounds)
    if device.type == "cuda":
        costs["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    result = {
        "schema_version": 1, "protocol": "cifar10_client_backdoor_common_v1",
        "implementation": "adapted" if method in {"fedosd", "fedup", "maverick", "fastfedul"} else "baseline",
        "method": method, "seed": config.seed, "config": config.to_dict(),
        "manifest_fingerprint": data.manifest["fingerprint"], "device": str(device),
        "history": history, "final_metrics": evaluate_all(model, data, config, device, retrain_model),
        "costs": costs,
    }
    return result


def execute(config, phase, methods, device_name, force):
    config.validate()
    seed_dir = Path(config.output_dir) / f"seed_{config.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = build_protocol(config, seed_dir)
    save_json(seed_dir / "environment.json", metadata(config, data))
    print(f"manifest: {seed_dir / 'manifest.json'}", flush=True)
    if phase == "prepare":
        return

    initial_path = seed_dir / "initial.pth"
    original_path = seed_dir / "original.pth"
    ledger_path = seed_dir / "target_updates.pth"
    if not initial_path.exists() or (force and phase in {"pretrain", "all"}):
        set_seed(config.seed)
        torch.save(cpu_state(make_model(config, torch.device("cpu"))), initial_path)

    if phase in {"pretrain", "all"} and (force or not original_path.exists()):
        set_seed(config.seed)
        model, history, ledger, seconds = train_federated(
            config, data, device, load_state(initial_path), store_target_updates=True)
        torch.save(cpu_state(model), original_path)
        torch.save(ledger, ledger_path)
        communication = model_bytes(model) * 2 * sum(len(x) for x in data.manifest["selected_clients_by_round"])
        payload = result_payload("original", config, data, model, history, {
            "pretraining_seconds": seconds, "communication_bytes_estimated": communication,
            "stored_target_update_bytes": ledger_path.stat().st_size, "additional_rounds": 0}, device)
        payload["checkpoint_sha256"] = digest(original_path)
        save_json(seed_dir / "original.json", payload)
        print(f"pretrained: {original_path}", flush=True)
    if phase == "pretrain":
        return

    original_state = load_state(original_path)
    if not ledger_path.exists() and "fastfedul" in methods:
        raise FileNotFoundError(f"Missing {ledger_path}; pretraining must save the update ledger.")
    retrain_path = seed_dir / "retrain.pth"
    if "retrain" in methods or any(x in methods for x in ("fine_tune", "fedosd", "fedup", "maverick", "fastfedul")):
        if force or not retrain_path.exists():
            set_seed(config.seed)
            retrained, history, _, seconds = train_federated(
                config, data, device, load_state(initial_path), exclude_target=True)
            torch.save(cpu_state(retrained), retrain_path)
            payload = result_payload("retrain", config, data, retrained, history,
                                     {"elapsed_seconds": seconds, "additional_rounds": config.rounds}, device)
            payload["checkpoint_sha256"] = digest(retrain_path)
            save_json(seed_dir / "retrain.json", payload)
            print(f"retrained: {retrain_path}", flush=True)
        retrain_model = make_model(config, device, load_state(retrain_path))
    else:
        retrain_model = None
    method_functions = {"fine_tune": fine_tune, "fedosd": fedosd,
                        "fedup": fedup, "maverick": maverick}
    for method in methods:
        if method in {"original", "retrain"}:
            continue
        output = seed_dir / f"{method}.json"
        if output.exists() and not force:
            print(f"skipping existing: {output}", flush=True)
            continue
        set_seed(config.seed + 71)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if method == "fastfedul":
            ledger = load_state(ledger_path)
            model, history, costs = fastfedul(config, data, device, original_state, ledger)
        else:
            model, history, costs = method_functions[method](config, data, device, original_state)
        checkpoint = seed_dir / f"{method}.pth"
        torch.save(cpu_state(model), checkpoint)
        payload = result_payload(method, config, data, model, history, costs, device, retrain_model)
        payload["original_checkpoint_sha256"] = digest(original_path)
        payload["checkpoint_sha256"] = digest(checkpoint)
        save_json(output, payload)
        print(f"completed: {method} -> {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cifar10_client_backdoor.json")
    parser.add_argument("--phase", choices=("prepare", "pretrain", "run", "all"), default="all")
    parser.add_argument("--methods", nargs="+", choices=METHODS + ("all",), default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true", help="rerun and replace this seed's checkpoints/results")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    # Relative paths are anchored to comparative_benchmark, independent of cwd.
    for field in ("data_root", "output_dir"):
        path = Path(getattr(config, field))
        if not path.is_absolute():
            setattr(config, field, str((ROOT / path).resolve()))
    methods = list(METHODS) if "all" in args.methods else args.methods
    for seed in args.seeds or [config.seed]:
        config.seed = seed
        execute(config, args.phase, methods, args.device, args.force)


if __name__ == "__main__":
    main()

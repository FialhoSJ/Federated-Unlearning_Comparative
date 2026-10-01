"""Aggregate paired seeds and create article tables/figures from common results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("test_accuracy", "retain_accuracy", "clean_forget_accuracy",
           "backdoor_asr", "test_loss", "mia_proxy_auc_clean", "mia_proxy_advantage",
           "mia_proxy_balanced_accuracy_clean",
           "test_probability_l1_to_retrain")
COSTS = ("elapsed_seconds", "pretraining_seconds", "unlearning_communication_bytes_estimated",
         "communication_bytes_estimated", "stored_bytes", "stored_target_update_bytes",
         "trainable_parameters", "checkpoint_model_bytes", "peak_cuda_memory_bytes")
METHODS = ("original", "retrain", "fine_tune", "fedosd", "fedup", "maverick", "fastfedul")


def paired_ci(values, rng, draws=2000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [None, None]
    boot = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/cifar10_client_backdoor")
    parser.add_argument("--require-seeds", type=int, default=1)
    parser.add_argument("--require-all", action="store_true", help="require every method for the same seeds")
    parser.add_argument("--min-original-validation-accuracy", type=float, default=None)
    parser.add_argument("--min-original-validation-asr", type=float, default=None)
    args = parser.parse_args()
    rows = {}
    for seed_dir in sorted(args.results.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        for method in METHODS:
            path = seed_dir / f"{method}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("protocol") != "cifar10_client_backdoor_common_v1":
                    continue
                rows.setdefault(method, {})[seed] = data
    if not rows:
        raise SystemExit("No common-protocol result JSONs found.")
    if args.require_all:
        missing = [method for method in METHODS if method not in rows]
        if missing:
            raise SystemExit(f"Missing methods: {missing}")
        common_seeds = set.intersection(*(set(rows[method]) for method in METHODS))
        if len(common_seeds) < args.require_seeds:
            raise SystemExit(f"Only {len(common_seeds)} seeds contain all methods; required {args.require_seeds}")
        rows = {method: {seed: record for seed, record in method_rows.items() if seed in common_seeds}
                for method, method_rows in rows.items()}
    fingerprints = {seed: {record["manifest_fingerprint"] for method_rows in rows.values()
                            for candidate, record in method_rows.items() if candidate == seed}
                    for seed in {s for method_rows in rows.values() for s in method_rows}}
    if any(len(values) != 1 for values in fingerprints.values()):
        raise SystemExit("Methods within a seed used different manifests; refusing to aggregate.")
    reference_configs = {json.dumps({key: value for key, value in record["config"].items()
                                     if key not in {"seed", "output_dir"}}, sort_keys=True)
                         for method_rows in rows.values() for record in method_rows.values()}
    if len(reference_configs) != 1:
        raise SystemExit("Results use different experimental configurations; refusing to aggregate.")
    for seed, record in rows.get("original", {}).items():
        scores = record["final_metrics"]
        if args.min_original_validation_accuracy is not None and scores["validation_accuracy"] < args.min_original_validation_accuracy:
            raise SystemExit(f"seed {seed}: original validation accuracy {scores['validation_accuracy']:.2f}% below gate")
        if args.min_original_validation_asr is not None and scores["validation_asr"] < args.min_original_validation_asr:
            raise SystemExit(f"seed {seed}: original validation ASR {scores['validation_asr']:.2f}% below gate")
    summary_dir = args.results / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260926)
    summary = []
    for method in METHODS:
        method_rows = rows.get(method, {})
        if not method_rows:
            continue
        if len(method_rows) < args.require_seeds:
            raise SystemExit(f"{method}: {len(method_rows)} seeds; required {args.require_seeds}")
        paired = sorted(set(method_rows) & set(rows.get("retrain", {})))
        paired_original = sorted(set(method_rows) & set(rows.get("original", {})))
        for metric in METRICS:
            values = [record["final_metrics"][metric] for _, record in sorted(method_rows.items())
                      if metric in record["final_metrics"]]
            if not values:
                continue
            differences = [method_rows[s]["final_metrics"][metric] -
                           rows["retrain"][s]["final_metrics"][metric] for s in paired
                           if metric in method_rows[s]["final_metrics"]
                           and metric in rows["retrain"][s]["final_metrics"]]
            original_differences = [method_rows[s]["final_metrics"][metric] -
                                    rows["original"][s]["final_metrics"][metric]
                                    for s in paired_original
                                    if metric in method_rows[s]["final_metrics"]
                                    and metric in rows["original"][s]["final_metrics"]]
            summary.append({"method": method, "metric": metric, "n": len(values),
                            "mean": float(np.mean(values)),
                            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                            "paired_n": len(differences),
                            "difference_to_retrain": float(np.mean(differences)) if differences else None,
                            "paired_ci95": paired_ci(differences, rng) if differences else [None, None],
                            "paired_original_n": len(original_differences),
                            "difference_to_original": float(np.mean(original_differences)) if original_differences else None,
                            "original_ci95": paired_ci(original_differences, rng) if original_differences else [None, None]})
        for metric in COSTS:
            values = [record["costs"][metric] for _, record in sorted(method_rows.items())
                      if metric in record.get("costs", {})]
            if values:
                summary.append({"method": method, "metric": f"cost_{metric}", "n": len(values),
                                "mean": float(np.mean(values)),
                                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                "paired_n": 0, "difference_to_retrain": None,
                                "paired_ci95": [None, None], "paired_original_n": 0,
                                "difference_to_original": None, "original_ci95": [None, None]})
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (summary_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHODS:
        if method not in rows:
            continue
        xs = [record["final_metrics"]["backdoor_asr"] for record in rows[method].values()]
        ys = [record["final_metrics"]["test_accuracy"] for record in rows[method].values()]
        ax.scatter(xs, ys, alpha=0.55)
        ax.scatter(np.mean(xs), np.mean(ys), s=90, label=f"{method} (n={len(xs)})")
    ax.set(xlabel="ASR condicional (%) - menor é melhor", ylabel="Acurácia no teste (%) - maior é melhor",
           title="Utilidade e esquecimento por seed")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(summary_dir / "utility_vs_asr.png", dpi=300)
    plt.close(fig)
    print(f"summary: {summary_dir}")


if __name__ == "__main__":
    main()

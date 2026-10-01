"""Create line and bar charts from a common-protocol pilot's saved JSON results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("original", "retrain", "fine_tune", "fedosd", "fedup", "maverick", "fastfedul")
LABELS = {
    "original": "Original",
    "retrain": "Retreinamento",
    "fine_tune": "Fine-tuning",
    "fedosd": "FedOSD",
    "fedup": "FedUP",
    "maverick": "Maverick",
    "fastfedul": "Fast-FedUL",
}
COLORS = {
    "original": "#4C78A8",
    "retrain": "#F58518",
    "fine_tune": "#54A24B",
    "fedosd": "#E45756",
    "fedup": "#72B7B2",
    "maverick": "#B279A2",
    "fastfedul": "#FF9DA6",
}


def load_records(results: Path, require_seeds: int):
    records = {method: {} for method in METHODS}
    for seed_dir in sorted(results.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        for method in METHODS:
            path = seed_dir / f"{method}.json"
            if path.exists():
                records[method][seed] = json.loads(path.read_text(encoding="utf-8"))

    missing = {method: require_seeds - len(values) for method, values in records.items()
               if len(values) < require_seeds}
    if missing:
        details = ", ".join(f"{method}: faltam {count}" for method, count in missing.items())
        raise SystemExit(f"Resultados incompletos em {results}: {details}")
    return records


def mean_std(values):
    array = np.asarray(values, dtype=float)
    mean = array.mean(axis=0)
    std = array.std(axis=0, ddof=1) if array.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def plot_round_history(records: dict, output: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = (
        ("validation_accuracy", "Acurácia de validação (%)", "maior é melhor"),
        ("validation_asr", "ASR de validação (%)", "menor é melhor"),
        ("train_loss", "Perda de treino", "menor é melhor"),
    )
    for ax, (key, ylabel, direction) in zip(axes, metrics):
        for method in ("original", "retrain"):
            series = []
            for record in records[method].values():
                history = sorted(record["history"], key=lambda row: row["round"])
                series.append([row[key] for row in history])
            if len({len(row) for row in series}) != 1:
                raise SystemExit(f"Históricos com números diferentes de rodadas para {method}.")
            mean, std = mean_std(series)
            rounds = np.arange(1, len(mean) + 1)
            color = COLORS[method]
            ax.plot(rounds, mean, marker="o", markersize=3.5, linewidth=1.8,
                    color=color, label=LABELS[method])
            ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.16)
        ax.set_xlabel("Rodada federada")
        ax.set_ylabel(ylabel)
        ax.set_title(direction)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Evolução do treino: média e desvio-padrão entre seeds", y=1.02)
    fig.tight_layout()
    path = output / "linha_historico_treino.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_final_metrics(records: dict, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = (
        (axes[0, 0], ("test_accuracy", "retain_accuracy"),
         ("Acurácia teste", "Acurácia retida"), "Acurácia (%) — maior é melhor", None),
        (axes[0, 1], ("backdoor_asr",),
         ("ASR backdoor",), "ASR (%) — menor é melhor", None),
        (axes[1, 0], ("mia_proxy_auc_clean",),
         ("AUC MIA",), "AUC — 0,5 indica ataque ao acaso", 0.5),
    )
    methods = [method for method in METHODS if records[method]]
    x = np.arange(len(methods))
    width = 0.34
    for ax, metric_keys, metric_labels, ylabel, reference in panels:
        for index, (metric, metric_label) in enumerate(zip(metric_keys, metric_labels)):
            values = [[record["final_metrics"][metric] for record in records[method].values()]
                      for method in methods]
            means, stds = zip(*(mean_std([[value] for value in group]) for group in values))
            means = np.asarray(means).reshape(-1)
            stds = np.asarray(stds).reshape(-1)
            offset = (index - (len(metric_keys) - 1) / 2) * width
            if metric == "backdoor_asr":
                yerr = np.vstack((means - np.maximum(0, means - stds),
                                  np.minimum(100, means + stds) - means))
            elif metric == "mia_proxy_auc_clean":
                yerr = np.vstack((means - np.maximum(0, means - stds),
                                  np.minimum(1, means + stds) - means))
            else:
                yerr = stds
            ax.bar(x + offset, means, width, yerr=yerr, capsize=3, label=metric_label,
                   color="#4C78A8" if index == 0 else "#F58518", alpha=0.9)
        if reference is not None:
            ax.axhline(reference, color="#555555", linestyle="--", linewidth=1,
                       label="Referência aleatória (0,5)")
        ax.set_ylabel(ylabel)
        if metric_keys == ("test_accuracy", "retain_accuracy"):
            ax.set_ylim(0, 100)
        elif metric_keys == ("backdoor_asr",):
            ax.set_ylim(0, 100)
        elif metric_keys == ("mia_proxy_auc_clean",):
            ax.set_ylim(0, 1)
        ax.set_xticks(x, [LABELS[method] for method in methods], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    # Compare model-side time for the full baseline training or the unlearning stage.
    time_values = []
    for method in methods:
        values = []
        for record in records[method].values():
            costs = record.get("costs", {})
            key = "pretraining_seconds" if method == "original" else "elapsed_seconds"
            if key in costs:
                values.append(costs[key])
        time_values.append(values)
    means, stds = zip(*(mean_std([[value] for value in group]) for group in time_values))
    means = np.asarray(means).reshape(-1)
    stds = np.asarray(stds).reshape(-1)
    ax = axes[1, 1]
    ax.bar(x, means, yerr=stds, capsize=3,
           color=[COLORS[method] for method in methods], alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("Tempo medido (segundos; escala log)")
    ax.set_xticks(x, [LABELS[method] for method in methods], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Original: pré-treino; demais: custo da etapa")

    fig.suptitle("Comparação final entre métodos — média ± desvio-padrão", y=1.01)
    fig.tight_layout()
    path = output / "barras_comparacao_final.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/pilot_balanced_v1")
    parser.add_argument("--require-seeds", type=int, default=3)
    args = parser.parse_args()
    records = load_records(args.results, args.require_seeds)
    output = args.results / "summary"
    output.mkdir(parents=True, exist_ok=True)
    for path in (plot_round_history(records, output), plot_final_metrics(records, output)):
        print(path)
    seeds = sorted(records["original"])
    print(f"Seeds incluídos: {seeds}")


if __name__ == "__main__":
    main()

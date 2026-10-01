"""Generate summary plots for the FastFedUL experiment shipped with this repo.

The script reads the JSON training log and the final-round pickle snapshots,
then writes PNGs to ``analysis/figures``. It does not rerun training.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_JSON = ROOT / "fedtask/mnist_cnum25_dist0_skew0_seed0/record/fast_fedUL_Mcnn_R51_B32_P0.30_TD1.00_GE0.07.json"
DEFAULT_SNAPSHOT_DIR = ROOT / "fedtasksave/mnist_cnum25_dist0_skew0_seed0/R51_P0.30_alpha0.07/record"
DEFAULT_OUT = ROOT / "analysis/figures"


class CPUUnpickler(pickle.Unpickler):
    """Load torch tensors saved on another device without requiring CUDA."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch

            return lambda data: torch.load(io.BytesIO(data), map_location="cpu")
        return super().find_class(module, name)


def pct(values):
    return 100 * np.asarray(values, dtype=float)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(log, out):
    rounds = np.arange(len(log["test_accs"]))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(rounds, pct(log["test_accs"]), label="Acurácia teste", lw=2.2)
    ax.plot(rounds, pct(log["mean_valid_accs"]), label="Acurácia validação média", lw=2)
    ax.plot(rounds, pct(log["backdoor_accs"]), label="Acurácia backdoor", lw=2, color="#d95f02")
    ax.set(xlabel="Rodada de comunicação", ylabel="Acurácia (%)", title="FastFedUL: evolução das acurácias")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    save(fig, out / "01_acuracias_por_rodada.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(rounds, log["train_losses"], label="Loss treino", lw=2.2)
    ax.plot(rounds, log["test_losses"], label="Loss teste", lw=2)
    ax.set(xlabel="Rodada de comunicação", ylabel="Loss", title="FastFedUL: evolução das perdas")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, out / "02_losses_por_rodada.png")


def plot_client_heterogeneity(log, out):
    clients = np.asarray(log["valid_accs"], dtype=float).T
    rounds = np.arange(clients.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(100 * clients, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=100)
    ax.set(xlabel="Rodada de comunicação", ylabel="Cliente", title="Acurácia de validação por cliente")
    ax.set_yticks(np.arange(clients.shape[0]))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Acurácia (%)")
    save(fig, out / "03_heatmap_validacao_clientes.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(rounds, pct(log["mean_curve"]), label="Média entre clientes", lw=2)
    ax.fill_between(
        rounds,
        pct(log["mean_curve"]) - pct(log["var_curve"]),
        pct(log["mean_curve"]) + pct(log["var_curve"]),
        alpha=0.22,
        label="± desvio-padrão",
    )
    ax.set(xlabel="Rodada de comunicação", ylabel="Acurácia de validação (%)", title="Heterogeneidade entre clientes")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, out / "04_media_e_variacao_clientes.png")


def read_snapshots(snapshot_dir):
    records = []
    for path in sorted(snapshot_dir.glob("history*.pkl"), key=lambda p: int(p.stem.removeprefix("history"))):
        with path.open("rb") as stream:
            item = CPUUnpickler(stream).load()
        records.append(
            {
                "round": int(path.stem.removeprefix("history")),
                "clean_main": item["accuracy"][0],
                "clean_backdoor": item["accuracy"][1],
                "unlearn_main": item["accuracy_unlearn"][0],
                "unlearn_backdoor": item["accuracy_unlearn"][1],
                "unlearn_time": item["unlearn_time"],
            }
        )
    return records


def plot_snapshots(records, out):
    rounds = [row["round"] for row in records]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(rounds, pct([row["clean_main"] for row in records]), "o-", label="Antes do unlearning")
    ax.plot(rounds, pct([row["unlearn_main"] for row in records]), "o-", label="Após o unlearning")
    ax.set(xlabel="Rodada final", ylabel="Acurácia principal (%)", title="Preservação do desempenho principal")
    ax.set_ylim(90, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, out / "05_main_accuracy_unlearning.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(rounds, pct([row["clean_backdoor"] for row in records]), "o-", label="Antes do unlearning")
    ax.plot(rounds, pct([row["unlearn_backdoor"] for row in records]), "o-", label="Após o unlearning")
    ax.set(xlabel="Rodada final", ylabel="Acurácia backdoor (%)", title="Remoção do comportamento backdoor")
    ax.set_ylim(0, 80)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, out / "06_backdoor_accuracy_unlearning.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(rounds, [row["unlearn_time"] for row in records], "o-", color="#7570b3")
    ax.set(xlabel="Rodada final", ylabel="Tempo (s)", title="Custo do unlearning nos snapshots finais")
    ax.grid(alpha=0.25)
    save(fig, out / "07_tempo_unlearning.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with args.json.open() as stream:
        log = json.load(stream)
    snapshots = read_snapshots(args.snapshots)
    plot_training_curves(log, args.out)
    plot_client_heterogeneity(log, args.out)
    plot_snapshots(snapshots, args.out)
    print(f"Gráficos gerados em: {args.out}")
    print(f"Rodadas no log: {len(log['test_accs'])}; snapshots: {len(snapshots)}")


if __name__ == "__main__":
    main()

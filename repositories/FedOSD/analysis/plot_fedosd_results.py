"""Generate plots from the FedOSD repository's available CIFAR-10 log.

The checked-in result is the FedAvg pre-training phase used to initialize the
unlearning experiment. It is intentionally plotted as pre-training data; no
FedOSD-stage result is inferred when it is absent from the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = next(ROOT.glob("UnlearningTask/**/FedAvg/*.json"))
DEFAULT_OUT = ROOT / "analysis/figures"


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_histories(path):
    with path.open() as stream:
        data = json.load(stream)
    histories = data["metric_log"]["client_metric_history"]
    return data, histories


def plot_client_accuracy(histories, out, unlearn_client):
    rounds = np.arange(len(histories[0]["test_accuracy"]))
    retained = [i for i in range(len(histories)) if i != unlearn_client]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for client_id, history in enumerate(histories):
        style = {"color": "#d95f02", "lw": 2.5, "label": f"Cliente {client_id} (unlearning)"} if client_id == unlearn_client else {"color": "#9ecae1", "alpha": 0.7, "lw": 1}
        ax.plot(rounds, history["test_accuracy"], **style)
    mean_retained = np.mean([histories[i]["test_accuracy"] for i in retained], axis=0)
    ax.plot(rounds, mean_retained, color="#1f4e79", lw=2.5, label="Média dos clientes retidos")
    ax.set(xlabel="Rodada de comunicação", ylabel="Acurácia local (%)", title="FedAvg/CIFAR-10: acurácia por cliente")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    save(fig, out / "01_acuracia_clientes.png")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    retained_values = np.asarray([histories[i]["test_accuracy"] for i in retained], dtype=float)
    mean = retained_values.mean(axis=0)
    std = retained_values.std(axis=0)
    ax.plot(rounds, mean, color="#1f4e79", lw=2.5, label="Média retidos")
    ax.fill_between(rounds, mean - std, mean + std, color="#6baed6", alpha=0.3, label="± desvio-padrão")
    ax.plot(rounds, histories[unlearn_client]["test_accuracy"], color="#d95f02", lw=2.5, label=f"Cliente {unlearn_client} (unlearning)")
    ax.set(xlabel="Rodada de comunicação", ylabel="Acurácia local (%)", title="Convergência e dispersão entre clientes")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    save(fig, out / "02_media_e_variacao_clientes.png")


def plot_losses(histories, out, unlearn_client):
    rounds = np.arange(len(histories[0]["test_loss"]))
    train = np.asarray([[np.nan if x is None else x for x in h["training_loss"]] for h in histories], dtype=float)
    test = np.asarray([h["test_loss"] for h in histories], dtype=float)
    train_count = np.sum(~np.isnan(train), axis=0)
    train_mean = np.divide(np.nansum(train, axis=0), train_count, out=np.full(train_count.shape, np.nan), where=train_count > 0)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(rounds, train_mean, label="Loss de treino médio", lw=2.2)
    ax.plot(rounds, test.mean(axis=0), label="Loss de teste médio", lw=2.2)
    ax.plot(rounds, test[unlearn_client], label=f"Loss teste cliente {unlearn_client}", lw=1.8, color="#d95f02")
    ax.set(xlabel="Rodada de comunicação", ylabel="Loss", title="FedAvg/CIFAR-10: evolução das perdas")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, out / "03_losses.png")


def plot_heatmap(histories, out):
    values = np.asarray([h["test_accuracy"] for h in histories], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=100)
    ax.set(xlabel="Rodada de comunicação", ylabel="Cliente", title="Heatmap da acurácia local")
    ax.set_yticks(np.arange(values.shape[0]))
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Acurácia (%)")
    save(fig, out / "04_heatmap_acuracia.png")


def plot_backdoor(history, out, client_id):
    if "backdoor_test_accuracy" not in history:
        return
    rounds = np.arange(len(history["backdoor_test_accuracy"]))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(rounds, history["test_accuracy"], label="Acurácia local", lw=2.2)
    ax.plot(rounds, history["backdoor_test_accuracy"], label="Acurácia backdoor (ASR)", lw=2.2, color="#d95f02")
    ax.set(xlabel="Rodada de comunicação", ylabel="Acurácia (%)", title=f"Cliente {client_id}: comportamento backdoor no pré-treinamento")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    save(fig, out / "05_backdoor_pretraining.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--unlearn-client", type=int, default=2)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data, histories = load_histories(args.json)
    plot_client_accuracy(histories, args.out, args.unlearn_client)
    plot_losses(histories, args.out, args.unlearn_client)
    plot_heatmap(histories, args.out)
    plot_backdoor(histories[args.unlearn_client], args.out, args.unlearn_client)
    print(f"Gráficos gerados em: {args.out}")
    print(f"Arquivo: {args.json}")
    print(f"Rodadas registradas: {len(histories[0]['test_accuracy'])}; clientes: {len(histories)}")
    print(f"Parâmetros: N={data['params']['N']}, NC={data['params']['NC']}, B={data['params']['B']}, C={data['params']['C']}")


if __name__ == "__main__":
    main()

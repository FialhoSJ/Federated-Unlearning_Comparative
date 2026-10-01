"""Generate isolated, reproducible robustness configurations from an article config."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ExperimentConfig


SCENARIOS = {
    "iid": {"partition": "iid"},
    "dirichlet_a01": {"partition": "dirichlet", "dirichlet_alpha": 0.1},
    "dirichlet_a05": {"partition": "dirichlet", "dirichlet_alpha": 0.5},
    "dirichlet_a1": {"partition": "dirichlet", "dirichlet_alpha": 1.0},
    "participation_03": {"client_fraction": 0.3},
    "participation_10": {"client_fraction": 1.0},
    "poison_02": {"poison_fraction": 0.2},
    "poison_08": {"poison_fraction": 0.8},
    "fedup_centroids_1": {"fedup_centroids_per_class": 1},
    "fedup_centroids_8": {"fedup_centroids_per_class": 8},
    "ledger_f32": {"ledger_dtype": "float32"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "configs/article_cifar10.json")
    parser.add_argument("--scenarios", nargs="+", choices=(*SCENARIOS, "all"), default=["all"])
    parser.add_argument("--out", type=Path, default=ROOT / "configs/generated")
    args = parser.parse_args()
    base = ExperimentConfig.from_json(args.base).to_dict()
    names = list(SCENARIOS) if "all" in args.scenarios else args.scenarios
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        config = {**base, **SCENARIOS[name], "output_dir": f"./results/scenarios/{name}"}
        ExperimentConfig(**config).validate()
        path = args.out / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ExperimentConfig
from common.data import apply_trigger, make_client_indices
from common.model import SmallCifarCNN
from common.seed import set_seed


def main() -> None:
    config = ExperimentConfig.from_json(ROOT / "configs/cifar10_client_backdoor.json")
    config.validate()
    set_seed(config.seed)
    first = make_client_indices(100, config.num_clients, config.seed)
    second = make_client_indices(100, config.num_clients, config.seed)
    assert first == second, "partition is not deterministic"
    assert sorted(index for chunk in first for index in chunk) == list(range(100))
    image = torch.zeros(3, config.image_size, config.image_size)
    triggered = apply_trigger(image, config.trigger_size, config.trigger_value)
    assert float(triggered[:, -config.trigger_size:, -config.trigger_size:].mean()) == config.trigger_value
    model = SmallCifarCNN(config.num_classes)
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, config.num_classes)
    print("comparative_benchmark smoke test: OK")
    print(f"clients={config.num_clients} target_client={config.target_client} seed={config.seed}")


if __name__ == "__main__":
    main()

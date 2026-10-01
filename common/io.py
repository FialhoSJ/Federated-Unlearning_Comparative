from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)

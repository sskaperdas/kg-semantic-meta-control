from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator


def iter_candidate_records(path: str | Path) -> Iterator[dict]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc
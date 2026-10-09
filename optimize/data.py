import json

import numpy  # noqa: F401
import dspy

from app.config import DATA_DIR


def load_split(name: str, limit: int | None = None) -> list[dspy.Example]:
    rows = [json.loads(line) for line in open(DATA_DIR / f"{name}.jsonl")][:limit]
    return [dspy.Example(**r).with_inputs("question") for r in rows]

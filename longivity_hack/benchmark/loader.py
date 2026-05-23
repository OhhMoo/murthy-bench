import json
from pathlib import Path
from typing import Iterator


def load_tasks(source: str, limit: int | None = None) -> Iterator[dict]:
    """
    Yield task dicts from either the LongeBench HF dataset or a local JSONL file.

    source: "longebench" | "longebench:extra" | path to a .jsonl file
    """
    if source.startswith("longebench"):
        yield from _load_longebench(source, limit)
    else:
        yield from _load_jsonl(source, limit)


def _load_longebench(source: str, limit: int | None) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    config_name = "extra" if source == "longebench:extra" else "benchmark"
    ds = load_dataset("insilicomedicine/longebench", config_name, split="eval")

    count = 0
    for row in ds:
        if limit is not None and count >= limit:
            break
        yield dict(row)
        count += 1


def _load_jsonl(path: str, limit: int | None) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")

    count = 0
    with p.open() as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1

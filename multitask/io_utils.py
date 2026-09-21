"""Small self-contained file readers shared by the builders.

Deliberately not importing dataset_stats/analyze_datasets.py — that module
loads the Qwen3-4B tokenizer (network fetch) as a side effect of import,
which we don't want here.
"""
import csv
import gzip
import json
from pathlib import Path


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_json_dict(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        yield from data.items()
    else:
        yield from enumerate(data)


def iter_csv(path):
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        yield from enumerate(csv.DictReader(f))


def iter_csv_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        yield from enumerate(csv.DictReader(f))


def read_csv_gz_dict(path, key_field):
    """Load a whole csv.gz keyed by one column -> row dict. For small target files."""
    out = {}
    for _, row in iter_csv_gz(path):
        out[row[key_field]] = row
    return out


def read_text(path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")

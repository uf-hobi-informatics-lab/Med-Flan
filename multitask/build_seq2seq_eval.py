"""
Standalone writer for the seq2seq-target eval jsonl files
(multitask/data/eval/2022n2c2_track2_{mimic,uw}_seq2seq.jsonl, test split
only) -- the new `sdoh_event_seq2seq` task's equivalent of what
build_unified_dataset.py's `main()` writes per-dataset for every REGISTRY
entry, but scoped to only these two new datasets.

Deliberately never imports or runs build_unified_dataset.py: this script
touches nothing under multitask/data/ except the two new files it writes,
so train_unified.jsonl/dev_unified.jsonl are provably untouched by this
feature (same additive-only guarantee the rest of this feature keeps).

Run with: python -m multitask.build_seq2seq_eval
"""
import json
from pathlib import Path

from multitask.builders.sdoh_2022_t2_seq2seq import build_sdoh_event_seq2seq

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "data" / "eval"


def _write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for source in ("mimic", "uw"):
        name = f"2022n2c2_track2_{source}_seq2seq"
        examples = list(build_sdoh_event_seq2seq(source, "test"))
        path = EVAL_DIR / f"{name}.jsonl"
        _write_jsonl(path, examples)
        print(f"[eval] {name} (test): {len(examples)} examples -> {path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()

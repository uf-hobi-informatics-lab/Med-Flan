from pathlib import Path

import pandas as pd

from multitask.schemas import example

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "emrqa-msquad" / "raw"
# HF ships train/validation only, no test -- "dev" here is HF's "validation",
# and (same as medmcqa) it also doubles as the final per-task eval split.
SPLIT_FILES = {"train": "train-00000-of-00001.parquet", "dev": "validation-00000-of-00001.parquet"}


def build_emrqa_msquad(split):
    path = DIR / SPLIT_FILES[split]
    if not path.exists():
        return
    df = pd.read_parquet(path)
    for i, row in df.iterrows():
        input_text = f"Context:\n{row['context']}\nQuestion: {row['question']}"
        answers = row["answers"]["text"]
        answer = answers[0] if len(answers) else ""
        yield example(
            "qa_span_extraction", "emrqa-msquad", split, f"{path.name}#{i}",
            input_text, {"answer": answer},
        )

"""
Save each dataset SEPARATELY as a HuggingFace `datasets.DatasetDict`
(train/dev/test, whichever splits exist), in OpenAI chat-messages format
(`{"messages": [{"role": ..., "content": ...}, ...]}`) so
`tokenizer.apply_chat_template()` can be applied directly.

Unlike build_unified_dataset.py, nothing is merged here -- each dataset is
written to its own directory under multitask/huggingface_format/<name>/ via
DatasetDict.save_to_disk(), so they can be reloaded individually
(`datasets.load_from_disk(...)`) and combined however you like later, e.g.:

    from datasets import load_from_disk, concatenate_datasets
    ds = [load_from_disk(p)["train"] for p in paths]
    merged = concatenate_datasets(ds)

Run with: python -m multitask.build_hf_datasets
"""
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict

from multitask.builders.cohort_eligibility_2018_t1 import build_cohort_eligibility
from multitask.builders.disorder_normalization_2015_t14 import build_disorder_normalization
from multitask.builders.family_history_2019_t2 import build_family_history
from multitask.builders.i2b2_2010 import build_concept_assertion_relation
from multitask.builders.medication_event_2022_t1 import build_medication_event
from multitask.builders.ner_re_2018_t2 import build_ner_re_medication
from multitask.builders.sdoh_2022_t2 import build_sdoh_event
from multitask.builders.simple_tasks import (
    build_discharge_me, build_litcovid, build_medmcqa, build_medqa,
    build_mednli, build_mmlu_pro_health, build_pubmedqa, build_umn_abbreviation,
)
from multitask.builders.sts_2019_t1 import build_sts

OUT_DIR = Path(__file__).resolve().parent / "huggingface_format"

# (original dataset name -> used as the save_to_disk subfolder name,
#  generator(split) -> iterator of schemas.example() dicts, splits it has)
REGISTRY = [
    ("MedNLI", build_mednli, ["train", "dev", "test"]),
    ("MedQA-USMLE-US", lambda s: build_medqa("MedQA-USMLE-US", s), ["train", "dev", "test"]),
    ("MedQA-USMLE-Mainland", lambda s: build_medqa("MedQA-USMLE-Mainland", s), ["train", "dev", "test"]),
    ("MedQA-USMLE-Taiwan", lambda s: build_medqa("MedQA-USMLE-Taiwan", s), ["train", "dev", "test"]),
    ("MMLU-Pro-Health", build_mmlu_pro_health, ["test"]),  # upstream ships only one split
    # medmcqa's public test.json ships no `cop` (gold answer) -> every test
    # target would be {"answer": null} (unscoreable, toxic to train on), so
    # drop it and let dev stand in as the eval split (same as build_unified_dataset.py).
    ("medmcqa", build_medmcqa, ["train", "dev"]),
    ("pubmedqa", build_pubmedqa, ["train", "dev", "test"]),
    ("covid_LitCovid", build_litcovid, ["train", "dev", "test"]),
    ("discharge-me", build_discharge_me, ["train", "dev", "test"]),
    ("2018_n2c2_track1", build_cohort_eligibility, ["train", "test"]),
    ("2018_n2c2_track2", build_ner_re_medication, ["train", "test"]),
    ("2019_n2c2_track1", build_sts, ["train", "test"]),
    ("2019_n2c2_track2", build_family_history, ["train", "test"]),
    # No test split released locally for these two -- dev is the only
    # held-out split available (same gap flagged in build_unified_dataset.py).
    ("2022n2c2_track1", build_medication_event, ["train", "dev"]),
    ("2022n2c2_track2", build_sdoh_event, ["train", "dev"]),
    ("semeval_2015", build_disorder_normalization, ["train", "test"]),
    ("umn_abbreviation", build_umn_abbreviation, ["train", "dev", "test"]),
    ("2010_i2b2_va", build_concept_assertion_relation, ["train", "test"]),
]


def _build_split(gen, split):
    rows = [
        {"messages": ex["messages"], "task": ex["task"], "dataset": ex["dataset"], "id": ex["id"]}
        for ex in gen(split)
    ]
    return Dataset.from_list(rows) if rows else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, gen, splits in REGISTRY:
        split_datasets = {}
        for split in splits:
            ds = _build_split(gen, split)
            if ds is not None and len(ds) > 0:
                split_datasets[split] = ds
        if not split_datasets:
            print(f"[skip] {name}: no data found", file=sys.stderr)
            continue
        dd = DatasetDict(split_datasets)
        out_path = OUT_DIR / name
        dd.save_to_disk(str(out_path))
        summary = ", ".join(f"{split}={len(ds)}" for split, ds in split_datasets.items())
        print(f"[saved] {name}: {summary} -> {out_path.relative_to(OUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()

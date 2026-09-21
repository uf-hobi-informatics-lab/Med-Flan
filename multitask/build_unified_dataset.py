"""
Build the merged multi-task training set (multitask/data/train_unified.jsonl)
and, separately and NOT merged, each dataset's own held-out eval split
(multitask/data/eval/<dataset>.jsonl) -- per the plan: train on the union,
evaluate per-dataset on that dataset's own original test data.

Run with: python -m multitask.build_unified_dataset
"""
import json
import random
from pathlib import Path

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

DATA_DIR = Path(__file__).resolve().parent / "data"
EVAL_DIR = DATA_DIR / "eval"

# (dataset name used for eval filenames, generator(split)->iterator, splits it has)
REGISTRY = [
    ("MedNLI", build_mednli, ["train", "dev", "test"]),
    ("MedQA-USMLE-US", lambda s: build_medqa("MedQA-USMLE-US", s), ["train", "dev", "test"]),
    ("MedQA-USMLE-Mainland", lambda s: build_medqa("MedQA-USMLE-Mainland", s), ["train", "dev", "test"]),
    ("MedQA-USMLE-Taiwan", lambda s: build_medqa("MedQA-USMLE-Taiwan", s), ["train", "dev", "test"]),
    ("MMLU-Pro-Health", build_mmlu_pro_health, ["test"]),  # upstream ships only one split
    # medmcqa's released test split has null answers for all examples (the
    # public test set never shipped gold labels) -- unlike every other
    # dataset here, "test" is unusable for scoring at all, so it's left out
    # of the splits list entirely and dev stands in as this dataset's final
    # eval split (same fallback pattern as CMED/SHAC below, different root
    # cause: those two have no test split, this one has an ungraded one).
    ("medmcqa", build_medmcqa, ["train", "dev"]),
    ("pubmedqa", build_pubmedqa, ["train", "dev", "test"]),
    ("covid (LitCovid)", build_litcovid, ["train", "dev", "test"]),
    ("discharge-me", build_discharge_me, ["train", "dev", "test"]),
    ("2018_n2c2_track1", build_cohort_eligibility, ["train", "test"]),
    ("2018_n2c2_track2", build_ner_re_medication, ["train", "test"]),
    ("2019_n2c2_track1", build_sts, ["train", "test"]),
    ("2019_n2c2_track2", build_family_history, ["train", "test"]),
    # No test split exists locally for these two -- "dev" stands in as the
    # held-out eval split; flagged as a gap in the plan (CMED/SHAC have no
    # released test set in this repo).
    ("2022n2c2_track1", build_medication_event, ["train", "dev"]),
    ("2022n2c2_track2", build_sdoh_event, ["train", "dev"]),
    ("semeval_2015", build_disorder_normalization, ["train", "test"]),
    ("umn_abbreviation", build_umn_abbreviation, ["train", "dev", "test"]),
    # No data locally yet -- DUA-gated, folder absent; yields 0 examples
    # safely until 2010_i2b2_va/ is populated (builder's rglob finds nothing).
    ("2010_i2b2_va", build_concept_assertion_relation, ["train", "test"]),
]


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def _write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main(seed: int = 13):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    train_examples = []
    dev_examples = []
    eval_counts = {}

    for name, gen, splits in REGISTRY:
        if "train" in splits:
            n_before = len(train_examples)
            train_examples.extend(gen("train"))
            print(f"[train] {name}: {len(train_examples) - n_before} examples")

        # The final per-task metric always comes from a genuine held-out test
        # split when one exists; only fall back to dev (CMED, SHAC) when this
        # dataset has no test split released locally at all.
        eval_split = "test" if "test" in splits else ("dev" if "dev" in splits else None)

        # A dataset's dev split feeds the MERGED validation set only when it's
        # distinct from whatever we just used as the final eval split above --
        # i.e. only for datasets with a genuine train/dev/test three-way split.
        # It is never used for final per-task numbers; that's test's job.
        if "dev" in splits and eval_split != "dev":
            n_before = len(dev_examples)
            dev_examples.extend(gen("dev"))
            print(f"[dev]   {name}: {len(dev_examples) - n_before} examples (merged validation only)")

        if eval_split is None:
            continue
        eval_examples = list(gen(eval_split))
        eval_path = EVAL_DIR / f"{safe_name(name)}.jsonl"
        _write_jsonl(eval_path, eval_examples)
        eval_counts[name] = (eval_split, len(eval_examples))
        print(f"[eval]  {name} ({eval_split}): {len(eval_examples)} examples -> {eval_path.relative_to(DATA_DIR.parent)}")

    rng = random.Random(seed)
    rng.shuffle(train_examples)
    rng.shuffle(dev_examples)

    train_path = DATA_DIR / "train_unified.jsonl"
    _write_jsonl(train_path, train_examples)
    dev_path = DATA_DIR / "dev_unified.jsonl"
    _write_jsonl(dev_path, dev_examples)

    print(f"\nWrote {len(train_examples)} merged training examples -> {train_path.relative_to(DATA_DIR.parent)}")
    print(f"Wrote {len(dev_examples)} merged validation examples -> {dev_path.relative_to(DATA_DIR.parent)}")
    print("(validation set is for monitoring joint training only -- never used for the per-task numbers below)")
    print("\nPer-dataset eval split summary (unmerged, final per-task metric):")
    for name, (split, n) in eval_counts.items():
        print(f"  {name}: {split} split, {n} examples")


if __name__ == "__main__":
    main()

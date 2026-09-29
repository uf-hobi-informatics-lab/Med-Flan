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
from multitask.builders.emrqa_msquad import build_emrqa_msquad
from multitask.builders.family_history_2019_t2 import build_family_history
from multitask.builders.i2b2_2010 import build_concept_assertion_relation
from multitask.builders.medication_event_2022_t1 import build_medication_event
from multitask.builders.ner_re_2018_t2 import build_ner_re_medication
from multitask.builders.sdoh_2022_t2 import build_sdoh_event
from multitask.builders.simple_tasks import (
    build_biosses, build_bionli, build_discharge_me, build_litcovid, build_medmcqa, build_medqa,
    build_mednli, build_mmlu_pro_health, build_nli4ct, build_pubmedqa, build_umn_abbreviation,
)
from multitask.builders.sts_2019_t1 import build_sts, build_sts_classification
from multitask.split_utils import carve_dev_from_test

DATA_DIR = Path(__file__).resolve().parent / "data"
EVAL_DIR = DATA_DIR / "eval"

# (dataset name used for eval filenames, generator(split)->iterator, splits it has)
#
# Tier A subtask-enrichment datasets for ner_re_medication/medication_event
# (design.md) are intentionally NOT registered here -- they're train-only,
# additive re-framings meant to be opted in/out of per training run via
# multitask/huggingface_format/<name>/, not merged unconditionally into this
# module's single train_unified.jsonl. See build_hf_datasets.py's REGISTRY.
REGISTRY = [
    ("MedNLI", build_mednli, ["train", "dev", "test"]),
    ("bionli", build_bionli, ["train", "dev", "test"]),
    ("nli4ct", build_nli4ct, ["train", "dev", "test"]),
    ("MedQA-USMLE-US", lambda s: build_medqa("MedQA-USMLE-US", s), ["train", "dev", "test"]),
    ("MedQA-USMLE-Mainland", lambda s: build_medqa("MedQA-USMLE-Mainland", s), ["train", "dev", "test"]),
    ("MedQA-USMLE-Taiwan", lambda s: build_medqa("MedQA-USMLE-Taiwan", s), ["train", "dev", "test"]),
    ("MMLU-Pro-Health", build_mmlu_pro_health, ["test"]),  # upstream ships only one split
    # OOD-only check (not clinical-note domain): a different topic-similarity
    # dataset from the other 11, added specifically to sanity-check the
    # model's output distribution isn't overfit to this repo's own tasks --
    # see multitask.builders.simple_tasks.build_biosses's module comment.
    # Upstream ships only one split too, so test-only, same as MMLU-Pro-Health.
    ("biosses", build_biosses, ["test"]),
    # medmcqa's released test split has null answers for all examples (the
    # public test set never shipped gold labels) -- unlike every other
    # dataset here, "test" is unusable for scoring at all, so it's left out
    # of the splits list entirely and dev stands in as this dataset's final
    # eval split (the only dataset in this REGISTRY that needs this
    # fallback -- every other dataset has a genuine, scoreable test split).
    ("medmcqa", build_medmcqa, ["train", "dev"]),
    ("pubmedqa", build_pubmedqa, ["train", "dev", "test"]),
    ("covid (LitCovid)", build_litcovid, ["train", "dev", "test"]),
    ("discharge-me", build_discharge_me, ["train", "dev", "test"]),
    ("2018_n2c2_track1", build_cohort_eligibility, ["train", "test"]),
    ("2018_n2c2_track2", build_ner_re_medication, ["train", "test"]),
    ("2019_n2c2_track1", build_sts, ["train", "test"]),
    ("2019_n2c2_track1_cls", build_sts_classification, ["train", "test"]),
    ("2019_n2c2_track2", build_family_history, ["train", "test"]),
    ("2022n2c2_track1", build_medication_event, ["train", "dev", "test"]),
    ("2022n2c2_track2_mimic", lambda s: build_sdoh_event("mimic", s), ["train", "dev", "test"]),
    ("2022n2c2_track2_uw", lambda s: build_sdoh_event("uw", s), ["train", "dev", "test"]),
    ("semeval_2015", build_disorder_normalization, ["train", "test"]),
    ("umn_abbreviation", build_umn_abbreviation, ["train", "dev", "test"]),
    ("2010_i2b2_va", build_concept_assertion_relation, ["train", "test"]),
    # HF ships train/validation only, no test -- dev stands in as the final
    # eval split, same fallback as medmcqa above.
    ("emrqa-msquad", build_emrqa_msquad, ["train", "dev"]),
]


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


# The smallest task slices in the whole training mix by 100-1000x (98-350
# examples each, vs. 18k-211k for the QA/classification tasks -- see
# `python -c "import json,collections; ..."` count over train_unified.jsonl,
# or just re-run this module and read the [train] lines below). Investigated
# this session as a likely contributor to these tasks' generation quality
# staying well below published baselines even after a separate
# frequency_penalty decoding bug was found and fixed (multitask.inference.
# run_generation.DEFAULT_GEN's comment has the full writeup) -- sdoh_event,
# the one IE task with a comparatively large slice (3067), recovered to
# near-baseline after that fix; these did not.
#
# Upsampled here by adding a SECOND training row per example, rendered via
# schemas.render_md/render_messages_md instead of render/render_messages --
# already computed by schemas.example() as "text_md"/"messages_md" on every
# example, just not previously emitted as its own row in train_unified.jsonl
# (build_hf_datasets.py already builds a separate "<name>_md" HF dataset
# per REGISTRY entry, but that's never merged into this file). This is
# genuine input-format diversity (different section-header/fenced-code
# framing of the same content), not an exact-duplicate row -- a real, if
# modest, augmentation rather than empty repetition. Train-only: eval/test
# is untouched, so no scored metric is affected by this, only what the SFT
# trainer sees. Add/remove task tags here to tune -- e.g. drop sdoh_event in
# since it's not struggling the same way, or add it back if 2x still isn't
# enough for the others and you want a matched multiplier across all IE
# tasks.
UPSAMPLE_MD_TASKS = {
    "family_history", "concept_assertion_relation", "disorder_normalization",
    "ner_re_medication", "medication_event",
}


def _markdown_duplicate(ex):
    """A second training row for `ex`, swapped to its own already-computed
    markdown rendering (see UPSAMPLE_MD_TASKS above) -- id gets a "#md"
    suffix so it never collides with the original row."""
    dup = dict(ex)
    dup["id"] = f"{ex['id']}#md"
    dup["text"] = ex["text_md"]
    dup["messages"] = ex["messages_md"]
    return dup


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
        # split when one exists; only fall back to dev (medmcqa) when this
        # dataset has no scoreable test split released locally at all.
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

        # No released dev split -- carve a monitoring-only dev sample out of
        # test instead. This overlaps test (test itself is left intact above,
        # so per-task scoring still runs against the full, official test set).
        if "dev" not in splits and eval_split == "test":
            carved = carve_dev_from_test(eval_examples)
            dev_examples.extend(carved)
            print(f"[dev]   {name}: {len(carved)} examples carved from test (merged validation only, overlaps test)")

    rng = random.Random(seed)

    md_dups = [_markdown_duplicate(ex) for ex in train_examples if ex["task"] in UPSAMPLE_MD_TASKS]
    train_examples.extend(md_dups)
    print(f"\n[upsample] added {len(md_dups)} markdown-rendering duplicate(s) for {sorted(UPSAMPLE_MD_TASKS)}")

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

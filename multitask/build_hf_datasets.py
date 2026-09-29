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
from multitask.builders.emrqa_msquad import build_emrqa_msquad
from multitask.builders.family_history_2019_t2 import build_family_history
from multitask.builders.i2b2_2010 import (
    build_concept_assertion_relation, build_concept_assertion_relation_ner_only,
    build_concept_assertion_relation_re_only, build_concept_assertion_relation_section,
)
from multitask.builders.medication_event_2022_t1 import (
    build_medication_event, build_medication_event_attr_only, build_medication_event_paragraph,
    build_medication_event_trigger_only,
)
from multitask.builders.ner_re_2018_t2 import (
    build_ner_re_medication, build_ner_re_medication_ner_only, build_ner_re_medication_paragraph,
    build_ner_re_medication_re_only,
)
from multitask.builders.sdoh_2022_t2 import build_sdoh_event
from multitask.builders.sdoh_2022_t2_seq2seq import (
    build_sdoh_event_seq2seq, build_sdoh_event_seq2seq_curriculum, build_sdoh_event_seq2seq_true_negative,
)
from multitask.builders.simple_tasks import (
    build_biosses, build_bionli, build_discharge_me, build_litcovid, build_medmcqa, build_medqa,
    build_mednli, build_mmlu_pro_health, build_nli4ct, build_pubmedqa, build_umn_abbreviation,
)
from multitask.builders.sts_2019_t1 import build_sts, build_sts_classification
from multitask.split_utils import carve_dev_from_test

OUT_DIR = Path(__file__).resolve().parent / "huggingface_format"

# (original dataset name -> used as the save_to_disk subfolder name,
#  generator(split) -> iterator of schemas.example() dicts, splits it has)
REGISTRY = [
    ("MedNLI", build_mednli, ["train", "dev", "test"]),
    ("bionli", build_bionli, ["train", "dev", "test"]),
    ("nli4ct", build_nli4ct, ["train", "dev", "test"]),
    ("MedQA-USMLE-US", lambda s: build_medqa("MedQA-USMLE-US", s), ["train", "dev", "test"]),
    ("MedQA-USMLE-Mainland", lambda s: build_medqa("MedQA-USMLE-Mainland", s), ["train", "dev", "test"]),
    ("MedQA-USMLE-Taiwan", lambda s: build_medqa("MedQA-USMLE-Taiwan", s), ["train", "dev", "test"]),
    ("MMLU-Pro-Health", build_mmlu_pro_health, ["test"]),  # upstream ships only one split
    # OOD-only check (not clinical-note domain) -- see build_unified_dataset.py's
    # REGISTRY comment and multitask.builders.simple_tasks.build_biosses.
    # Test-only, upstream ships only one split.
    ("biosses", build_biosses, ["test"]),
    # medmcqa's public test.json ships no `cop` (gold answer) -> every test
    # target would be {"answer": null} (unscoreable, toxic to train on), so
    # drop it and let dev stand in as the eval split (same as build_unified_dataset.py).
    ("medmcqa", build_medmcqa, ["train", "dev"]),
    ("pubmedqa", build_pubmedqa, ["train", "dev", "test"]),
    ("covid_LitCovid", build_litcovid, ["train", "dev", "test"]),
    ("discharge-me", build_discharge_me, ["train", "dev", "test"]),
    ("2018_n2c2_track1", build_cohort_eligibility, ["train", "test"]),
    ("2018_n2c2_track2", build_ner_re_medication, ["train", "test"]),
    # Tier A subtask-enrichment datasets (design.md) -- train-only, additive
    # re-framings of ner_re_medication's own gold annotations. No dev/test,
    # no run_eval.py dispatch entry, and deliberately excluded from
    # build_unified_dataset.py's merged train_unified.jsonl -- include or
    # exclude these for a training run by choosing whether to
    # concatenate_datasets() their directory alongside the parent's.
    ("2018_n2c2_track2_ner_only", build_ner_re_medication_ner_only, ["train"]),
    ("2018_n2c2_track2_re_only", build_ner_re_medication_re_only, ["train"]),
    # Paragraph-window dataset (design.md's paragraph-chunking section) --
    # HF-only, additive, distinct mechanism from Tier A above (new text
    # spans, not new labels on old text). train+test both built (per the
    # "prepare all" instruction) for future experimentation flexibility;
    # dev is auto-carved from test below just like the parent dataset.
    # Never wired into build_unified_dataset.py/run_eval.py -- doc-level
    # prediction/scoring stays authoritative (see design.md).
    ("2018_n2c2_track2_paragraph", build_ner_re_medication_paragraph, ["train", "test"]),
    ("2019_n2c2_track1", build_sts, ["train", "test"]),
    ("2019_n2c2_track1_cls", build_sts_classification, ["train", "test"]),
    ("2019_n2c2_track2", build_family_history, ["train", "test"]),
    ("2022n2c2_track1", build_medication_event, ["train", "dev", "test"]),
    # Tier A subtask-enrichment datasets for medication_event -- same
    # train-only/HF-only convention as the two above.
    ("2022n2c2_track1_trigger_only", build_medication_event_trigger_only, ["train"]),
    ("2022n2c2_track1_attr_only", build_medication_event_attr_only, ["train"]),
    # Paragraph-window dataset for medication_event (design.md's paragraph-
    # chunking section) -- HF-only, additive, same mechanism/convention as
    # 2018_n2c2_track2_paragraph above (new text spans, not relabeled ones).
    # track1 ships a real dev split, so all three are built (unlike track2's
    # paragraph, which auto-carves dev from test). Never wired into
    # build_unified_dataset.py/run_eval.py -- doc-level scoring stays
    # authoritative (see design.md).
    ("2022n2c2_track1_paragraph", build_medication_event_paragraph, ["train", "dev", "test"]),
    ("2022n2c2_track2_mimic", lambda s: build_sdoh_event("mimic", s), ["train", "dev", "test"]),
    ("2022n2c2_track2_uw", lambda s: build_sdoh_event("uw", s), ["train", "dev", "test"]),
    # Microsoft-Nuance-style seq2seq target format for sdoh_event (task tag
    # sdoh_event_seq2seq, see multitask/seq2seq_grammar.py) -- brand-new,
    # additive alongside the JSON-based rows above, never merged with them.
    # Train-only augmentation variants (_trueneg/_curriculum) follow the
    # same HF-only/train-only convention as the Tier A datasets above.
    ("2022n2c2_track2_mimic_seq2seq", lambda s: build_sdoh_event_seq2seq("mimic", s), ["train", "dev", "test"]),
    ("2022n2c2_track2_uw_seq2seq", lambda s: build_sdoh_event_seq2seq("uw", s), ["train", "dev", "test"]),
    ("2022n2c2_track2_mimic_seq2seq_trueneg", lambda s: build_sdoh_event_seq2seq_true_negative("mimic", s), ["train"]),
    ("2022n2c2_track2_uw_seq2seq_trueneg", lambda s: build_sdoh_event_seq2seq_true_negative("uw", s), ["train"]),
    ("2022n2c2_track2_mimic_seq2seq_curriculum", lambda s: build_sdoh_event_seq2seq_curriculum("mimic", s), ["train"]),
    ("2022n2c2_track2_uw_seq2seq_curriculum", lambda s: build_sdoh_event_seq2seq_curriculum("uw", s), ["train"]),
    ("semeval_2015", build_disorder_normalization, ["train", "test"]),
    ("umn_abbreviation", build_umn_abbreviation, ["train", "dev", "test"]),
    ("2010_i2b2_va", build_concept_assertion_relation, ["train", "test"]),
    # Tier A subtask-enrichment datasets for concept_assertion_relation --
    # same train-only/HF-only convention as the two above.
    ("2010_i2b2_va_ner_only", build_concept_assertion_relation_ner_only, ["train"]),
    ("2010_i2b2_va_re_only", build_concept_assertion_relation_re_only, ["train"]),
    # Section-window dataset (design.md's paragraph-chunking section) --
    # same convention as 2018_n2c2_track2_paragraph above, but boundaries
    # are section headers instead of blank lines (this format has none).
    # train+test both built (per the "prepare all" instruction); never
    # wired into build_unified_dataset.py/run_eval.py.
    ("2010_i2b2_va_section", build_concept_assertion_relation_section, ["train", "test"]),
    # HF ships train/validation only, no test -- dev stands in as the final
    # eval split, same fallback as medmcqa above.
    ("emrqa-msquad", build_emrqa_msquad, ["train", "dev"]),
]


def _build_split(examples, messages_key="messages"):
    rows = [
        {
            "messages": ex[messages_key],
            "task": ex["task"],
            "category": ex["category"],
            "dataset": ex["dataset"],
            "id": ex["id"],
        }
        for ex in examples
    ]
    return Dataset.from_list(rows) if rows else None


def _build_and_save(name, examples_by_split, splits, messages_key):
    split_datasets = {}
    for split in splits:
        ds = _build_split(examples_by_split[split], messages_key)
        if ds is not None and len(ds) > 0:
            split_datasets[split] = ds
    # No released dev split -- carve a monitoring-only dev sample out of
    # test instead (overlaps test; test itself is left unchanged so
    # per-task scoring still runs against the full, official test set).
    if "dev" not in splits and "train" in split_datasets and "test" in split_datasets:
        dev_rows = carve_dev_from_test(split_datasets["test"].to_list())
        if dev_rows:
            split_datasets["dev"] = Dataset.from_list(dev_rows)
    if not split_datasets:
        print(f"[skip] {name}: no data found", file=sys.stderr)
        return
    dd = DatasetDict(split_datasets)
    out_path = OUT_DIR / name
    dd.save_to_disk(str(out_path))
    summary = ", ".join(f"{split}={len(ds)}" for split, ds in split_datasets.items())
    print(f"[saved] {name}: {summary} -> {out_path.relative_to(OUT_DIR.parent.parent)}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, gen, splits in REGISTRY:
        # gen(split) does the actual BRAT/XML/TSV parsing+alignment work;
        # schemas.example() already populates both "messages" and
        # "messages_md" in one pass, so materializing each split once and
        # reusing it for both variants below avoids re-running that work twice.
        examples_by_split = {split: list(gen(split)) for split in splits}
        _build_and_save(name, examples_by_split, splits, "messages")
        _build_and_save(f"{name}_md", examples_by_split, splits, "messages_md")


if __name__ == "__main__":
    main()

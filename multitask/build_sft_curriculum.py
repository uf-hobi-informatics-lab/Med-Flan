"""Build Stage-1 (subtasks) and Stage-2 (end-to-end combined) curriculum
DatasetDicts for SFT.

Owner decision (design.md's "Owner decision: Tier A as a 2-stage SFT
curriculum, not additive-only"): decompose ner_re_medication/medication_event
into their Tier A pipeline-stage sub-skills (NER-only, RE-only,
trigger-detection-only, attribute-classification-only -- already built,
see design.md's "Implementation status") and train on those sub-skills
FIRST (alongside the other easy/foundational tasks), then train on the
full end-to-end tasks (combined with everything else, including the
sub-skill datasets again as anti-forgetting replay) in Stage 2. This is a
genuine 2-stage curriculum, not the additive-single-pass approach
design.md originally recommended -- see that section for the accepted
train/inference-mismatch tradeoff (RE-only/attr-only train on gold
entities, but Stage 1 is a skill-acquisition warm-up, not a literal
chained inference pipeline; the full end-to-end task is still the only
thing ever scored).

Same overall shape as build_curriculum_datasets.py (that one is the
existing 2-stage RL/GRPO curriculum -- unmodified, untouched by this
script), but scoped to the med_flan_v1 mixture (see build_med_flan_mix.py's
MED_FLAN_DATASETS -- run_custom_model_eval.ipynb's MAIN_DATASET_SPECS) plus
the Tier A subtask datasets, and with NO render-based upsampling: every
dataset contributes exactly its own (capped) train split once, no
duplicate rows. `2010_i2b2_va` is deliberately NOT included (not part of
the med_flan_v1 mixture at all).

Combines existing per-task HF datasets under multitask/huggingface_format/
into two mixes. Both stages keep the `task` column.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk

SRC = Path(__file__).parent / "huggingface_format"
CAP = 4000
SEED = 42

# Stage 1: the original easy-task TIER1 plus the four Tier A subtask
# datasets (design.md) -- train-only, no `dev` split (confirmed on disk:
# {"train": 303}/{"train": 350}), handled by build() below.
TIER1 = [
    "MedNLI_md",
    "MedQA-USMLE-US_md",
    "umn_abbreviation_md",
    "2018_n2c2_track1_md",
    "2019_n2c2_track1_cls_md",
    "2018_n2c2_track2_ner_only_md",
    "2018_n2c2_track2_re_only_md",
    "2022n2c2_track1_trigger_only_md",
    "2022n2c2_track1_attr_only_md",
]
# Stage 2 additions: the full end-to-end tasks (the ones Stage 1's subtask
# datasets decompose) plus the two IE tasks with no decomposition
# available (sdoh_event, family_history -- design.md: "never evaluated
# either way, stays out of scope" for family_history; sdoh_event
# deliberately excluded from Tier A).
TIER2 = [
    "2018_n2c2_track2_md",
    "2018_n2c2_track2_paragraph_md",
    "2022n2c2_track1_md",
    "2022n2c2_track2_mimic_md",
    "2022n2c2_track2_uw_md",
    "2019_n2c2_track2_md",
]

EXPECTED_COLUMNS = ["messages", "task", "category", "dataset", "id"]


def build(names, out_path: Path):
    train_parts, dev_parts = [], []
    for name in names:
        d = load_from_disk(str(SRC / name))
        tr = d["train"].shuffle(seed=SEED)
        tr = tr.select(range(min(CAP, len(tr))))
        train_parts.append(tr)
        if "dev" in d:
            dev_parts.append(d["dev"])

    out = DatasetDict(
        {
            "train": concatenate_datasets(train_parts).shuffle(seed=SEED),
            "dev": concatenate_datasets(dev_parts).shuffle(seed=SEED),
        }
    )

    if out_path.exists():
        shutil.rmtree(out_path)
    out.save_to_disk(str(out_path))
    return out


def verify(stage_name: str, out_path: Path, expected_tags: set, min_tier2_share: float | None):
    dd = load_from_disk(str(out_path))
    assert set(dd.keys()) == {"train", "dev"}, f"{stage_name}: unexpected splits {list(dd.keys())}"

    for split in ("train", "dev"):
        cols = dd[split].column_names
        assert cols == EXPECTED_COLUMNS, f"{stage_name}/{split}: columns {cols} != {EXPECTED_COLUMNS}"

    for split in ("train", "dev"):
        counts = Counter(dd[split]["task"])
        print(f"\n[{stage_name}] {split} task counts ({len(dd[split])} total):")
        for tag, n in counts.most_common():
            print(f"    {tag}: {n}")

        tags = set(counts)
        if split == "dev":
            # Stage 1's Tier A subtask datasets contribute no dev rows (see
            # build()), so dev's tag set is a subset of train's, not equal.
            assert tags <= expected_tags, f"{stage_name}/{split}: tags {tags} not subset of {expected_tags}"
        else:
            assert tags == expected_tags, f"{stage_name}/{split}: tags {tags} != expected {expected_tags}"

        if split == "train":
            total = sum(counts.values())
            top_tag, top_n = counts.most_common(1)[0]
            top_share = top_n / total
            print(f"    largest task: {top_tag} = {top_share:.1%} of train")
            assert top_share <= 0.5, f"{stage_name}: {top_tag} is {top_share:.1%} of train (>50%)"

            if min_tier2_share is not None:
                tier2_tags = {TASK_TAG_OF_DIR[n] for n in TIER2}
                tier2_n = sum(n for tag, n in counts.items() if tag in tier2_tags)
                tier2_share = tier2_n / total
                print(f"    tier-2 tasks share: {tier2_share:.1%} of train")
                assert tier2_share >= min_tier2_share, (
                    f"{stage_name}: tier-2 share {tier2_share:.1%} < {min_tier2_share:.0%}"
                )

    # spot-check 2 examples per task per split: gold answer parses as JSON
    for split in ("train", "dev"):
        checked = Counter()
        remaining = set(Counter(dd[split]["task"]))
        for ex in dd[split]:
            tag = ex["task"]
            if checked[tag] >= 2:
                continue
            checked[tag] += 1
            if checked[tag] >= 2:
                remaining.discard(tag)
            try:
                json.loads(ex["messages"][-1]["content"])
            except json.JSONDecodeError as e:
                raise AssertionError(
                    f"{stage_name}/{split}: task={tag} id={ex['id']} gold content is not valid JSON: {e}"
                )
            if not remaining:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-base",
        default=str(Path(__file__).parent / "huggingface_format" / "curriculum"),
        help="Base directory to write the two stage DatasetDicts into.",
    )
    args = parser.parse_args()

    out_base = Path(args.out_base)
    stage1_path = out_base / "med_flan_v1_md_sft_stage1_subtasks"
    stage2_path = out_base / "med_flan_v1_md_sft_stage2_full"

    print(f"Building SFT Stage 1 (easy + subtasks) -> {stage1_path}")
    build(TIER1, stage1_path)

    print(f"Building SFT Stage 2 (full end-to-end, combined) -> {stage2_path}")
    build(TIER1 + TIER2, stage2_path)

    global TASK_TAG_OF_DIR
    all_names = TIER1 + TIER2
    TASK_TAG_OF_DIR = {}
    for name in all_names:
        d = load_from_disk(str(SRC / name))
        TASK_TAG_OF_DIR[name] = Counter(d["train"]["task"]).most_common(1)[0][0]

    stage1_tags = {TASK_TAG_OF_DIR[n] for n in TIER1}
    stage2_tags = {TASK_TAG_OF_DIR[n] for n in TIER1 + TIER2}

    verify("sft_stage1_subtasks", stage1_path, stage1_tags, min_tier2_share=None)
    verify("sft_stage2_full", stage2_path, stage2_tags, min_tier2_share=0.25)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

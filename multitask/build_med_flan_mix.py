"""
Build med_flan_v1_md_upsampled: the curated 11-dataset SFT mixture (same
list as run_custom_model_eval.ipynb's MAIN_DATASET_SPECS / test.ipynb's
med_flan_v1 prototype), all-markdown base, with a genuinely-different
plain-text-rendered copy of the train split added for the smallest/most
training-starved information-extraction tasks in the mixture -- real (not
exact-duplicate) upsampling. `2010_i2b2_va` is never in scope: it is not
part of this 11-dataset mixture at all (it's the OOD-only concept_assertion_
relation check, evaluated but never trained on in this mix).

Why these three tasks specifically: this session found `ner_re_medication`
(2018_n2c2_track2, 303 train examples), `family_history` (2019_n2c2_track2,
98), and `medication_event` (2022n2c2_track1, 350) are the smallest slices
in this 11-dataset mixture by 10-100x (MedNLI/MedQA-USMLE-US/umn_abbreviation
are all 10k+; 2022n2c2_track2_mimic/uw are 1300+/500+; 2018_n2c2_track1/
2019_n2c2_track1_cls are 1600-2600) -- see multitask.build_unified_dataset's
UPSAMPLE_MD_TASKS comment for the full writeup and the frequency_penalty
decoding-bug investigation that came first. 2018_n2c2_track2_paragraph is
NOT upsampled here despite being in the same family: its paragraph-level
splitting already gives it thousands of train rows, so it isn't
training-starved the way the other three are.

Mixed rendering, not an exact markdown duplicate: the base mix uses each
dataset's already-built huggingface_format/<name>_md (markdown-rendered);
the upsample addition uses huggingface_format/<name> (plain) for the same 3
tasks -- two markdown renders of the same example would be byte-identical
(schemas.render_md/render_messages_md are deterministic), so using the
OTHER rendering is what makes this a real augmentation rather than empty
repetition. This script does no rendering itself, only concatenation --
run `python -m multitask.build_hf_datasets` first if any <name>/<name>_md
directory is missing.

dev is never touched by the upsampling (only the train bucket gets the
extra rows), so monitoring never sees a duplicate-flavored row.

Run with: python -m multitask.build_med_flan_mix
"""
from collections import defaultdict
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk

ROOT = Path(__file__).resolve().parent / "huggingface_format"

# Same 11 names as run_custom_model_eval.ipynb's MAIN_DATASET_SPECS / this
# session's test.ipynb prototype.
MED_FLAN_DATASETS = [
    "2018_n2c2_track1",
    "2018_n2c2_track2",
    "2018_n2c2_track2_paragraph",
    "2019_n2c2_track1_cls",
    "2019_n2c2_track2",
    "2022n2c2_track1",
    "2022n2c2_track2_mimic",
    "2022n2c2_track2_uw",
    "MedNLI",
    "MedQA-USMLE-US",
    "umn_abbreviation",
]

# The training-starved IE tasks within MED_FLAN_DATASETS -- see module
# docstring. Upsampled by adding each one's PLAIN train split on top of the
# all-markdown base mix's own train bucket.
UPSAMPLE_DATASETS = {"2018_n2c2_track2", "2019_n2c2_track2", "2022n2c2_track1"}

SPLIT_MAP = {"train": "train", "dev": "dev", "validation": "dev"}


def _bucket(names, suffix=""):
    buckets = defaultdict(list)
    for name in names:
        for split_name, split in load_from_disk((ROOT / f"{name}{suffix}").as_posix()).items():
            target = SPLIT_MAP.get(split_name.lower())
            if target:
                buckets[target].append(split)
    return buckets


def build():
    buckets = _bucket(MED_FLAN_DATASETS, suffix="_md")
    for name in UPSAMPLE_DATASETS:
        plain_train = load_from_disk((ROOT / name).as_posix())["train"]
        buckets["train"].append(plain_train)
    return DatasetDict({name: concatenate_datasets(parts).shuffle(seed=42) for name, parts in buckets.items()})


def main():
    dd = build()
    out_path = (ROOT / "med_flan_v1_md_upsampled").as_posix()
    dd.save_to_disk(out_path)
    print(f"[saved] med_flan_v1_md_upsampled: {dict((k, len(v)) for k, v in dd.items())} -> {out_path}")


if __name__ == "__main__":
    main()

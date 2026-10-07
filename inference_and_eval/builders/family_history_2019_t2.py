"""
2019 n2c2 Track 2 (family history) builder.

Not BRAT -- purely TSV-labeled, no character offsets. The official release
defines two subtasks: Subtask 1 is bare entity identification (family
members + side of family; observation mentions with no attributes) and is
explicitly a component task, not the end product -- Subtask 2 is what the
task's own README calls the actual goal: "evaluate participating systems
as end-to-end family history summarization systems," producing each family
member's side, living-status score, and observation(s) with negation in
one combined structure.

Trained/scored target is Subtask 2 ONLY (`living_status` + `observations`
below) -- not both subtasks merged into one JSON object as an earlier
version of this builder did. That earlier version asked the model to
predict `family_members`/`observations_subtask1` as a second, separately-
shaped view of largely the same information Subtask 2's own `member`/
`side`/`text` fields already carry (Subtask 1's family-member list is
redundant with the member/side already attached to every Subtask 2 row;
its bare observation strings are a subset of Subtask 2's negation-labeled
ones -- measured, 930 subtask-1 observations vs 740 subtask-2 ones on the
train split). That is two different tasks bolted into one generation
target, not one end-to-end task -- restated per the project's task-tag
design (`multitask/schemas.py`'s docstring): every task tag should be one
coherent output contract, not a byte-for-byte instruction: JSON: shape.

Cost of dropping Subtask 1: family members mentioned in the note but never
assigned a living-status score or observation in Subtask 2 become
invisible to this target -- measured, 70/667 (10.5%) of train and 63/638
(9.9%) of test Subtask-1 family members have no Subtask-2 counterpart.
Accepted: those members are, by the task's own Subtask 2 definition, not
part of the "end-to-end family history" being summarized, so training on
Subtask 2 alone matches what is actually scored as the real task rather
than a superset the official eval never rewards.

`multitask/convert_back/family_history_2019_t2.py` still exposes
`to_subtask1_lines`/gold-file paths for anyone who wants to run the
official Subtask 1 scorer separately (e.g. to report both numbers), but
Subtask 1 is no longer part of what this builder trains on or what the
default `_self_check` verifies.
"""
import csv
from pathlib import Path

from inference_and_eval.io_utils import read_text
from inference_and_eval.schemas import example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "raw_datasets" / "2019_n2c2_track2"
SPLIT_FILES = {
    "train": {"subtask1": BASE / "train_subtask1_2.tsv", "subtask2": BASE / "train_subtask2_2.tsv"},
    "test": {"subtask1": BASE / "test_subtask1.tsv", "subtask2": BASE / "test_subtask2.tsv"},
}
SPLIT_TEXT_DIRS = {"train": BASE / "bioc-FH-training", "test": BASE / "bioc-FH-testing"}


def _load_tsv_by_doc(path):
    by_doc = {}
    if not path.exists():
        return by_doc
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        # QUOTE_NONE: these files are plain tab-delimited text, not quoted
        # CSV -- a handful of observation strings contain literal `"`
        # characters (e.g. `"pyromaniac`) that csv's default dialect would
        # otherwise misparse as quoting syntax, silently corrupting the text.
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if not row:
                continue
            by_doc.setdefault(row[0], []).append(row)
    return by_doc


def build_family_history(split):
    files = SPLIT_FILES.get(split)
    text_dir = SPLIT_TEXT_DIRS.get(split)
    if files is None or text_dir is None:
        return
    s2_by_doc = _load_tsv_by_doc(files["subtask2"])
    for doc_id in sorted(s2_by_doc):
        txt_path = text_dir / f"{doc_id}.txt"
        if not txt_path.exists():
            continue
        doc_text = read_text(txt_path)

        living_status, observations = [], []
        for row in s2_by_doc.get(doc_id, []):
            member, side, rtype = row[1], row[2], row[3]
            if rtype == "LivingStatus" and len(row) >= 5:
                living_status.append({"member": member, "side": side, "score": row[4]})
            elif rtype == "Observation" and len(row) >= 6:
                observations.append({"member": member, "side": side, "text": row[4], "negation": row[5]})

        target = {"living_status": living_status, "observations": observations}
        yield example("family_history", "2019_n2c2_track2", split, doc_id, doc_text, target)


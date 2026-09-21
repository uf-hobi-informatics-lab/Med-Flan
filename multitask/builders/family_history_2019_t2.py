"""
2019 n2c2 Track 2 (family history) builder.

Not BRAT -- purely TSV-labeled, no character offsets, and official scoring
(`eval.py`) matches family-member/status tuples exactly but observation text
by token overlap. Subtask 1 gives family members (+ side of family) and bare
observation mentions; subtask 2 gives per-member living-status scores and
per-member negated/non-negated observations. Both subtasks are merged into
one JSON target per document so a single generation captures everything;
the back-converter splits it into the two separate TSV formats `eval.py`
expects.
"""
import csv
from pathlib import Path

from multitask.io_utils import read_text
from multitask.schemas import example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "2019_n2c2_track2"
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
    s1_by_doc = _load_tsv_by_doc(files["subtask1"])
    s2_by_doc = _load_tsv_by_doc(files["subtask2"])
    doc_ids = sorted(set(s1_by_doc) | set(s2_by_doc))

    for doc_id in doc_ids:
        txt_path = text_dir / f"{doc_id}.txt"
        if not txt_path.exists():
            continue
        doc_text = read_text(txt_path)

        family_members, observations_subtask1 = [], []
        for row in s1_by_doc.get(doc_id, []):
            if row[1] == "FamilyMember" and len(row) >= 4:
                family_members.append({"member": row[2], "side": row[3]})
            elif row[1] == "Observation" and len(row) >= 3:
                observations_subtask1.append(row[2])

        living_status, observations = [], []
        for row in s2_by_doc.get(doc_id, []):
            member, side, rtype = row[1], row[2], row[3]
            if rtype == "LivingStatus" and len(row) >= 5:
                living_status.append({"member": member, "side": side, "score": row[4]})
            elif rtype == "Observation" and len(row) >= 6:
                observations.append({"member": member, "side": side, "text": row[4], "negation": row[5]})

        target = {
            "family_members": family_members,
            "observations_subtask1": observations_subtask1,
            "living_status": living_status,
            "observations": observations,
        }
        yield example("family_history", "2019_n2c2_track2", split, doc_id, doc_text, target)

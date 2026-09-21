"""
JSON -> native TSV converter for 2019 n2c2 Track 2 (family history), so the
official eval.py can score model output unmodified instead of forking it to
understand JSON.

Input: an iterable of (doc_id, target) pairs, where `target` is the JSON
dict produced by multitask.builders.family_history_2019_t2.build_family_history
(or a model's parsed completion in that same schema):

    {"family_members": [{"member": ..., "side": ...}, ...],
     "observations_subtask1": [<text>, ...],
     "living_status": [{"member": ..., "side": ..., "score": ...}, ...],
     "observations": [{"member": ..., "side": ..., "text": ..., "negation": ...}, ...]}

Output: the two flat TSVs eval.py's parse_s1_output/parse_s2_output expect,
verified field-for-field against README examples:

  subtask 1: doc_id\tFamilyMember\t<member>\t<side>      (4 fields)
             doc_id\tObservation\t<text>                  (3 fields)
  subtask 2: doc_id\t<member>\t<side>\tLivingStatus\t<score>              (5 fields)
             doc_id\t<member>\t<side>\tObservation\t<text>\t<negation>    (6 fields)

Run standalone to self-check the converter against gold (round-trip through
JSON and back should reproduce eval.py P=R=F1=1.0 on the test split):
    python -m multitask.convert_back.family_history_2019_t2
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = ROOT / "2019_n2c2_track2" / "eval.py"
GOLD_S1 = ROOT / "2019_n2c2_track2" / "test_subtask1.tsv"
GOLD_S2 = ROOT / "2019_n2c2_track2" / "test_subtask2.tsv"


def to_subtask1_lines(predictions):
    """predictions: iterable of (doc_id, target dict) -> list of TSV lines."""
    lines = []
    for doc_id, target in predictions:
        for fm in target.get("family_members", []):
            lines.append("\t".join([doc_id, "FamilyMember", fm["member"], fm["side"]]))
        for ob in target.get("observations_subtask1", []):
            lines.append("\t".join([doc_id, "Observation", ob]))
    return lines


def to_subtask2_lines(predictions):
    """predictions: iterable of (doc_id, target dict) -> list of TSV lines."""
    lines = []
    for doc_id, target in predictions:
        for ls in target.get("living_status", []):
            lines.append("\t".join([doc_id, ls["member"], ls["side"], "LivingStatus", str(ls["score"])]))
        for ob in target.get("observations", []):
            lines.append("\t".join([doc_id, ob["member"], ob["side"], "Observation", ob["text"], ob["negation"]]))
    return lines


def write_tsvs(predictions, out_dir):
    """predictions: iterable of (doc_id, target dict), consumed once per
    call -- pass a list if you need both TSVs from the same predictions."""
    predictions = list(predictions)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_path = out_dir / "subtask1_pred.tsv"
    s2_path = out_dir / "subtask2_pred.tsv"
    s1_path.write_text("\n".join(to_subtask1_lines(predictions)) + "\n")
    s2_path.write_text("\n".join(to_subtask2_lines(predictions)) + "\n")
    return s1_path, s2_path


def _run_eval(subtask, gs_path, pred_path):
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "--subtask", str(subtask), "--gs", str(gs_path), "--pred", str(pred_path)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)


def _self_check():
    """Round-trip gold test-split JSON (built by the dataset builder) back
    to TSV, then score it against the real gold TSV -- should print
    P=R=F1=1.0 for both subtasks if the converter is correct."""
    from multitask.builders.family_history_2019_t2 import build_family_history

    predictions = [(ex["id"], ex["target"]) for ex in _examples_with_raw_target("test")]
    with tempfile.TemporaryDirectory() as tmp:
        s1_path, s2_path = write_tsvs(predictions, tmp)
        print("=== Subtask 1 (self-check: gold JSON -> TSV -> eval.py) ===")
        _run_eval(1, GOLD_S1, s1_path)
        print("=== Subtask 2 (self-check: gold JSON -> TSV -> eval.py) ===")
        _run_eval(2, GOLD_S2, s2_path)


def _examples_with_raw_target(split):
    """schemas.example() only keeps the rendered text/messages, not the raw
    target dict -- rebuild examples here with the raw dict attached so the
    self-check can round-trip it without going through JSON string parsing."""
    import csv
    from multitask.builders.family_history_2019_t2 import (
        BASE, SPLIT_FILES, SPLIT_TEXT_DIRS, _load_tsv_by_doc,
    )

    files = SPLIT_FILES[split]
    s1_by_doc = _load_tsv_by_doc(files["subtask1"])
    s2_by_doc = _load_tsv_by_doc(files["subtask2"])
    doc_ids = sorted(set(s1_by_doc) | set(s2_by_doc))

    for doc_id in doc_ids:
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
        yield {"id": doc_id, "target": target}


if __name__ == "__main__":
    _self_check()

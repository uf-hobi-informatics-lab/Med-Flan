"""
JSON -> native TSV converter for 2019 n2c2 Track 2 (family history), so the
official eval.py can score model output unmodified instead of forking it to
understand JSON.

Only Subtask 2 (the task's own "end-to-end family history summarization"
framing) is trained/scored -- see multitask.builders.family_history_2019_t2's
module docstring for why Subtask 1 (bare family-member/observation
identification, redundant with Subtask 2's own member/side/text fields) was
dropped from the trainable target rather than merged alongside it.

Input: an iterable of (doc_id, target) pairs, where `target` is the JSON
dict produced by multitask.builders.family_history_2019_t2.build_family_history
(or a model's parsed completion in that same schema):

    {"living_status": [{"member": ..., "side": ..., "score": ...}, ...],
     "observations": [{"member": ..., "side": ..., "text": ..., "negation": ...}, ...]}

Output: the flat TSV eval.py's parse_s2_output expects, verified
field-for-field against the README examples:

  doc_id\t<member>\t<side>\tLivingStatus\t<score>              (5 fields)
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
GOLD_S2 = ROOT / "2019_n2c2_track2" / "test_subtask2.tsv"


def to_subtask2_lines(predictions):
    """predictions: iterable of (doc_id, target dict) -> list of TSV lines."""
    lines = []
    for doc_id, target in predictions:
        for ls in target.get("living_status", []):
            lines.append("\t".join([doc_id, ls["member"], ls["side"], "LivingStatus", str(ls["score"])]))
        for ob in target.get("observations", []):
            lines.append("\t".join([doc_id, ob["member"], ob["side"], "Observation", ob["text"], ob["negation"]]))
    return lines


def write_tsv(predictions, out_dir):
    """predictions: iterable of (doc_id, target dict), consumed once."""
    predictions = list(predictions)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s2_path = out_dir / "subtask2_pred.tsv"
    s2_path.write_text("\n".join(to_subtask2_lines(predictions)) + "\n")
    return s2_path


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
    P=R=F1=1.0 if the converter is correct."""
    predictions = [(ex["id"], ex["target"]) for ex in _examples_with_raw_target("test")]
    with tempfile.TemporaryDirectory() as tmp:
        s2_path = write_tsv(predictions, tmp)
        print("=== Subtask 2 (self-check: gold JSON -> TSV -> eval.py) ===")
        _run_eval(2, GOLD_S2, s2_path)


def _examples_with_raw_target(split):
    """schemas.example() only keeps the rendered text/messages, not the raw
    target dict -- rebuild examples here with the raw dict attached so the
    self-check can round-trip it without going through JSON string parsing."""
    from multitask.builders.family_history_2019_t2 import build_family_history

    for ex in build_family_history(split):
        import json
        yield {"id": ex["id"], "target": json.loads(ex["messages"][-1]["content"])}


if __name__ == "__main__":
    _self_check()

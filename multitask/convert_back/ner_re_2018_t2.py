"""
JSON -> BRAT .ann converter for 2018 n2c2 Track 2 (Drug-ADE), so
Track2-evaluate-ver4.py can score model output unmodified.

Input target schema (see multitask.builders.ner_re_2018_t2):
    {"entities": [{"id": "T1", "type": "Drug", "text": "Lipitor"}, ...],
     "relations": [{"type": "Reason-Drug", "arg1": "T5", "arg2": "T1"}, ...]}

Entity ids are reused verbatim as T-line ids (relations reference them
directly), so no id-remapping is needed -- only offsets need recovering.
Note that Track2-evaluate-ver4.py's own T-line parser collapses a
discontinuous span ("start1 end1;start2 end2") down to (start1, end2)
anyway, so a single contiguous (start, end) from reanchor() is exactly what
it would have derived from the original multi-span gold line too.

Run standalone to self-check against gold (should print P=R=F1=1.0 for
tags and relations on the test split):
    python -m multitask.convert_back.ner_re_2018_t2
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from multitask.reanchor import reanchor

ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = ROOT / "2018_n2c2_track2" / "Track2-evaluate-ver4.py"
GOLD_DIR = ROOT / "2018_n2c2_track2" / "gold_standard_test"


def to_ann(target, doc_text):
    """target: the JSON dict (entities/relations). Returns .ann file text."""
    used_spans = []
    cursor = 0
    lines = []
    valid_ids = set()
    for ent in target.get("entities", []):
        span = reanchor(ent["text"], doc_text, used_spans, min_start=cursor)
        if span is None:
            continue
        start, end = span
        cursor = max(cursor, end)
        # Use the JSON-provided text (already newline-free) rather than a
        # raw doc_text slice: a discontinuous span's slice can embed a
        # literal newline, which would corrupt the single-physical-line
        # BRAT T-line format when written to disk.
        lines.append(f"{ent['id']}\t{ent['type']} {start} {end}\t{ent['text']}")
        valid_ids.add(ent["id"])

    rid = 0
    for rel in target.get("relations", []):
        if rel["arg1"] not in valid_ids or rel["arg2"] not in valid_ids:
            continue  # dropped entity -- can't emit a relation pointing at it
        rid += 1
        lines.append(f"R{rid}\t{rel['type']} Arg1:{rel['arg1']} Arg2:{rel['arg2']}")

    return "\n".join(lines) + ("\n" if lines else "")


def write_anns(predictions, out_dir):
    """predictions: iterable of (doc_id, target, doc_text). Writes one
    <doc_id>.ann per prediction under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, target, doc_text in predictions:
        (out_dir / f"{doc_id}.ann").write_text(to_ann(target, doc_text), encoding="utf-8")
    return out_dir


def _run_eval(gold_dir, system_dir):
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), str(gold_dir), str(system_dir)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)


def _self_check():
    """Round-trip gold test-split JSON (built by the dataset builder) back
    to .ann, then score against the real gold -- should be P=R=F1=1.0."""
    from multitask.builders.ner_re_2018_t2 import build_ner_re_medication
    from multitask.io_utils import read_text

    predictions = []
    for ex in _examples_with_raw_target("test"):
        doc_text = read_text(GOLD_DIR / f"{ex['id']}.txt")
        predictions.append((ex["id"], ex["target"], doc_text))

    with tempfile.TemporaryDirectory() as tmp:
        write_anns(predictions, tmp)
        print("=== self-check: gold JSON -> .ann -> Track2-evaluate-ver4.py ===")
        _run_eval(GOLD_DIR, tmp)


def _examples_with_raw_target(split):
    """Rebuild examples with the raw target dict attached (schemas.example()
    only keeps rendered text/messages)."""
    from multitask.brat import parse_ann
    from multitask.builders.ner_re_2018_t2 import SPLIT_DIRS

    split_dir = SPLIT_DIRS[split]
    for txt_path in sorted(split_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        doc = parse_ann(ann_path)
        # Match the builder: emit in true document order, not T-id order.
        entities = [
            {"id": eid, "type": e.type, "text": e.text}
            for eid, e in sorted(doc.entities.items(), key=lambda kv: (kv[1].start, kv[1].end))
        ]
        relations = [
            {"type": r.type, "arg1": r.arg1, "arg2": r.arg2}
            for r in doc.relations.values()
            if r.arg1 in doc.entities and r.arg2 in doc.entities
        ]
        yield {"id": txt_path.stem, "target": {"entities": entities, "relations": relations}}


if __name__ == "__main__":
    _self_check()

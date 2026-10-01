"""
Seq2seq -> BRAT .ann converter for medication_event_seq2seq (see
multitask/medication_event_seq2seq_grammar.py for the grammar). `to_ann`
parses the tag-delimited text back into the exact same {"events": [...]}
shape multitask.convert_back.medication_event_2022_t1.to_ann already
accepts, then delegates to that function unchanged -- a single reanchor()
call per event, no new span-resolution logic needed (see that module's
docstring for why this task's ceiling is inherently below 1.0: repeated
medication mentions within a doc can make reanchor() pick a same-text-but-
wrong occurrence).

Run standalone to self-check:
    python -m multitask.convert_back.medication_event_2022_t1_seq2seq
"""
import tempfile
from pathlib import Path

from multitask.convert_back.medication_event_2022_t1 import GOLD_DIR
from multitask.convert_back.medication_event_2022_t1 import score_dirs
from multitask.convert_back.medication_event_2022_t1 import to_ann as _to_ann_json
from multitask.medication_event_seq2seq_grammar import parse_and_resolve, serialize_events

GOLD_DIR_BY_DATASET = {"2022n2c2_track1_seq2seq": GOLD_DIR}


def to_ann(generated_text, doc_text):
    return _to_ann_json(parse_and_resolve(generated_text), doc_text)


def write_anns(predictions, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, generated_text, doc_text in predictions:
        (out_dir / f"{doc_id}.ann").write_text(to_ann(generated_text, doc_text), encoding="utf-8")
    return out_dir


def _examples_with_raw_target(split):
    from multitask.builders.medication_event_2022_t1 import _build_events, _iter_docs

    for txt_path, _doc_text, doc in _iter_docs(split):
        events = _build_events(doc)
        yield {"id": txt_path.stem, "target": serialize_events({"events": events})}


def _self_check():
    """Round-trip gold test-split seq2seq text (built by the dataset
    builder) back to .ann, then score against the real gold -- should
    match medication_event_2022_t1's own current self-check numbers
    exactly (triggers F1=0.9792, attributes F1=0.9146, combined F1=0.9479)."""
    from multitask.io_utils import read_text

    predictions = []
    for ex in _examples_with_raw_target("test"):
        doc_text = read_text(GOLD_DIR / f"{ex['id']}.txt")
        predictions.append((ex["id"], ex["target"], doc_text))

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = write_anns(predictions, tmp)
        print("=== self-check: gold seq2seq text -> .ann -> brat_scorer ===")
        score_dirs(GOLD_DIR, out_dir)


if __name__ == "__main__":
    _self_check()

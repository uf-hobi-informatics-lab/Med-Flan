"""
Seq2seq -> BRAT .ann converter for ner_re_medication_seq2seq (see
multitask/ner_re_medication_seq2seq_grammar.py for the grammar). `to_ann`
parses the tag-delimited text back into the exact same {"drugs": [...],
"unlinked": [...]} shape multitask.convert_back.ner_re_2018_t2.to_ann
already accepts, then delegates to that function unchanged -- so drug-span
disambiguation (`_pick_drug_span`'s filler-support heuristic), the
two-tier exclusive/non-exclusive filler reanchor, and the unlinked-mention
reanchor-with-cursor all reuse the exact same, already-validated logic as
the JSON path. This module adds no new span-resolution code of its own
(see the grammar module's docstring for why).

GOLD_DIR_BY_DATASET mirrors sdoh_2022_t2_seq2seq.py's convention so
eval/run_eval.py's generic _run_brat_seq2seq dispatch works unchanged.

Run standalone to self-check against gold -- same "Known limitation"
ceiling as the JSON path's own self-check (see that module's docstring):
    python -m multitask.convert_back.ner_re_2018_t2_seq2seq
"""
import tempfile
from pathlib import Path

from inference_and_eval.convert_back.ner_re_2018_t2 import GOLD_DIR
from inference_and_eval.convert_back.ner_re_2018_t2 import score_dirs
from inference_and_eval.convert_back.ner_re_2018_t2 import to_ann as _to_ann_json
from inference_and_eval.ner_re_medication_seq2seq_grammar import parse_and_resolve, serialize_events

GOLD_DIR_BY_DATASET = {"2018_n2c2_track2_seq2seq": GOLD_DIR}


def to_ann(generated_text, doc_text):
    return _to_ann_json(parse_and_resolve(generated_text), doc_text)


def write_anns(predictions, out_dir):
    """predictions: iterable of (doc_id, generated_text, doc_text). Writes
    one <doc_id>.ann per prediction under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, generated_text, doc_text in predictions:
        (out_dir / f"{doc_id}.ann").write_text(to_ann(generated_text, doc_text), encoding="utf-8")
    return out_dir


def _examples_with_raw_target(split):
    """Rebuild examples with the raw seq2seq target string attached
    (schemas.example() only keeps rendered text/messages). Delegates to
    the JSON-path builder's own _iter_docs/_build_ner_re_target instead of
    reimplementing them -- same anti-duplication rationale as that
    module's own _examples_with_raw_target."""
    from inference_and_eval.builders.ner_re_2018_t2 import _build_ner_re_target, _iter_docs

    for txt_path, _doc_text, doc in _iter_docs(split):
        target = _build_ner_re_target(doc)
        yield {"id": txt_path.stem, "target": serialize_events(target)}


def _self_check():
    """Round-trip gold test-split seq2seq text (built by the dataset
    builder) back to .ann, then score against the real gold -- expect the
    same sub-1.0 F1 ceiling as ner_re_2018_t2.py's own self-check (see that
    module's "Known limitation" docstring section), since resolution is
    delegated straight to its to_ann."""
    from inference_and_eval.io_utils import read_text

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

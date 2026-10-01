"""
Seq2seq -> target-dict converter for concept_assertion_relation_seq2seq (see
multitask/concept_assertion_relation_seq2seq_grammar.py for the grammar).
`to_target` parses the tag-delimited text back into the exact same
{"concepts": [...], "relations": [...]} shape
inference_and_eval.convert_back.i2b2_2010.score already accepts; `score_dataset`
delegates to that module's own score_dataset verbatim -- no new scoring
logic here, since i2b2_2010.py's scorer already re-resolves every predicted
concept's span from text via reanchor() regardless of predicted id (see
that module's docstring for why this task's ceiling is inherently below
1.0 on concepts: a handful of short, repeated tokens like "T"/"P"/"rate"
can make reanchor() pick a same-text-but-wrong occurrence).

Run standalone to self-check:
    python -m inference_and_eval.convert_back.i2b2_2010_seq2seq
"""
from inference_and_eval.concept_assertion_relation_seq2seq_grammar import parse_and_resolve, serialize_target
from inference_and_eval.convert_back.i2b2_2010 import score_dataset as _score_dataset_json


def to_target(generated_text):
    return parse_and_resolve(generated_text)


def score_dataset(predictions, split="test"):
    """predictions: {doc_id: raw generated text}. Parses each completion
    back into the target-dict shape, then delegates to
    i2b2_2010.score_dataset unchanged."""
    parsed = {doc_id: to_target(text) for doc_id, text in predictions.items()}
    return _score_dataset_json(parsed, split=split)


def _examples_with_raw_target(split):
    from inference_and_eval.builders.i2b2_2010 import _build_target, _iter_records

    for doc_id, _doc_text, concepts, relations in _iter_records(split):
        yield {"id": doc_id, "target": serialize_target(_build_target(concepts, relations))}


def _self_check():
    """Round-trip gold test-split seq2seq text (built by the dataset
    builder) back to a target dict, then score against the real gold.
    Concept F1/assertion_accuracy match i2b2_2010.py's own current numbers
    EXACTLY (0.9996149032444401 / 1.0), since concept scoring fully
    delegates and text is copied verbatim with no lossy step. Relation F1
    is 0.9998410932782457, not quite the JSON path's exact 1.0: because
    relations reference concepts by (type, text) rather than a raw id
    (the model never invents/echoes one -- see
    multitask/concept_assertion_relation_seq2seq_grammar.py's docstring),
    a relation whose non-anchor argument has two truly identical
    (type, text) concepts elsewhere in the same document (measured: 1 case
    in the whole test split, out of ~6300 relations -- two identical
    "KLEBSIELLA PNEUMONIAE" problem concepts) can't be disambiguated from
    text alone without doc-text/sentence-boundary access, which this
    module deliberately doesn't have. Same category of inherent ceiling as
    the concept side's own repeated-short-token ambiguity -- not a
    converter bug."""
    predictions = {ex["id"]: ex["target"] for ex in _examples_with_raw_target("test")}
    if not predictions:
        print("i2b2_2010_seq2seq: no data found under 2010i2b2/ -- nothing to self-check (see .gitignore's 2010i2b2/ entry).")
        return
    import json
    print("=== self-check: gold seq2seq text -> target dict -> i2b2_2010.score ===")
    print(json.dumps(score_dataset(predictions, split="test"), indent=2))


if __name__ == "__main__":
    _self_check()

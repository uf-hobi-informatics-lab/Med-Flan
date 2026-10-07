"""
Convert-back + scorer for SemEval-2015 Task 14 disorder normalization.

No confirmed persistent public official scorer exists for this task (the
QCRI task page only linked a validation script during the evaluation
period, with no lasting public link, and the downloaded package ships no
bundled eval script). Implements strict F (exact span match + correct CUI)
and relaxed F (any word overlap + correct CUI) directly from the paper's
own formulas (Elhadad et al. 2015, SemEval-2015 Task 14, SS4.1) -- flagged
in the README as self-implemented, not an official binary (same treatment
as discharge_me.py's missing MEDCON metric).

Gold spans come from the builder's own `_char_span` (exact, from the
released .pipe offsets -- see builders/disorder_normalization_2015_t14.py),
never reanchored. Predicted spans are recovered via reanchor(), since a
generative model can only be expected to reproduce text it saw, not exact
offsets. Gold used to be reanchored through the same lossy path as
predictions -- comparing a lossy reconstruction against an equally lossy
reconstruction rather than against the truth, which is why the old
self-check's "Strict F1 = Relaxed F1 = 1.0" was vacuous: it could not fail
for any span reason. Confirmed by that bug once fixed here: on the real
test split, 638/8,003 gold entities (8.0%, mostly discontinuous BRAT-style
mentions whose space-joined text can never re-match a contiguous span) no
longer trivially match, so gold-round-trip strict F1 no longer needs to be
1.0 -- see _self_check's assertion for the honest floor.
"""
from pathlib import Path

from inference_and_eval.builders.disorder_normalization_2015_t14 import _iter_docs
from inference_and_eval.reanchor import reanchor

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "raw_datasets" / "semeval_2015"


def _examples_with_raw_target(split):
    for doc_id, doc_text, entities in _iter_docs(split):
        # Strip the internal "_char_span" -- what a real model produces is
        # text+cui only, and this dict stands in for a model's parsed
        # completion in the self-check below.
        target = {"entities": [{"text": e["text"], "cui": e["cui"]} for e in entities]}
        yield {"id": doc_id, "doc_text": doc_text, "target": target}


def _words(text):
    return set(text.lower().split())


def _resolve_predicted(entities, doc_text):
    """entities: a prediction's "entities" list (text/cui only, no
    offsets). Returns [(start, end, cui, text), ...], resolved left-to-right
    via reanchor() -- see module docstring for why predictions (unlike
    gold) must go through this lossy path.

    An entity whose text can't be found anywhere in doc_text is NOT
    dropped -- it gets a poison span (past doc_text, monotonically
    increasing) that can never equal a real gold span, so it still lands
    in the returned list. Two reasons this matters, not one: (a) dropping
    it meant a hallucinated entity cost nothing in the strict-match
    precision denominator (`len(pred_resolved)`), same "hallucination is
    free" gap as every other convert_back module; (b) relaxed matching
    (_score_doc's word-overlap check) doesn't even need a real span, only
    the text/cui -- dropping the entity here threw away information
    relaxed matching could have used, which silently capped relaxed
    recall below 1.0 even for the gold-round-trip self-check."""
    used_spans = []
    resolved = []
    cursor = 0
    poison = len(doc_text) + 1
    for ent in entities if isinstance(entities, list) else []:
        if not isinstance(ent, dict):
            continue
        text = ent.get("text") or ""
        span = reanchor(text, doc_text, used_spans, min_start=cursor)
        if span is None:
            start, end = poison, poison + 1
            poison += 1
        else:
            start, end = span
            cursor = end
        resolved.append((start, end, ent.get("cui"), text))
    return resolved


def _resolve_gold(entities):
    """entities: the builder's own gold list, each carrying an exact
    "_char_span" -- no reanchoring, no lossy reconstruction."""
    return [(e["_char_span"][0], e["_char_span"][1], e["cui"], e["text"]) for e in entities]


def _score_doc(gold_entities, doc_text, predicted_entities):
    gold_resolved = _resolve_gold(gold_entities)
    pred_resolved = _resolve_predicted(predicted_entities, doc_text)

    gold_remaining_strict = list(gold_resolved)
    gold_remaining_relaxed = list(gold_resolved)
    strict_tp = 0
    relaxed_tp = 0
    for p_start, p_end, p_cui, p_text in pred_resolved:
        for g in gold_remaining_strict:
            if g[0] == p_start and g[1] == p_end and g[2] == p_cui:
                strict_tp += 1
                gold_remaining_strict.remove(g)
                break
        for g in gold_remaining_relaxed:
            if g[2] == p_cui and (_words(g[3]) & _words(p_text)):
                relaxed_tp += 1
                gold_remaining_relaxed.remove(g)
                break

    return strict_tp, relaxed_tp, len(gold_resolved), len(pred_resolved)


def _prf(tp, n_gold, n_pred):
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score(gold_docs, predictions):
    """gold_docs: {doc_id: (doc_text, gold_entities)} -- gold_entities carry
    "_char_span" (see builders/disorder_normalization_2015_t14.py).
    predictions: {doc_id: parsed target dict, e.g. {"entities": [...]}}."""
    strict_tp = relaxed_tp = n_gold = n_pred = 0
    for doc_id, (doc_text, gold_entities) in gold_docs.items():
        predicted = predictions.get(doc_id)
        predicted_entities = predicted.get("entities", []) if isinstance(predicted, dict) else []
        s_tp, r_tp, g, p = _score_doc(gold_entities, doc_text, predicted_entities)
        strict_tp += s_tp
        relaxed_tp += r_tp
        n_gold += g
        n_pred += p

    return {"strict": _prf(strict_tp, n_gold, n_pred), "relaxed": _prf(relaxed_tp, n_gold, n_pred)}


def score_dataset(predictions, split="test"):
    """predictions: {doc_id: parsed target dict}. Reloads gold for `split`
    directly from semeval_2015/<split>/*.pipe+.text, independent of the
    builder's rendered eval jsonl."""
    gold_docs = {doc_id: (doc_text, entities) for doc_id, doc_text, entities in _iter_docs(split)}
    return score(gold_docs, predictions)


def _self_check():
    predictions = {ex["id"]: ex["target"] for ex in _examples_with_raw_target("test")}
    result = score_dataset(predictions, split="test")
    print(result)
    # Relaxed matching is span-independent (word-overlap on the entity
    # TEXT only, see _score_doc) -- gold-as-prediction always shares every
    # word with itself, so this stays exactly 1.0 regardless of the span
    # fix above.
    assert result["relaxed"]["f1"] == 1.0, result
    # Strict is no longer trivially 1.0 now that gold uses its real spans
    # (see module docstring): a real, non-bug ceiling remains from
    # discontinuous mentions, whose space-joined text can never re-match a
    # single contiguous span via reanchor(). Floor set from the measured
    # value at the time of this fix (0.92) with headroom, not hand-tuned to
    # the exact number, so an unrelated regression still trips this.
    assert result["strict"]["f1"] >= 0.90, result
    print("OK")


if __name__ == "__main__":
    _self_check()

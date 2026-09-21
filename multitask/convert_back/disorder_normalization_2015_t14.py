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
"""
from pathlib import Path

from multitask.builders.disorder_normalization_2015_t14 import _iter_docs
from multitask.reanchor import reanchor

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "semeval_2015"


def _examples_with_raw_target(split):
    for doc_id, doc_text, entities in _iter_docs(split):
        yield {"id": doc_id, "doc_text": doc_text, "target": {"entities": entities}}


def _words(text):
    return set(text.lower().split())


def _resolve_entities(entities, doc_text):
    """Recover (start, end, cui, text) for each entity via reanchor(), fed
    in the given order (min_start floor advances left-to-right -- entities
    are always supplied in true document order by the builder, and gold
    round-trips through the same order at self-check time)."""
    used_spans = []
    resolved = []
    cursor = 0
    for ent in entities if isinstance(entities, list) else []:
        if not isinstance(ent, dict):
            continue
        text = ent.get("text") or ""
        span = reanchor(text, doc_text, used_spans, min_start=cursor)
        if span is None:
            continue
        cursor = span[1]
        resolved.append((span[0], span[1], ent.get("cui"), text))
    return resolved


def _score_doc(gold_entities, doc_text, predicted_entities):
    gold_resolved = _resolve_entities(gold_entities, doc_text)
    pred_resolved = _resolve_entities(predicted_entities, doc_text)

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
    """gold_docs: {doc_id: (doc_text, gold_entities)}.
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
    assert result["strict"]["f1"] == 1.0, result
    assert result["relaxed"]["f1"] == 1.0, result
    print("OK")


if __name__ == "__main__":
    _self_check()

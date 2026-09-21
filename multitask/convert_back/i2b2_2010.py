"""
Native Python P/R/F1 scorer for the 2010 i2b2/VA concept/assertion/relation
challenge. Unlike every other convert_back/ module, this doesn't shell out
to an official eval tool -- no persistent Perl/Java scorer ships with this
release, and the user chose a from-scratch Python reimplementation over
sourcing/porting one (same situation, and same style of solution, as
disorder_normalization_2015_t14.py's self-implemented strict/relaxed F).

Input target schema (see multitask.builders.i2b2_2010):
    {"concepts": [{"id": "c0", "type": "problem", "text": ..., "assertion": ...}, ...],
     "relations": [{"type": "TeCP", "arg1": "c1", "arg2": "c0"}, ...]}

Predicted concept text is reanchored onto the raw document with
multitask.reanchor.reanchor() (the same left-to-right greedy matching every
BRAT-family convert_back module uses) to recover a comparable span, since a
generative model can only be expected to reproduce the text it saw, not the
exact line:token offsets. Gold spans are NOT reanchored -- the builder
already derives them exactly from the released .con/.ast/.rel offsets.

Scoring, standard i2b2-2010 methodology:
  - concepts: exact-match P/R/F1 on (span, type).
  - assertion: accuracy, restricted to problem concepts whose span+type were
    already correctly matched (an assertion on a concept nobody correctly
    predicted isn't a meaningful comparison).
  - relations: P/R/F1 on (type, arg1, arg2), matched via each relation's
    endpoints' gold identity -- a predicted relation only counts if both
    endpoints reanchored onto a correctly-matched gold concept.

No real data has been obtained yet (DUA-gated) -- self-check degrades to a
no-op until 2010_i2b2_va/ is populated; see _self_check() below.

Run standalone to self-check against gold on the test split (once data
exists -- should give concept/relation P=R=F1=1.0 and assertion accuracy=1.0):
    python -m multitask.convert_back.i2b2_2010
"""
from pathlib import Path

from multitask.reanchor import reanchor

ROOT = Path(__file__).resolve().parents[2]


def _prf(tp, n_gold, n_pred):
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _resolve_predicted_concepts(concepts, doc_text):
    """concepts: predicted target's "concepts" list (no offsets -- only
    text). Returns {id: (start, end, type, assertion)}, resolved
    left-to-right via reanchor()."""
    used_spans, cursor = [], 0
    resolved = {}
    for c in concepts if isinstance(concepts, list) else []:
        if not isinstance(c, dict):
            continue
        span = reanchor(c.get("text") or "", doc_text, used_spans, min_start=cursor)
        if span is None:
            continue
        cursor = span[1]
        resolved[c.get("id")] = (span[0], span[1], c.get("type"), c.get("assertion"))
    return resolved


def _score_doc(gold, doc_text, pred):
    """gold: {"concepts": [...with "_char_span"...], "relations": [...]}.
    pred: parsed model target dict (or {} if unparseable)."""
    gold_by_id = {
        c["id"]: (c["_char_span"][0], c["_char_span"][1], c["type"], c.get("assertion"))
        for c in gold["concepts"]
    }
    pred_by_id = _resolve_predicted_concepts(pred.get("concepts", []) if isinstance(pred, dict) else [], doc_text)

    gold_remaining = list(gold_by_id.items())  # [(gid, (start, end, type, assertion)), ...]
    concept_tp = assertion_correct = assertion_total = 0
    matched_pred_to_gold = {}  # predicted concept id -> gold concept id
    for pid, (p_start, p_end, p_type, p_assertion) in pred_by_id.items():
        for gid, g in gold_remaining:
            if g[0] == p_start and g[1] == p_end and g[2] == p_type:
                concept_tp += 1
                gold_remaining.remove((gid, g))
                matched_pred_to_gold[pid] = gid
                if g[2] == "problem" and g[3] is not None:
                    assertion_total += 1
                    if g[3] == p_assertion:
                        assertion_correct += 1
                break

    # Relations are compared by their endpoints' GOLD identity, not raw ids
    # (predicted ids are the model's own generation order, unrelated to
    # gold's) -- a predicted relation only counts if both endpoints
    # reanchored onto a correctly-matched gold concept.
    gold_rel_keys = [
        (r.get("type"), r.get("arg1"), r.get("arg2"))
        for r in gold.get("relations", [])
        if r.get("arg1") in gold_by_id and r.get("arg2") in gold_by_id
    ]

    pred_rel_keys = []
    for r in pred.get("relations", []) if isinstance(pred, dict) else []:
        if not isinstance(r, dict):
            continue
        gid1 = matched_pred_to_gold.get(r.get("arg1"))
        gid2 = matched_pred_to_gold.get(r.get("arg2"))
        if gid1 is not None and gid2 is not None:
            pred_rel_keys.append((r.get("type"), gid1, gid2))

    gold_rel_remaining = list(gold_rel_keys)
    rel_tp = 0
    for k in pred_rel_keys:
        if k in gold_rel_remaining:
            rel_tp += 1
            gold_rel_remaining.remove(k)

    return {
        "concept": (concept_tp, len(gold_by_id), len(pred_by_id)),
        "assertion": (assertion_correct, assertion_total),
        "relation": (rel_tp, len(gold_rel_keys), len(pred_rel_keys)),
    }


def score(gold_docs, predictions):
    """gold_docs: {doc_id: (doc_text, gold_target_with_char_spans)}.
    predictions: {doc_id: parsed target dict}."""
    c_tp = c_gold = c_pred = 0
    a_correct = a_total = 0
    r_tp = r_gold = r_pred = 0
    for doc_id, (doc_text, gold) in gold_docs.items():
        pred = predictions.get(doc_id)
        result = _score_doc(gold, doc_text, pred if isinstance(pred, dict) else {})
        ctp, cg, cp = result["concept"]
        ac, at = result["assertion"]
        rtp, rg, rp = result["relation"]
        c_tp += ctp; c_gold += cg; c_pred += cp
        a_correct += ac; a_total += at
        r_tp += rtp; r_gold += rg; r_pred += rp

    return {
        "concept": _prf(c_tp, c_gold, c_pred),
        "assertion_accuracy": (a_correct / a_total) if a_total else 0.0,
        "relation": _prf(r_tp, r_gold, r_pred),
    }


def score_dataset(predictions, split="test"):
    """predictions: {doc_id: parsed target dict}. Reloads gold for `split`
    directly from 2010_i2b2_va/, independent of the builder's rendered eval
    jsonl (mirrors disorder_normalization_2015_t14.score_dataset)."""
    from multitask.builders.i2b2_2010 import _iter_records

    gold_docs = {
        doc_id: (doc_text, {"concepts": concepts, "relations": relations})
        for doc_id, doc_text, concepts, relations in _iter_records(split)
    }
    return score(gold_docs, predictions)


def _examples_with_raw_target(split):
    from multitask.builders.i2b2_2010 import _iter_records

    for doc_id, _doc_text, concepts, relations in _iter_records(split):
        target = {
            "concepts": [
                {"id": c["id"], "type": c["type"], "text": c["text"], "assertion": c["assertion"]}
                for c in concepts
            ],
            "relations": relations,
        }
        yield {"id": doc_id, "target": target}


def _self_check():
    predictions = {ex["id"]: ex["target"] for ex in _examples_with_raw_target("test")}
    if not predictions:
        print("i2b2_2010: no data found under 2010_i2b2_va/test/ -- nothing to self-check yet (DUA-gated, not yet obtained).")
        return
    result = score_dataset(predictions, split="test")
    print(result)
    assert result["concept"]["f1"] == 1.0, result
    assert result["assertion_accuracy"] == 1.0, result
    assert result["relation"]["f1"] == 1.0, result
    print("OK")


if __name__ == "__main__":
    _self_check()

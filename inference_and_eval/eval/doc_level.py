"""
Doc-level, offset-free scoring for the IE tasks -- an ADDON axis alongside
the real offset-based scorers (multitask.brat_scorer, convert_back/
i2b2_2010.py, convert_back/disorder_normalization_2015_t14.py), not a
replacement for them.

Modeled on how 2019_n2c2_track2/eval.py already scores family_history: no
character offsets anywhere, just per-document Python sets of tuples
compared by plain set membership (TP = |gold & pred|, FP = |pred - gold|,
FN = |gold - pred|). family_history's target schema has no offsets to begin
with, so it needs no module here -- this generalizes that same idea to the
tasks whose targets DO carry span text that would otherwise need
reanchor()-based offset recovery to score.

Why this is worth having as a second signal: every offset-based scorer in
this repo has to resolve predicted text back to a character span before it
can be compared to gold, via multitask.reanchor.reanchor() or
brat.align_entity_text(). That resolution step is real, useful, and mostly
solid, but it is not free -- see convert_back/ner_re_2018_t2.py's own
"Known limitation" section, and multitask.target_schema's docstring on
serialization drift. A doc-level, set-of-tuples score answers a narrower,
more robust question -- "did the model name the right concepts/relations/
events" -- without any dependency on span reconstruction succeeding.

Text fields are compared after _norm() (lowercase + collapse whitespace),
exact match only -- no fuzzy/word-overlap fallback (unlike family_history's
own Observation axis, which is a bespoke behavior of that one official
script, not generalized here).

Run standalone to self-check against real gold (gold fed back as its own
prediction -- a wiring smoke test, not a meaningful accuracy result, since
gold-vs-itself is trivially perfect for every axis):
    python -m inference_and_eval.eval.doc_level
"""
from inference_and_eval.brat_scorer import Measures
from inference_and_eval.target_schema import CMED_ATTR_NAMES, MEDICATION_ATTR_TYPES


def _norm(text):
    return " ".join((text or "").split()).lower()


def _score_axis(gold_by_doc, pred_by_doc):
    """gold_by_doc/pred_by_doc: {doc_id: set of tuples}. Returns one
    micro-summed Measures across every doc id in either dict."""
    total = Measures()
    for doc_id in set(gold_by_doc) | set(pred_by_doc):
        gold = gold_by_doc.get(doc_id, set())
        pred = pred_by_doc.get(doc_id, set())
        total = total + Measures(tp=len(gold & pred), fp=len(pred - gold), fn=len(gold - pred))
    return total


def _tuples_ner_re_medication(target):
    if not isinstance(target, dict):
        return {"drugs": set(), "attr_links": set()}
    drugs = [d for d in target.get("drugs") or [] if isinstance(d, dict)]
    unlinked = [o for o in target.get("unlinked") or [] if isinstance(o, dict)]

    entity_tuples = {("Drug", _norm(d.get("text"))) for d in drugs}
    entity_tuples |= {(o.get("type"), _norm(o.get("text"))) for o in unlinked}

    link_tuples = {
        (_norm(d.get("text")), attr_type, _norm(v))
        for d in drugs
        for attr_type in MEDICATION_ATTR_TYPES
        for v in (d.get(attr_type) or [])
    }
    return {"drugs": entity_tuples, "attr_links": link_tuples}


def _tuples_medication_event(target):
    if not isinstance(target, dict):
        return {"events": set(), "attributes": set()}
    events = [e for e in target.get("events") or [] if isinstance(e, dict)]

    event_tuples = {(e.get("type"), _norm(e.get("trigger_text"))) for e in events}
    attr_tuples = {
        (_norm(e.get("trigger_text")), name, value)
        for e in events
        for name, value in (e.get("attributes") or {}).items()
        if name in CMED_ATTR_NAMES
    }
    return {"events": event_tuples, "attributes": attr_tuples}


def _tuples_sdoh_event(target):
    if not isinstance(target, dict):
        return {"events": set(), "roles": set(), "attributes": set()}
    events = [e for e in target.get("events") or [] if isinstance(e, dict)]

    event_tuples = {(e.get("type"), _norm(e.get("trigger_text"))) for e in events}
    role_tuples = {
        (_norm(e.get("trigger_text")), role, _norm(filler.get("text")))
        for e in events
        for role, filler in (e.get("roles") or {}).items()
        if isinstance(filler, dict)
    }
    attr_tuples = {
        (_norm(e.get("trigger_text")), a.get("role"), a.get("name"), a.get("value"))
        for e in events
        for a in (e.get("attributes") or [])
        if isinstance(a, dict)
    }
    return {"events": event_tuples, "roles": role_tuples, "attributes": attr_tuples}


def _tuples_concept_assertion_relation(target):
    if not isinstance(target, dict):
        return {"concepts": set(), "relations": set()}
    concepts = [c for c in target.get("concepts") or [] if isinstance(c, dict)]
    relations = [r for r in target.get("relations") or [] if isinstance(r, dict)]

    concept_tuples = {(c.get("type"), _norm(c.get("text")), c.get("assertion")) for c in concepts}
    relation_tuples = {(r.get("type"), _norm(r.get("arg1")), _norm(r.get("arg2"))) for r in relations}
    return {"concepts": concept_tuples, "relations": relation_tuples}


def _tuples_disorder_normalization(target):
    if not isinstance(target, dict):
        return {"entities": set()}
    entities = [e for e in target.get("entities") or [] if isinstance(e, dict)]
    return {"entities": {(_norm(e.get("text")), e.get("cui")) for e in entities}}


_TUPLE_FUNCS = {
    "ner_re_medication": _tuples_ner_re_medication,
    "medication_event": _tuples_medication_event,
    "sdoh_event": _tuples_sdoh_event,
    "concept_assertion_relation": _tuples_concept_assertion_relation,
    "disorder_normalization": _tuples_disorder_normalization,
}


def score(task, gold_by_id, pred_by_id):
    """gold_by_id/pred_by_id: {doc_id: target dict}. Prints per-axis P/R/F1
    (same line shape as brat_scorer.score_dirs) and returns
    {axis: Measures.as_dict()}. Raises ValueError for any task with no
    doc-level axis defined (this module intentionally covers only the 5
    tasks above -- family_history already has its own offset-free official
    eval.py and needs no axis here)."""
    tuple_func = _TUPLE_FUNCS.get(task)
    if tuple_func is None:
        raise ValueError(f"no doc-level axis for task {task!r} -- see inference_and_eval.eval.doc_level")

    gold_axes = {doc_id: tuple_func(target) for doc_id, target in gold_by_id.items()}
    pred_axes = {doc_id: tuple_func(target) for doc_id, target in pred_by_id.items()}

    axis_names = set().union(*(a.keys() for a in gold_axes.values())) if gold_axes else set(tuple_func({}).keys())

    results = {}
    for axis in sorted(axis_names):
        gold_by_doc = {doc_id: axes[axis] for doc_id, axes in gold_axes.items()}
        pred_by_doc = {doc_id: axes.get(axis, set()) for doc_id, axes in pred_axes.items()}
        m = _score_axis(gold_by_doc, pred_by_doc)
        results[axis] = m.as_dict()
        print(f"{axis}: P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f} (TP={m.tp} FP={m.fp} FN={m.fn})")
    return results


def _self_check():
    """Gold fed back as its own prediction for all 5 covered tasks -- a
    wiring smoke test (every axis must be exactly P=R=F1=1.0), not a
    meaningful accuracy result."""
    from inference_and_eval.convert_back import (
        disorder_normalization_2015_t14, i2b2_2010, medication_event_2022_t1,
        ner_re_2018_t2, sdoh_2022_t2,
    )

    jobs = [
        ("ner_re_medication", lambda: {ex["id"]: ex["target"] for ex in ner_re_2018_t2._examples_with_raw_target("test")}),
        ("medication_event", lambda: {ex["id"]: ex["target"] for ex in medication_event_2022_t1._examples_with_raw_target("test")}),
        ("concept_assertion_relation", lambda: {ex["id"]: ex["target"] for ex in i2b2_2010._examples_with_raw_target("test")}),
        ("disorder_normalization", lambda: {ex["id"]: ex["target"] for ex in disorder_normalization_2015_t14._examples_with_raw_target("test")}),
    ]
    for task, load_gold in jobs:
        print(f"\n=== {task} (gold vs itself) ===")
        gold = load_gold()
        if not gold:
            print(f"{task}: no data found -- skipping (see that dataset's own self-check for why)")
            continue
        results = score(task, gold, gold)
        for axis, m in results.items():
            assert m["f1"] == 1.0, (task, axis, m)

    for source in ("mimic", "uw"):
        print(f"\n=== sdoh_event ({source}, gold vs itself) ===")
        gold = {ex["id"]: ex["target"] for ex in sdoh_2022_t2._examples_with_raw_target(source, "test")}
        if not gold:
            print(f"sdoh_event ({source}): no data found -- skipping")
            continue
        results = score("sdoh_event", gold, gold)
        for axis, m in results.items():
            assert m["f1"] == 1.0, (source, axis, m)

    print("\ndoc_level._self_check(): all axes P=R=F1=1.0")


if __name__ == "__main__":
    _self_check()

"""Deterministic, per-task GRPO reward for the meddata multitask mix.

This is the RL counterpart of `multitask/eval/` in the meddata repo: instead of the
soft string-similarity rewards the bioleaflets GRPO used (`semantic_similarity_reward`,
`meteor_reward_function`), it scores each rollout against the gold answer with the SAME
deterministic metric that task's official eval uses -- exact-match for the
classification/QA tasks, offset-free doc-level micro-F1 for the IE tasks -- routed per
example by the `task` column that `convert_to_grpo` now carries through.

The scoring logic is ported (self-contained, no import from meddata -- the meddata eval
code is not guaranteed to be on the training cluster) from:
  - `multitask/eval/simple_tasks.py` -- `parse_completion` (first `{...}` JSON object),
    `_norm`, and the per-task key for `_accuracy` (nli->"label", mcqa->"answer", ...).
  - `multitask/eval/doc_level.py` -- the per-task tuple sets + micro-F1 over pooled
    TP/FP/FN (offset-free, so no `reanchor()` span recovery is needed -- ideal for a
    rollout reward where the model emits JSON, not spans).
  - `multitask/target_schema.py` -- `MEDICATION_ATTR_TYPES`, `CMED_ATTR_NAMES`.
  - `multitask/convert_back/family_history_2019_t2.py` -- the family-history tuple
    schema (`doc_level.py` deliberately excludes family_history, which has its own
    offset-free official eval; the tuple shape is `(member, side, score)` for
    living_status and `(member, side, text, negation)` for observations).

TRL's `GRPOTrainer` passes every dataset column other than `prompt`/`completion` as a
kwarg to reward functions, so `med_multitask_reward` receives `ground_truth` (the gold
JSON string, as produced by `convert_to_grpo`) and `task` (the task tag) aligned 1:1
with `completions`.

Reward range: [0, 1]. 1.0 = perfect (exact match, or F1=1); 0.0 = total miss or
unparseable JSON. An unparseable completion scores 0.0 on every task, which still
carries gradient signal toward emitting valid JSON in the task's shape -- no separate
format reward is needed.

Run the self-check (`python -m alignment.med_grpo_rewards`) to confirm the scoring
ports are correct: gold-vs-itself must be 1.0 for every task, gold-vs-wrong 0.0.
"""

import json
import re


# Inlined copy of `grpo_reward_functions.get_predicted_results` (same logic; inlined
# here so this module has no import back into grpo_reward_functions, which would be
# circular -- grpo_reward_functions imports this module to register the reward).
def _get_predicted_results(completions):
    """Extract the assistant text from each completion (TRL passes completions as a
    list of conversations, each a list of {role, content} message dicts)."""
    predicted_results = []
    for completion in completions:
        for each in completion:
            if each["role"] == "assistant":
                predicted_results.append(each["content"])
                break
    return predicted_results


# --- JSON parse + normalize (ported from multitask/eval/simple_tasks.py:77-98) ----
# Greedy first-{...}-to-last-} match, same as the eval scorer: a single well-formed
# JSON object is captured whole; anything unparseable -> {} (a full miss).
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def parse_completion(raw):
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    m = _JSON_OBJ_RE.search(raw)
    if m is None:
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {}


def _norm(text):
    """Collapse whitespace + lowercase (multitask/eval/doc_level.py:40). A superset of
    simple_tasks._norm (which only strips) -- safe for the closed-vocab exact-match
    fields too, since those are single tokens where strip == collapse-whitespace."""
    return " ".join((text or "").split()).lower()


# --- Closed vocabularies (ported from multitask/target_schema.py) -----------------
# Inlined rather than imported so this module has no meddata dependency on the cluster.
_MED_ATTR_TYPES = (
    "Strength",
    "Frequency",
    "Dosage",
    "Route",
    "Duration",
    "Form",
    "Reason",
    "ADE",
)
_CMED_ATTR_NAMES = ("Action", "Certainty", "Temporality", "Actor", "Negation")


# --- Doc-level tuple sets (ported from multitask/eval/doc_level.py:54-116) ---------
# Each returns a single set of tuples for the example (all axes pooled -- a 2-tuple and
# a 3-tuple never collide, so pooling is safe and gives one micro-F1 scalar).
def _tuples_ner_re_medication(target):
    tuples = set()
    if not isinstance(target, dict):
        return tuples
    drugs = [d for d in target.get("drugs") or [] if isinstance(d, dict)]
    unlinked = [o for o in target.get("unlinked") or [] if isinstance(o, dict)]
    for d in drugs:
        tuples.add(("Drug", _norm(d.get("text"))))
        for attr_type in _MED_ATTR_TYPES:
            for v in d.get(attr_type) or []:
                tuples.add((_norm(d.get("text")), attr_type, _norm(v)))
    for o in unlinked:
        tuples.add((o.get("type"), _norm(o.get("text"))))
    return tuples


def _tuples_medication_event(target):
    tuples = set()
    if not isinstance(target, dict):
        return tuples
    for e in target.get("events") or []:
        if not isinstance(e, dict):
            continue
        trig = _norm(e.get("trigger_text"))
        tuples.add((e.get("type"), trig))
        for name, value in (e.get("attributes") or {}).items():
            if name in _CMED_ATTR_NAMES:
                tuples.add((trig, name, value))
    return tuples


def _tuples_sdoh_event(target):
    tuples = set()
    if not isinstance(target, dict):
        return tuples
    for e in target.get("events") or []:
        if not isinstance(e, dict):
            continue
        trig = _norm(e.get("trigger_text"))
        tuples.add((e.get("type"), trig))
        for role, filler in (e.get("roles") or {}).items():
            if isinstance(filler, dict):
                tuples.add((trig, role, _norm(filler.get("text"))))
        for a in e.get("attributes") or []:
            if isinstance(a, dict):
                tuples.add((trig, a.get("role"), a.get("name"), _norm(a.get("value"))))
    return tuples


def _tuples_concept_assertion_relation(target):
    tuples = set()
    if not isinstance(target, dict):
        return tuples
    for c in target.get("concepts") or []:
        if isinstance(c, dict):
            tuples.add((c.get("type"), _norm(c.get("text")), c.get("assertion")))
    for r in target.get("relations") or []:
        if isinstance(r, dict):
            tuples.add((r.get("type"), _norm(r.get("arg1")), _norm(r.get("arg2"))))
    return tuples


def _tuples_family_history(target):
    """family_history is excluded from doc_level.py (it has its own offset-free
    official eval.py); the tuple shape is taken from
    convert_back/family_history_2019_t2.to_subtask2_lines -- (member, side, score) for
    living_status, (member, side, text, negation) for observations -- with a leading
    axis tag so the two axes are distinguishable in the pooled set."""
    tuples = set()
    if not isinstance(target, dict):
        return tuples
    for ls in target.get("living_status") or []:
        if isinstance(ls, dict):
            tuples.add(
                (
                    "LivingStatus",
                    _norm(ls.get("member")),
                    _norm(ls.get("side")),
                    _norm(str(ls.get("score"))),
                )
            )
    for ob in target.get("observations") or []:
        if isinstance(ob, dict):
            tuples.add(
                (
                    "Observation",
                    _norm(ob.get("member")),
                    _norm(ob.get("side")),
                    _norm(ob.get("text")),
                    _norm(ob.get("negation")),
                )
            )
    return tuples


def _micro_f1(gold_set, pred_set):
    if not gold_set and not pred_set:
        return 1.0
    tp = len(gold_set & pred_set)
    if tp == 0:
        return 0.0
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


# --- Task dispatch ---------------------------------------------------------------
# task tag -> gold/pred key for an exact-match score (ported from
# simple_tasks.score_dataset's dispatch). qa_yesnomaybe/multilabel_cls/mcqa_explain are
# not in the v1_md 12-task mix but are wired for robustness.
_EXACT_MATCH_KEYS = {
    "nli": "label",
    "mcqa": "answer",
    "mcqa_explain": "answer",
    "qa_yesnomaybe": "decision",
    "abbreviation_disambiguation": "expansion",
    "cohort_eligibility": "label",
}

# task tag -> doc-level tuple extractor (offset-free micro-F1). Paragraph/section
# variants reuse their base task's extractor (same target shape -- see
# target_schema.SPECS).
_IE_TUPLE_FUNCS = {
    "ner_re_medication": _tuples_ner_re_medication,
    "ner_re_medication_paragraph": _tuples_ner_re_medication,
    "medication_event": _tuples_medication_event,
    "medication_event_paragraph": _tuples_medication_event,
    "sdoh_event": _tuples_sdoh_event,
    "concept_assertion_relation": _tuples_concept_assertion_relation,
    "concept_assertion_relation_section": _tuples_concept_assertion_relation,
    "family_history": _tuples_family_history,
}


def _score_one(task_tag, pred_text, gold_text):
    """Deterministic score in [0, 1] for one (completion, gold, task) triple."""
    pred = parse_completion(pred_text)
    gold = parse_completion(gold_text)

    key = _EXACT_MATCH_KEYS.get(task_tag)
    if key is not None:
        return float(_norm(pred.get(key)) == _norm(gold.get(key)))

    if task_tag in ("sts_classification", "similarity_classification"):
        return float(_norm(pred.get("category")) == _norm(gold.get("category")))

    if task_tag == "multilabel_cls":
        glabels = {_norm(l) for l in gold.get("labels") or []}
        plabels = {_norm(l) for l in pred.get("labels") or []}
        return _micro_f1(glabels, plabels)

    tuple_func = _IE_TUPLE_FUNCS.get(task_tag)
    if tuple_func is not None:
        return _micro_f1(tuple_func(gold), tuple_func(pred))

    # Unknown task tag: no signal (and intentionally not 1.0, so a novel task can't
    # be reward-hacked to a free 1.0). Logged once at first encounter via the
    # warning below in med_multitask_reward.
    return 0.0


_SEEN_UNKNOWN_TASKS = set()


def med_multitask_reward(completions, ground_truth, task=None, **kwargs):
    """GRPO reward routing each rollout to its task's deterministic eval metric.

    Args:
        completions: list of conversations (each a list of {role, content} dicts),
            as TRL passes to reward functions.
        ground_truth: list of gold answer strings (the assistant turn from the
            SFT example, carried through by `convert_to_grpo`), aligned 1:1 with
            `completions`.
        task: list of task tags (the `task` dataset column, carried through by
            `convert_to_grpo`), aligned 1:1 with `completions`.

    Returns: list[float] in [0, 1], one per completion.
    """
    preds = _get_predicted_results(completions)
    tasks = task if task is not None else [""] * len(preds)
    rewards = []
    for pred_text, gold_text, task_tag in zip(preds, ground_truth, tasks):
        rewards.append(_score_one(task_tag, pred_text, gold_text))
        if (
            task_tag
            and task_tag not in _EXACT_MATCH_KEYS
            and task_tag
            not in ("sts_classification", "similarity_classification", "multilabel_cls")
            and task_tag not in _IE_TUPLE_FUNCS
            and task_tag not in _SEEN_UNKNOWN_TASKS
        ):
            _SEEN_UNKNOWN_TASKS.add(task_tag)
            print(
                f"[med_multitask_reward] WARNING: unknown task tag {task_tag!r} -- scoring 0.0"
            )
    return rewards


# --- Self-check: gold-vs-itself must be 1.0, gold-vs-wrong 0.0 (no dataset needed) --
_EXAMPLES = {
    "nli": {"label": "entailment"},
    "mcqa": {"answer": "C"},
    "abbreviation_disambiguation": {"expansion": "as needed"},
    "cohort_eligibility": {"label": "met"},
    "sts_classification": {"category": "mostly_equivalent"},
    "multilabel_cls": {"labels": ["Treatment", "Diagnosis"]},
    "ner_re_medication": {
        "drugs": [
            {"text": "Lipitor", "Strength": ["10 mg"], "Reason": ["hyperlipidemia"]}
        ],
        "unlinked": [{"type": "Reason", "text": "hyperlipidemia"}],
    },
    "medication_event": {
        "events": [
            {
                "id": "E1",
                "type": "Disposition",
                "trigger_text": "started",
                "attributes": {"Action": "Start", "Certainty": "Certain"},
            }
        ],
    },
    "sdoh_event": {
        "events": [
            {
                "id": "E2",
                "type": "Tobacco",
                "trigger_text": "smokes",
                "roles": {"Status": {"type": "StatusTime", "text": "current"}},
                "attributes": [
                    {"role": "Status", "name": "StatusTimeVal", "value": "current"}
                ],
            }
        ],
    },
    "concept_assertion_relation": {
        "concepts": [
            {"id": "c0", "type": "problem", "text": "diabetes", "assertion": "present"}
        ],
        "relations": [{"type": "TrIP", "arg1": "c1", "arg2": "c0"}],
    },
    "family_history": {
        "living_status": [{"member": "Father", "side": "Paternal", "score": "0"}],
        "observations": [
            {
                "member": "Father",
                "side": "Paternal",
                "text": "heart disease",
                "negation": "Non_Negated",
            }
        ],
    },
}


def _self_check():
    ok = True
    for task_tag, gold in _EXAMPLES.items():
        gold_str = json.dumps(gold)
        perfect = _score_one(task_tag, gold_str, gold_str)
        wrong_str = json.dumps(
            {
                k: (
                    "WRONG"
                    if isinstance(v, str)
                    else ["WRONG"] if isinstance(v, list) else {}
                )
                for k, v in gold.items()
            }
        )
        wrong = _score_one(task_tag, wrong_str, gold_str)
        status = "OK" if perfect == 1.0 and wrong == 0.0 else "FAIL"
        if status == "FAIL":
            ok = False
        print(
            f"  {task_tag:30s} gold-vs-self={perfect:.3f}  gold-vs-wrong={wrong:.3f}  [{status}]"
        )
    # unparseable completion -> 0.0 on a known task
    unparseable = _score_one(
        "nli", "not json at all", json.dumps({"label": "entailment"})
    )
    print(
        f"  {'(unparseable nli)':30s} score={unparseable:.3f}  [{'OK' if unparseable == 0.0 else 'FAIL'}]"
    )
    ok = ok and unparseable == 0.0
    print("self-check:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _self_check()

"""
Local scorer for qa_span_extraction (emrqa-msquad): standard SQuAD-style
normalized exact-match + token-F1, the same metric the emrQA-msquad paper
itself reports. No convert_back/ step is needed -- the target is always a
verbatim substring of the context shown in the prompt (confirmed by direct
inspection of the released parquet: context[answer_start:answer_end] ==
text on every row checked), so gold/prediction are just a small JSON dict
like the tasks in multitask.eval.simple_tasks, not a round-trip-to-native-
format problem.

Gold lives in multitask/data/eval/emrqa-msquad.jsonl (built by
build_unified_dataset.py), in each row's messages[-1]["content"]. Predictions
are parsed the same lenient way multitask.eval.simple_tasks.parse_completion
does (reused here, not reimplemented).

Run standalone to self-check by feeding gold back as its own "prediction" --
expect exact_match == f1 == 1.0:
    python -m multitask.eval.qa_span
"""
import json
import re
import string
from pathlib import Path

from multitask.eval.simple_tasks import parse_completion

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "multitask" / "data" / "eval"
GOLD_PATH = EVAL_DIR / "emrqa-msquad.jsonl"

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(s):
    """Standard SQuAD evaluate-v2.0.py normalization: lowercase, strip
    punctuation, strip articles, collapse whitespace."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def exact_match(pred, gold):
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def f1_score(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for t in gold_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in pred_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _load_gold():
    rows = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["id"], json.loads(d["messages"][-1]["content"])))
    return rows


def score_dataset(predictions):
    """predictions: {id: raw completion string}. Returns {"n", "exact_match",
    "f1"} (both averaged over all rows, matching the standard SQuAD metric)."""
    rows = _load_gold()
    n = len(rows)
    if n == 0:
        return {"n": 0, "exact_match": 0.0, "f1": 0.0}
    em_total = f1_total = 0.0
    for doc_id, gold in rows:
        pred = parse_completion(predictions.get(doc_id))
        gold_answer = gold.get("answer", "")
        pred_answer = pred.get("answer", "")
        em_total += exact_match(pred_answer, gold_answer)
        f1_total += f1_score(pred_answer, gold_answer)
    return {"n": n, "exact_match": em_total / n, "f1": f1_total / n}


def _self_check():
    if not GOLD_PATH.exists():
        print(f"emrqa-msquad: skipped (no eval file at {GOLD_PATH})")
        return
    rows = _load_gold()
    predictions = {doc_id: json.dumps(gold) for doc_id, gold in rows}
    print(f"emrqa-msquad: {score_dataset(predictions)}")


if __name__ == "__main__":
    _self_check()

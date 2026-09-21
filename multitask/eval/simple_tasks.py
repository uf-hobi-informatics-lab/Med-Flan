"""
Local scorers for the five "trivial" task tags -- nli, mcqa, mcqa_explain,
qa_yesnomaybe, multilabel_cls. Per the plan, none of these datasets ships
(or needs) an official evaluation script: gold and prediction are both
just a small JSON dict, so scoring is a direct field comparison, not a
format-conversion problem like the BRAT/XML/TSV datasets in convert_back/.

Gold for each dataset lives in multitask/data/eval/<name>.jsonl, built by
build_unified_dataset.py -- specifically in each row's
messages[-1]["content"] (the assistant turn), which is exactly the target
JSON string. score_dataset() reads gold from there and expects predictions
as {id: <raw model completion string>}, parsed the same lenient way a real
model's output would need to be (strip code fences, take the first
{...} JSON object found).

mcqa_explain's "explanation" field is intentionally not scored -- there's
no official metric for free-text rationale quality here, and the plan
never called for one; only "answer" is compared.

Run standalone to self-check every dataset by feeding gold back as its own
"prediction" -- should give 1.0/perfect on all of them:
    python -m multitask.eval.simple_tasks
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "multitask" / "data" / "eval"

# tag -> (eval jsonl filename stem, gold/pred key(s) that matter for scoring)
DATASETS = {
    "nli": ["MedNLI"],
    "mcqa": ["MedQA-USMLE-US", "MedQA-USMLE-Mainland", "MedQA-USMLE-Taiwan", "MMLU-Pro-Health"],
    "mcqa_explain": ["medmcqa"],
    "qa_yesnomaybe": ["pubmedqa"],
    "multilabel_cls": ["covid_LitCovid"],
    "abbreviation_disambiguation": ["umn_abbreviation"],
}

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def parse_completion(raw):
    """raw: a model's raw completion string (or already-parsed dict/None,
    passed through). Returns a dict, or {} if nothing JSON-shaped is found
    (counts as a full miss on every field, the conservative default)."""
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


def _norm(x):
    return str(x).strip().lower() if x is not None else None


def _load_gold(name):
    path = EVAL_DIR / f"{name}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["id"], d["task"], json.loads(d["messages"][-1]["content"])))
    return rows


def _accuracy(rows, predictions, key):
    n = correct = 0
    for doc_id, _, gold in rows:
        pred = parse_completion(predictions.get(doc_id))
        n += 1
        if _norm(pred.get(key)) == _norm(gold.get(key)):
            correct += 1
    return {"n": n, "accuracy": correct / n if n else 0.0}


def _multilabel_prf(rows, predictions):
    tp = fp = fn = 0
    for doc_id, _, gold in rows:
        pred = parse_completion(predictions.get(doc_id))
        gold_labels = {_norm(l) for l in gold.get("labels", []) or []}
        pred_labels = {_norm(l) for l in pred.get("labels", []) or []}
        tp += len(gold_labels & pred_labels)
        fp += len(pred_labels - gold_labels)
        fn += len(gold_labels - pred_labels)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"n": len(rows), "precision": p, "recall": r, "f1": f1}


def score_dataset(name, predictions):
    """name: one of the dataset names in DATASETS' values (e.g. "MedNLI").
    predictions: {id: raw completion string}. Returns a metrics dict."""
    rows = _load_gold(name)
    if not rows:
        return {"n": 0}
    task = rows[0][1]
    if task in ("nli",):
        return _accuracy(rows, predictions, "label")
    if task in ("mcqa", "mcqa_explain"):
        return _accuracy(rows, predictions, "answer")
    if task == "qa_yesnomaybe":
        return _accuracy(rows, predictions, "decision")
    if task == "multilabel_cls":
        return _multilabel_prf(rows, predictions)
    if task == "abbreviation_disambiguation":
        return _accuracy(rows, predictions, "expansion")
    raise ValueError(f"no scorer wired up for task {task!r}")


def _self_check():
    for names in DATASETS.values():
        for name in names:
            path = EVAL_DIR / f"{name}.jsonl"
            if not path.exists():
                print(f"{name}: skipped (no eval file at {path})")
                continue
            rows = _load_gold(name)
            predictions = {doc_id: json.dumps(gold) for doc_id, _, gold in rows}
            metrics = score_dataset(name, predictions)
            print(f"{name}: {metrics}")


if __name__ == "__main__":
    _self_check()

"""
Local scorers for the seven "trivial" task tags -- nli, mcqa, mcqa_explain,
qa_yesnomaybe, multilabel_cls, abbreviation_disambiguation,
sts_classification. Per the plan, none of these datasets ships (or needs)
an official evaluation script: gold and prediction are both just a small
JSON dict, so scoring is a direct field comparison, not a format-conversion
problem like the BRAT/XML/TSV datasets in convert_back/.

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

sts_classification (see multitask.builders.sts_2019_t1) is the discretized,
additive alternative to sts_regression -- scored with accuracy + macro
precision/recall/F1 (flat, per-category) *and* quadratic-weighted Cohen's
kappa. The kappa is the point of adding this task at all: plain accuracy/
macro-F1 scores a near-miss (predicted "mostly_equivalent", gold
"highly_equivalent") identically to a wild miss (predicted "dissimilar",
gold "highly_equivalent"), throwing away the ordinal "how close" signal
that a similarity scale should be judged on -- kappa's quadratic-distance
weighting penalizes far misses more than near ones, same rationale as its
use for essay/Likert-scale scoring elsewhere.

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
    "nli": ["MedNLI", "bionli", "nli4ct"],
    "mcqa": ["MedQA-USMLE-US", "MedQA-USMLE-Mainland", "MedQA-USMLE-Taiwan", "MMLU-Pro-Health"],
    "mcqa_explain": ["medmcqa"],
    "qa_yesnomaybe": ["pubmedqa"],
    "multilabel_cls": ["covid_LitCovid"],
    "abbreviation_disambiguation": ["umn_abbreviation"],
    "sts_classification": ["2019_n2c2_track1_cls"],
}

# Must match multitask.builders.sts_2019_t1.STS_CATEGORIES /
# _ANCHOR_CATEGORY exactly (5 ordinal categories, grouped from the 11
# half-point anchors 0.0, 0.5, ..., 5.0) -- fixed here rather than imported
# so this module has no dependency on the builders/ package for scoring.
_STS_CATEGORIES = (
    "dissimilar", "weakly_related", "partially_equivalent",
    "mostly_equivalent", "highly_equivalent",
)
_STS_ANCHOR_CATEGORY = (
    "dissimilar", "dissimilar",
    "weakly_related", "weakly_related", "weakly_related",
    "partially_equivalent", "partially_equivalent",
    "mostly_equivalent", "mostly_equivalent",
    "highly_equivalent", "highly_equivalent",
)

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


def _quadratic_weighted_kappa(gold_idx, pred_idx, n_classes):
    """Standard weighted-kappa formula: kappa = 1 - sum(w*O)/sum(w*E), with
    quadratic weights w_ij = (i-j)^2/(n_classes-1)^2, O the observed
    confusion counts and E the counts expected under independence (row
    marginal * col marginal / n). n_classes is fixed at the full category
    count (11), not just categories observed in this sample, so kappa is
    comparable across runs/subsets rather than rescaled per-run."""
    n = len(gold_idx)
    if n == 0:
        return 0.0
    conf = [[0] * n_classes for _ in range(n_classes)]
    hist_gold = [0] * n_classes
    hist_pred = [0] * n_classes
    for g, p in zip(gold_idx, pred_idx):
        conf[g][p] += 1
        hist_gold[g] += 1
        hist_pred[p] += 1
    num = den = 0.0
    denom_scale = (n_classes - 1) ** 2 or 1
    for i in range(n_classes):
        for j in range(n_classes):
            w = ((i - j) ** 2) / denom_scale
            num += w * conf[i][j]
            den += w * hist_gold[i] * hist_pred[j] / n
    if den == 0:
        return 1.0  # every gold/pred in the same single category -- no possible disagreement
    return 1 - num / den


def _ordinal_classification_metrics(rows, predictions, key, categories):
    """Accuracy + macro precision/recall/F1 (flat, per-category) +
    quadratic-weighted kappa (see module docstring for why kappa is the
    point of this scorer). Predictions/gold outside `categories` (a model
    hallucinating a raw numeric score instead of a category name) are
    snapped to the nearest of the 11 half-point anchors and mapped through
    _STS_ANCHOR_CATEGORY, same conservative fallback convention as
    sts_regression's parse_similarity()."""
    index_of = {c: i for i, c in enumerate(categories)}
    mid = categories[len(categories) // 2]

    def to_index(raw):
        norm = _norm(raw)
        if norm in index_of:
            return index_of[norm]
        try:
            value = float(norm)
        except (TypeError, ValueError):
            return index_of[mid]
        anchor_idx = max(0, min(10, round(value * 2)))
        return index_of[_STS_ANCHOR_CATEGORY[anchor_idx]]

    n_classes = len(categories)
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    gold_idx, pred_idx = [], []
    correct = 0
    for doc_id, _, gold in rows:
        pred = parse_completion(predictions.get(doc_id))
        g_idx = to_index(gold.get(key))
        p_idx = to_index(pred.get(key))
        gold_idx.append(g_idx)
        pred_idx.append(p_idx)
        if g_idx == p_idx:
            correct += 1
            tp[g_idx] += 1
        else:
            fp[p_idx] += 1
            fn[g_idx] += 1

    n = len(rows)
    precisions, recalls, f1s = [], [], []
    for i in range(n_classes):
        p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "macro_precision": sum(precisions) / n_classes,
        "macro_recall": sum(recalls) / n_classes,
        "macro_f1": sum(f1s) / n_classes,
        "quadratic_weighted_kappa": _quadratic_weighted_kappa(gold_idx, pred_idx, n_classes),
    }


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
    if task == "sts_classification":
        return _ordinal_classification_metrics(rows, predictions, "category", _STS_CATEGORIES)
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

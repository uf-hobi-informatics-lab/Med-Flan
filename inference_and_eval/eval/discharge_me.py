"""
Best-effort scorer for `summarization` (discharge-me / BioNLP-ACL24 "Discharge
Me!" shared task). Target is plain text (see multitask.builders.simple_tasks
.build_discharge_me), not JSON -- JSON-wrapping free text would only hurt a
ROUGE/BLEU-style metric, per the plan.

The task's target has two labeled sections in one string:
    DISCHARGE INSTRUCTIONS:
    <...>

    BRIEF HOSPITAL COURSE:
    <...>
A model's completion is expected to follow the same two-section shape (it's
literally what the training targets look like); parse_sections() splits on
the same two header lines and is tolerant of a missing section (empty
string, scored as a full miss on that field -- the same conservative
default used everywhere else in this project).

The official shared task metric is a 4-way composite: ROUGE-1/2/L, BLEU,
BERTScore, and MEDCON (a clinical-concept-overlap metric via QuickUMLS).
This module computes the first three (rouge_score, sacrebleu, bert_score --
all already installed in this venv) macro-averaged over the two sections.
MEDCON is intentionally NOT implemented: it needs a QuickUMLS index built
from UMLS data, which is licensed and not available in this environment --
not something to fake or approximate silently. Treat these numbers as a
partial stand-in for the real composite, not the official score.

BERTScore requires downloading a ~1.4GB model on first use; it's computed
only when `include_bertscore=True` (default in score_dataset(), off in the
_self_check() sample below to keep the standalone run fast) and is skipped
with a printed note, not a crash, if the download/model load fails for any
reason (offline environment, etc).

Run standalone to self-check against gold on a small sample of the test
split (feeding gold's own target back as the "prediction") -- ROUGE/BLEU
should come back at (or extremely near) their maximum:
    python -m inference_and_eval.eval.discharge_me
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "multitask" / "data" / "eval" / "discharge-me.jsonl"

_SECTION_RE = re.compile(
    r"DISCHARGE INSTRUCTIONS:\s*(.*?)\s*(?:BRIEF HOSPITAL COURSE:\s*(.*))?$", re.S,
)


def parse_sections(text):
    """Returns (discharge_instructions, brief_hospital_course), either half
    "" if its header isn't found at all."""
    if not text:
        return "", ""
    m = _SECTION_RE.search(text)
    if m is None:
        return "", ""
    return (m.group(1) or "").strip(), (m.group(2) or "").strip()


def _rouge(preds, golds):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = 0
    for pred, gold in zip(preds, golds):
        if not gold:
            continue
        scores = scorer.score(gold, pred)
        for k in sums:
            sums[k] += scores[k].fmeasure
        n += 1
    return {k: v / n if n else 0.0 for k, v in sums.items()}


def _bleu(preds, golds):
    import sacrebleu

    pairs = [(p, g) for p, g in zip(preds, golds) if g]
    if not pairs:
        return 0.0
    preds_f, golds_f = zip(*pairs)
    return sacrebleu.corpus_bleu(list(preds_f), [list(golds_f)]).score


def _bertscore(preds, golds):
    from bert_score import score as bert_score

    pairs = [(p, g) for p, g in zip(preds, golds) if g]
    if not pairs:
        return 0.0
    preds_f, golds_f = zip(*pairs)
    _, _, f1 = bert_score(list(preds_f), list(golds_f), lang="en", verbose=False)
    return f1.mean().item()


def _score_field(preds, golds, include_bertscore):
    metrics = _rouge(preds, golds)
    metrics["bleu"] = _bleu(preds, golds)
    if include_bertscore:
        try:
            metrics["bertscore_f1"] = _bertscore(preds, golds)
        except Exception as e:  # offline / model download failure -- skip, don't crash
            print(f"  (bertscore skipped: {e})")
    return metrics


def score_dataset(predictions, include_bertscore=True):
    """predictions: {id: raw model completion string}, id matching
    multitask/data/eval/discharge-me.jsonl's "id" field. Returns
    {"discharge_instructions": {...}, "brief_hospital_course": {...},
    "overall": {...}} (overall = macro mean of the two fields' metrics).

    NOT the official 4-way composite -- see module docstring (no MEDCON)."""
    gold_di, gold_bhc = [], []
    pred_di, pred_bhc = [], []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["id"] not in predictions:
                continue
            gold_text = d["messages"][-1]["content"]
            g_di, g_bhc = parse_sections(gold_text)
            p_di, p_bhc = parse_sections(predictions[d["id"]])
            gold_di.append(g_di); gold_bhc.append(g_bhc)
            pred_di.append(p_di); pred_bhc.append(p_bhc)

    di_metrics = _score_field(pred_di, gold_di, include_bertscore)
    bhc_metrics = _score_field(pred_bhc, gold_bhc, include_bertscore)
    overall = {k: (di_metrics[k] + bhc_metrics[k]) / 2 for k in di_metrics if k in bhc_metrics}
    return {"n": len(gold_di), "discharge_instructions": di_metrics,
            "brief_hospital_course": bhc_metrics, "overall": overall}


def _self_check(sample_size=25):
    """Feeds gold's own target text back as the "prediction" for a small
    sample (BERTScore's model load makes the full ~11k-row test split slow;
    ROUGE/BLEU logic is what's actually being checked here, not runtime)."""
    predictions = {}
    with open(EVAL_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            d = json.loads(line)
            predictions[d["id"]] = d["messages"][-1]["content"]

    print(f"=== self-check: {len(predictions)} gold examples fed back as their own prediction ===")
    metrics = score_dataset(predictions, include_bertscore=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _self_check()

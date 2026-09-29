"""
Builders for datasets that are already close to decoder-only generation
format — no structural reformatting needed, just wrapping the existing
label/answer into the JSON target shape for their task tag.
"""
import json
import re
from pathlib import Path

from multitask.io_utils import iter_csv, iter_csv_gz, iter_json_dict, iter_jsonl, read_csv_gz_dict
from multitask.schemas import example

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# MedNLI
# ---------------------------------------------------------------------------
MEDNLI_DIR = ROOT / "MedNLI/mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0"
MEDNLI_SPLITS = {"train": "mli_train_v1.jsonl", "dev": "mli_dev_v1.jsonl", "test": "mli_test_v1.jsonl"}


def build_mednli(split):
    path = MEDNLI_DIR / MEDNLI_SPLITS[split]
    for i, rec in iter_jsonl(path):
        input_text = f"Premise: {rec['sentence1']}\nHypothesis: {rec['sentence2']}"
        yield example("nli", "MedNLI", split, f"{path.name}#{i}", input_text, {"label": rec["gold_label"]})


# ---------------------------------------------------------------------------
# bionli -- adversarially-generated biomedical NLI (same "nli" task/schema
# as MedNLI: Premise/Hypothesis input, {"label": "entailment"|"contradiction"}
# target, binary only -- no neutral class here, confirmed by exhaustive scan
# of label_cat below)
# ---------------------------------------------------------------------------
BIONLI_DIR = ROOT / "bionli" / "raw"
BIONLI_SPLITS = {"train": "train_balanced.csv", "dev": "dev_balanced.csv", "test": "test.csv"}
# <el>/<le> and <re>/<er> are literal unescaped entity-highlight markup
# around mentions in `conclusion` -- confirmed by direct sampling -- stripped
# so the hypothesis reads as plain prose like MedNLI's.


def _strip_bionli_tags(text):
    return re.sub(r"\s+", " ", re.sub(r"<el>|<le>|<re>|<er>", "", text)).strip()


def build_bionli(split):
    path = BIONLI_DIR / BIONLI_SPLITS[split]
    for i, rec in iter_csv(path):
        hypothesis = _strip_bionli_tags(rec["conclusion"])
        # Only label_cat == "pos" is a true entailment; every other value is
        # one of bionli's adversarial-negative-generation categories (SEN,
        # SEP, SREO, LPR, swap_number, generation, generation_nd, posToNeg,
        # negToPos, generation_nd_SEN, generation_nd_SRE, SRE, ...) --
        # confirmed by exhaustively scanning label_cat across all three
        # splits -- so this is a binary entailment/contradiction task.
        label = "entailment" if rec["label_cat"] == "pos" else "contradiction"
        input_text = f"Premise: {rec['supp_set']}\nHypothesis: {hypothesis}"
        yield example("nli", "bionli", split, f"{path.name}#{i}", input_text, {"label": label})


# ---------------------------------------------------------------------------
# MedQA-USMLE (US / Mainland / Taiwan) -- MCQA
# ---------------------------------------------------------------------------
MEDQA_DIRS = {
    "MedQA-USMLE-US": ROOT / "MedQA-USMLE/questions/US",
    "MedQA-USMLE-Mainland": ROOT / "MedQA-USMLE/questions/Mainland",
    "MedQA-USMLE-Taiwan": ROOT / "MedQA-USMLE/questions/Taiwan",
}
MEDQA_SPLITS = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}


def _format_options(options: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in options.items())


def build_medqa(dataset_name, split):
    path = MEDQA_DIRS[dataset_name] / MEDQA_SPLITS[split]
    for i, rec in iter_jsonl(path):
        input_text = f"Question: {rec['question']}\nOptions:\n{_format_options(rec['options'])}"
        yield example("mcqa", dataset_name, split, f"{path.name}#{i}", input_text, {"answer": rec["answer_idx"]})


# ---------------------------------------------------------------------------
# MMLU-Pro-Health -- MCQA (HF arrow dataset)
# ---------------------------------------------------------------------------
def build_mmlu_pro_health(split):
    from datasets import load_from_disk

    ds = load_from_disk(str(ROOT / "MMLU-Pro-Health"))
    # Single-split arrow dataset; treat it all as "test" (no train/dev given upstream).
    if split != "test":
        return
    for i, rec in enumerate(ds):
        # rec["options"] already ships with the letter prefix baked in
        # ("A. Herbal remedies", ...) -- don't add a second one.
        opts = "\n".join(rec["options"])
        input_text = f"Question: {rec['question']}\nOptions:\n{opts}"
        yield example("mcqa", "MMLU-Pro-Health", split, f"MMLU-Pro-Health#{i}", input_text, {"answer": rec["answer"]})


# ---------------------------------------------------------------------------
# BIOSSES -- biomedical-literature sentence-pair similarity (HF arrow
# dataset), discretized to classification the same way
# multitask.builders.sts_2019_t1.build_sts_classification discretizes 2019
# n2c2 Track 1's clinical STS -- but a different 5-category vocabulary
# (very_weak/.../very_strong, the standard correlation-strength bands) since
# BIOSSES' 0-4 SemEval-2012 scale and domain (TAC Biomedical Summarization
# Track literature sentences, not clinical notes) don't match STS_CATEGORIES.
# Added as an out-of-domain eval-only check (test-only, never merged into
# train_unified.jsonl) -- see multitask.eval.simple_tasks for the scorer.
# A companion candidate, EBMSASS, was evaluated and rejected: its own
# annotations skew heavily to "unrelated", which would collapse most of any
# discretization into one dominant class -- a poor discriminative OOD test.
# ---------------------------------------------------------------------------
BIOSSES_CATEGORIES = ("very_weak", "weak", "moderate", "strong", "very_strong")


def _bucket_biosses(score):
    """Raw 0-4 BIOSSES score -> one of BIOSSES_CATEGORIES, via the standard
    correlation-strength bands over the normalized (score/4) value: [0,.2)
    very_weak, [.2,.4) weak, [.4,.6) moderate, [.6,.8) strong, [.8,1] very_strong.
    Must stay in sync with the duplicate in multitask/eval/simple_tasks.py."""
    norm = score / 4.0
    if norm < 0.2:
        return "very_weak"
    if norm < 0.4:
        return "weak"
    if norm < 0.6:
        return "moderate"
    if norm < 0.8:
        return "strong"
    return "very_strong"


def build_biosses(split):
    from datasets import load_from_disk

    ds = load_from_disk(str(ROOT / "biosses"))
    # Single-split arrow dataset (100 pairs); treat it all as "test" (no train/dev given upstream).
    if split != "test":
        return
    for i, rec in enumerate(ds):
        input_text = f"Sentence 1: {rec['sentence1']}\nSentence 2: {rec['sentence2']}"
        target = {"category": _bucket_biosses(rec["score"])}
        yield example("similarity_classification", "biosses", split, f"biosses#{i}", input_text, target)


# ---------------------------------------------------------------------------
# medmcqa -- MCQA + explanation
# ---------------------------------------------------------------------------
MEDMCQA_DIR = ROOT / "medmcqa/raw"
MEDMCQA_SPLITS = {"train": "train.json", "dev": "dev.json", "test": "test.json"}
_COP_LETTERS = ["A", "B", "C", "D"]  # cop is 1-indexed onto opa..opd


def build_medmcqa(split):
    path = MEDMCQA_DIR / MEDMCQA_SPLITS[split]
    for i, rec in iter_jsonl(path):
        opts = {
            "A": rec.get("opa", ""), "B": rec.get("opb", ""),
            "C": rec.get("opc", ""), "D": rec.get("opd", ""),
        }
        input_text = f"Question: {rec['question']}\nOptions:\n{_format_options(opts)}"
        cop = rec.get("cop")
        answer = _COP_LETTERS[cop - 1] if isinstance(cop, int) and 1 <= cop <= 4 else None
        target = {"answer": answer, "explanation": rec.get("exp") or ""}
        yield example("mcqa_explain", "medmcqa", split, f"{path.name}#{i}", input_text, target)


# ---------------------------------------------------------------------------
# pubmedqa -- yes/no/maybe QA
# ---------------------------------------------------------------------------
PUBMEDQA_DIR = ROOT / "pubmedqa/raw"
PUBMEDQA_SPLITS = {"train": "ori_pqaa.json", "dev": "dev_set.json", "test": "test_set.json"}


def build_pubmedqa(split):
    path = PUBMEDQA_DIR / PUBMEDQA_SPLITS[split]
    for pmid, rec in iter_json_dict(path):
        contexts = rec.get("CONTEXTS") or []
        ctx_str = "\n".join(contexts) if isinstance(contexts, list) else str(contexts)
        input_text = f"Question: {rec['QUESTION']}\nContext:\n{ctx_str}"
        decision = rec.get("final_decision") or (rec.get("LABELS") or [None])[0]
        target = {"decision": decision, "long_answer": rec.get("LONG_ANSWER") or ""}
        yield example("qa_yesnomaybe", "pubmedqa", split, f"{path.name}#{pmid}", input_text, target)


# ---------------------------------------------------------------------------
# covid (LitCovid) -- multi-label classification
# ---------------------------------------------------------------------------
LITCOVID_DIR = ROOT / "covid/raw"
LITCOVID_SPLITS = {
    "train": "BC7-LitCovid-Train.csv", "dev": "BC7-LitCovid-Dev.csv", "test": "BC7-LitCovid-Test-GS.csv",
}


def build_litcovid(split):
    path = LITCOVID_DIR / LITCOVID_SPLITS[split]
    for i, row in iter_csv(path):
        input_text = f"Title: {row.get('title', '')}\nAbstract: {row.get('abstract', '')}"
        labels = [l for l in (row.get("label") or "").split(";") if l]
        yield example("multilabel_cls", "covid (LitCovid)", split, f"{path.name}#{i}", input_text, {"labels": labels})


# ---------------------------------------------------------------------------
# discharge-me -- summarization (plain-text target, ROUGE-scored)
# ---------------------------------------------------------------------------
DISCHARGE_DIR = ROOT / "discharge-me-bionlp-acl24-shared-task-on-streamlining-discharge-documentation-1.3"
DISCHARGE_SPLIT_DIRS = {"train": "train", "dev": "valid", "test": "test_phase_2"}


def build_discharge_me(split):
    split_dir = DISCHARGE_DIR / DISCHARGE_SPLIT_DIRS[split]
    targets = read_csv_gz_dict(split_dir / "discharge_target.csv.gz", "note_id")
    for i, row in iter_csv_gz(split_dir / "discharge.csv.gz"):
        note_id = row["note_id"]
        tgt = targets.get(note_id)
        target_text = (
            f"DISCHARGE INSTRUCTIONS:\n{tgt['discharge_instructions']}\n\n"
            f"BRIEF HOSPITAL COURSE:\n{tgt['brief_hospital_course']}"
        ) if tgt else ""
        yield example("summarization", "discharge-me", split, f"discharge.csv.gz#{note_id}",
                      row.get("text", ""), target_text)


# ---------------------------------------------------------------------------
# umn_abbreviation -- clinical abbreviation sense disambiguation
# ---------------------------------------------------------------------------
UMN_ABBR_DIR = ROOT / "umn_abbreviation/raw"
UMN_ABBR_SPLITS = {"train": "train.txt", "dev": "valid.txt", "test": "test.txt"}


def build_umn_abbreviation(split):
    # Latin-1, not UTF-8 -- confirmed by decode failure under utf-8. The
    # released start/end columns are offsets into the original full
    # document, not this context snippet, so they're dropped rather than
    # used for span alignment (verified: values exceed the snippet length).
    path = UMN_ABBR_DIR / UMN_ABBR_SPLITS[split]
    with open(path, encoding="latin-1") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("|")
            if len(parts) != 7:
                continue
            abbr, expansion, _surface, _start, _end, section, text = parts
            header = f"Section: {section}\n" if section else ""
            input_text = f"Abbreviation: {abbr}\n{header}Context: {text}"
            yield example("abbreviation_disambiguation", "umn_abbreviation", split,
                          f"{path.name}#{i}", input_text, {"expansion": expansion})


# ---------------------------------------------------------------------------
# nli4ct -- clinical-trial-report NLI (same "nli" task/schema as MedNLI:
# Premise/Hypothesis input, {"label": "entailment"|"contradiction"} target,
# binary only -- confirmed no neutral class across train/dev/Gold_test)
# ---------------------------------------------------------------------------
NLI4CT_DIR = ROOT / "nli4ct" / "raw" / "Complete_dataset"
NLI4CT_SPLITS = {"train": "train.json", "dev": "dev.json", "test": "Gold_test.json"}
NLI4CT_CT_DIR = NLI4CT_DIR / "CT json"


def _nli4ct_ct_cache():
    cache = {}
    for path in NLI4CT_CT_DIR.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            cache[path.stem] = json.load(f)
    return cache


def _nli4ct_section_text(ct, section_id):
    lines = ct.get(section_id) or []
    return "\n".join(lines) if isinstance(lines, list) else str(lines)


def build_nli4ct(split):
    path = NLI4CT_DIR / NLI4CT_SPLITS[split]
    ct_by_id = _nli4ct_ct_cache()
    for rec_id, rec in iter_json_dict(path):
        primary_ct = ct_by_id.get(rec["Primary_id"], {})
        primary_text = _nli4ct_section_text(primary_ct, rec["Section_id"])
        if rec["Type"] == "Comparison":
            secondary_ct = ct_by_id.get(rec["Secondary_id"], {})
            secondary_text = _nli4ct_section_text(secondary_ct, rec["Section_id"])
            premise = (
                f"Trial 1 ({rec['Primary_id']}):\n{primary_text}\n\n"
                f"Trial 2 ({rec['Secondary_id']}):\n{secondary_text}"
            )
        else:
            premise = primary_text
        input_text = f"Premise: {premise}\nHypothesis: {rec['Statement']}"
        label = rec["Label"].lower()
        yield example("nli", "nli4ct", split, rec_id, input_text, {"label": label})

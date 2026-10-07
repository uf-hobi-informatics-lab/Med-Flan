"""
Survey every dataset under the repo root, tag each with its NLP task type,
and compute token-length statistics (tiktoken cl100k_base + Qwen3-4B if
available) per dataset. Writes dataset_report.json and dataset_report.md
next to this script.
"""
import gzip
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw_datasets"
OUT_DIR = Path(__file__).resolve().parent

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------
tk_enc = tiktoken.get_encoding("cl100k_base")

qw_tokenizer = None
qw_error = None
try:
    from transformers import AutoTokenizer
    import transformers
    transformers.logging.set_verbosity_error()
    qw_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
except Exception as e:  # noqa: BLE001
    qw_error = f"{type(e).__name__}: {e}"

print(f"tiktoken cl100k_base: ready", file=sys.stderr)
print(f"Qwen3-4B tokenizer: {'ready' if qw_tokenizer else 'UNAVAILABLE (' + str(qw_error) + ')'}", file=sys.stderr)


def count_tk(text: str) -> int:
    if not text:
        return 0
    return len(tk_enc.encode(text, disallowed_special=()))


def count_qw(text: str) -> int:
    if not text or qw_tokenizer is None:
        return 0
    return len(qw_tokenizer.encode(text, add_special_tokens=False))


# --------------------------------------------------------------------------
# Accumulator
# --------------------------------------------------------------------------
class Accumulator:
    def __init__(self):
        self.tk_lens = []
        self.qw_lens = []
        self.max_tk = (-1, None)
        self.max_qw = (-1, None)

    def add(self, text: str, locator: str):
        n_tk = count_tk(text)
        self.tk_lens.append(n_tk)
        if n_tk > self.max_tk[0]:
            self.max_tk = (n_tk, locator)
        if qw_tokenizer is not None:
            n_qw = count_qw(text)
            self.qw_lens.append(n_qw)
            if n_qw > self.max_qw[0]:
                self.max_qw = (n_qw, locator)

    def summary(self):
        def stats(lens):
            if not lens:
                return None
            arr = np.array(lens)
            return {
                "n": int(arr.size),
                "min": int(arr.min()),
                "mean": round(float(arr.mean()), 1),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "max": int(arr.max()),
            }

        return {
            "tiktoken_cl100k": stats(self.tk_lens),
            "qwen3_4b": stats(self.qw_lens) if qw_tokenizer is not None else None,
            "longest_locator_tiktoken": self.max_tk[1],
            "longest_locator_qwen3_4b": self.max_qw[1] if qw_tokenizer is not None else None,
        }


# --------------------------------------------------------------------------
# Text extraction per dataset shape
# --------------------------------------------------------------------------
def nli_text(r):
    return f"{r.get('sentence1', '') or ''} {r.get('sentence2', '') or ''}"


def mcqa_text(r):
    opts = r.get("options")
    opts_str = " ".join(f"{k}: {v}" for k, v in opts.items()) if isinstance(opts, dict) else ""
    return f"{r.get('question', '') or ''} {opts_str}"


def mmlu_text(r):
    opts = r.get("options")
    opts_str = " ".join(str(o) for o in opts) if isinstance(opts, list) else ""
    return f"{r.get('question', '') or ''} {opts_str} {r.get('cot_content', '') or ''}"


def medmcqa_text(r):
    opts = " ".join(str(r.get(k, "") or "") for k in ("opa", "opb", "opc", "opd"))
    return f"{r.get('question', '') or ''} {opts} {r.get('exp', '') or ''}"


def pubmedqa_text(r):
    ctx = r.get("CONTEXTS")
    ctx_str = " ".join(ctx) if isinstance(ctx, list) else str(ctx or "")
    return f"{r.get('QUESTION', '') or ''} {ctx_str} {r.get('LONG_ANSWER', '') or ''}"


def title_abstract_text(r):
    return f"{r.get('title', '') or ''} {r.get('abstract', '') or ''}"


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------
def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_json_dict_or_list(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k, v in data.items():
            yield k, v
    else:
        for i, v in enumerate(data):
            yield i, v


def iter_csv(path, usecols, chunksize=5000):
    idx = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype=str, keep_default_na=False,
                              engine="python", on_bad_lines="skip"):
        for _, row in chunk.iterrows():
            yield idx, row.to_dict()
            idx += 1


def run_jsonl_entry(paths, text_fn):
    acc = Accumulator()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for i, rec in iter_jsonl(p):
            acc.add(text_fn(rec), f"{p.relative_to(ROOT)}#{i}")
    return acc


def run_json_dict_entry(paths, text_fn):
    acc = Accumulator()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for k, rec in iter_json_dict_or_list(p):
            if not isinstance(rec, dict):
                continue
            acc.add(text_fn(rec), f"{p.relative_to(ROOT)}#{k}")
    return acc


def run_csv_entry(paths, text_fn, usecols):
    acc = Accumulator()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for i, row in iter_csv(p, usecols):
            acc.add(text_fn(row), f"{p.relative_to(ROOT)}#{i}")
    return acc


def run_whole_files_entry(paths):
    acc = Accumulator()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        acc.add(text, str(p.relative_to(ROOT)))
    return acc


TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.S)


def run_n2c2_2018_track1_entry(dirs):
    acc = Accumulator()
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.glob("*.xml")):
            content = p.read_text(encoding="utf-8", errors="replace")
            m = TEXT_RE.search(content)
            text = m.group(1) if m else content
            text = text.replace("<![CDATA[", "").replace("]]>", "")
            acc.add(text, str(p.relative_to(ROOT)))
    return acc


def run_sts_entry(paths):
    acc = Accumulator()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 2:
                    continue
                acc.add(f"{cols[0]} {cols[1]}", f"{p.relative_to(ROOT)}#{i}")
    return acc


def run_cord_fulltext_entry(dirs):
    acc = Accumulator()
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                data = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            body = data.get("body_text", [])
            text = " ".join(seg.get("text", "") for seg in body if isinstance(seg, dict))
            acc.add(text, str(p.relative_to(ROOT)))
    return acc


def run_arrow_entry(path, text_fn):
    from datasets import load_from_disk
    ds = load_from_disk(str(path))
    acc = Accumulator()
    for i, rec in enumerate(ds):
        acc.add(text_fn(rec), f"{Path(path).relative_to(ROOT)}#{i}")
    return acc


def run_csv_gz_field_entry(paths, field_names, joiner=" "):
    acc = Accumulator()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for i, row in iter_csv(p, field_names):
            text = joiner.join(str(row.get(fn, "") or "") for fn in field_names)
            acc.add(text, f"{p.relative_to(ROOT)}#{i}")
    return acc


# --------------------------------------------------------------------------
# Dataset registry: (dataset, subset, task_type, format, run_fn)
# --------------------------------------------------------------------------
MEDQA_US = RAW / "MedQA-USMLE/questions/US"
MEDQA_ML = RAW / "MedQA-USMLE/questions/Mainland"
MEDQA_TW = RAW / "MedQA-USMLE/questions/Taiwan"
DISCHARGE = RAW / "discharge-me-bionlp-acl24-shared-task-on-streamlining-discharge-documentation-1.3"
N2C2_2018_T2 = RAW / "2018_n2c2_track2"
N2C2_2019_T2 = RAW / "2019_n2c2_track2"
N2C2_2022_T1 = RAW / "2022n2c2_track1/trainingdata_v3"
N2C2_2022_T2 = RAW / "2022n2c2_track2/Track2_SubtaskA/Annotations"

ENTRIES = []


def entry(dataset, subset, task_type, fmt, fn, *args, **kwargs):
    ENTRIES.append({"dataset": dataset, "subset": subset, "task_type": task_type, "format": fmt,
                     "fn": fn, "args": args, "kwargs": kwargs})


entry("MedNLI", "sentence pairs", "NLI", "jsonl", run_jsonl_entry,
      [RAW / "MedNLI/mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0" / f
       for f in ("mli_train_v1.jsonl", "mli_dev_v1.jsonl", "mli_test_v1.jsonl")],
      nli_text)

entry("MedQA-USMLE-US", "question+options", "MCQA", "jsonl", run_jsonl_entry,
      [MEDQA_US / f for f in ("train.jsonl", "dev.jsonl", "test.jsonl", "US_qbank.jsonl")],
      mcqa_text)

entry("MedQA-USMLE-Mainland", "question+options (zh)", "MCQA", "jsonl", run_jsonl_entry,
      [MEDQA_ML / f for f in ("train.jsonl", "dev.jsonl", "test.jsonl", "chinese_qbank.jsonl")],
      mcqa_text)

entry("MedQA-USMLE-Taiwan", "question+options (zh-TW)", "MCQA", "jsonl", run_jsonl_entry,
      [MEDQA_TW / f for f in ("train.jsonl", "dev.jsonl", "test.jsonl", "taiwanese_qbank.jsonl")],
      mcqa_text)

entry("MedQA-USMLE-textbooks-en", "whole textbook", "raw corpus (long-form)", "txt", run_whole_files_entry,
      sorted((RAW / "MedQA-USMLE/textbooks/en").glob("*.txt")))

entry("MedQA-USMLE-textbooks-zh", "whole textbook (zh)", "raw corpus (long-form)", "txt", run_whole_files_entry,
      [RAW / "MedQA-USMLE/textbooks/zh_paragraph/all_books.txt",
       RAW / "MedQA-USMLE/textbooks/zh_sentence/all_books.txt"])

entry("MMLU-Pro-Health", "question+options+cot", "MCQA", "arrow", run_arrow_entry,
      RAW / "MMLU-Pro-Health", mmlu_text)

entry("medmcqa", "question+options+explanation", "MCQA + explanation", "jsonl", run_jsonl_entry,
      [RAW / "medmcqa/raw" / f for f in ("train.json", "dev.json", "test.json")],
      medmcqa_text)

entry("pubmedqa", "question+contexts+long_answer", "QA / yes-no-maybe classification", "json-dict",
      run_json_dict_entry,
      [RAW / "pubmedqa/raw" / f for f in ("ori_pqaa.json", "dev_set.json", "test_set.json")],
      pubmedqa_text)

entry("covid (LitCovid)", "title+abstract", "multi-label text classification", "csv", run_csv_entry,
      [RAW / "covid/raw" / f for f in ("BC7-LitCovid-Train.csv", "BC7-LitCovid-Dev.csv", "BC7-LitCovid-Test-GS.csv")],
      title_abstract_text, ["title", "abstract"])

entry("CORD-19-metadata", "title+abstract", "raw corpus / IR", "csv", run_csv_entry,
      [RAW / "2019_COVID_NLP/all_sources_metadata_2020-03-13.csv"],
      title_abstract_text, ["title", "abstract"])

entry("CORD-19-fulltext", "full paper body text", "raw corpus / IR", "json", run_cord_fulltext_entry,
      [RAW / "2019_COVID_NLP/biorxiv_medrxiv/biorxiv_medrxiv",
       RAW / "2019_COVID_NLP/comm_use_subset/comm_use_subset",
       RAW / "2019_COVID_NLP/noncomm_use_subset/noncomm_use_subset",
       RAW / "2019_COVID_NLP/pmc_custom_license/pmc_custom_license"])

for split in ("train", "valid", "test_phase_1", "test_phase_2"):
    entry("discharge-me", f"{split}: discharge note text", "summarization", "csv.gz", run_csv_gz_field_entry,
          [DISCHARGE / split / "discharge.csv.gz"], ["text"])
    entry("discharge-me", f"{split}: radiology report text", "summarization (source)", "csv.gz",
          run_csv_gz_field_entry, [DISCHARGE / split / "radiology.csv.gz"], ["text"])
    entry("discharge-me", f"{split}: target (instructions+course)", "summarization (target)", "csv.gz",
          run_csv_gz_field_entry, [DISCHARGE / split / "discharge_target.csv.gz"],
          ["discharge_instructions", "brief_hospital_course"])

entry("2018_n2c2_track1", "note <TEXT>", "text classification (cohort eligibility)", "xml",
      run_n2c2_2018_track1_entry,
      [RAW / "2018_n2c2_track1/train", RAW / "2018_n2c2_track1/test_notags"])

entry("2018_n2c2_track2", "clinical note text", "NER + RE (ADE / medication)", "txt", run_whole_files_entry,
      sorted((N2C2_2018_T2 / "2018n2c2_track2_training").glob("*.txt")) +
      sorted((N2C2_2018_T2 / "gold_standard_test").glob("*.txt")))

entry("2019_n2c2_track1", "sentence1+sentence2", "semantic textual similarity (regression)", "tsv-in-txt",
      run_sts_entry,
      [RAW / "2019_n2c2_track1" / f for f in ("clinicalSTS2019.train.txt", "clinicalSTS2019.test.txt")])

entry("2019_n2c2_track2", "clinical note text", "NER + attribute/relation extraction (family history)", "txt",
      run_whole_files_entry,
      sorted((N2C2_2019_T2 / "bioc-FH-training").glob("*.txt")) +
      sorted((N2C2_2019_T2 / "bioc-FH-testing").glob("*.txt")))

entry("2022n2c2_track1", "clinical note text", "event extraction / classification (medication disposition)", "txt",
      run_whole_files_entry,
      sorted((N2C2_2022_T1 / "train").glob("*.txt")) + sorted((N2C2_2022_T1 / "dev").glob("*.txt")))

entry("2022n2c2_track2", "clinical note text", "NER + attribute extraction (social history)", "txt",
      run_whole_files_entry,
      sorted(N2C2_2022_T2.glob("train/**/*.txt")) + sorted(N2C2_2022_T2.glob("dev/**/*.txt")))

# medqa/raw/*.jsonl confirmed byte-identical to MedQA-USMLE/questions/US/*.jsonl -> not re-tokenized,
# flagged as duplicate in the report instead.
DUPLICATES = [
    {"dataset": "medqa", "subset": "train/dev/test.jsonl", "task_type": "MCQA",
     "note": "byte-identical duplicate of MedQA-USMLE-US (question/train,dev,test.jsonl); not re-tokenized"},
]


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main():
    results = []
    for e in ENTRIES:
        print(f"[analyze] {e['dataset']} :: {e['subset']}", file=sys.stderr)
        acc = e["fn"](*e["args"], **e["kwargs"])
        results.append({
            "dataset": e["dataset"],
            "subset": e["subset"],
            "task_type": e["task_type"],
            "format": e["format"],
            **acc.summary(),
        })

    # global longest across everything
    global_longest_tk = max(
        (r for r in results if r["tiktoken_cl100k"]),
        key=lambda r: r["tiktoken_cl100k"]["max"],
        default=None,
    )
    global_longest_qw = None
    if qw_tokenizer is not None:
        global_longest_qw = max(
            (r for r in results if r["qwen3_4b"]),
            key=lambda r: r["qwen3_4b"]["max"],
            default=None,
        )

    report = {
        "qwen3_4b_tokenizer_available": qw_tokenizer is not None,
        "qwen3_4b_error": qw_error,
        "duplicates_skipped": DUPLICATES,
        "datasets": results,
        "global_longest_tiktoken_cl100k": global_longest_tk,
        "global_longest_qwen3_4b": global_longest_qw,
    }

    (OUT_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2))
    write_markdown(report)
    print("Wrote dataset_stats/dataset_report.json and dataset_report.md", file=sys.stderr)


def write_markdown(report):
    lines = ["# Dataset report", ""]
    lines.append(f"Qwen3-4B tokenizer available: **{report['qwen3_4b_tokenizer_available']}**"
                 + (f" (error: {report['qwen3_4b_error']})" if report["qwen3_4b_error"] else ""))
    lines.append("")
    if report["duplicates_skipped"]:
        lines.append("## Duplicates skipped (not re-tokenized)")
        for d in report["duplicates_skipped"]:
            lines.append(f"- **{d['dataset']}** ({d['subset']}): {d['note']}")
        lines.append("")

    lines.append("## Per-dataset stats")
    lines.append("")
    header = ["dataset", "subset", "task_type", "n", "tk_mean", "tk_p95", "tk_max", "qw_mean", "qw_p95", "qw_max"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in report["datasets"]:
        tk = r["tiktoken_cl100k"] or {}
        qw = r["qwen3_4b"] or {}
        lines.append("| " + " | ".join(str(x) for x in [
            r["dataset"], r["subset"], r["task_type"],
            tk.get("n", "-"), tk.get("mean", "-"), tk.get("p95", "-"), tk.get("max", "-"),
            qw.get("mean", "-"), qw.get("p95", "-"), qw.get("max", "-"),
        ]) + " |")
    lines.append("")

    lines.append("## Longest document overall")
    lines.append("")
    glt = report["global_longest_tiktoken_cl100k"]
    if glt:
        lines.append(f"- **tiktoken cl100k_base**: {glt['tiktoken_cl100k']['max']} tokens — "
                      f"`{glt['longest_locator_tiktoken']}` (dataset: {glt['dataset']} / {glt['subset']})")
    glq = report["global_longest_qwen3_4b"]
    if glq:
        lines.append(f"- **Qwen3-4B**: {glq['qwen3_4b']['max']} tokens — "
                      f"`{glq['longest_locator_qwen3_4b']}` (dataset: {glq['dataset']} / {glq['subset']})")
    lines.append("")

    (OUT_DIR / "dataset_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

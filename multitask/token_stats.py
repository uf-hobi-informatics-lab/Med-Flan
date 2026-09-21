"""
Token-length + metadata stats for the reformatted multi-task dataset
(multitask/data/train_unified.jsonl and multitask/data/eval/*.jsonl).

Companion to dataset_stats/analyze_datasets.py, which measured the RAW
source datasets; this measures the REFORMATTED text-to-text examples
(task tag + instruction + INPUT + OUTPUT), per task tag and per dataset,
using the same two tokenizers (tiktoken cl100k_base, Qwen3-4B).

Run with: python -m multitask.token_stats
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import tiktoken

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = DATA_DIR

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

print(f"Qwen3-4B tokenizer: {'ready' if qw_tokenizer else 'UNAVAILABLE (' + str(qw_error) + ')'}", file=sys.stderr)


def count_tk(text: str) -> int:
    return len(tk_enc.encode(text, disallowed_special=())) if text else 0


def count_qw(text: str) -> int:
    return len(qw_tokenizer.encode(text, add_special_tokens=False)) if (text and qw_tokenizer) else 0


class Bucket:
    def __init__(self):
        self.tk, self.qw = [], []
        self.n = 0

    def add(self, text):
        self.n += 1
        self.tk.append(count_tk(text))
        if qw_tokenizer is not None:
            self.qw.append(count_qw(text))

    def summary(self):
        def stats(lens):
            if not lens:
                return None
            a = np.array(lens)
            return {"n": int(a.size), "min": int(a.min()), "mean": round(float(a.mean()), 1),
                    "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                    "p99": float(np.percentile(a, 99)), "max": int(a.max())}
        return {"n": self.n, "tiktoken_cl100k": stats(self.tk),
                "qwen3_4b": stats(self.qw) if qw_tokenizer is not None else None}


def iter_examples(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    report = {"qwen3_4b_available": qw_tokenizer is not None, "qwen3_4b_error": qw_error}

    # ---- merged train + dev sets: stats per task tag ----
    for merged_name, filename in (("train_unified", "train_unified.jsonl"), ("dev_unified", "dev_unified.jsonl")):
        path = DATA_DIR / filename
        by_task = {}
        total = Bucket()
        for ex in iter_examples(path):
            by_task.setdefault(ex["task"], Bucket()).add(ex["text"])
            total.add(ex["text"])
        report[merged_name] = {
            "path": str(path.relative_to(DATA_DIR.parent.parent)),
            "overall": total.summary(),
            "by_task": {t: b.summary() for t, b in sorted(by_task.items())},
        }

    # ---- per-dataset eval files ----
    eval_dir = DATA_DIR / "eval"
    by_dataset = {}
    for p in sorted(eval_dir.glob("*.jsonl")):
        b = Bucket()
        task_tag = None
        split = None
        for ex in iter_examples(p):
            b.add(ex["text"])
            task_tag = ex["task"]
            split = ex["split"]
        by_dataset[p.stem] = {"task": task_tag, "split": split, **b.summary()}
    report["eval_by_dataset"] = by_dataset

    (OUT_DIR / "token_stats.json").write_text(json.dumps(report, indent=2))
    write_markdown(report)
    print("Wrote multitask/data/token_stats.json and token_stats.md", file=sys.stderr)


def write_markdown(report):
    lines = ["# Multi-task reformatted data: token stats + metadata", ""]
    lines.append(f"Qwen3-4B tokenizer available: **{report['qwen3_4b_available']}**")
    lines.append("")

    for merged_name, title in (("train_unified", "train_unified.jsonl (merged train)"),
                                ("dev_unified", "dev_unified.jsonl (merged validation, monitoring only)")):
        lines.append(f"## {title} -- overall")
        ov = report[merged_name]["overall"]
        tk, qw = ov["tiktoken_cl100k"], ov["qwen3_4b"] or {}
        lines.append(f"- n = {ov['n']} examples")
        lines.append(f"- tiktoken: mean {tk['mean']}, p95 {tk['p95']}, max {tk['max']}")
        if qw:
            lines.append(f"- Qwen3-4B: mean {qw['mean']}, p95 {qw['p95']}, max {qw['max']}")
        lines.append("")

        lines.append(f"## {title} -- by task tag")
        lines.append("")
        header = ["task", "n", "tk_mean", "tk_p95", "tk_max", "qw_mean", "qw_p95", "qw_max"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for task, s in report[merged_name]["by_task"].items():
            tk, qw = s["tiktoken_cl100k"] or {}, s["qwen3_4b"] or {}
            lines.append("| " + " | ".join(str(x) for x in [
                task, s["n"], tk.get("mean", "-"), tk.get("p95", "-"), tk.get("max", "-"),
                qw.get("mean", "-"), qw.get("p95", "-"), qw.get("max", "-"),
            ]) + " |")
        lines.append("")

    lines.append("## Per-dataset eval split (unmerged, original test/dev)")
    lines.append("")
    header = ["dataset", "task", "split", "n", "tk_mean", "tk_p95", "tk_max", "qw_mean", "qw_p95", "qw_max"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for name, s in report["eval_by_dataset"].items():
        tk, qw = s["tiktoken_cl100k"] or {}, s["qwen3_4b"] or {}
        lines.append("| " + " | ".join(str(x) for x in [
            name, s["task"], s["split"], s["n"], tk.get("mean", "-"), tk.get("p95", "-"), tk.get("max", "-"),
            qw.get("mean", "-"), qw.get("p95", "-"), qw.get("max", "-"),
        ]) + " |")
    lines.append("")

    (OUT_DIR / "token_stats.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Report max/percentile chat-template token lengths for a dataset, across
the three loop_transformer tokenizer families (Qwen3, Ouro, HRM-Text).

Applies each model's own chat template to the full conversation (prompt +
response, add_generation_prompt=False -- i.e. the same sequence length the
SFT trainer actually sees) and reports length statistics, so max_seq_length
in the recipes can be sized correctly.

Usage (run on a host where the dataset + model paths are mounted):
  python scripts/analyze_token_lengths.py \\
      --dataset-path /project/med_flan/data/med_flan_v1_md \\
      --splits train dev \\
      --qwen-path /project/models/Qwen3-8B \\
      --ouro-path /project/models/Ouro-2.6B-Thinking \\
      --hrm-path /project/models/HRM-Text-1B

--dataset-path also accepts multiple per-task dataset dirs (each a
DatasetDict saved via save_to_disk, e.g. multitask/huggingface_format/<name>
in the meddata repo), in which case each split is concatenated across all of
them -- mirroring alignment.data.mix_datasets with frac=1.0 for every task:
  python scripts/analyze_token_lengths.py \\
      --dataset-path path/to/task_a path/to/task_b \\
      --splits train dev --skip ouro
"""

import argparse
import os
import sys

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alignment.hrm.data import format_hrm_sample  # noqa: E402
from alignment.hrm.tokenizer import load_hrm_tokenizer  # noqa: E402

THRESHOLDS = [4096, 8192, 16384, 32768, 65536]


def load_splits(dataset_paths, splits):
    if len(dataset_paths) == 1:
        ds = load_from_disk(dataset_paths[0])
        if isinstance(ds, DatasetDict):
            return {s: ds[s] for s in splits if s in ds}
        return {s: load_from_disk(os.path.join(dataset_paths[0], s)) for s in splits}

    per_split = {s: [] for s in splits}
    for path in dataset_paths:
        ds = load_from_disk(path)
        for s in splits:
            if s in ds:
                per_split[s].append(ds[s])
    return {s: concatenate_datasets(parts) for s, parts in per_split.items() if parts}


def report(name, lengths):
    lengths = np.asarray(lengths)
    print(f"\n=== {name} (n={len(lengths)}) ===")
    print(
        f"  min={lengths.min()}  mean={lengths.mean():.1f}  "
        f"p50={np.percentile(lengths, 50):.0f}  p95={np.percentile(lengths, 95):.0f}  "
        f"p99={np.percentile(lengths, 99):.0f}  max={lengths.max()}"
    )
    for t in THRESHOLDS:
        n_over = int((lengths > t).sum())
        if n_over:
            print(
                f"  {n_over} / {len(lengths)} examples (> {t} tokens, "
                f"{100 * n_over / len(lengths):.2f}%)"
            )
    return int(lengths.max())


def qwen_ouro_length(example, tokenizer, enable_thinking):
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    ids = tokenizer.apply_chat_template(
        example["messages"], tokenize=True, add_generation_prompt=False, **kwargs
    )
    # Some transformers versions return a BatchEncoding (dict-like) here instead
    # of a plain list of ids -- len() on that gives the number of dict keys, not
    # the sequence length, so unwrap "input_ids" first.
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return {"length": len(ids)}


def hrm_length(example, tokenizer, condition):
    out = format_hrm_sample(
        example,
        tokenizer=tokenizer,
        condition=condition,
        max_seq_length=10_000_000,  # sentinel: effectively disables truncation
        use_chat_template=True,
    )
    return {"length": len(out["input_ids"])}


def analyze_family(name, dataset_by_split, length_fn, num_proc):
    all_lengths = []
    overall_max = 0
    for split, ds in dataset_by_split.items():
        mapped = ds.map(length_fn, num_proc=num_proc, desc=f"{name}:{split}")
        lengths = mapped["length"]
        overall_max = max(overall_max, report(f"{name} / {split}", lengths))
        all_lengths.extend(lengths)
    if len(dataset_by_split) > 1:
        report(f"{name} / ALL SPLITS COMBINED", all_lengths)
    return overall_max


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-path",
        nargs="+",
        default=["/project/med_flan/data/med_flan_v1_md"],
    )
    p.add_argument("--splits", nargs="+", default=["train", "dev"])
    p.add_argument("--qwen-path", default="/project/models/Qwen3-8B")
    p.add_argument("--ouro-path", default="/project/models/Ouro-2.6B-Thinking")
    p.add_argument("--hrm-path", default="/project/models/HRM-Text-1B")
    p.add_argument("--condition", default="direct", help="HRM condition token")
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--skip", nargs="*", default=[], choices=["qwen", "ouro", "hrm"])
    args = p.parse_args()

    print(f"Loading dataset from: {args.dataset_path} (splits={args.splits})")
    dataset_by_split = load_splits(args.dataset_path, args.splits)
    for split, ds in dataset_by_split.items():
        print(f"  {split}: {len(ds)} examples, columns={ds.column_names}")

    overall_max = {}

    if "qwen" not in args.skip:
        tok = AutoTokenizer.from_pretrained(args.qwen_path)
        overall_max["qwen"] = analyze_family(
            "Qwen3",
            dataset_by_split,
            lambda ex: qwen_ouro_length(ex, tok, enable_thinking=False),
            args.num_proc,
        )

    if "ouro" not in args.skip:
        tok = AutoTokenizer.from_pretrained(args.ouro_path, trust_remote_code=True)
        for attr, val in {
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
            "unk_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
        }.items():
            setattr(tok, attr, val)
        overall_max["ouro"] = analyze_family(
            "Ouro",
            dataset_by_split,
            lambda ex: qwen_ouro_length(ex, tok, enable_thinking=None),
            args.num_proc,
        )

    if "hrm" not in args.skip:
        tok = load_hrm_tokenizer(args.hrm_path, trust_remote_code=True)
        overall_max["hrm"] = analyze_family(
            "HRM-Text",
            dataset_by_split,
            lambda ex: hrm_length(ex, tok, condition=args.condition),
            args.num_proc,
        )

    print("\n=== SUMMARY (max tokens per family, across all splits) ===")
    for name, m in overall_max.items():
        print(f"  {name}: {m}")
    if overall_max:
        print(f"  overall max across all families: {max(overall_max.values())}")


if __name__ == "__main__":
    main()

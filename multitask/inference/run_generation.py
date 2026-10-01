"""
Single-file vLLM generation sweep: loads one model once, runs it over every
dataset in DATASET_SPECS via the batched multitask.inference.vllm_structured_infer.run()
(see that module for the actual generation/retry/fallback logic -- this file
just packs together the model config, per-dataset generation settings, and
the sweep loop that used to live spread across run_custom_model_eval.ipynb's
cells.

Usage:
    python -m multitask.inference.run_generation \\
        --model /path/to/checkpoint --backend xgrammar \\
        --out-dir /path/to/outputs/completions/

    # restrict to a subset while testing:
    python -m multitask.inference.run_generation --datasets 2018_n2c2_track1 biosses --limit 20

Writes one <out-dir>/<model_tag>__<dataset>.json per dataset -- {id: completion},
exactly vllm_structured_infer.run()'s own out_path format (dict/list for
JSON-shaped tasks, plain string for PLAIN_TEXT_TASKS) -- the same shape
multitask.eval.run_eval already expects and every other script in this repo
already produces/consumes, so it drops straight into
`python -m multitask.eval.run_eval <dataset> <out-dir>/<model_tag>__<dataset>.json`
or multitask.inference.run_eval_xlsx (this file's companion) with no
reshaping. model_tag is derived from --model (see model_tag()'s docstring
below) -- e.g. ".../models/ouro/1.4b/checkpoint-25860" -> "ouro_checkpoint-25860"
-- so results from different checkpoints/models never collide or get
overwritten in the same --out-dir.
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_from_disk

from multitask.inference.vllm_structured_infer import _make_llm, _parse_sampling_kwargs, run as run_inference

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "multitask" / "huggingface_format"

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def model_tag(model_path):
    """Short filename-safe tag identifying model + checkpoint, e.g.
    ".../models/v1_md/ouro_1_4b_std_plw005_1/checkpoint-25860" -> "ouro_1_4b_std_plw005_1_checkpoint-25860",
    ".../models/qwen/1_7b_std_v1/checkpoint-16165" -> "1_7b_std_v1_checkpoint-16165",
    or a bare "ouro/checkpoint-25860" (no variant dir, e.g. a shorter test-env
    path) -> "ouro_checkpoint-25860". Model name is always the directory that
    directly contains "checkpoint-<N>" (the variant dir), regardless of how
    many "models/<main_folder>/..." segments precede it -- picking the
    segment right after "models/" instead would collide across every variant
    that shares a main_folder + checkpoint number (this repo's actual
    on-cluster layout is models/<main_folder>/<variant>/checkpoint-<N>, so
    that used to silently drop the variant name here). Falls back to a
    sanitized version of the whole path/Hub id (e.g.
    "Qwen/Qwen2.5-7B-Instruct" -> "Qwen_Qwen2.5-7B-Instruct") when there's no
    checkpoint-<N> segment at all (a stock, non-fine-tuned Hub model)."""
    parts = Path(str(model_path).rstrip("/")).parts
    ckpt = next((p for p in parts if _CKPT_RE.match(p)), None)
    if ckpt is not None:
        idx = parts.index(ckpt)
        model_name = parts[idx - 1] if idx >= 1 else None
    else:
        model_name = None
    if model_name and ckpt:
        return f"{model_name}_{ckpt}"
    if ckpt:
        return ckpt
    return re.sub(r"[^\w.-]+", "_", str(model_path).strip("/"))

# (hf_format_dir_name, eval_kind, eval_name_or_None) -- eval_kind/eval_name
# are only consumed by run_eval_xlsx.py (generation itself doesn't need them),
# kept alongside the dataset list here so the two scripts share one
# definition instead of drifting apart.
MAIN_DATASET_SPECS = [
    ("2018_n2c2_track1", "standard", "2018_n2c2_track1"),
    ("2018_n2c2_track2", "standard", "2018_n2c2_track2"),
    ("2018_n2c2_track2_paragraph", "paragraph", None),
    ("2019_n2c2_track1_cls", "standard", "2019_n2c2_track1_cls"),
    ("2019_n2c2_track2", "standard", "2019_n2c2_track2"),
    ("2022n2c2_track1", "standard", "2022n2c2_track1"),
    ("2022n2c2_track2_mimic", "standard", "2022n2c2_track2_mimic"),
    ("2022n2c2_track2_uw", "standard", "2022n2c2_track2_uw"),
    ("MedNLI", "standard", "MedNLI"),
    ("MedQA-USMLE-US", "standard", "MedQA-USMLE-US"),
    ("umn_abbreviation", "standard", "umn_abbreviation"),
]

# Out-of-domain / output-distribution checks -- see run_custom_model_eval.ipynb's
# DATASET_SPECS cell for the per-dataset rationale (2010_i2b2_va, nli4ct,
# MMLU-Pro-Health, biosses).
OOD_DATASET_SPECS = [
    ("2010_i2b2_va", "standard", "2010_i2b2_va"),
    ("nli4ct", "standard", "nli4ct"),
    ("MMLU-Pro-Health", "standard", "MMLU-Pro-Health"),
    ("biosses", "standard", "biosses"),
]

DATASET_SPECS = MAIN_DATASET_SPECS + OOD_DATASET_SPECS

# Same values as run_custom_model_eval.ipynb's gen-settings cell -- see that
# cell's comment block for how these were measured (tiktoken cl100k over
# each dataset's real assistant-turn gold target).
#
# frequency_penalty deliberately REMOVED (was 0.3): confirmed via a real
# side-by-side run to corrupt ~21% of extracted span text with a spurious
# leading "-" character (e.g. "-Alcohol", "-quit", "- IVDU.") that isn't in
# the source document -- verified against this same dataset's own training
# gold, which has the same artifact in only 0.06% of spans (2/3471), so this
# was a decoding-time penalty effect, not something the model learned from
# noisy labels. write_anns's reanchor step then fails to match the
# corrupted text against the source note and silently drops that
# entity/role -- directly costing recall. Removing it took
# 2022n2c2_track2_mimic/uw's events.combined F1 from ~0.50 to ~0.78 (vs. the
# Microsoft-Nuance 2022 n2c2 Track 2 paper's T5 baseline of ~0.88 and
# RoBERTa baseline of ~0.79 on the closest-matching subtask) -- see this
# session's investigation. repetition_penalty=1.05 was tested removed at
# the same time as frequency_penalty and not independently re-confirmed, so
# it's left in place here; revisit if a future run isolates it as also
# contributing to the corruption.
DEFAULT_GEN = dict(
    max_tokens=12800,
    sampling_kwargs=dict(repetition_penalty=1.05, seed=1234),
)
GEN_OVERRIDES = {
    "2018_n2c2_track1": dict(max_tokens=128, sampling_kwargs=dict(repetition_penalty=1.05, seed=1234)),
    "2010_i2b2_va": dict(max_tokens=24000, sampling_kwargs=dict(repetition_penalty=1.05, seed=1234)),
    "2018_n2c2_track2_paragraph": dict(max_tokens=4096, sampling_kwargs=dict(repetition_penalty=1.05, seed=1234)),
    **{
        name: dict(max_tokens=128, sampling_kwargs=dict(repetition_penalty=1.05, seed=1234))
        for name in (
            "2019_n2c2_track1_cls", "MedNLI", "MedQA-USMLE-US", "umn_abbreviation",
            "nli4ct", "MMLU-Pro-Health", "biosses",
        )
    },
}


def gen_settings(name):
    return GEN_OVERRIDES.get(name, DEFAULT_GEN)


def run_sweep(
    model, backend="xgrammar", root=DEFAULT_ROOT, dataset_specs=None, limit=None,
    max_model_len=32768, gpu_memory_utilization=0.95, tensor_parallel_size=None,
    retries=2, retry_temperature=None, fallback_model=None, fallback_backend=None,
    fallback_retries=1, out_dir=None, batch_size=None, enable_thinking=False,
    sampling_kwarg_overrides=None, temperature=0.0, top_p=1.0, top_k=-1,
    enable_prefix_caching=False, disable_any_whitespace=True,
):
    dataset_specs = dataset_specs if dataset_specs is not None else DATASET_SPECS
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    llm_kwargs = {
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        # OFF by default -- see vllm_structured_infer._make_llm's docstring:
        # unverified against this vLLM install whether prefix-caching +
        # grammar-constrained decoding is actually correct here (a real,
        # documented bug class in some vLLM versions). Pass
        # --enable-prefix-caching once you've A/B-tested a small dataset
        # and confirmed byte-identical completions with it on vs. off.
        "enable_prefix_caching": enable_prefix_caching,
    }
    if tensor_parallel_size:
        llm_kwargs["tensor_parallel_size"] = tensor_parallel_size

    print(f"loading {model!r} (backend={backend!r}, llm_kwargs={llm_kwargs})")
    llm = _make_llm(model, backend, disable_any_whitespace=disable_any_whitespace, llm_kwargs=llm_kwargs)

    tag = model_tag(model)

    fallback_llm = None
    if fallback_model:
        print(f"loading fallback model {fallback_model!r} (backend={fallback_backend or backend!r})")
        fallback_llm = _make_llm(
            fallback_model, fallback_backend or backend,
            disable_any_whitespace=disable_any_whitespace, llm_kwargs=llm_kwargs,
        )

    completions_by_name = {}
    errors = {}

    for name, _eval_kind, _eval_name in dataset_specs:
        print("=" * 88)
        print(name)
        print("=" * 88)
        ds = load_from_disk((root / name).as_posix())["test"]
        gen = gen_settings(name)
        sampling_kwargs = {**gen["sampling_kwargs"], **(sampling_kwarg_overrides or {})}
        out_path = (out_dir / f"{tag}__{name}.json") if out_dir else None
        # Resume support: skip a dataset whose completion JSON already exists
        # and is valid, re-running only what didn't finish. The validity check
        # (json.loads, not just .exists()) guards against a non-atomic write
        # (vllm_structured_infer.run writes via Path.write_text -- no temp +
        # rename) leaving a 0-byte / truncated file after an OOM/preemption,
        # which a bare existence check would silently skip forever. Delete the
        # JSON (or let the next run find it invalid) to force a regenerate.
        if out_path and out_path.exists():
            try:
                json.loads(out_path.read_text(encoding="utf-8"))
                print(f"[skip] {out_path.name} already exists for {name}; skipping")
                continue
            except (OSError, ValueError):
                print(f"[skip] {out_path.name} exists but is not valid JSON; regenerating")
        try:
            completions = run_inference(
                ds, model, backend, llm=llm,
                max_tokens=gen["max_tokens"], sampling_kwargs=sampling_kwargs,
                temperature=temperature, top_p=top_p, top_k=top_k,
                limit=limit, retries=retries, retry_temperature=retry_temperature,
                fallback_llm=fallback_llm, fallback_retries=fallback_retries,
                out_path=out_path.as_posix() if out_path else None, batch_size=batch_size,
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            print(f"[generation FAILED] {name}: {e!r}")
            errors[name] = str(e)
            continue
        completions_by_name[name] = completions

    print(f"\n{len(completions_by_name)}/{len(dataset_specs)} datasets generated successfully.")
    if errors:
        print(f"{len(errors)} dataset(s) FAILED generation entirely: {list(errors)}")

    return completions_by_name, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend", default="xgrammar", choices=["xgrammar", "guidance", "outlines", "auto"],
        help="vLLM structured-output backend. Only xgrammar/guidance/auto support "
             "disable-any-whitespace (the default -- see --allow-any-whitespace to use "
             "outlines instead).",
    )
    parser.add_argument(
        "--allow-any-whitespace", action="store_true", default=False,
        help="Turn OFF disable_any_whitespace (see vllm_structured_infer._make_llm's docstring) "
             "-- required for --backend outlines, which does not support the flag and would "
             "otherwise raise. Only needed for a non-xgrammar/guidance backend.",
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="huggingface_format/ directory")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="restrict to these dataset names (default: every entry in DATASET_SPECS)",
    )
    parser.add_argument("--limit", type=int, default=None, help="per-dataset example cap, for testing")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=None,
        help="shard the model across N GPUs (production B300 supports up to 2; "
             "unset/1 for the single-GPU H100 test env)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="sampling temperature applied to every dataset in this run (default 0.0 = greedy/"
             "deterministic, matching this repo's reproducible-eval default -- vLLM's own default "
             "is 1.0). e.g. --temperature 0.3 --top-p 0.95 to sample instead of decoding greedily.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-temperature", type=float, default=None)
    parser.add_argument("--fallback-model", default=None)
    parser.add_argument("--fallback-backend", default=None)
    parser.add_argument("--fallback-retries", type=int, default=1)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="cap examples per llm.chat() call (default: everyone in one call per round). "
             "Set to 1 to reproduce the original fully-sequential behavior -- useful for "
             "A/B-testing suspiciously low scores against the batched default: at "
             "temperature=0 both must produce byte-identical completions for the same "
             "examples, since batching only changes throughput, not per-example sampling.",
    )
    parser.add_argument(
        "--enable-prefix-caching", action="store_true", default=False,
        help="OFF by default -- there's a real, documented history of vLLM bugs at the "
             "prefix-caching + grammar-constrained/structured-output decoding intersection, "
             "unverified against your installed vLLM version. Only turn this on after "
             "confirming a small dataset produces byte-identical completions with it on vs. "
             "off at temperature=0 (a real speed win once verified safe).",
    )
    parser.add_argument(
        "--sampling-kwarg", action="append", default=[], metavar="KEY=VALUE",
        help="override any OTHER vllm.SamplingParams field from GEN_OVERRIDES/DEFAULT_GEN's "
             "sampling_kwargs, applied to EVERY dataset in this run, repeatable, e.g. "
             "--sampling-kwarg frequency_penalty=0 --sampling-kwarg repetition_penalty=1.0 "
             "-- e.g. to test whether frequency_penalty is suppressing recall on the "
             "IE tasks without editing GEN_OVERRIDES in this file. Use --temperature/--top-p/"
             "--top-k for those (own flags below), not this.",
    )
    parser.add_argument(
        "--enable-thinking", choices=["auto", "true", "false"], default="false",
        help="Qwen3-style chat-template reasoning toggle. 'false' (default) disables thinking "
             "-- render_messages's own training targets never include a <think> block, so a "
             "fine-tune trained on them should be served the same way; leaving this at the "
             "tokenizer's own default ('auto', thinking ON for stock Qwen3 templates) can make "
             "the model continue past a reasoning scaffold it was never trained to produce, "
             "which shows up as high teacher-forced training accuracy but near-chance real "
             "generation quality. Silently ignored with a printed note on an older vLLM/template "
             "that doesn't define enable_thinking (e.g. Qwen2.5) -- see vllm_structured_infer.py's "
             "module docstring.",
    )
    parser.add_argument("--out-dir", required=True, help="directory to write <model_tag>__<dataset>.json completions into")
    args = parser.parse_args()

    dataset_specs = None
    if args.datasets:
        dataset_specs = [spec for spec in DATASET_SPECS if spec[0] in args.datasets]
        missing = set(args.datasets) - {s[0] for s in dataset_specs}
        if missing:
            raise SystemExit(f"unknown dataset name(s): {sorted(missing)} -- see DATASET_SPECS in this file")

    enable_thinking = {"auto": None, "true": True, "false": False}[args.enable_thinking]
    run_sweep(
        args.model, args.backend, root=Path(args.root), dataset_specs=dataset_specs, limit=args.limit,
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size, retries=args.retries,
        retry_temperature=args.retry_temperature, fallback_model=args.fallback_model,
        fallback_backend=args.fallback_backend, fallback_retries=args.fallback_retries,
        out_dir=args.out_dir, batch_size=args.batch_size, enable_thinking=enable_thinking,
        sampling_kwarg_overrides=_parse_sampling_kwargs(args.sampling_kwarg),
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_any_whitespace=not args.allow_any_whitespace,
    )


if __name__ == "__main__":
    main()

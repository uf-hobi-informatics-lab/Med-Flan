"""
Single-file transformers (non-vLLM) generation sweep -- same structure and
output contract as run_generation.py, generation swapped to
inference_and_eval.inference.hf_structured_infer.run() (see that module's
docstring for why: HRM needs trust_remote_code + PrefixLM token_type_ids
handling vLLM's generic serving path can't apply, and has been unreliable
under vLLM's prefix-caching + grammar-decoding combination).

Reuses run_generation.py's DATASET_SPECS/gen_settings/model_tag directly --
no separate dataset list or per-task max_tokens table to drift out of sync.

Usage:
    python -m inference_and_eval.inference.run_generation_hf \\
        --model /path/to/hrm/checkpoint --condition direct --out-dir /path/to/outputs/completions/

    # restrict to a subset while testing:
    python -m inference_and_eval.inference.run_generation_hf --model ... --condition direct --datasets 2018_n2c2_track1 --limit 20 --batch-size 1

Writes one <out-dir>/<model_tag>__<dataset>.json per dataset, identical
shape to run_generation.py's output -- drops straight into
inference_and_eval.inference.run_eval_csv with no reshaping.
"""
import argparse
import json
from pathlib import Path

from datasets import load_from_disk

from inference_and_eval.inference.hf_structured_infer import _load_model, run as run_inference
from inference_and_eval.inference.run_generation import DATASET_SPECS, DEFAULT_ROOT, gen_settings, model_tag


def run_sweep(
    model, condition, root=DEFAULT_ROOT, dataset_specs=None, limit=None, retries=2, retry_temperature=None,
    out_dir=None, batch_size=8, temperature=0.0, top_p=1.0, markdown=False,
    dtype="bfloat16", device="cuda", attn_implementation=None,
):
    dataset_specs = dataset_specs if dataset_specs is not None else DATASET_SPECS
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {model!r} (dtype={dtype!r}, device={device!r}, attn_implementation={attn_implementation!r})")
    tokenizer, hf_model = _load_model(model, dtype=dtype, device=device, attn_implementation=attn_implementation)

    tag = model_tag(model)

    completions_by_name = {}
    errors = {}

    for name, _eval_kind, _eval_name in dataset_specs:
        print("=" * 88)
        print(name)
        print("=" * 88)
        # see run_generation.run_sweep's own comment -- same markdown/plain
        # variant selection, output filenames still keyed off the bare name.
        ds_dir = f"{name}_md" if markdown else name
        ds = load_from_disk((root / ds_dir).as_posix())["test"]
        gen = gen_settings(name)
        out_path = (out_dir / f"{tag}__{name}.json") if out_dir else None
        # Resume support -- see run_generation.run_sweep for the full rationale:
        # skip a dataset whose completion JSON already exists and is valid,
        # re-running only what didn't finish. The validity check guards against
        # a non-atomic write (hf_structured_infer.run writes via Path.write_text)
        # leaving a truncated/0-byte file after an OOM/preemption, which a bare
        # existence check would silently skip forever.
        if out_path and out_path.exists():
            try:
                json.loads(out_path.read_text(encoding="utf-8"))
                print(f"[skip] {out_path.name} already exists for {name}; skipping")
                continue
            except (OSError, ValueError):
                print(f"[skip] {out_path.name} exists but is not valid JSON; regenerating")
        try:
            completions = run_inference(
                ds, model, condition, tokenizer=tokenizer, hf_model=hf_model,
                max_tokens=gen["max_tokens"], temperature=temperature, top_p=top_p,
                limit=limit, retries=retries, retry_temperature=retry_temperature,
                out_path=out_path.as_posix() if out_path else None, batch_size=batch_size,
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
        "--condition", required=True,
        help="HRM condition value this checkpoint was trained with (its recipe yaml's "
             "ModelArguments.condition -- e.g. 'direct'/'cot'/'noisy'/'synth', or a composite "
             "like 'synth,cot'), threaded into tokenizer.apply_chat_template()'s condition kwarg "
             "(HRM_CHAT_TEMPLATE's cond_map). Required, no default -- getting this wrong silently "
             "mismatches train/eval prompt formatting.",
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="huggingface_format/ directory")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="restrict to these dataset names (default: every entry in DATASET_SPECS)",
    )
    parser.add_argument("--limit", type=int, default=None, help="per-dataset example cap, for testing")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-temperature", type=float, default=None)
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="examples per model.generate() call (plain padded batch, not vLLM continuous "
             "batching -- set to 1 to isolate a suspicious example / A-B against a larger batch).",
    )
    parser.add_argument("--dtype", default="bfloat16", help="torch dtype name (bfloat16/float16/float32) or shorthand alias (bf16/fp16/fp32/half).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation", default=None,
        help="passed to AutoModelForCausalLM.from_pretrained, e.g. 'flash_attention_2'/'sdpa'/'eager' "
             "-- unset by default (transformers' own default), since HRM's custom trust_remote_code "
             "modeling code isn't guaranteed to support every implementation. Verify on a small "
             "--limit run before relying on it for a full sweep.",
    )
    parser.add_argument(
        "--markdown", action="store_true", default=False,
        help="load the markdown-rendered '<name>_md' huggingface_format/ variant of every "
             "dataset instead of the plain '<name>' one -- see run_generation.py's --markdown.",
    )
    parser.add_argument("--out-dir", required=True, help="directory to write <model_tag>__<dataset>.json completions into")
    args = parser.parse_args()

    dataset_specs = None
    if args.datasets:
        dataset_specs = [spec for spec in DATASET_SPECS if spec[0] in args.datasets]
        missing = set(args.datasets) - {s[0] for s in dataset_specs}
        if missing:
            raise SystemExit(f"unknown dataset name(s): {sorted(missing)} -- see DATASET_SPECS in run_generation.py")

    run_sweep(
        args.model, args.condition, root=Path(args.root), dataset_specs=dataset_specs, limit=args.limit,
        retries=args.retries, retry_temperature=args.retry_temperature, out_dir=args.out_dir,
        batch_size=args.batch_size, temperature=args.temperature, top_p=args.top_p,
        markdown=args.markdown, dtype=args.dtype, device=args.device,
        attn_implementation=args.attn_implementation,
    )


if __name__ == "__main__":
    main()

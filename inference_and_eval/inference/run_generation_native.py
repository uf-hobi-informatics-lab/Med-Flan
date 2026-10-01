"""
Single-file native-transformers generation sweep for Ouro checkpoints with
adaptive early-exit (the *_gated_* variant folders) -- same structure and
output contract as run_generation.py / run_generation_hf.py, generation
swapped to inference_and_eval.inference.native_infer.run().

Why this exists (and why BACKEND_ENGINE=vllm is wrong for a gated Ouro
checkpoint): vLLM's OuroForCausalLM (vllm/model_executor/models/ouro.py)
loads the early_exit_gate module (so checkpoint weights match) but NEVER
calls it in forward() -- it unconditionally runs the full
range(total_ut_steps) loop every time, ignoring the adaptive-early-exit
feature the checkpoint was trained to use. Ouro's own model card confirms
vLLM always executes the full total_ut_steps and does not support adaptive
exit. native_infer.run() instead runs the model's own trust_remote_code
generate() path, which DOES respect the early_exit_gate -- so a gated
checkpoint actually exercises its dynamic-round behavior here, instead of
silently degrading to the fixed-loop behavior vLLM imposes.

Reuses run_generation.py's DATASET_SPECS/gen_settings/model_tag and
native_infer.run()'s own model loading -- no separate dataset list or
per-task max_tokens table to drift out of sync.

Deliberately the SAME feature surface as native_infer.run() (see that
module's docstring) -- this is the native path promoted into the sweep, not
a second vLLM-equivalent pipeline:
  - NO grammar-constrained JSON decoding (transformers has no xgrammar
    wiring here; completions are free-form text -- expect a higher
    bad-JSON rate on IE tasks than the vLLM run, which is the honest
    tradeoff for seeing the raw gated behavior).
  - NO retry/fallback-model escalation, NO repetition-loop salvage, NO
    target_schema.validate() pass. The eval step scores whatever parses.
  - One example at a time (native_infer.run() loops examples sequentially,
    no batching) -- meaningfully slower per token than vLLM's continuous
    batching; fine for a gated-checkpoint eval sweep, not a throughput race.

The runai sweep auto-routes a *_gated_* variant folder to this path (see
infer_eval_multitask.runai.sh's _gated_ auto-detection); BACKEND_ENGINE=native
selects it explicitly too.

Usage:
    python -m inference_and_eval.inference.run_generation_native \\
        --model /path/to/ouro_1_4b_gated_std/checkpoint-25860 --out-dir /path/to/outputs/completions/

    # restrict to a subset while testing:
    python -m inference_and_eval.inference.run_generation_native --model ... --datasets 2018_n2c2_track1 --limit 20

Writes one <out-dir>/<model_tag>__<dataset>.json per dataset, identical
shape to run_generation.py's output -- drops straight into
inference_and_eval.inference.run_eval_csv with no reshaping.
"""
import argparse
import json
from pathlib import Path

from datasets import load_from_disk

from inference_and_eval.inference.native_infer import _make_native_model, run as run_inference
from inference_and_eval.inference.run_generation import DATASET_SPECS, DEFAULT_ROOT, gen_settings, model_tag


def run_sweep(
    model, root=DEFAULT_ROOT, dataset_specs=None, limit=None, out_dir=None,
    temperature=0.0, top_p=1.0, top_k=None, markdown=False, dtype="auto",
    device_map="auto", trust_remote_code=True, enable_thinking=False,
):
    dataset_specs = dataset_specs if dataset_specs is not None else DATASET_SPECS
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {model!r} (native trust_remote_code={trust_remote_code}, dtype={dtype!r}, device_map={device_map!r})")
    hf_model, tokenizer = _make_native_model(
        model, trust_remote_code=trust_remote_code, device_map=device_map, torch_dtype=dtype,
    )

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
        # a non-atomic write (native_infer.run writes via Path.write_text)
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
                ds, model, max_tokens=gen["max_tokens"],
                out_path=out_path.as_posix() if out_path else None,
                model_obj=hf_model, tokenizer=tokenizer,
                trust_remote_code=trust_remote_code, device_map=device_map, torch_dtype=dtype,
                temperature=temperature, top_p=top_p, top_k=top_k, limit=limit,
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
        "--trust-remote-code", action="store_true", default=True,
        help="required for Ouro (ships custom modeling code not merged into transformers). "
             "ON by default for this wrapper -- pass --no-trust-remote-code to disable "
             "(no current use case; Ouro needs it).",
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="huggingface_format/ directory")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="restrict to these dataset names (default: every entry in DATASET_SPECS)",
    )
    parser.add_argument("--limit", type=int, default=None, help="per-dataset example cap, for testing")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--dtype", default="auto",
        help="torch dtype name (bfloat16/float16/float32) or 'auto' (default -- let the "
             "checkpoint's config.json torch_dtype decide, same as native_infer).",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--markdown", action="store_true", default=False,
        help="load the markdown-rendered '<name>_md' huggingface_format/ variant of every "
             "dataset instead of the plain '<name>' one -- see run_generation.py's --markdown.",
    )
    parser.add_argument(
        "--enable-thinking", choices=["auto", "true", "false"], default="false",
        help="Qwen3-style chat-template reasoning toggle, see native_infer.run()'s docstring. "
             "'false' (default) disables thinking to match render_messages's thinking-free "
             "training targets.",
    )
    parser.add_argument("--out-dir", required=True, help="directory to write <model_tag>__<dataset>.json completions into")
    args = parser.parse_args()

    dataset_specs = None
    if args.datasets:
        dataset_specs = [spec for spec in DATASET_SPECS if spec[0] in args.datasets]
        missing = set(args.datasets) - {s[0] for s in dataset_specs}
        if missing:
            raise SystemExit(f"unknown dataset name(s): {sorted(missing)} -- see DATASET_SPECS in run_generation.py")

    enable_thinking = {"auto": None, "true": True, "false": False}[args.enable_thinking]
    run_sweep(
        args.model, root=Path(args.root), dataset_specs=dataset_specs, limit=args.limit,
        out_dir=args.out_dir, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        markdown=args.markdown, dtype=args.dtype, device_map=args.device_map,
        trust_remote_code=args.trust_remote_code, enable_thinking=enable_thinking,
    )


if __name__ == "__main__":
    main()

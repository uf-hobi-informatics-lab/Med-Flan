"""
Plain `transformers` (`AutoModelForCausalLM.generate()`) inference -- an
alternative to vllm_structured_infer.run(), specifically for comparing the
two on the same model/dataset.

Why this exists: ByteDance/Ouro's paper-described inference-time behavior
(adaptive early-exit -- stop the recurrent loop once accumulated exit
probability crosses `early_exit_threshold`) is NOT implemented in vLLM's
`OuroForCausalLM` (vllm/model_executor/models/ouro.py) -- its
`early_exit_gate` module is loaded (so checkpoint weights match) but never
called in `forward()`, which unconditionally runs the full `range(total_ut_steps)`
loop every time. Confirmed both by reading that source directly and by
Ouro's own model card, which states vLLM always executes the full
`total_ut_steps` and does not support the adaptive-exit feature. This script
runs the model's own native code path (via `trust_remote_code=True`)
instead, so its behavior -- and, on models where it matters, whether early
exit actually triggers -- can be compared against the vLLM run directly.

Same prompts (the example's own {system, user} turns, gold assistant turn
dropped -- see vllm_structured_infer._prompt_messages, imported and reused
here so there is exactly one prompt-construction path) and the same
{id: completion} JSON output shape as vllm_structured_infer.run(), so
either file scores with the unchanged multitask.eval.run_eval, and the two
can be diffed directly, completion-by-completion, id-by-id.

Deliberately NOT a full reimplementation of vllm_structured_infer.py's
feature set -- this is a comparison tool, not a second production pipeline:
  - No grammar-constrained JSON decoding. `transformers` has no equivalent
    wired up here to vLLM's xgrammar/outlines StructuredOutputsParams.
    Completions are free-form text; expect a higher bad-JSON rate than the
    vLLM run on the JSON-shaped IE tasks -- that's a real, informative part
    of the comparison, not a bug in this script.
  - No retry/fallback-model escalation, no repetition detection, no
    target_schema.validate() pass. Add those back in only if this script
    stops being a one-off comparison tool and starts being relied on for
    real numbers.
  - One example at a time, no batching. `generate()` here has none of
    vLLM's continuous batching/PagedAttention, so this is meaningfully
    slower per token than vLLM -- fine for a --limit-bounded comparison
    run, impractical for a full multi-thousand-example sweep.

Usage:
    python -m multitask.inference.native_infer 2018_n2c2_track1 \\
        --model ByteDance/Ouro-1.4B --trust-remote-code --limit 20 \\
        --out completions_native.json

    # score exactly like a vLLM run:
    python -m multitask.eval.run_eval 2018_n2c2_track1 completions_native.json
"""
import argparse
import json
import time
from pathlib import Path

from multitask.inference.vllm_structured_infer import (
    DEFAULT_MAX_TOKENS,
    _DEFAULT_MAX_TOKENS,
    _load_examples,
    _prompt_messages,
    _strip_lone_surrogates,
)

ROOT = Path(__file__).resolve().parents[2]


def _make_native_model(model, trust_remote_code=False, device_map="auto", torch_dtype="auto", model_kwargs=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Deliberately NOT passing fix_mistral_regex=True -- see
    # vllm_structured_infer._make_llm's docstring/comment: transformers'
    # "might be Mistral" detection is a false positive for any local
    # checkpoint whose config.json lacks "transformers_version" (plausible
    # for a custom training pipeline), regardless of actual model_type, and
    # forcing the flag on a false positive replaces the tokenizer's real
    # pre_tokenizer with Mistral's regex-split pattern -- wrong for a
    # non-Mistral model, not a safe no-op.
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=torch_dtype,
        **(model_kwargs or {}),
    )
    return hf_model, tokenizer


def run(
    dataset, model, limit=None, max_tokens=None, out_path=None,
    model_obj=None, tokenizer=None, trust_remote_code=False,
    device_map="auto", torch_dtype="auto", model_kwargs=None,
    temperature=0.0, top_p=1.0, top_k=None, sampling_kwargs=None,
    enable_thinking=False,
):
    """Same {id: completion} contract as vllm_structured_infer.run(), minus
    grammar-constrained decoding/retry (see module docstring for why). Pass
    an already-loaded (model_obj, tokenizer) to reuse across calls, same
    convention as run()'s llm= param.

    enable_thinking: False (default) passes chat_template_kwargs={"enable_thinking":
    False} to apply_chat_template -- render_messages's own training targets
    (see multitask.schemas) never include a <think> block, so a fine-tune
    trained on them should be served the same way; leaving the tokenizer's
    own chat-template default in place (None -- thinking ON for stock Qwen3
    templates) can make the model continue past a reasoning scaffold it was
    never trained to produce, which looks exactly like high teacher-forced
    training accuracy paired with near-chance real generation quality.
    Silently ignored with a printed note if this tokenizer's chat template
    doesn't accept enable_thinking at all (e.g. Qwen2.5)."""
    import torch

    examples = _load_examples(dataset, limit)

    if model_obj is None or tokenizer is None:
        model_obj, tokenizer = _make_native_model(
            model, trust_remote_code=trust_remote_code, device_map=device_map,
            torch_dtype=torch_dtype, model_kwargs=model_kwargs,
        )

    chat_kwargs = {}
    if enable_thinking is not None:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    _supports_chat_template_kwargs = True
    if chat_kwargs:
        try:
            tokenizer.apply_chat_template([], tokenize=False, **chat_kwargs)
        except TypeError:
            _supports_chat_template_kwargs = False
            print(
                "this tokenizer's apply_chat_template() has no chat_template_kwargs/"
                "enable_thinking param -- ignoring and using the template's own default."
            )
        except Exception:
            pass  # an empty-conversation rendering error is fine -- we only
                  # needed to know whether the *signature* accepts the kwarg.

    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs = dict(do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
    gen_kwargs.update(sampling_kwargs or {})

    completions = {}
    truncated_ids = []
    surrogate_ids = []
    latencies = {}  # doc_id -> (wall_seconds, new_tokens) -- see docstring
    for ex in examples:
        doc_id, task = ex["id"], ex["task"]
        mt = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS.get(task, _DEFAULT_MAX_TOKENS)

        conv = _prompt_messages(ex)
        if chat_kwargs and _supports_chat_template_kwargs:
            prompt_text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, **chat_kwargs)
        else:
            prompt_text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model_obj.device)
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model_obj.generate(**inputs, max_new_tokens=mt, **gen_kwargs)
        elapsed = time.perf_counter() - t0
        new_ids = out_ids[0][prompt_len:]
        latencies[doc_id] = (elapsed, new_ids.shape[0])
        text = tokenizer.decode(new_ids, skip_special_tokens=True)

        if new_ids.shape[0] >= mt:
            truncated_ids.append(doc_id)

        clean, had_surr = _strip_lone_surrogates(text)
        if had_surr:
            surrogate_ids.append(doc_id)
        completions[doc_id] = clean

    print(f"ran {len(completions)}/{len(examples)} examples")
    if latencies:
        per_token = sorted(sec / max(n, 1) for sec, n in latencies.values())
        mid = per_token[len(per_token) // 2]
        print(
            f"per-example seconds/token: min={per_token[0]:.4f} median={mid:.4f} "
            f"max={per_token[-1]:.4f} -- if early exit is really triggering per-example, "
            "expect real spread here (fewer UT steps -> lower seconds/token on some "
            "examples); a flat distribution suggests every example is still running the "
            "full loop depth, same as vLLM's unconditional behavior."
        )
    if truncated_ids:
        print(
            f"{len(truncated_ids)}/{len(examples)} completions hit max_new_tokens "
            f"(truncated, likely unclosed JSON). First few ids: {truncated_ids[:5]}"
        )
    if surrogate_ids:
        print(f"{len(surrogate_ids)}/{len(examples)} completions had lone surrogates stripped: {surrogate_ids[:5]}")

    if out_path:
        # Same ensure_ascii=True belt-and-suspenders as vllm_structured_infer.run().
        Path(out_path).write_text(json.dumps(completions, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"wrote {len(completions)} completions to {out_path}")
    return completions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset name matching multitask/data/eval/<dataset>.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--trust-remote-code", action="store_true", default=False,
        help="required for models shipping custom modeling code not merged into transformers, "
             "e.g. ByteDance/Ouro-*",
    )
    parser.add_argument("--limit", type=int, default=None, help="only run the first N examples")
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="override the per-task default in vllm_structured_infer.DEFAULT_MAX_TOKENS uniformly",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="0.0 (default) = greedy (do_sample=False), matching vllm_structured_infer.py's "
             "reproducible-eval default",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--enable-thinking", choices=["auto", "true", "false"], default="false",
        help="Qwen3-style chat-template reasoning toggle, see run()'s docstring. 'false' "
             "(default) disables thinking to match render_messages's thinking-free training "
             "targets.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    enable_thinking = {"auto": None, "true": True, "false": False}[args.enable_thinking]
    run(
        args.dataset, args.model, limit=args.limit, max_tokens=args.max_tokens,
        out_path=args.out, trust_remote_code=args.trust_remote_code,
        device_map=args.device_map, torch_dtype=args.torch_dtype,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        enable_thinking=enable_thinking,
    )


if __name__ == "__main__":
    main()

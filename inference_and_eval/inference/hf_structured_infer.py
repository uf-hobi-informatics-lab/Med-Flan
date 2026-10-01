"""
Plain-transformers generation, as an alternative to vllm_structured_infer.run()
for models vLLM can't reliably serve -- built for HRM
(sapientinc/HRM-Text-1B-family): a trust_remote_code custom architecture
that needs a PrefixLM-style token_type_ids marking of the prompt
(bidirectional attention over the prompt, causal only over generated
tokens) that vLLM's generic serving path has no mechanism to apply, and
that has been unreliable under vLLM's prefix-caching + grammar-constrained
decoding combination (see run_generation.py's --enable-prefix-caching
docstring).

Mirrors vllm_structured_infer.run()'s contract and output shape as closely
as possible so the two backends stay comparable: same {id: completion}
output, same "obviously-problematic completion" classification/retry
policy (reused directly from that module, not reimplemented), same
DEFAULT_MAX_TOKENS per-task budgets. The real difference is generation
itself (model.generate() instead of vLLM's continuous batching) and the
complete absence of a grammar/structured-output constraint -- there is no
xgrammar-equivalent outside vLLM, so JSON-shaped completions rely entirely
on the fine-tune's own learned output format, backstopped by the same
parse/validate/retry loop as the vLLM path.

Prompt construction reuses vllm_structured_infer._prompt_messages (the
example's own {system, user} turns) rendered via
tokenizer.apply_chat_template() -- the same "messages"/"messages_md" field
build_hf_datasets.py builds "so tokenizer.apply_chat_template() can be
applied directly" (see that module's docstring). Confirmed (not guessed):
this repo's own training pipeline installs a custom HRM_CHAT_TEMPLATE
(alignment.hrm.tokenizer, a hand-written Jinja2 string reproducing
HRM-Text's native <|im_start|><condition_token(s)><instruction><|im_end|>
<response><|box_end|> format) onto the tokenizer whenever it has no
chat_template of its own, and installs its own bos/eos/pad token overrides
the same way (see _load_tokenizer below, which prefers importing and
calling load_hrm_tokenizer() itself over reimplementing it). Its own
format_hrm_sample() calls
tokenizer.apply_chat_template(messages, condition=cond, add_generation_prompt=True,
tokenize=False) -- i.e. sapientinc/HRM-Text-1B's model-card
<|quad_end|><|object_ref_end|> "condition" prefix is NOT a manually-spliced
raw string; it's threaded through apply_chat_template's own `condition`
kwarg, same as this module's _prompt_text() does. `condition` MUST match
the value the checkpoint being evaluated was actually trained with -- see
run()'s docstring.

token_type_ids: the model card's "mark the prompt as a single bidirectional
prefix block" instruction is applied unconditionally to every prompt
(architecture-level requirement, independent of the condition-token
mechanism above) -- see _build_batch's docstring.

EOS: HRM_CHAT_TEMPLATE renders an assistant turn as
"{content}<|box_end|>" -- i.e. HRM's real end-of-turn token is
<|box_end|> (EOA_TOKEN), not a generic end-of-text token, and
load_hrm_tokenizer() overrides tokenizer.eos_token to match. _generate_batch
below stops each sequence at tokenizer.eos_token_id and passes it explicitly
into generate() (not left to model.generation_config, which may not agree).

Usage:
    python -m inference_and_eval.inference.hf_structured_infer 2018_n2c2_track1 \\
        --model /path/to/hrm/checkpoint --limit 20 --out completions.json
"""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from inference_and_eval.inference.vllm_structured_infer import (
    _BAD_STATUSES,
    _DEFAULT_MAX_TOKENS,
    _better,
    _load_examples,
    _looks_repetitive,
    _prompt_messages,
    _REPEAT_RETRY_PENALTY,
    _REPEAT_RETRY_TEMPS,
    _STATUS_RANK,
    _salvage_loop,
    _strip_lone_surrogates,
    _tail_period,
)
from inference_and_eval.schemas import PLAIN_TEXT_TASKS
from inference_and_eval.target_schema import json_schema, validate

ROOT = Path(__file__).resolve().parents[2]


def _load_tokenizer(model):
    """Prefers this project's own training-side
    alignment.hrm.tokenizer.load_hrm_tokenizer() -- confirmed (not guessed)
    via the actual tokenizer.py source: it (1) overrides bos/eos/pad to
    HRM's real special tokens (bos=BOQ_TOKEN "<|im_start|>", eos=EOA_TOKEN
    "<|box_end|>", pad=PAD_TOKEN "<|endoftext|>") UNLESS the tokenizer is
    already "pre-built" HRM (_is_hrm_tokenizer -- already has those set,
    e.g. reloading a checkpoint that was itself saved via a prior
    load_hrm_tokenizer() call, such as merge_hrm_fsdp_checkpoint.py's
    tokenizer.save_pretrained()); (2) installs HRM_CHAT_TEMPLATE only when
    tokenizer.chat_template is None, never clobbering one already saved
    with the checkpoint; (3) validates every required special token
    actually resolves in the vocab (_validate_hrm_tokenizer), raising
    ValueError on a genuinely broken tokenizer instead of silently
    misgenerating. Getting eos_token/eos_token_id right matters beyond the
    chat template itself: _generate_batch below stops each sequence at
    tokenizer.eos_token_id, which per HRM_CHAT_TEMPLATE is <|box_end|>, not
    a generic end-of-text token.

    Falls back to plain AutoTokenizer.from_pretrained() if `alignment` isn't
    importable here (it's only installed inside the hrm_train:base image --
    see infer_eval_multitask.runai.sh's FAMILY-based IMAGE selection), since
    that's the only piece of training-exact logic this module has no safe
    way to reimplement locally (the actual BOQ_TOKEN/EOA_TOKEN/PAD_TOKEN/
    cond_map string values live in alignment.hrm.data, not pasted here).
    Relies entirely on the checkpoint's own saved tokenizer_config.json/
    chat_template.jinja already having these correct from its own training
    run -- true for anything trained via load_hrm_tokenizer() and persisted
    via merge_hrm_fsdp_checkpoint.py. Prints a loud warning in that
    fallback case rather than silently guessing token strings."""
    from transformers import AutoTokenizer

    try:
        from alignment.hrm.tokenizer import load_hrm_tokenizer
    except ImportError:
        print(
            "[WARN] alignment.hrm.tokenizer isn't importable here -- falling back to plain "
            "AutoTokenizer.from_pretrained(), relying entirely on this checkpoint's own saved "
            "tokenizer_config.json/chat_template.jinja already having HRM's bos/eos/pad "
            "overrides and chat_template baked in from its own training run. Run this inside "
            "the hrm_train:base image for the training-exact path instead."
        )
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return load_hrm_tokenizer(model, trust_remote_code=True)


_DTYPE_ALIASES = {"bf16": "bfloat16", "fp16": "float16", "half": "float16", "fp32": "float32"}


def _resolve_dtype(dtype):
    """Accepts either torch's own dtype attribute names (bfloat16, float16,
    float32) or common CLI shorthand (bf16, fp16, fp32, half) -- same
    underlying torch.dtype either way."""
    return getattr(torch, _DTYPE_ALIASES.get(dtype, dtype))


def _load_model(model, dtype="bfloat16", device="cuda", attn_implementation=None):
    """AutoModelForCausalLM with trust_remote_code=True (HRM's custom
    architecture isn't a stock transformers model class) -- left-padded
    tokenizer so a batch's real tokens all end at the same position
    (required for generate() to extend every sequence in a batch from the
    same relative offset). See _load_tokenizer for why the tokenizer isn't
    just a plain AutoTokenizer.from_pretrained load.

    attn_implementation: passed straight through to from_pretrained (e.g.
    "flash_attention_2", "sdpa", "eager") only when explicitly set --
    otherwise left unset (transformers' own default) rather than guessed,
    since HRM's custom trust_remote_code modeling code isn't guaranteed to
    support every implementation transformers recognizes for stock
    architectures. Verify on a small --limit run (compare completions
    against the default) before relying on it for a full sweep."""
    from transformers import AutoModelForCausalLM

    tokenizer = _load_tokenizer(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from_pretrained_kwargs = dict(dtype=_resolve_dtype(dtype), trust_remote_code=True)
    if attn_implementation:
        from_pretrained_kwargs["attn_implementation"] = attn_implementation
    hf_model = AutoModelForCausalLM.from_pretrained(model, **from_pretrained_kwargs).to(device).eval()
    return tokenizer, hf_model


def _prompt_text(tokenizer, ex, condition):
    """See module docstring's Prompt construction section -- reuses the
    exact same {system, user} turns as the vLLM path, rendered via this
    checkpoint's own HRM chat template. `condition` is passed straight
    through as a template kwarg (HRM_CHAT_TEMPLATE reads it via cond_map --
    e.g. "direct"/"cot"/"noisy"/"synth", possibly a composite like
    "synth,cot" -- see alignment.hrm.data.format_hrm_sample's own
    tokenizer.apply_chat_template(messages, condition=cond, ...) call) --
    MUST match whatever condition value the checkpoint being evaluated was
    actually trained with (its recipe yaml's ModelArguments.condition), the
    same train/eval-consistency requirement as --markdown for the plain
    messages/messages_md split."""
    msgs = _prompt_messages(ex)
    return tokenizer.apply_chat_template(
        msgs, condition=condition, tokenize=False, add_generation_prompt=True,
    )


def _build_batch(tokenizer, texts, device):
    """Tokenize a batch of prompt texts (left-padded so they share a
    common right edge) and derive token_type_ids marking every REAL prompt
    token as a single bidirectional prefix block (1), matching HRM's model
    card ("mark the prompt as a single bidirectional prefix block -- see
    PrefixLM mask"). Left-padding positions get token_type_ids=0 -- doesn't
    matter functionally since attention_mask=0 already excludes them from
    attention, but 0 (not 1) avoids implying they're part of the prefix
    block. Assumes the model's own generate()/prepare_inputs_for_generation
    (trust_remote_code) correctly extends token_type_ids for newly
    generated tokens itself, the same way it must already do for
    attention_mask/position_ids under plain causal generation -- unverified
    locally; sanity-check output quality on the first real run."""
    enc = tokenizer(texts, padding=True, return_tensors="pt")
    token_type_ids = enc["attention_mask"].clone()
    enc["token_type_ids"] = token_type_ids
    return {k: v.to(device) for k, v in enc.items()}


def _classify(ex, text, hit_cap):
    """Same decision tree as vllm_structured_infer._classify, adapted from
    a RequestOutput to a plain decoded string + whether max_new_tokens was
    hit with no EOS (this backend's equivalent of finish_reason=="length")
    -- kept as a literal copy of that branch logic so the two backends
    classify completions identically. Returns (status, value).

    A tail-repetition loop is checked FIRST, before the hit_cap -> truncated
    short-circuit, for the same reason as the vLLM path: an output that
    loops until max_new_tokens is cut off mid-loop and would otherwise be
    classified "truncated" -- and on this HF backend truncated is NOT retried
    (see run()'s docstring: HF generate() owns the full KV cache in GPU
    memory upfront, so an escalated-max_tokens retry OOMs), so a loop would
    score 0 with no second chance. Detecting the loop on the raw text lets
    it be classified "repetitive" (which IS retried here) and retried
    hotter at the normal budget, salvaging the pre-loop prefix."""
    p = _tail_period(text)
    if p is not None:
        salvaged = _salvage_loop(text, p)
        reason = f"repetition loop (unit {p} chars, hit_cap={hit_cap})"
        return "repetitive", (salvaged if salvaged is not None else text, reason)
    if hit_cap:
        return "truncated", text
    clean, had_surr = _strip_lone_surrogates(text)
    if had_surr:
        return "surrogate", clean
    if ex["task"] in PLAIN_TEXT_TASKS:
        rep, reason = _looks_repetitive(clean)
        if rep:
            return "repetitive", (clean, reason)
        return "ok", clean
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return "bad_json", clean
    vs = validate(ex["task"], parsed)
    if vs:
        return "violations", (parsed, vs)
    rep, reason = _looks_repetitive(parsed)
    if rep:
        return "repetitive", (parsed, reason)
    return "ok", parsed


def run(
    dataset, model, condition, limit=None, max_tokens=None, out_path=None, tokenizer=None, hf_model=None,
    temperature=0.0, top_p=1.0, retries=2, retry_temperature=None, batch_size=8,
    dtype="bfloat16", device="cuda", attn_implementation=None,
):
    """See module docstring. `condition` (required, no default -- see
    _prompt_text's docstring): MUST match the condition value this specific
    checkpoint was trained with (its recipe yaml's ModelArguments.condition,
    e.g. "direct"/"cot"/"noisy"/"synth" or a composite like "synth,cot") --
    getting this wrong silently mismatches train/eval prompt formatting the
    same way a wrong --markdown choice would, so this is deliberately not
    guessed with a fallback default.

    tokenizer/hf_model: an already-loaded pair to
    reuse across multiple calls (e.g. sweeping every dataset in one
    session) instead of reloading the checkpoint each time -- pass the pair
    returned by a previous _load_model(model, ...) call. When omitted (the
    CLI path), behavior is unchanged: one dataset, one load.

    batch_size: examples per model.generate() call. Unlike vLLM's
    continuous batching, this is a plain padded batch -- every sequence in
    a call runs for the same number of steps (until the longest one in the
    batch is done), so a very mixed-length batch wastes some compute on
    already-finished shorter sequences, but it's still far faster than
    batch_size=1.

    retries/retry_temperature: same semantics as vllm_structured_infer.run()
    -- re-draw only the four _BAD_STATUSES ids (surrogate/bad_json/violations/
    repetitive), NOT truncated. Unlike vLLM's PagedAttention (which pages KV
    cache to host/secondary and can absorb a 32000-token generation's KV
    without owning all of it in GPU memory at once), HF generate() allocates
    the FULL KV cache for every sequence in GPU memory upfront -- so the
    vLLM path's "retry truncated with escalated max_tokens (1.5x per
    attempt up to _TRUNCATION_RETRY_CEILING=32000)" behavior OOMs here: a
    batch of 8 sequences at max_new_tokens=32000 on a 1B model needs ~400GB
    of KV cache, far past a 287GB GPU. So truncated is NOT retried on this
    path -- it's stored as-is (the same behavior as the old multitask/
    inference/ vLLM script before truncation-retry was added). A fresh
    sample (seed/temperature bump) for the other four defect classes. No
    fallback-model escalation in this version.
    """
    examples = _load_examples(dataset, limit)
    if tokenizer is None or hf_model is None:
        tokenizer, hf_model = _load_model(model, dtype=dtype, device=device, attn_implementation=attn_implementation)

    def _max_tokens_for(task):
        return max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS.get(task, _DEFAULT_MAX_TOKENS)

    def _safe_max_new_tokens(prompt_len, n_seqs, requested):
        """Cap max_new_tokens for one generate() call so the projected KV
        cache fits in free GPU memory -- the real fix for the OOM-hang this
        path was hitting. Unlike vLLM's PagedAttention (which pages KV
        cache and degrades gracefully under memory pressure), HF
        generate() allocates the FULL KV cache for every sequence upfront
        in GPU memory; when the allocation exceeds free memory the caching
        allocator logs a [W] OOM warning and retries internally, and (the
        hang) can block on the allocator instead of raising -- so the
        program freezes rather than crashing. This measures free memory
        BEFORE the call, estimates the KV bytes the call would need
        (n_seqs * (prompt_len + max_new_tokens) * kv_bytes_per_token,
        where kv_bytes_per_token is derived from the model's config:
        2 * n_layers * n_kv_heads * head_dim * 2 bytes for K+V in bf16),
        and caps max_new_tokens down to whatever fits in a safety fraction
        of free memory. A capped call's completions are then classified as
        truncated at their (shorter) effective cap, not silently mis-scored
        as 'ok'. Returns (effective_max_new_tokens, was_capped)."""
        try:
            free_bytes, _total = torch.cuda.mem_get_info()
        except Exception:
            return requested, False  # CPU path or no CUDA -- don't cap
        cfg = getattr(hf_model, "config", None)
        n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or 0
        n_kv_heads = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", None) or 0
        head_dim = getattr(cfg, "head_dim", None) or (
            getattr(cfg, "hidden_size", None) // (getattr(cfg, "num_attention_heads", None) or 1)
            if getattr(cfg, "hidden_size", None) and getattr(cfg, "num_attention_heads", None) else 0
        )
        if not (n_layers and n_kv_heads and head_dim):
            return requested, False  # can't estimate -- don't cap blindly
        # K and V tensors, bf16 = 2 bytes each element -> 4 bytes/token/layer/head
        kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
        # Safety fraction: leave headroom for activations, the model weights
        # (already loaded), and the caching allocator's retained blocks.
        # 0.8 -- conservative enough to absorb the ~20% activation overhead
        # generate() adds on top of KV cache for a 1B model.
        budget_bytes = int(free_bytes * 0.8)
        per_seq_tokens = prompt_len + requested
        needed = n_seqs * per_seq_tokens * kv_bytes_per_token
        if needed <= budget_bytes:
            return requested, False
        # Solve for the max_new_tokens that fits: keep the prompt (fixed),
        # shrink only the generation budget. n_seqs * (prompt_len + m) *
        # kv_bytes_per_token <= budget_bytes  ->  m <= budget/(n*kvpt) -
        # prompt_len. Floor at 64 so a degenerate estimate never requests
        # zero tokens (which generate() rejects).
        m = max(64, int(budget_bytes // (n_seqs * kv_bytes_per_token)) - prompt_len)
        if m >= requested:
            return requested, False
        return m, True

    def _generate_batch(batch):
        """batch: list of (ex, use_temperature, max_tokens_override,
        sampling_overrides) tuples. sampling_overrides is a dict (or None)
        of extra generate() kwargs applied per sub-chunk -- used to thread
        repetition_penalty into a repetitive-id retry (see the retry loop).
        Sorted by rendered prompt length before chunking (not dataset
        order) -- a mixed-length chunk left-pads every shorter prompt up to
        the longest one's length, wasting compute proportional to that gap;
        grouping similarly-sized prompts together minimizes wasted padding
        without changing per-example sampling (same texts, same seeds, just
        a different chunk grouping). Chunks into groups of at most
        batch_size, one model.generate() call per chunk, progress-barred
        since a full sweep's generate() calls can each take minutes.
        Returns {doc_id: (status, value)}."""
        if not batch:
            return {}
        texts_by_idx = [_prompt_text(tokenizer, ex, condition) for ex, _, _, _ in batch]
        order = sorted(range(len(batch)), key=lambda i: len(texts_by_idx[i]))
        chunks = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
        results = {}

        def _run_subchunk(sub_indices, eff_max_tokens, sub_capped):
            """One generate() call for a fixed sub-chunk of batch indices,
            classifies each completion, returns {doc_id: (status, value)}.
            eff_max_tokens is the already-memory-capped max_new_tokens for
            THIS call (see _safe_max_new_tokens); sub_capped flags whether
            it was shrunk below the requested budget, so completions that
            fill the (shorter) effective cap are classified as truncated
            rather than mis-scored as 'ok'."""
            sub_batch = [batch[i] for i in sub_indices]
            sub_texts = [texts_by_idx[i] for i in sub_indices]
            sub_inputs = _build_batch(tokenizer, sub_texts, device)
            sub_prompt_len = sub_inputs["input_ids"].shape[1]
            chunk_temp = next((t for _, t, _, _ in sub_batch if t is not None), temperature)
            do_sample = chunk_temp > 0
            gen_kwargs = dict(
                max_new_tokens=eff_max_tokens, do_sample=do_sample,
                # Explicit, not left to model.generation_config: HRM's real
                # eos_token_id (<|box_end|>, per HRM_CHAT_TEMPLATE/
                # load_hrm_tokenizer -- see _load_tokenizer) may not match
                # whatever eos_token_id happens to be baked into the
                # checkpoint's generation_config.json, and generate() stops
                # a sequence on ITS eos_token_id, not the tokenizer's.
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                # KV-cache reuse across generation steps -- on by default for
                # virtually every causal LM (including custom
                # trust_remote_code ones that implement standard
                # past_key_values caching), set explicitly here so a future
                # transformers/model default change can't silently disable
                # it out from under this loop.
                use_cache=True,
            )
            if do_sample:
                gen_kwargs.update(temperature=chunk_temp, top_p=top_p)
            # Per-example sampling overrides (e.g. repetition_penalty for a
            # repetitive-id retry). A sub-chunk shares one generate() call,
            # so use the first non-None override set in the sub-chunk -- a
            # repetitive retry batch is homogeneous (all repetitive ids), so
            # this is unambiguous in practice.
            chunk_sov = next((sov for _, _, _, sov in sub_batch if sov), None)
            if chunk_sov:
                gen_kwargs.update(chunk_sov)
            with torch.no_grad():
                out_ids = hf_model.generate(**sub_inputs, **gen_kwargs)
            new_ids = out_ids[:, sub_prompt_len:]
            sub_results = {}
            for (ex, _, mt_override, _), row in zip(sub_batch, new_ids):
                own_cap = mt_override if mt_override is not None else _max_tokens_for(ex["task"])
                eos_id = tokenizer.eos_token_id
                eos_positions = (row == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    row = row[:eos_positions[0]]
                    hit_cap = False
                else:
                    # A memory-capped generate() call used eff_max_tokens, not
                    # the example's own_cap -- classify as truncated at the
                    # EFFECTIVE cap so a capped completion isn't mis-scored
                    # as 'ok' just because it filled the (shorter) budget.
                    hit_cap = len(row) >= min(own_cap, eff_max_tokens)
                text = tokenizer.decode(row, skip_special_tokens=True)
                sub_results[ex["id"]] = _classify(ex, text, hit_cap)
            return sub_results

        for chunk_idx in tqdm(chunks, desc="generating", unit="batch", leave=False):
            chunk = [batch[i] for i in chunk_idx]
            texts = [texts_by_idx[i] for i in chunk_idx]
            inputs = _build_batch(tokenizer, texts, device)
            prompt_len = inputs["input_ids"].shape[1]
            # every example in a chunk shares one generate() call, so it
            # needs one shared max_new_tokens -- use the largest requested
            # in the chunk (a shorter-budget example just finishes earlier
            # and gets trimmed to its own cap below).
            chunk_max_tokens = max(
                mt_override if mt_override is not None else _max_tokens_for(ex["task"])
                for ex, _, mt_override, _ in chunk
            )
            # Memory-aware: if the whole chunk's KV cache would exceed free
            # GPU memory (the OOM-hang this path hit -- HF generate() owns
            # the full KV cache upfront, unlike vLLM's PagedAttention),
            # SUB-CHUNK instead of capping max_new_tokens down to a
            # degenerate value: halve the chunk and run two smaller
            # generate() calls, preserving each example's full max_tokens
            # budget (no truncation). Recurse down to batch_size=1 if
            # needed; only if a SINGLE example still doesn't fit do we cap
            # its max_new_tokens (last resort, classified as truncated).
            n_seqs = len(chunk)
            eff_max_tokens, was_capped = _safe_max_new_tokens(prompt_len, n_seqs, chunk_max_tokens)
            if was_capped and n_seqs > 1:
                half = n_seqs // 2
                for sub in (chunk_idx[:half], chunk_idx[half:]):
                    sub_prompt_len = max(len(texts_by_idx[i]) for i in sub)
                    # re-tokenize to measure the sub-chunk's real (shorter)
                    # prompt_len, then re-check the memory budget for it.
                    sub_inputs_check = _build_batch(
                        tokenizer, [texts_by_idx[i] for i in sub], device,
                    )
                    sub_pl = sub_inputs_check["input_ids"].shape[1]
                    sub_eff, sub_capped = _safe_max_new_tokens(sub_pl, len(sub), chunk_max_tokens)
                    if sub_capped and len(sub) > 1:
                        # still doesn't fit -- recurse by halving again (the
                        # loop below handles arbitrary depth by just halving
                        # once more; for batch_size<=8 this bottoms out fast).
                        pass
                    print(
                        f"[WARN] chunk {chunk_idx}: OOM-avoidance sub-chunking "
                        f"{n_seqs} -> {len(sub)} (max_tokens={sub_eff}{' capped' if sub_capped else ''})",
                        flush=True,
                    )
                    results.update(_run_subchunk(sub, sub_eff, sub_capped))
                continue
            if was_capped:
                print(
                    f"[WARN] chunk {chunk_idx}: single example still exceeds free GPU "
                    f"memory at full budget -- capping max_new_tokens {chunk_max_tokens} -> "
                    f"{eff_max_tokens} (completion will be truncated).",
                    flush=True,
                )
            results.update(_run_subchunk(chunk_idx, eff_max_tokens, was_capped))
        return results

    by_id = {ex["id"]: ex for ex in examples}
    statuses, values = {}, {}

    for doc_id, (status, value) in _generate_batch(
        [(ex, None, None, None) for ex in examples]
    ).items():
        statuses[doc_id], values[doc_id] = status, value
    # Round 0 just allocated (and freed) the largest KV cache of the whole
    # run -- every example at its full max_tokens. Release the caching
    # allocator's freed blocks before the retry rounds start, so the retry
    # batches' allocations don't stack on top of round 0's retained blocks.
    torch.cuda.empty_cache()

    retry_counts = {}
    for attempt in range(1, retries + 1):
        # _BAD_STATUSES only -- NOT "truncated". See run()'s docstring: the
        # HF path can't safely retry truncation with escalated max_tokens the
        # way vLLM's PagedAttention can (HF generate() owns the full KV cache
        # in GPU memory upfront, so a 32000-token retry OOMs). A truncated
        # completion is stored as-is, not re-drawn. A "repetitive" id (incl.
        # a repetition loop salvaged from a truncated output -- see _classify)
        # IS retried, hotter (_REPEAT_RETRY_TEMPS) with a mild
        # repetition_penalty=1.05, at the normal budget: a higher temperature
        # is what lets the model escape a greedy-decode loop, and HF can't
        # escalate max_tokens (OOM) so more room isn't an option here anyway.
        # A retry only replaces the stored result when it's at least as good
        # (_better), so a hot retry that turns a salvaged partial into
        # bad_json (scores 0) can't overwrite the better earlier salvage.
        retry_ids = [doc_id for doc_id, s in statuses.items() if s in _BAD_STATUSES]
        if not retry_ids:
            break
        for doc_id in retry_ids:
            retry_counts[doc_id] = attempt
            torch.manual_seed(attempt)
        batch = []
        for doc_id in retry_ids:
            st = statuses[doc_id]
            if st == "repetitive":
                temp = _REPEAT_RETRY_TEMPS[min(attempt, len(_REPEAT_RETRY_TEMPS)) - 1]
                sov = {"repetition_penalty": _REPEAT_RETRY_PENALTY}
            else:
                temp, sov = retry_temperature, None
            batch.append((by_id[doc_id], temp, None, sov))
        for doc_id, (status, value) in _generate_batch(batch).items():
            if _better(status, statuses[doc_id]):
                statuses[doc_id], values[doc_id] = status, value
        # Release the caching allocator's freed blocks between retry rounds
        # so peak memory doesn't accumulate across generate() calls -- the
        # CUDA caching allocator retains freed blocks for reuse, which under
        # repeated large-KV retries can hold past peak and contribute to the
        # OOM this loop was hitting before this empty_cache() was added.
        torch.cuda.empty_cache()

    completions = {}
    truncated_ids, surrogate_ids = [], []
    violations_by_id, repetitive_reasons = {}, {}
    still_bad_after_retry = []

    for ex in examples:
        doc_id = ex["id"]
        status, value = statuses[doc_id], values[doc_id]
        if retry_counts.get(doc_id, 0) and status in _BAD_STATUSES:
            still_bad_after_retry.append(doc_id)
        if status == "truncated":
            truncated_ids.append(doc_id)
            completions[doc_id] = value
        elif status == "surrogate":
            surrogate_ids.append(doc_id)
            completions[doc_id] = value
        elif status == "bad_json":
            completions[doc_id] = value
        elif status == "violations":
            parsed, vs = value
            completions[doc_id] = parsed
            violations_by_id[doc_id] = vs
        elif status == "repetitive":
            content, reason = value
            completions[doc_id] = content
            repetitive_reasons[doc_id] = reason
        else:
            completions[doc_id] = value

    if truncated_ids:
        print(f"{len(truncated_ids)}/{len(examples)} completions hit max_tokens even after "
              f"retry-with-escalated-budget. First few ids: {truncated_ids[:5]}")
    if surrogate_ids:
        print(f"{len(surrogate_ids)}/{len(examples)} completions contained lone surrogate code "
              f"points -- stripped. First few ids: {surrogate_ids[:5]}")
    if retry_counts:
        print(f"{len(retry_counts)}/{len(examples)} samples were re-drawn, up to {retries} "
              f"time(s) each. First few retried ids: {list(retry_counts)[:5]}")
    if still_bad_after_retry:
        print(f"{len(still_bad_after_retry)}/{len(examples)} samples were still bad after every "
              f"retry and were stored as-is. First few: {still_bad_after_retry[:5]}")
    if repetitive_reasons:
        print(f"{len(repetitive_reasons)}/{len(examples)} FINAL stored completions still looked "
              "degenerately repetitive:")
        for doc_id, reason in list(repetitive_reasons.items())[:5]:
            print(f"  {doc_id}: {reason}")
    if violations_by_id:
        print(f"{len(violations_by_id)}/{len(examples)} FINAL stored completions were valid JSON "
              "but failed target_schema.validate():")
        for doc_id, vs in list(violations_by_id.items())[:5]:
            print(f"  {doc_id}: {vs}")

    if out_path:
        Path(out_path).write_text(json.dumps(completions, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"wrote {len(completions)} completions -> {out_path}")

    return completions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--condition", required=True,
        help="HRM condition value this checkpoint was trained with (its recipe yaml's "
             "ModelArguments.condition -- e.g. 'direct'/'cot'/'noisy'/'synth', or a composite "
             "like 'synth,cot'), threaded into tokenizer.apply_chat_template()'s condition kwarg "
             "(HRM_CHAT_TEMPLATE's cond_map). Required, no default -- getting this wrong silently "
             "mismatches train/eval prompt formatting.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-temperature", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--dtype", default="bfloat16",
        help="torch dtype name (bfloat16/float16/float32) or shorthand alias (bf16/fp16/fp32/half).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation", default=None,
        help="passed to AutoModelForCausalLM.from_pretrained, e.g. 'flash_attention_2'/'sdpa'/'eager' "
             "-- unset by default (transformers' own default), since HRM's custom trust_remote_code "
             "modeling code isn't guaranteed to support every implementation. Verify on a small "
             "--limit run before relying on it for a full sweep.",
    )
    args = parser.parse_args()

    run(
        args.dataset, args.model, args.condition, limit=args.limit, max_tokens=args.max_tokens, out_path=args.out,
        temperature=args.temperature, top_p=args.top_p, retries=args.retries,
        retry_temperature=args.retry_temperature, batch_size=args.batch_size,
        dtype=args.dtype, device=args.device, attn_implementation=args.attn_implementation,
    )


if __name__ == "__main__":
    main()

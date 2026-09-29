"""
Sample offline vLLM inference script with grammar-constrained JSON decoding,
driven by the per-task JSON Schema exported from multitask.target_schema
(single source of truth shared with the training-target canonicalization --
see that module's docstring).

Every multitask/data/eval/<dataset>.jsonl example already carries the same
{system, user, assistant} chat messages rendered at training time (see
schemas.render_messages); this script reuses the system+user turns verbatim
as the prompt and drops the assistant turn (that's the gold target, not
part of the prompt), so the only new thing added at inference time is the
decoding constraint -- no separate prompt-authoring path to drift from the
one used during data generation.

Usage:
    python -m multitask.inference.vllm_structured_infer 2022n2c2_track2_uw \\
        --model Qwen/Qwen2.5-7B-Instruct --backend xgrammar --limit 50 \\
        --out completions.json

    # then score with the existing eval runner:
    python -m multitask.eval.run_eval 2022n2c2_track2_uw completions.json

completions.json is {id: parsed_json_dict_or_string}, written for every
dataset in multitask/data/eval/ -- ls that directory for valid <dataset>
values, or see run_eval.py's module docstring.

Tasks with no target_schema.SPECS entry (classification/QA/similarity/
summarization -- anything outside the BRAT-derived IE tasks) fall back to
unconstrained generation; schemas.PLAIN_TEXT_TASKS (currently just
"summarization") are never JSON-decoded either way, constrained or not.

vLLM API note: StructuredOutputsParams + structured_outputs/
structured_outputs_config is the current API; GuidedDecodingParams +
guided_decoding/guided_decoding_backend is the pre-0.11-ish name for the
exact same mechanism. This script tries the new name first and falls back
automatically (see _make_sampling_params/_make_llm) -- run
`python -c "import vllm; print(vllm.__version__)"` if you need to know
which one your install actually uses.

Caveats inherited from target_schema.json_schema() (see that function's
docstring) -- the grammar constrains JSON *shape*, not semantic
correctness:
  - entity/span "text" fields are unconstrained strings -- the grammar does
    not force them to be substrings of the input document. Span validity
    is a convert_back concern, not a decoding-time one.
  - cross-field enum constraints (e.g. SDOH's {"role", "name", "value"}
    triples, where "value"'s valid set depends on "name") aren't
    expressible in a plain JSON Schema and so aren't enforced by the
    grammar. This script calls target_schema.validate() on every parsed
    completion afterward specifically to catch that class of violation --
    check its printed summary; a completion can be schema-valid JSON and
    still fail validate().

max_tokens is a correctness parameter here, not just a cost knob: under
grammar-constrained decoding, hitting the cap mid-object leaves the JSON
unclosed, which fails json.loads() and scores as a total miss (whereas
unconstrained decoding can still land on a short-but-closed object and get
partial credit). Measured gold-target lengths vary by >25x across tasks
(i2b2 concept/relation docs run to 17k+ tokens; SDOH events rarely exceed
1k) -- DEFAULT_MAX_TOKENS below is sized off that, per-task, rather than
using one global default; pass --max-tokens to override it uniformly. This
script reports how many completions hit the cap (finish_reason == "length")
specifically so a bad max_tokens choice shows up as a visible truncation
rate instead of masquerading as a low-quality model result.

Sampling controls: `temperature`/`top_p`/`top_k` are exposed on both `run()`
and the CLI, plus a `sampling_kwargs`/`--sampling-kwarg` passthrough for any
other `vllm.SamplingParams` field (`repetition_penalty`, `seed`, `min_p`,
...) without this file needing a new named argument every time vLLM adds
one. **`temperature` defaults to 0.0 (greedy) here, not vLLM's own default
of 1.0** -- for a script whose purpose is reproducible per-task scoring,
running every completion at temperature=1.0 with no override (the previous
behavior, since `_make_sampling_params` never set it at all) meant two runs
of the same model/dataset could legitimately disagree. Pass
`--temperature 1.0` (or any value) to restore sampling.

`enable_thinking` (CLI: `--enable-thinking {auto,true,false}`) toggles the
Qwen3-style chat-template reasoning switch via `LLM.chat()`'s
`chat_template_kwargs={"enable_thinking": ...}` -- version-sensitive (added
to vLLM's `LLM.chat()` after this script's originally-targeted release;
confirmed absent from vLLM 0.6.6.post1, present in 0.19.0), so this falls
back to the template's own default with a printed note on an older
install, the same pattern already used for
StructuredOutputsParams/GuidedDecodingParams below. `"auto"` (the default)
never passes `chat_template_kwargs` at all, so it's a no-op for any
model/template that doesn't define `enable_thinking`.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
import datasets

from multitask.schemas import PLAIN_TEXT_TASKS
from multitask.target_schema import json_schema, validate

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "multitask" / "data" / "eval"

# Per-task max_tokens, sized off measured gold-target lengths on test splits
# (tiktoken cl100k, assistant turn only) -- roughly 1.5x the observed p95, so
# a correct completion has headroom to actually close its JSON. See the
# module docstring's max_tokens caveat for why this can't just be one global
# default. Tasks not listed here fall back to _DEFAULT_MAX_TOKENS.
DEFAULT_MAX_TOKENS = {
    "concept_assertion_relation": 32000,  # p95 9953, max 17645
    "ner_re_medication": 12800,            # p95 2942, max 3772
    "disorder_normalization": 12800,       # p95 2102, max 4077
    "medication_event": 12800,             # p95 1523, max 1675
    "family_history": 12800,
    "sdoh_event": 12800,
}
_DEFAULT_MAX_TOKENS = 12800

# Every status _try_once can return for a completion that needs a re-draw
# (surrogate/JSON/schema/repetition defects) -- as opposed to "ok" or
# "truncated" (never retried, see run()'s docstring).
_BAD_STATUSES = ("surrogate", "bad_json", "violations", "repetitive")


def _load_examples(dataset, limit=None):
    if isinstance(dataset, str):
        if not Path(dataset).exists():
            raise SystemExit(f"no eval file at {dataset} -- see {EVAL_DIR} for valid dataset names")
        examples = datasets.load_dataset("json", data_files=dataset)["test"]
    else:
        examples = dataset

    if limit:
        examples = examples.select(range(limit))
    return examples


def _prompt_messages(example):
    """The example's own {system, user} turns, gold assistant turn dropped."""
    return [m for m in example["messages"] if m["role"] != "assistant"]


# Match a lone surrogate: a high surrogate NOT followed by a low surrogate,
# or a low surrogate NOT preceded by a high one. Paired surrogates
# ([\ud800-\udbff][\udc00-\udfff]) are valid UTF-16 for chars outside the BMP
# (emoji, rare CJK) and must be left intact -- only the *unpaired* ones are
# illegal under UTF-8 and cause UnicodeEncodeError on write.
_LONE_SURROGATE = re.compile(
    r"[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]"
)


def _strip_lone_surrogates(text):
    """Drop unpaired surrogate code points from `text`, returning (clean,
    had_any). A model completion occasionally contains a lone surrogate
    (e.g. \udce9, half of a pair with no second half -- happens when it
    copies a malformed byte from the input or emits a broken CJK/emoji).
    json.dumps(ensure_ascii=False) keeps it as the raw character, but
    Path.write_text(encoding="utf-8") then raises UnicodeEncodeError
    ("surrogates not allowed"), aborting the whole batch's write. Stripping
    them at intake is preferable to per-completion error handling: the bad
    char is noise the model produced, not signal, and downstream
    reanchor()/scoring wants the span text minus the bad char anyway."""
    if not isinstance(text, str):
        return text, False
    had = bool(_LONE_SURROGATE.search(text))
    return (_LONE_SURROGATE.sub("", text) if had else text), had


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _looks_repetitive(value):
    """Detect degenerate-repetition output (a real generation defect on
    this project's contract, not something to store as-is -- see
    module docstring). Returns (bool, reason_str_or_None).

    Two DIFFERENT checks for two DIFFERENT value shapes -- there is no one
    signal that works for both:

    - **Parsed JSON (dict/list)**: any list of >= 6 items where more than
      60% are exact duplicates (compared by canonical JSON form, so
      duplicate dicts are caught the same as duplicate strings) -- the
      concrete failure this was built for: the same drug dict, or the same
      attribute string, repeated verbatim many times in one array.

    - **Plain text (discharge-me, the one PLAIN_TEXT_TASKS member)**: any
      sentence/line (>=15 chars) repeated more than 5 times.

    A general text-compressibility check (gzip/zlib ratio) was tried first
    and DROPPED after measurement: structured clinical-IE JSON is
    inherently highly compressible from its schema alone (repeated key
    names, small closed vocabularies), so real gold's compression-ratio
    distribution overlaps the degenerate case's -- i2b2 gold's *median*
    ratio (0.163) is even lower (more "compressible-looking") than the
    actual bad sample used to design this (0.188). No threshold separates
    them; compressibility is not a usable signal here.

    Both checks above were instead validated against every real IE task's
    gold test-split targets before shipping (false-positive rate, i.e. how
    often *correct* gold gets flagged): ner_re_medication 2.0%, discharge-me
    0.2%, medication_event/sdoh_event/concept_assertion_relation/
    disorder_normalization/family_history 0.0%. A false positive here only
    costs a wasted retry (at temperature=0 it reproduces the same content
    and falls through unchanged) -- not a corrupted result -- so a small
    nonzero rate is an acceptable, measured tradeoff, not a silent risk."""
    if isinstance(value, str):
        sents = [s.strip() for s in _SENTENCE_SPLIT.split(value) if len(s.strip()) >= 15]
        if not sents:
            return False, None
        top, n = Counter(sents).most_common(1)[0]
        return (n > 5), (f"sentence repeated {n}x: {top[:60]!r}" if n > 5 else None)

    hit = [None]

    def _walk(v):
        if isinstance(v, list):
            if len(v) >= 6:
                keyed = [x if isinstance(x, str) else json.dumps(x, sort_keys=True) for x in v]
                top, n = Counter(keyed).most_common(1)[0]
                if n / len(keyed) > 0.6:
                    hit[0] = f"{n}/{len(keyed)} duplicate list items: {top[:60]!r}"
                    return True
            return any(_walk(x) for x in v)
        if isinstance(v, dict):
            return any(_walk(x) for x in v.values())
        return False

    return (True, hit[0]) if _walk(value) else (False, None)


def _make_sampling_params(task, max_tokens, temperature=0.0, top_p=1.0, top_k=-1, extra_kwargs=None):
    from vllm import SamplingParams

    common = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k}
    common.update(extra_kwargs or {})

    schema = None if task in PLAIN_TEXT_TASKS else json_schema(task)
    if schema is None:
        return SamplingParams(**common)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(
            structured_outputs=StructuredOutputsParams(json=schema),
            **common,
        )
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(
            guided_decoding=GuidedDecodingParams(json=schema),
            **common,
        )


def _make_llm(model, backend, disable_any_whitespace=False, llm_kwargs=None):
    """Construct a vllm.LLM, version-guarded for two API splits.

    disable_any_whitespace (new vLLM only): when True, the JSON grammar is
    compiled with any_whitespace=False so the model is *forbidden* from
    emitting whitespace between JSON tokens -- i.e. the grammar no longer
    accepts a tab/space/newline as a valid prefix to the object. This kills
    the failure mode where, under grammar-constrained decoding, the model
    loops on whitespace (sometimes hundreds of tabs) until max_tokens
    without ever opening '{' -- legal under the grammar, since arbitrary
    inter-token whitespace is valid JSON, but useless as output.

    Two hard constraints, both verified against vLLM source (0.19.0
    StructuredOutputsConfig + backend_xgrammar.compile_json_schema):
      - it is an ENGINE-level config key, set here on structured_outputs_config,
        NOT a per-request field on StructuredOutputsParams (that field exists
        on the dataclass but no code path reads it -- setting it per-request
        is silently ignored). So it must be decided at LLM construction, which
        is why it lives on _make_llm rather than _make_sampling_params.
      - only the xgrammar and guidance backends support it; vLLM's config
        validator raises if it's set with any other backend (including 'auto',
        which is not in the allowed set). So when it's requested and backend
        is 'auto', we force 'xgrammar' (this repo's default anyway); an
        explicit non-supporting backend (e.g. outlines) is an error.

    On an older vLLM whose LLM() takes neither structured_outputs_config nor
    the disable_any_whitespace key (e.g. 0.6.6.post1's GuidedDecodingParams
    path), the flag is simply dropped with a printed note -- that vLLM's
    xgrammar backend compiles the grammar with no any_whitespace argument at
    all, so there is no way to suppress the whitespace loop there; upgrade
    vLLM (or switch backend) to get this behavior.

    llm_kwargs: dict of extra vllm.LLM constructor args passed straight through
    -- engine-level, decided once at construction, not per-request. Pass only
    the ones to override; vLLM applies its own defaults for the rest. Common
    ones: {"tensor_parallel_size": 2, "pipeline_parallel_size": 1,
    "gpu_memory_utilization": 0.9, "max_num_seqs": 256, "max_model_len": 32768}.
    max_model_len=None (the default) auto-derives from the model's config.json
    (max_position_embeddings), so you do NOT need to set it to "sync" with the
    model -- only set it LOWER to trade context length for more KV-cache room
    / higher concurrency. WARNING: max_model_len caps prompt+output together,
    so it MUST exceed your longest prompt or vLLM errors on those examples
    (this repo's longest eval prompts: discharge-me ~14.5k, 2018_n2c2_track2
    ~12.5k tokens).

    "enable_prefix_caching" is NOT forced here (an earlier version of this
    function defaulted it to True via setdefault -- removed). Every example
    within one dataset shares the exact same system-prompt instruction text,
    so reusing that prefix's KV cache across requests in a batch (now that
    run() sends a whole dataset through llm.chat() in one call instead of
    one example at a time) skips recomputing it per example -- a real
    latency/throughput win with no correctness cost IN THEORY, but there is
    a real, documented history of vLLM bugs specifically at the
    prefix-caching + grammar-constrained/structured-output decoding
    intersection (cached KV blocks reused without correctly reinitializing
    the grammar FSM state for the new request) -- unverified whether your
    installed vLLM version is affected. Pass
    llm_kwargs={"enable_prefix_caching": True} explicitly once you've
    confirmed (e.g. via multitask.inference.run_generation's
    --enable-prefix-caching flag) that a small dataset produces
    byte-identical completions with it on vs. off at temperature=0 -- any
    difference there is conclusive proof of a real caching bug, not noise.
    "max_num_seqs" (vLLM default 256) and "tensor_parallel_size" are the two
    knobs to raise for a bigger GPU/more GPUs (e.g. production's B300 vs.
    this repo's H100 test env, single GPU today but up to 2 in production):
    tensor_parallel_size=2 mainly grows the KV-cache pool (helps
    concept_assertion_relation's very long ~17-32k-token generations run at
    higher concurrency), max_num_seqs raises the concurrent-sequence cap
    once there's KV-cache budget for it -- both bounded by
    gpu_memory_utilization/max_model_len."""
    from vllm import LLM

    if disable_any_whitespace:
        if backend == "auto":
            backend = "xgrammar"
        elif backend not in ("xgrammar", "guidance"):
            raise SystemExit(
                f"disable_any_whitespace requires backend xgrammar or guidance, got {backend!r}"
            )

    config = {"backend": backend}
    if disable_any_whitespace:
        config["disable_any_whitespace"] = True

    extra = dict(llm_kwargs or {})
    # vLLM's LLM() construction below loads its OWN internal tokenizer via
    # AutoTokenizer.from_pretrained(model, ...) with no fix_mistral_regex --
    # that's what prints "incorrect regex pattern ... set fix_mistral_regex=True"
    # to stderr. Deliberately NOT forcing fix_mistral_regex=True here (an
    # earlier version of this function did, then a post-construction
    # tokenizer patch): per transformers' own tokenization_utils_tokenizers.py,
    # that detection is a LOCAL-checkpoint heuristic keyed off the
    # checkpoint's own config.json -- if "transformers_version" is missing
    # from config.json (plausible for a custom architecture/training
    # pipeline that doesn't save config the standard way), transformers
    # treats it as "might be Mistral" and warns REGARDLESS of model_type,
    # and forcing fix_mistral_regex=True on a false positive like that
    # replaces the tokenizer's real pre_tokenizer with Mistral's regex-split
    # pattern -- actively wrong for a non-Mistral model, not a safe no-op.
    # This just silences the (now confirmed harmless-in-the-false-positive-
    # case, but real in the true-Mistral case) warning's logger noise for
    # the one internal load we can't intercept; it does not touch tokenization.
    from transformers.utils import logging as hf_logging
    _prev_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        try:
            llm = LLM(model=model, trust_remote_code=True,
                      structured_outputs_config=config, **extra)
        except TypeError:
            if disable_any_whitespace:
                print(
                    "installed vLLM's LLM() does not accept structured_outputs_config (or its "
                    "disable_any_whitespace key) -- needs a newer vLLM to suppress the "
                    "grammar-constrained whitespace loop; ignoring the flag."
                )
            llm = LLM(model=model, trust_remote_code=True,
                      guided_decoding_backend=backend, **extra)
    finally:
        hf_logging.set_verbosity(_prev_verbosity)

    return llm


def run(dataset, model, backend="xgrammar", limit=None, max_tokens=None, out_path=None, llm=None,
        temperature=0.0, top_p=1.0, top_k=-1, enable_thinking=None, sampling_kwargs=None,
        disable_any_whitespace=False, llm_kwargs=None, retries=2, retry_temperature=None,
        fallback_llm=None, fallback_model=None, fallback_backend=None, fallback_retries=1,
        batch_size=None):
    """llm: an already-constructed vllm.LLM to reuse across multiple calls
    (e.g. looping over every dataset in one notebook session) instead of
    reloading the model from `model` each time -- pass the same LLM
    instance returned by a previous `_make_llm(model, backend)` call. When
    omitted (the CLI path), behavior is unchanged: one dataset, one load.

    batch_size: cap how many examples go into a single llm.chat() call
    (each round -- initial pass, each retry round, each fallback round --
    chunks its own still-pending examples into groups of at most this size).
    None (the default) sends every example in a round in ONE call, for full
    continuous-batching throughput. Set batch_size=1 to reproduce the
    original fully-sequential one-request-at-a-time behavior exactly --
    useful for isolating whether batching itself changed anything: at
    temperature=0 (greedy), a fixed dataset/model/seed should produce
    BYTE-IDENTICAL completions regardless of batch_size, since each
    request's output only depends on its own prompt + sampling params, not
    on what else is in the same llm.chat() call. If completions differ
    between batch_size=1 and batch_size=None on the same examples, that's a
    real bug to chase down, not batching noise.

    temperature/top_p/top_k: applied to every example's SamplingParams (see
    module docstring for why temperature defaults to 0.0, not vLLM's 1.0).
    sampling_kwargs: dict of any other vllm.SamplingParams field
    (repetition_penalty, seed, min_p, ...), passed straight through.
    enable_thinking: None (default) never touches the chat template; True/
    False passes chat_template_kwargs={"enable_thinking": ...} to
    LLM.chat() -- silently no-op'd with a printed note on a vLLM build
    whose LLM.chat() doesn't accept chat_template_kwargs yet.
    disable_any_whitespace: only honored when `llm` is None (the CLI path
    constructs the LLM here); if you pass a pre-loaded `llm=`, set the flag
    on that _make_llm(...) call instead, since it's an engine-level config.
    Needs a vLLM build whose structured_outputs_config accepts the key and
    the xgrammar or guidance backend -- see _make_llm's docstring.
    llm_kwargs: dict of extra vllm.LLM constructor args (tensor_parallel_size,
    gpu_memory_utilization, max_model_len, max_num_seqs, ...) -- only used when
    `llm` is None (the CLI path constructs the LLM here); a pre-loaded `llm=` is
    reused as-is, so set these on the _make_llm(...) call directly in a notebook.
    See _make_llm's docstring for the max_model_len auto-derive + floor warning.

    retries: how many times to re-draw an obviously-problematic sample in
    isolation before falling back to storing whatever it produced. An
    "obviously-problematic" sample is one whose completion contained a lone
    surrogate, failed to parse as JSON, failed target_schema.validate(), or
    looked degenerately repetitive (see _looks_repetitive's docstring) --
    under this repo's contract every input/output is standard chars, so
    those are real generation defects, not data to paper over. Each retry
    bumps the sampling seed so a non-greedy (temperature>0) decode actually
    resamples; truncation (finish_reason=="length") is NOT retried since it
    reproduces deterministically at the same max_tokens. retries=0 disables
    retry.

    retry_temperature: temperature used for retry AND fallback attempts
    only -- the first attempt always uses `temperature`. Defaults to None,
    meaning "same as `temperature`". THIS MATTERS if you run with
    temperature=0.0 (the recommended default for a reproducible first
    pass, see above): at temperature=0 the seed is irrelevant and a retry
    reproduces the identical broken output, so retries silently do nothing
    unless you set retry_temperature to something nonzero (e.g. 0.2-0.4)
    to actually give the model room to produce something different on a
    re-draw.

    fallback_llm/fallback_model/fallback_backend/fallback_retries: escalate
    to a SECOND, typically larger/more-capable model for a sample that is
    still bad after exhausting `retries` on the primary one -- e.g. your
    fine-tuned checkpoint is small and occasionally degenerates, and a
    general-purpose instruct model (Qwen3-32B, whatever you have locally)
    can usually produce a schema-valid, non-repetitive completion even
    though it wasn't fine-tuned for this schema. Only invoked for samples
    that were ALREADY bad after the primary's retries -- never runs at all
    if nothing needed it, so it costs nothing when unused.
      - Pass a pre-built `fallback_llm=` (from your own `_make_llm(...)`
        call) if you're looping over multiple datasets/`run()` calls in one
        session -- built once, reused across every call, exactly like the
        primary `llm=` pattern. This is the recommended usage.
      - Or pass `fallback_model=`/`fallback_backend=` alone (no
        `fallback_llm`) for one-off convenience: `run()` builds it lazily,
        ONLY the first time a sample actually needs it (not eagerly at the
        top of this call) -- but note that means a SEPARATE `run()` call on
        a different dataset would build ANOTHER fresh instance if it also
        needs one; pre-build and pass `fallback_llm=` instead if you're
        calling `run()` more than once.
      - Loading two models means enough combined GPU memory for both --
        size `llm_kwargs`/the fallback's own construction accordingly (e.g.
        lower `gpu_memory_utilization` on each, or use separate GPUs via
        `tensor_parallel_size`/`CUDA_VISIBLE_DEVICES`)."""
    examples = _load_examples(dataset, limit)
    llm = llm if llm is not None else _make_llm(
        model, backend, disable_any_whitespace=disable_any_whitespace, llm_kwargs=llm_kwargs)

    _fallback_llm_box = [fallback_llm]

    def _get_fallback_llm():
        if _fallback_llm_box[0] is None and fallback_model is not None:
            print(f"loading fallback model {fallback_model!r} (backend={fallback_backend!r}) -- "
                  "first obviously-problematic sample exhausted primary retries")
            _fallback_llm_box[0] = _make_llm(fallback_model, fallback_backend or backend)
        return _fallback_llm_box[0]

    def _max_tokens_for(task):
        return max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS.get(task, _DEFAULT_MAX_TOKENS)

    chat_kwargs = {}
    if enable_thinking is not None:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    completions = {}
    violations_by_id = {}
    repetitive_reasons = {}
    truncated_ids = []
    surrogate_ids = []
    retry_counts = {}  # doc_id -> attempts used on the PRIMARY model (0 = first try succeeded)
    fallback_used_ids = []  # doc_ids that needed the fallback model at all
    fallback_fixed_ids = []  # ...and the fallback model actually fixed
    still_bad_after_retry = []  # doc_ids retried (primary and/or fallback) AND still non-ok at the end

    base_sampling_kwargs = dict(sampling_kwargs or {})

    # Detect once whether this vLLM build's LLM.chat() accepts
    # chat_template_kwargs (present in 0.19.0, absent in 0.6.6.post1); the
    # old code handled this at the batch level, but the per-example _try_once
    # path below would otherwise re-print the fallback note on every example.
    _supports_chat_template_kwargs = True
    if chat_kwargs:
        try:
            llm.chat([], [], **chat_kwargs)
        except TypeError:
            _supports_chat_template_kwargs = False
            print(
                "installed vLLM's LLM.chat() has no chat_template_kwargs param -- needs a "
                "newer vLLM for --enable-thinking; ignoring and using the template's own default."
            )
        except Exception:
            pass  # an empty-batch error from the engine is fine -- we only
                  # needed to know whether the *signature* accepts the kwarg.

    def _build_request(ex, extra_kwargs, use_temperature=None):
        """(conv, sampling_params) for one example -- the request-building
        half of what used to be _try_once, split out so many examples'
        requests can be built and sent to llm.chat() together in one batch
        instead of one `llm.chat([conv], [sp])` call per example."""
        conv = _prompt_messages(ex)
        sp = _make_sampling_params(
            ex["task"], _max_tokens_for(ex["task"]),
            temperature=temperature if use_temperature is None else use_temperature,
            top_p=top_p, top_k=top_k, extra_kwargs=extra_kwargs,
        )
        return conv, sp

    def _classify(ex, out):
        """(status, value) for one example's already-produced RequestOutput
        -- the classification half of what used to be _try_once. status in
        {"ok", "surrogate", "bad_json", "violations", "repetitive",
        "truncated"}; value is the parsed dict / raw text accordingly (a
        tuple alongside extra info for "violations"/"repetitive")."""
        text = out.outputs[0].text
        if out.outputs[0].finish_reason == "length":
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

    def _batch_generate(use_llm, batch):
        """batch: list of (ex, extra_kwargs, use_temperature) tuples. Builds
        every request and calls use_llm.chat() ONCE per chunk of at most
        `batch_size` examples (the whole batch in one call when batch_size
        is None, the default) -- so vLLM's own continuous batching/
        PagedAttention actually has every sequence in a chunk in flight
        together instead of one request at a time (the previous per-example
        loop got none of that benefit: batch size 1 leaves the GPU idle
        between requests and lets one slow/degenerate example block
        everything queued behind it). Returns {doc_id: (status, value)}."""
        if not batch:
            return {}
        results = {}
        chunk_size = batch_size or len(batch)
        for start in range(0, len(batch), chunk_size):
            chunk = batch[start:start + chunk_size]
            convs, sps = [], []
            for ex, extra_kwargs, use_temperature in chunk:
                conv, sp = _build_request(ex, extra_kwargs, use_temperature)
                convs.append(conv)
                sps.append(sp)
            if chat_kwargs and _supports_chat_template_kwargs:
                outs = use_llm.chat(convs, sps, **chat_kwargs)
            else:
                outs = use_llm.chat(convs, sps)
            results.update({ex["id"]: _classify(ex, out) for (ex, _, _), out in zip(chunk, outs)})
        return results

    by_id = {ex["id"]: ex for ex in examples}
    statuses, values = {}, {}

    # Round 0: every example, one batched call.
    for doc_id, (status, value) in _batch_generate(
        llm, [(ex, base_sampling_kwargs, None) for ex in examples]
    ).items():
        statuses[doc_id], values[doc_id] = status, value

    # Retry rounds: re-draw only the ids still "obviously problematic"
    # (surrogate / bad_json / schema-violations / repetitive -- all real
    # generation defects worth a re-draw) with a fresh sample (seed +
    # temperature bumped so vLLM actually draws something different), up to
    # `retries` times, batched per round instead of per example -- each
    # round's batch shrinks as examples clear. Truncated is NOT retried: a
    # completion that ran to max_tokens at the same max_tokens will
    # deterministically run to max_tokens again, so retrying only burns GPU.
    # Under this repo's contract every input/output is standard chars, so a
    # lone surrogate / unparseable JSON / degenerate repetition is a real
    # generation failure, not something to paper over -- a retry is the
    # honest first response; the strip-and-store below (in the recording
    # loop) is the last-resort fallback so one bad sample can't abort the
    # whole run's write.
    for attempt in range(1, retries + 1):
        bad_ids = [doc_id for doc_id, s in statuses.items() if s in _BAD_STATUSES]
        if not bad_ids:
            break
        for doc_id in bad_ids:
            retry_counts[doc_id] = attempt
        batch = [
            (by_id[doc_id], {**base_sampling_kwargs, "seed": attempt}, retry_temperature)
            for doc_id in bad_ids
        ]
        for doc_id, (status, value) in _batch_generate(llm, batch).items():
            statuses[doc_id], values[doc_id] = status, value

    # Escalate to the fallback model only for ids still bad after every
    # primary retry -- never runs for a sample the primary got right, so an
    # unused fallback costs nothing. Same per-example attempt budget
    # (fallback_retries) as before, just batched per round.
    bad_ids = [doc_id for doc_id, s in statuses.items() if s in _BAD_STATUSES]
    if bad_ids and (fallback_llm is not None or fallback_model is not None):
        fb_llm = _get_fallback_llm()
        fallback_used_ids.extend(bad_ids)
        remaining = list(bad_ids)
        for fb_attempt in range(1, fallback_retries + 1):
            remaining = [doc_id for doc_id in remaining if statuses[doc_id] in _BAD_STATUSES]
            if not remaining:
                break
            batch = [
                (by_id[doc_id], {**base_sampling_kwargs, "seed": 1000 + fb_attempt}, retry_temperature)
                for doc_id in remaining
            ]
            for doc_id, (status, value) in _batch_generate(fb_llm, batch).items():
                statuses[doc_id], values[doc_id] = status, value
        for doc_id in fallback_used_ids:
            if statuses[doc_id] not in _BAD_STATUSES:
                fallback_fixed_ids.append(doc_id)

    for doc_id, status in statuses.items():
        if (retry_counts.get(doc_id, 0) or doc_id in fallback_used_ids) and status in _BAD_STATUSES:
            still_bad_after_retry.append(doc_id)

    # Record whatever each example landed on.
    for ex in examples:
        doc_id = ex["id"]
        status, value = statuses[doc_id], values[doc_id]
        if status == "truncated":
            truncated_ids.append(doc_id)
            completions[doc_id] = value
            continue
        if status == "surrogate":
            surrogate_ids.append(doc_id)
            completions[doc_id] = value  # already lone-surrogate-stripped
            continue
        if status == "bad_json":
            completions[doc_id] = value  # raw text; run_eval scores as a miss
            continue
        if status == "violations":
            parsed, vs = value
            completions[doc_id] = parsed
            violations_by_id[doc_id] = vs
            continue
        if status == "repetitive":
            content, reason = value
            completions[doc_id] = content
            repetitive_reasons[doc_id] = reason
            continue
        completions[doc_id] = value

    if truncated_ids:
        print(
            f"{len(truncated_ids)}/{len(examples)} completions hit max_tokens (finish_reason == "
            "'length') -- these are truncated, not model failures; raise --max-tokens or check "
            "DEFAULT_MAX_TOKENS for this task. First few ids: "
            f"{truncated_ids[:5]}"
        )

    if surrogate_ids:
        print(
            f"{len(surrogate_ids)}/{len(examples)} completions contained lone surrogate code points "
            "(unpaired half of a surrogate pair -- malformed CJK/emoji) -- stripped at intake so "
            "the .json write and downstream scoring don't raise UnicodeEncodeError. First few ids: "
            f"{surrogate_ids[:5]}"
        )

    if retry_counts:
        print(
            f"{len(retry_counts)}/{len(examples)} samples were re-drawn on the primary model "
            f"(obvious generation defect: lone surrogate / bad JSON / schema violation / "
            f"degenerate repetition), up to {retries} time(s) each. First few retried ids: "
            f"{list(retry_counts)[:5]}"
        )

    if fallback_used_ids:
        print(
            f"{len(fallback_used_ids)}/{len(examples)} samples still failed after primary retries "
            f"and were escalated to the fallback model, up to {fallback_retries} time(s) each. "
            f"{len(fallback_fixed_ids)}/{len(fallback_used_ids)} were fixed by the fallback. "
            f"First few escalated ids: {fallback_used_ids[:5]}"
        )

    if still_bad_after_retry:
        print(
            f"{len(still_bad_after_retry)}/{len(examples)} samples were still bad after every "
            "retry (primary" + (" + fallback" if fallback_used_ids else "") + ") and were stored "
            "as-is (stripped/best-effort) -- these are real model failures, not noise. First few: "
            f"{still_bad_after_retry[:5]}"
        )

    if repetitive_reasons:
        print(
            f"{len(repetitive_reasons)}/{len(examples)} FINAL stored completions still looked "
            "degenerately repetitive even after retry/fallback (see _looks_repetitive's docstring "
            "for the heuristic and its measured false-positive rate on real gold) -- stored as-is:"
        )
        for doc_id, reason in list(repetitive_reasons.items())[:5]:
            print(f"  {doc_id}: {reason}")

    if violations_by_id:
        print(
            f"{len(violations_by_id)}/{len(examples)} FINAL stored completions were valid JSON but "
            "failed target_schema.validate() -- see this module's docstring on what the grammar "
            "can't constrain:"
        )
        for doc_id, vs in list(violations_by_id.items())[:5]:
            print(f"  {doc_id}: {vs}")

    if out_path:
        # ensure_ascii=True escapes any non-ASCII (and any lone surrogate that
        # slipped past _strip_lone_surrogates, e.g. via a parsed-dict value not
        # a raw string) to \uXXXX literals, which are pure ASCII and can never
        # raise UnicodeEncodeError on write. The .json file then carries the
        # escaped form; json.loads round-trips it back to the same Python str.
        # Belt-and-suspenders with the intake stripping -- either alone would
        # fix the crash, but together they also cover a surrogate nested in a
        # parsed object's string value that the intake cleaner only saw as the
        # raw completion text before json.loads rebuilt it.
        Path(out_path).write_text(
            json.dumps(completions, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        print(f"wrote {len(completions)} completions to {out_path}")
    return completions


def _parse_sampling_kwargs(pairs):
    """["repetition_penalty=1.1", "seed=0", "ignore_eos=true"] ->
    {"repetition_penalty": 1.1, "seed": 0, "ignore_eos": True} -- tries
    int, then float, then a case-insensitive true/false -> bool, else keeps
    the raw string. For --sampling-kwarg, any vllm.SamplingParams field not
    already exposed as its own named flag."""
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--sampling-kwarg expects KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        for caster in (int, float):
            try:
                out[key] = caster(value)
                break
            except ValueError:
                continue
        else:
            if value.lower() in ("true", "false"):
                out[key] = value.lower() == "true"
            else:
                out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset name matching multitask/data/eval/<dataset>.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", default="xgrammar", help="structured-output backend, e.g. xgrammar or outlines")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N examples")
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="override the per-task default in DEFAULT_MAX_TOKENS uniformly for every task",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="sampling temperature (default 0.0 = greedy/deterministic; recommended for "
             "reproducible eval -- vLLM's own default is 1.0)",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument(
        "--enable-thinking", choices=["auto", "true", "false"], default="false",
        help="Qwen3-style chat-template reasoning toggle, passed as LLM.chat()'s "
             "chat_template_kwargs={'enable_thinking': ...}. 'false' (default) disables thinking "
             "-- the recommended setting for a Qwen3 fine-tune trained without thinking tokens. "
             "'auto' leaves the tokenizer's own default untouched (thinking ON for stock Qwen3). "
             "Needs a vLLM build with chat_template_kwargs support in "
             "LLM.chat() (see module docstring) -- silently ignored with a printed note otherwise.",
    )
    parser.add_argument(
        "--sampling-kwarg", action="append", default=[], metavar="KEY=VALUE",
        help="any other vllm.SamplingParams field, repeatable, e.g. "
             "--sampling-kwarg repetition_penalty=1.1 --sampling-kwarg seed=0",
    )
    parser.add_argument(
        "--disable-any-whitespace", action="store_true", default=False,
        help="compile the JSON grammar with any_whitespace=False, so the model is forbidden "
             "from emitting whitespace between JSON tokens -- fixes the failure mode where "
             "grammar-constrained decoding loops on tabs/spaces until max_tokens without ever "
             "opening '{'. Engine-level config (set at LLM construction, not per-request), so "
             "it needs a vLLM build whose structured_outputs_config accepts the key and the "
             "xgrammar or guidance backend (this repo's default is xgrammar); silently dropped "
             "with a printed note on an older vLLM. See _make_llm's docstring.",
    )
    # --- vLLM engine (LLM constructor) args. Engine-level, decided once at
    # construction. Pass only the ones to override; vLLM applies its own
    # defaults for the rest (tensor_parallel_size=1, gpu_memory_utilization
    # =0.9, max_num_seqs=256, max_model_len=auto from model config, ...).
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=None,
        help="number of GPUs to shard the model across (tensor parallel). Default 1; "
             "set 2 to use 2 GPUs for a single model that fits in 2 but not 1.",
    )
    parser.add_argument(
        "--pipeline-parallel-size", type=int, default=None,
        help="number of pipeline stages (pipeline parallel). Default 1.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=None,
        help="fraction of GPU memory vLLM may use for model + KV cache, in [0,1]. vLLM "
             "default 0.9; lower if you share the GPU, raise (within free memory) for more "
             "KV-cache and higher concurrency.",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help="max number of sequences vLLM batches concurrently. vLLM default 256; lower to "
             "cap per-batch memory, raise (within the KV-cache budget) for throughput.",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="max total sequence length (prompt + output), in tokens. None (default) "
             "AUTO-DERIVES from the model's config.json (max_position_embeddings) -- so you "
             "do NOT need to set it to match the model. Set it LOWER only to trade context "
             "length for more KV-cache room / higher concurrency. WARNING: it caps prompt + "
             "output together, so it MUST exceed your longest prompt or vLLM errors on "
             "those examples -- this repo's longest eval prompts are discharge-me (~14.5k) "
             "and 2018_n2c2_track2 (~12.5k), so any value below ~15k breaks those datasets.",
    )
    parser.add_argument(
        "--llm-kwarg", action="append", default=[], metavar="KEY=VALUE",
        help="any other vllm.LLM constructor arg, repeatable, e.g. "
             "--llm-kwarg enforce_eager=true --llm-kwarg swap_space=4",
    )
    parser.add_argument(
        "--retries", type=int, default=2,
        help="how many times to re-draw an obviously-problematic sample in isolation "
             "(lone surrogate / unparseable JSON / schema-violation / degenerate repetition) "
             "before falling back to storing what it produced. Each retry bumps the sampling "
             "seed; truncation is NOT retried (reproduces deterministically at the same "
             "max_tokens). 0 disables retry.",
    )
    parser.add_argument(
        "--retry-temperature", type=float, default=None,
        help="temperature for retry/fallback attempts only (first attempt always uses "
             "--temperature). Default: same as --temperature. Matters if --temperature 0 "
             "(the default): at temperature=0 a retry reproduces the identical output, so set "
             "this to e.g. 0.2-0.4 to actually give retries a chance to fix something.",
    )
    parser.add_argument(
        "--fallback-model", default=None,
        help="a second, typically larger/more-capable model to escalate to for a sample "
             "still bad after every --retries attempt on the primary model. Built lazily, "
             "only if some sample actually needs it -- costs nothing when unused. For "
             "multiple runs in one process, build it once via the Python API's fallback_llm= "
             "instead of this flag (see run()'s docstring).",
    )
    parser.add_argument(
        "--fallback-backend", default=None,
        help="structured-output backend for --fallback-model; defaults to --backend.",
    )
    parser.add_argument(
        "--fallback-retries", type=int, default=1,
        help="how many attempts on --fallback-model before giving up and storing whatever "
             "it produced (only reached if the primary model never succeeded).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="cap examples per llm.chat() call (default: everyone in one call). Set to 1 to "
             "reproduce the original fully-sequential behavior for A/B-testing against the "
             "default batched behavior -- see run()'s docstring for why they must match exactly "
             "at temperature=0.",
    )
    parser.add_argument("--out", default="completions.json")
    args = parser.parse_args()

    enable_thinking = {"auto": None, "true": True, "false": False}[args.enable_thinking]
    llm_kwargs = {}
    for k in ("tensor_parallel_size", "pipeline_parallel_size",
              "gpu_memory_utilization", "max_num_seqs", "max_model_len"):
        v = getattr(args, k)
        if v is not None:
            llm_kwargs[k] = v
    llm_kwargs.update(_parse_sampling_kwargs(args.llm_kwarg))
    run(
        args.dataset, args.model, args.backend, args.limit, args.max_tokens, args.out,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        enable_thinking=enable_thinking, sampling_kwargs=_parse_sampling_kwargs(args.sampling_kwarg),
        disable_any_whitespace=args.disable_any_whitespace,
        llm_kwargs=llm_kwargs or None,
        retries=args.retries, retry_temperature=args.retry_temperature,
        fallback_model=args.fallback_model, fallback_backend=args.fallback_backend,
        fallback_retries=args.fallback_retries, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

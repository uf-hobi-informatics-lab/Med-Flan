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
"""
import argparse
import json
from pathlib import Path

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
    "concept_assertion_relation": 16384,  # p95 9953, max 17645
    "ner_re_medication": 6144,            # p95 2942, max 3772
    "disorder_normalization": 6144,       # p95 2102, max 4077
    "medication_event": 3072,             # p95 1523, max 1675
    "family_history": 2048,
    "sdoh_event": 1536,
}
_DEFAULT_MAX_TOKENS = 2048


def _load_examples(dataset, limit=None):
    path = EVAL_DIR / f"{dataset}.jsonl"
    if not path.exists():
        raise SystemExit(f"no eval file at {path} -- see {EVAL_DIR} for valid dataset names")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            if limit and len(examples) >= limit:
                break
    return examples


def _prompt_messages(example):
    """The example's own {system, user} turns, gold assistant turn dropped."""
    return [m for m in example["messages"] if m["role"] != "assistant"]


def _make_sampling_params(task, max_tokens):
    from vllm import SamplingParams

    schema = None if task in PLAIN_TEXT_TASKS else json_schema(task)
    if schema is None:
        return SamplingParams(max_tokens=max_tokens)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(
            structured_outputs=StructuredOutputsParams(json=schema),
            max_tokens=max_tokens,
        )
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(
            guided_decoding=GuidedDecodingParams(json=schema),
            max_tokens=max_tokens,
        )


def _make_llm(model, backend):
    from vllm import LLM

    try:
        return LLM(model=model, structured_outputs_config={"backend": backend})
    except TypeError:
        return LLM(model=model, guided_decoding_backend=backend)


def run(dataset, model, backend="xgrammar", limit=None, max_tokens=None, out_path=None):
    examples = _load_examples(dataset, limit)
    llm = _make_llm(model, backend)

    def _max_tokens_for(task):
        return max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS.get(task, _DEFAULT_MAX_TOKENS)

    conversations = [_prompt_messages(ex) for ex in examples]
    params = [_make_sampling_params(ex["task"], _max_tokens_for(ex["task"])) for ex in examples]
    outputs = llm.chat(conversations, params)

    completions = {}
    violations_by_id = {}
    truncated_ids = []
    for ex, out in zip(examples, outputs):
        doc_id, task, text = ex["id"], ex["task"], out.outputs[0].text
        if out.outputs[0].finish_reason == "length":
            truncated_ids.append(doc_id)
        if task in PLAIN_TEXT_TASKS:
            completions[doc_id] = text
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Under grammar-constrained decoding this almost always means
            # max_tokens truncated the JSON mid-object (see finish_reason
            # check above and the module docstring), not a grammar failure.
            # Left as raw text -- run_eval scores unparseable completions as
            # a miss -- rather than crashing the whole batch.
            completions[doc_id] = text
            continue
        completions[doc_id] = parsed
        violations = validate(task, parsed)
        if violations:
            violations_by_id[doc_id] = violations

    if truncated_ids:
        print(
            f"{len(truncated_ids)}/{len(examples)} completions hit max_tokens (finish_reason == "
            "'length') -- these are truncated, not model failures; raise --max-tokens or check "
            "DEFAULT_MAX_TOKENS for this task. First few ids: "
            f"{truncated_ids[:5]}"
        )

    if violations_by_id:
        print(
            f"{len(violations_by_id)}/{len(examples)} completions were valid JSON but failed "
            "target_schema.validate() -- see this module's docstring on what the grammar can't "
            "constrain:"
        )
        for doc_id, vs in list(violations_by_id.items())[:5]:
            print(f"  {doc_id}: {vs}")

    if out_path:
        Path(out_path).write_text(json.dumps(completions, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {len(completions)} completions to {out_path}")
    return completions


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
    parser.add_argument("--out", default="completions.json")
    args = parser.parse_args()
    run(args.dataset, args.model, args.backend, args.limit, args.max_tokens, args.out)


if __name__ == "__main__":
    main()

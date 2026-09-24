# Constrained-decoding inference (vLLM)

Sample offline-inference script for running a trained model against this
repo's own eval sets (`multitask/data/eval/*.jsonl`) with grammar-constrained
JSON decoding, so the model's output is guaranteed on-schema at generation
time instead of being hoped for and cleaned up after.

## Why a schema is needed at all

Training examples repeat the target JSON schema in prose (see
`multitask/schemas.py`'s per-task instructions), but prose is not a
guarantee — a model can still emit malformed JSON, extra/missing keys, or
values outside a closed vocabulary. vLLM (and other engines, via
XGrammar/Outlines/Guidance) can constrain decoding to only ever produce
tokens consistent with a JSON Schema, closing that gap at inference time.

`multitask/target_schema.py` (single source of truth for every task's
target shape — see its module docstring) exports exactly that schema via
`json_schema(task)`, so the grammar used at inference and the shape used to
canonicalize training targets never drift apart.

## Files

- `vllm_structured_infer.py` — run a model over one eval dataset with
  `json_schema(task)` attached as a per-example decoding constraint, write
  `{id: completion}` out as JSON. See its module docstring for the vLLM
  API-version note (`StructuredOutputsParams` vs. the older
  `GuidedDecodingParams`) and for what the grammar can and can't enforce.

## Usage

```bash
python -m multitask.inference.vllm_structured_infer 2022n2c2_track2_uw \
    --model Qwen/Qwen2.5-7B-Instruct --backend xgrammar --limit 50 \
    --out completions.json

python -m multitask.eval.run_eval 2022n2c2_track2_uw completions.json
```

Swap `2022n2c2_track2_uw` for any dataset name under `multitask/data/eval/`
(drop the `.jsonl`).

## Caveats

The grammar constrains JSON *shape*, not semantic correctness:

- Entity/span `"text"` fields are left unconstrained strings — the grammar
  doesn't force them to be substrings of the input note. Span validity is
  checked downstream in `convert_back/*`, not at decode time.
- Cross-field enum constraints (e.g. SDOH's `{"role", "name", "value"}`
  triples, where `"value"`'s valid set depends on `"name"`) can't be
  expressed in a plain JSON Schema, so the grammar won't catch a
  syntactically valid but semantically wrong pairing. The sample script
  calls `target_schema.validate()` on every parsed completion specifically
  to catch this class of issue — read its printed summary.
- **`max_tokens` is a correctness parameter, not just a cost knob.** Gold
  target lengths vary by >25x across tasks (measured on test splits,
  tiktoken cl100k, assistant turn only: i2b2 concept/relation p95 ≈ 9953
  tokens / max 17645; SDOH p95 ≈ 560 / max 1016). Under grammar-constrained
  decoding, hitting the cap mid-object leaves the JSON unclosed — that
  fails to parse and scores as a total miss, worse than unconstrained
  decoding's occasional short-but-closed partial credit. `DEFAULT_MAX_TOKENS`
  in the script is sized per-task off those measurements; the script also
  reports how many completions hit the cap (`finish_reason == "length"`) so
  a bad choice shows up as a truncation rate instead of masquerading as a
  low-quality model result.

This script is not runnable in a sandbox without `vllm` installed and a
model available; it's meant to be copied into an environment that has both.

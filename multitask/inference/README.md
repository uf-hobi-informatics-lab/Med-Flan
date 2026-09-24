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
  Its `run()` accepts an optional `llm=` to reuse an already-loaded model
  across multiple calls instead of reloading per dataset.
- `run_inference_eval.ipynb` — the multi-task driver: loads the model once,
  loops `run()` over every dataset under `multitask/data/eval/` (skipping
  the one stale/empty file), then feeds each dataset's completions straight
  into `multitask.eval.run_eval.run_eval()` — that dataset's own real
  scorer, official where one exists. Prints each dataset's full native
  output (JSON metrics dict, or the official script's own table) plus a
  best-effort single-number summary row per dataset — deliberately not a
  unified leaderboard score, since the tasks genuinely use different
  metrics on different scales. See its own first cell and "Known caveats"
  section for the full rationale and gotchas (`max_tokens`, discharge-me's
  literal header-format requirement, grammar-vs-semantic-correctness).

## Usage

Single dataset, from the command line:

```bash
python -m multitask.inference.vllm_structured_infer 2022n2c2_track2_uw \
    --model Qwen/Qwen2.5-7B-Instruct --backend xgrammar --limit 50 \
    --temperature 0 --enable-thinking false \
    --sampling-kwarg repetition_penalty=1.1 --sampling-kwarg seed=0 \
    --out completions.json

python -m multitask.eval.run_eval 2022n2c2_track2_uw completions.json
```

`--temperature`/`--top-p`/`--top-k` and repeatable `--sampling-kwarg KEY=VALUE`
(any other `vllm.SamplingParams` field) are all optional — default is greedy
decoding (`--temperature 0`, **not** vLLM's own default of `1.0`; see the
module docstring for why). `--enable-thinking {auto,true,false}` toggles the
Qwen3-style chat-template reasoning switch; needs a vLLM build whose
`LLM.chat()` accepts `chat_template_kwargs` (confirmed present in 0.19.0,
absent in 0.6.6.post1) — silently ignored with a printed note otherwise.
All of these are also `run()` keyword arguments for the Python API.

Swap `2022n2c2_track2_uw` for any dataset name under `multitask/data/eval/`
(drop the `.jsonl`).

Every task at once: open `run_inference_eval.ipynb`, set `MODEL`/`BACKEND`/
`LIMIT` in its Config cell, and run top to bottom.

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

This script (and the notebook that drives it over every task) is not
runnable in a sandbox without `vllm` installed and a model available;
they're meant to be copied into an environment that has both.

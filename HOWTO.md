# HOWTO: running inference and evaluation

Practical run-book for generating model completions and scoring them. For
the *why* behind the eval numbers and the per-task scorer status table, see
`README.md`'s "Running an eval" section; for the *why* behind the inference
design (grammar-constrained decoding, retry/fallback, batching), see
`multitask/inference/README.md` and each script's module docstring — this
file is the "what do I actually type" companion to both.

## Which script do I want?

| Goal | Script |
|---|---|
| Generate completions for every eval dataset, one model, one command | `multitask.inference.run_generation` |
| Generate completions for **one** dataset, more low-level control | `multitask.inference.vllm_structured_infer` |
| Compare vLLM output against plain `transformers.generate()` (e.g. checking Ouro's early-exit) | `multitask.inference.native_infer` |
| Generate + score `2018_n2c2_track2_paragraph` (paragraph-chunked, no `data/eval/*.jsonl`) | see "Paragraph-window inference" below |
| Score one dataset's completions, or self-check every scorer against gold | `multitask.eval.run_eval` |
| Score a whole `run_generation` output directory into one CSV | `multitask.inference.run_eval_csv` |

All of these assume `vllm`/`transformers`/a GPU are actually available —
they are not runnable in a sandbox; copy into an environment that has both.

## Inference

### Full sweep — `run_generation.py` (recommended for a real run)

Loads the model **once**, generates over every dataset in `DATASET_SPECS`,
writes `<out-dir>/<model_tag>__<dataset>.json` per dataset (`{id:
completion}`, ready for `run_eval.py` / `run_eval_csv.py` with no reshaping).

```bash
python -m multitask.inference.run_generation \
    --model /path/to/checkpoint --backend xgrammar \
    --out-dir outputs/completions/
```

```bash
# quick smoke test on two datasets, 20 examples each
python -m multitask.inference.run_generation \
    --model /path/to/checkpoint --datasets 2018_n2c2_track1 biosses --limit 20 \
    --out-dir outputs/smoke/
```

```bash
# sample instead of greedy-decode, with retries actually able to produce
# something different on a re-draw
python -m multitask.inference.run_generation \
    --model /path/to/checkpoint --temperature 0.3 --top-p 0.95 \
    --retry-temperature 0.3 \
    --out-dir outputs/sampled/
```

All CLI options:

| Flag | Default | Meaning |
|---|---|---|
| `--model` (required) | — | HF Hub id or local checkpoint path. `model_tag()` derives the output filename prefix from this — see its docstring for the naming rule. |
| `--backend` | `xgrammar` | Structured-output backend (`xgrammar`, `guidance`, `outlines`, `auto`). |
| `--root` | `multitask/huggingface_format/` | Directory of per-dataset HF `Dataset`s (`load_from_disk`). |
| `--datasets` | every entry in `DATASET_SPECS` | Restrict to these dataset names (space-separated). Unknown names raise immediately. |
| `--limit` | none | Per-dataset example cap — use while testing, not for real numbers. |
| `--out-dir` (required) | — | Where `<model_tag>__<dataset>.json` files land. |
| `--temperature` | `0.0` | Sampling temperature for every dataset in this run. `0.0` = greedy/deterministic (this repo's reproducible-eval default; vLLM's own default is `1.0`). |
| `--top-p` | `1.0` | Nucleus sampling threshold. |
| `--top-k` | `-1` (disabled) | Top-k sampling cutoff. |
| `--sampling-kwarg KEY=VALUE` (repeatable) | — | Any **other** `vllm.SamplingParams` field (`repetition_penalty`, `frequency_penalty`, `min_p`, `seed`, `ignore_eos`, ...), applied on top of `GEN_OVERRIDES`/`DEFAULT_GEN`'s per-dataset `sampling_kwargs` for every dataset in the run. Use `--temperature`/`--top-p`/`--top-k` for those, not this. |
| `--retries` | `2` | Re-draw attempts for an "obviously problematic" sample (lone surrogate / bad JSON / schema violation / degenerate repetition) before storing it as-is. `0` disables. |
| `--retry-temperature` | none (= same as `--temperature`) | Temperature used **only** for retry/fallback attempts. Matters at `--temperature 0`: a retry at temperature 0 reproduces the identical broken output, so set this to e.g. `0.2`–`0.4` to actually give a retry a chance. |
| `--fallback-model` | none | A second, typically larger model to escalate to for a sample still bad after every primary retry. Built lazily — costs nothing if never needed. |
| `--fallback-backend` | = `--backend` | Structured-output backend for the fallback model. |
| `--fallback-retries` | `1` | Attempts on the fallback model before giving up and storing whatever it produced. |
| `--batch-size` | none (whole round in one call) | Cap examples per `llm.chat()` call. Set `1` to reproduce fully-sequential behavior — useful for A/B-testing the batched default (must be byte-identical at `temperature=0`). |
| `--enable-thinking` | `false` | Qwen3-style chat-template reasoning toggle (`auto`/`true`/`false`). `false` matches training targets (no `<think>` block); needs a vLLM build with `chat_template_kwargs` support in `LLM.chat()` (silently ignored otherwise). |
| `--max-model-len` | `32768` | Engine-level max prompt+output tokens. Must exceed the longest prompt in your data (discharge-me ~14.5k, 2018_n2c2_track2 ~12.5k) or vLLM errors on those examples. |
| `--gpu-memory-utilization` | `0.95` | Fraction of GPU memory vLLM may use. |
| `--tensor-parallel-size` | none (1 GPU) | Shard the model across N GPUs. |

`--datasets`, per-dataset `max_tokens`, and per-dataset `sampling_kwargs`
come from `MAIN_DATASET_SPECS`/`OOD_DATASET_SPECS`/`DEFAULT_GEN`/
`GEN_OVERRIDES` at the top of `run_generation.py` — edit those in-file for a
permanent per-task change; use the CLI flags above for a one-off sweep-wide
override.

### Single dataset — `vllm_structured_infer.py` (more low-level control)

```bash
python -m multitask.inference.vllm_structured_infer 2022n2c2_track2_uw \
    --model Qwen/Qwen2.5-7B-Instruct --backend xgrammar --limit 50 \
    --temperature 0.3 --top-p 0.95 --enable-thinking false \
    --disable-any-whitespace \
    --sampling-kwarg repetition_penalty=1.05 --sampling-kwarg seed=0 \
    --out completions.json

python -m multitask.eval.run_eval 2022n2c2_track2_uw completions.json
```

Same sampling/retry/fallback/batching/engine flags as `run_generation.py`
above (this is the module those are forwarded to), plus:

| Flag | Default | Meaning |
|---|---|---|
| `dataset` (positional, required) | — | Name matching `multitask/data/eval/<dataset>.jsonl`. |
| `--max-tokens` | per-task `DEFAULT_MAX_TOKENS` | Override uniformly for every task in this run. See "max_tokens" in the tuning cookbook below — this is a correctness knob, not just cost. |
| `--disable-any-whitespace` | off | Compiles the JSON grammar with `any_whitespace=False`, forbidding whitespace between JSON tokens — fixes the failure mode where grammar-constrained decoding loops on tabs/spaces until `max_tokens` without ever opening `{`. Needs `xgrammar` or `guidance` backend + a vLLM build that supports the config key; silently dropped otherwise. `run_generation.py` always sets this. |
| `--pipeline-parallel-size` | none (1) | Pipeline-parallel stages. |
| `--max-num-seqs` | vLLM default `256` | Max concurrently-batched sequences. Raise once you have KV-cache budget for it. |
| `--llm-kwarg KEY=VALUE` (repeatable) | — | Any other `vllm.LLM` constructor arg (`enforce_eager=true`, `swap_space=4`, ...). |
| `--out` | `completions.json` | Output path. |

Python API (e.g. from a notebook, reusing one loaded model across many
`run()` calls — this is what `run_generation.run_sweep()` does internally):

```python
from multitask.inference.vllm_structured_infer import _make_llm, run

llm = _make_llm("Qwen/Qwen2.5-7B-Instruct", "xgrammar",
                 disable_any_whitespace=True, llm_kwargs={"max_model_len": 32768})
completions = run("2022n2c2_track2_uw", model=None, llm=llm,
                   temperature=0.3, top_p=0.95, sampling_kwargs={"repetition_penalty": 1.05},
                   out_path="completions.json")
```

### Native (`transformers`) comparison — `native_infer.py`

Not a production path — a one-off comparison tool (no grammar-constrained
decoding, no retry/fallback, no batching; see its module docstring for why).
Same `{id: completion}` output shape, so it scores with the same
`run_eval.py`:

```bash
python -m multitask.inference.native_infer 2018_n2c2_track1 \
    --model ByteDance/Ouro-1.4B --trust-remote-code --limit 20 \
    --temperature 0 --out completions_native.json
```

| Flag | Default | Meaning |
|---|---|---|
| `dataset` (positional, required) | — | Same dataset-name convention as `vllm_structured_infer.py`. |
| `--model` (required) | — | HF Hub id or local path. |
| `--trust-remote-code` | off | Required for custom modeling code not merged into `transformers` (e.g. `ByteDance/Ouro-*`). |
| `--limit` | none | Cap examples — this script is one-at-a-time, no batching, so keep it small. |
| `--max-tokens` | per-task default | Same meaning as the vLLM script. |
| `--device-map` | `auto` | `transformers` device placement. |
| `--torch-dtype` | `auto` | Model dtype. |
| `--temperature` | `0.0` | `0.0` = greedy (`do_sample=False`), matching the vLLM script's default. |
| `--top-p` / `--top-k` | `1.0` / none | Sampling controls. |
| `--enable-thinking` | `false` | Same meaning as the vLLM script. |
| `--out` | none | Output path. |

### Paragraph-window inference + eval (`2018_n2c2_track2_paragraph`)

This dataset is HF-only (no `multitask/data/eval/*.jsonl`) and keyed
one-per-paragraph, so it needs its own pipeline instead of `run_eval.py`:

```python
from datasets import load_from_disk
from multitask.inference.vllm_structured_infer import run
from multitask.eval.paragraph_eval import run_paragraph_eval

test_split = load_from_disk("multitask/huggingface_format/2018_n2c2_track2_paragraph")["test"]
completions = run(test_split, model="Qwen/Qwen2.5-7B-Instruct", backend="xgrammar")
run_paragraph_eval(completions)  # merges paragraphs back into whole-note .ann, then scores
```

`run()`'s `dataset` argument accepts an already-loaded `datasets.Dataset`
directly. `run_paragraph_eval` merges every document's paragraph completions
into one whole-note `.ann` (offset-shifted, ids renumbered) and scores it
against the same gold/`brat_scorer` the doc-level `ner_re_medication` task
uses, so the two are directly comparable.

## Evaluation

### One dataset, or self-check everything — `run_eval.py`

```bash
python -m multitask.eval.run_eval 2022n2c2_track2_uw completions.json

# offset-free doc-level addon (5 span-carrying tasks only) alongside the real score
python -m multitask.eval.run_eval 2022n2c2_track2_uw completions.json --doc-level
```

Run with no arguments to self-check **every** dataset end-to-end (gold fed
back as its own prediction — slow, this is a pre-flight/CI check, not
routine use):

```bash
python -m multitask.eval.run_eval
```

Python API:

```python
from multitask.eval.run_eval import run_eval, run_doc_level_eval
run_eval("2018_n2c2_track2", {"100130": '{"drugs": [...], "unlinked": [...]}', ...})
```

Dataset names and what each one's completion should look like are in
`README.md`'s status table / "Running an eval" section — this file doesn't
repeat that.

### Whole `run_generation` output directory → one CSV — `run_eval_csv.py`

Scores every `<model_tag>__<dataset>.json` file a `run_generation.py` sweep
wrote, using each task's real scorer directly (not `run_eval.py`'s
print-only path), into one long-format CSV (`dataset, metric, value`):

```bash
python -m multitask.inference.run_eval_csv \
    --completions-dir outputs/completions/ \
    --out outputs/eval_results.csv
```

| Flag | Default | Meaning |
|---|---|---|
| `--completions-dir` (required) | — | Directory from `run_generation.py`. |
| `--model-tag` | most-recently-modified match per dataset | Disambiguate when the directory holds runs from more than one model/checkpoint — matches `model_tag()`'s filename prefix. |
| `--datasets` | every `DATASET_SPECS` entry present in the directory | Restrict to these. |
| `--out` (required) | — | Output `.csv` path. |

Pivot to a wide table downstream if wanted:
`pandas.read_csv(...).pivot(index="dataset", columns="metric", values="value")`.

## Parameter-tuning cookbook

Ordered roughly by how often you'll actually reach for each one.

**`temperature` / `top_p` / `top_k`** — start at `temperature=0` (greedy) for
any run whose numbers you intend to compare across checkpoints; it's
reproducible and both this repo's scripts default to it (vLLM's own default
is `1.0`, deliberately overridden here). Move to sampling
(`--temperature 0.2`–`0.5`, `--top-p 0.9`–`0.95`) when you specifically want
retries to have a chance at producing something different (pair with
`--retry-temperature`, see below), or when you're studying output diversity
rather than scoring a fixed checkpoint. `top_k` is rarely needed alongside
`top_p`; leave it disabled (`-1`) unless a specific model is prone to
long-tail garbage tokens under `top_p` alone.

**`--retry-temperature`** — a footgun if left at the default while
`--temperature 0`: retries re-draw with the *same* greedy decode and
therefore reproduce the identical broken completion, silently wasting
compute. Whenever `retries > 0` (the default is `2`) and the main pass is
greedy, set `--retry-temperature 0.2`–`0.4` so a re-draw is actually a
different sample.

**`repetition_penalty` / `frequency_penalty`** — per-dataset in
`GEN_OVERRIDES`/`DEFAULT_GEN` (`run_generation.py`), or via
`--sampling-kwarg` for a one-off override. **`frequency_penalty` was
deliberately removed from this repo's defaults** — confirmed via a
side-by-side run to corrupt ~21% of extracted IE span text with a spurious
leading `"-"` character not present in the source document (see the comment
above `DEFAULT_GEN` in `run_generation.py`); removing it took
`2022n2c2_track2_mimic/uw`'s Combined F1 from ~0.50 to ~0.78. Don't
re-introduce it on the IE tasks without re-measuring. `repetition_penalty`
(currently `1.05` everywhere) is safer to nudge — try `1.0` (off) first if
you suspect it's suppressing recall on a specific task, `1.1`–`1.15` if a
model is looping/repeating.

**`max_tokens`** — a *correctness* parameter under grammar-constrained
decoding, not just a cost knob: hitting the cap mid-object leaves the JSON
unclosed, which fails to parse and scores as a total miss (vs.
unconstrained decoding, which can still land on a short-but-closed object).
Per-task defaults in `vllm_structured_infer.DEFAULT_MAX_TOKENS` are sized
off measured gold-target lengths (tiktoken cl100k, ~1.5x p95); both
`run_generation.py`'s sweep and `vllm_structured_infer.py`'s single-dataset
run print how many completions hit the cap (`finish_reason == "length"`) —
if you see a nonzero truncation rate, raise `--max-tokens` for that task
before concluding the model is bad. `--limit`-bounded test runs shouldn't
need to touch this.

**`--retries` / `--fallback-model`** — `retries=2` (the default) catches
most one-off degenerate samples cheaply. Reach for `--fallback-model` (a
larger/more general instruct model) only when a *fine-tuned, possibly
undertrained* checkpoint has a persistent bad-sample rate that retries alone
don't fix — it's built lazily so it costs nothing when unused, but loading
it needs enough combined GPU memory for both models (lower
`--gpu-memory-utilization` on each, or split across GPUs with
`--tensor-parallel-size`/`CUDA_VISIBLE_DEVICES`).

**`--batch-size`** — leave unset (whole round in one `llm.chat()` call) for
throughput. Only set it to `1` when you need to A/B-test the batched default
against strictly sequential behavior (e.g. chasing a suspiciously low
score) — at `temperature=0` the two must produce byte-identical completions,
since batching only changes throughput, not per-example sampling; a
difference there is a real bug to chase, not batching noise.

**`--enable-thinking`** — leave at `false` (the default) for any model
fine-tuned on this repo's training targets (`render_messages` never emits a
`<think>` block). Only use `auto`/`true` when deliberately evaluating a
stock, not-fine-tuned reasoning model in its native mode.

**`--disable-any-whitespace`** — leave it on (both scripts already default
to it / always set it) unless you're on an older vLLM that doesn't support
the config key, in which case it's silently dropped with a printed note —
no action needed either way.

**Engine args (`--max-model-len`, `--gpu-memory-utilization`,
`--tensor-parallel-size`, `--max-num-seqs`)** — decided once at model load,
not per request. Raise `--tensor-parallel-size` (up to 2 in production, 1 in
the single-GPU test env) mainly to grow the KV-cache pool for tasks with
very long generations (`concept_assertion_relation` runs 17k–32k tokens).
Raise `--max-num-seqs` once you have KV-cache headroom for more concurrency.
Only lower `--max-model-len` below its auto-derived default to trade context
length for KV-cache room — never raise it needlessly, and never drop it
below your longest prompt (discharge-me ~14.5k tokens, 2018_n2c2_track2
~12.5k) or those examples error out.

## End-to-end example

```bash
# 1. generate over every dataset, sampled (not greedy), with a fallback model
python -m multitask.inference.run_generation \
    --model /path/to/checkpoint --backend xgrammar \
    --temperature 0.3 --top-p 0.95 --retry-temperature 0.3 \
    --fallback-model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 \
    --out-dir outputs/completions/

# 2. score everything into one CSV
python -m multitask.inference.run_eval_csv \
    --completions-dir outputs/completions/ \
    --out outputs/eval_results.csv

# 3. spot-check one dataset's real scorer output directly
python -m multitask.eval.run_eval 2022n2c2_track2_uw \
    outputs/completions/<model_tag>__2022n2c2_track2_uw.json
```

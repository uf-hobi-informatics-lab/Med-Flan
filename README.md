# Med-Flan

A multi-task clinical-NLP instruction-tuning framework: train a single
decoder-only LLM on the union of clinical and biomedical NLP datasets, framing
every task uniformly as text-to-text generation, and evaluate per-task on each
dataset's own held-out split with its own official (or closest available)
metric. Each example is prefixed with a two-level `<<Category: Task>>` tag so
one model can be trained and scored across extraction, classification, QA,
similarity regression, and summarization in a shared text-to-text contract.

This repo is a fork/derivative of
[`uf-hobi-informatics-lab/Med-Flan`](https://github.com/uf-hobi-informatics-lab/Med-Flan).

## What it does

- **One model, 26 datasets, all text-to-text.** A single decoder-only LLM is
  trained on a union of clinical/biomedical NLP datasets (n2c2/i2b2 shared
  tasks, MedNLI/BioNLI/NLI4CT, MedQA, MedMCQA, PubMedQA, MMLU-Pro-Health,
  LitCovid, emrQA-msquad, Discharge Me!, SemEval-2015, UMN abbreviation, and
  more). Structured tasks produce a JSON target; summarization produces plain
  text.
- **5 categories, 25 task tags** — `Information_Extraction`,
  `Classification`, `Question_Answering`, `Similarity_Regression`,
  `Summarization` — with the two-level `<<Category: Task>>` tag split across
  chat roles: reusable category framing in the system turn, dataset-specific
  instruction (label set + JSON output schema) in the user turn. See
  [`multitask/design.md`](multitask/design.md) for the rationale.
- **~716K merged training examples, ~474M tokens** (cl100k). See the
  per-dataset breakdown and balance analysis in
  [`multitask/README.md`](multitask/README.md).
- **One source of truth for every target shape.** `multitask/target_schema.py`
  declares each task's JSON shape (canonical key order, closed vocabularies,
  required keys) once, and drives both training-target canonicalization and
  grammar-constrained decoding at inference — so what a model is trained to
  emit and what it's allowed to emit at decode time can never drift.
- **Per-task evaluation with real scorers.** Each dataset is scored against
  its own released test split using the task's official scorer where one
  exists (BRAT standoff, n2c2 XML/TSV/float-lines, ROUGE/BLEU, SQuAD-style
  EM/F1) and a self-implemented equivalent otherwise. A gold-as-prediction
  self-check (`python -m multitask.eval.run_eval`) verifies the whole
  converter/scorer wiring before any model output is scored.

## Repository layout

| Path | What it is | Read for detail |
|---|---|---|
| [`multitask/`](multitask/README.md) | Core pipeline: per-dataset builders, two-level task-tag prompt rendering, JSON target schemas, format converters (JSON → BRAT/XML/TSV), scorers, vLLM grammar-constrained inference, and the combined eval dispatcher. | [`multitask/README.md`](multitask/README.md), [`multitask/design.md`](multitask/design.md) |
| [`med_flan_training/`](med_flan_training/README.md) | SFT and GRPO training framework (derived from `alignment-handbook`) for three model families — HRM-Text-1B, Ouro-2.6B-Thinking, Qwen3-8B — with prompt-loss-weighted variants, weighted-JSON loss, a 2-stage curriculum, and RunAI/SLURM launch scripts. | [`med_flan_training/README.md`](med_flan_training/README.md) |
| [`inference_and_eval/`](inference_and_eval/HOWTO.md) | A cluster-deployment copy of the multitask inference and eval code (HuggingFace-native inference paths added), with a practical "what do I type" run-book. | [`inference_and_eval/HOWTO.md`](inference_and_eval/HOWTO.md) |
| [`raw_datasets/`](raw_datasets/sample_data) | Raw source data for every dataset. **Not committed** (see `.gitignore`) — only `raw_datasets/sample_data/` (1-5 records per dataset) is tracked, so the on-disk schema is visible without a download. | "Data sources" table in [`multitask/README.md`](multitask/README.md#data-sources) |

## Quickstart

Download every raw dataset into its matching `raw_datasets/<name>/` folder
first (see [Data sources](multitask/README.md#data-sources)), then build and
self-check the data pipeline:

```bash
python -m multitask.build_unified_dataset   # merged train + per-dataset eval jsonl
python -m multitask.build_hf_datasets       # per-dataset HF DatasetDict, saved separately
python -m multitask.hf_dataset_stats        # token-count stats, regenerates the stats appendix
python -m multitask.eval.run_eval           # full self-check: every real scorer, gold-vs-gold
```

Generate completions for every eval dataset with one model, then score the
whole output directory into one CSV (assumes a GPU + `vllm`/`transformers`
environment — copy into a host that has both):

```bash
python -m multitask.inference.run_generation \
    --model /path/to/checkpoint --backend xgrammar --out-dir outputs/completions/
python -m multitask.inference.run_eval_csv \
    --completions-dir outputs/completions/ --out eval_results.csv
```

## Where to go next

- **Data, task taxonomy, data sources, per-task eval status, data balance,
  known limitations, HF dataset stats** → [`multitask/README.md`](multitask/README.md)
- **Two-level task-tag mechanism and JSON target schemas (design rationale)**
  → [`multitask/design.md`](multitask/design.md)
- **Inference + evaluation run-book** (which script to use, full CLI flag
  tables, parameter tuning, end-to-end examples) →
  [`inference_and_eval/HOWTO.md`](inference_and_eval/HOWTO.md) and
  [`multitask/inference/README.md`](multitask/inference/README.md)
- **SFT / GRPO training, the three model families, curriculum staging, and
  cluster (RunAI/SLURM) launch** → [`med_flan_training/README.md`](med_flan_training/README.md)

## Environment

Two environments, by task:

- **Data building, eval, and self-check** run in the local project venv (see
  `CLAUDE.md`) as `python -m multitask.*` modules — no package install needed.
- **GPU inference and training** run inside NVIDIA vLLM containers, which
  already ship vLLM, PyTorch, and a pinned `transformers` (do **not** install
  or upgrade `transformers` from pip — use the container's version):
  - Ouro model family → `nvcr.io/nvidia/vllm:26.04-py3`
  - Qwen + HRM-Text family → `nvcr.io/nvidia/vllm:26.08-py3`

  Install the HuggingFace ecosystem packages on top of the container image with
  the root [`requirement.txt`](requirement.txt) (`trl`, `accelerate`,
  `datasets<5.0`, `bitsandbytes`, `einops`, `packaging`, `tensorboard`, `wandb`,
  `safetensors`, `peft`). Training-only extras are in
  [`med_flan_training/requirements.txt`](med_flan_training/requirements.txt).

## License

MIT License — see [`LICENSE`](LICENSE). Copyright © 2026
uf-hobi-informatics-lab.

## Citation

Per-dataset source papers for every task are listed in the "Citations"
section of [`multitask/README.md`](multitask/README.md#citations). Please cite
the original dataset/shared-task papers for any task you use.

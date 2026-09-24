# design.md

Design notes for the two-level task taxonomy and system/user prompt split
introduced into `multitask/schemas.py`. This is the "design plan" referenced
from `README.md`.

## Motivation

This repo trains a single decoder-only LLM on ~15 clinical/biomedical NLP
tasks — NER, relation/event extraction, classification, QA, similarity
regression, summarization — across 17-20 datasets, treating every task
uniformly as text generation: `<<tag>>\n<instruction>\nINPUT: ...\nOUTPUT: ...`.

Before this change, all prompt structure lived in one flat dict,
`INSTRUCTIONS: {task_tag: instruction_string}`, keyed by a single-level tag
`<<TASK:tag>>`. Every instruction string mixed two things that should be
separable:

1. **Generic task-family framing** — "extract entities/events with their
   attributes and return JSON", "classify the input into one of these
   labels", "answer the question" — reusable across every task with the
   same output-contract shape.
2. **Dataset-specific formatting** — the exact label set, JSON key names,
   attribute enumerations for *this* corpus's annotation scheme.

Both lived in the system message; the user message was bare input text.
There was also no category/grouping concept anywhere in the codebase or
README — only a flat task-tag↔dataset table and a few inconsistent,
undocumented prose clusters.

The goal: make the "treat everything as generation" framing explicit at the
prompt level, not just the pipeline level. A decoder model conditions on a
**category**-level system framing it can learn once across many tasks, then
reads a **task**-level, dataset-specific output contract in the user turn
right next to the input it applies to. Track both levels as metadata so they
can later drive template selection, sampling/curriculum weighting, or
analysis, without touching anything that depends on the existing low-level
`task` string (eval dispatch, convert_back schema selection).

## Task Taxonomy

### Category → Task mapping

Five categories, grouping the 15 task tags by **output-contract shape**
(what kind of thing the model is asked to produce), not by clinical
subdomain or source-annotation format — the same axis Super-NaturalInstructions
/ FLAN-style task-type taxonomies use.

| Category (code / display) | Task tags in category |
|---|---|
| `information_extraction` / `Information_Extraction` | `ner_re_medication` (NER_RE), `medication_event` (Medication_Event), `sdoh_event` (SDoH), `family_history` (Family_History), `disorder_normalization` (Disorder_Normalization), `concept_assertion_relation` (Concept_Assertion_Relation) |
| `classification` / `Classification` | `nli` (NLI), `multilabel_cls` (Multilabel_Topic), `cohort_eligibility` (Cohort_Eligibility), `abbreviation_disambiguation` (Abbreviation_Disambiguation) |
| `question_answering` / `Question_Answering` | `mcqa` (MCQA), `mcqa_explain` (MCQA_Explain), `qa_yesnomaybe` (QA_YesNoMaybe) |
| `similarity_regression` / `Similarity_Regression` | `sts_regression` (STS_Regression) |
| `summarization` / `Summarization` | `summarization` (Summarization) |

All 15 pre-existing snake_case `task` keys are unchanged — they remain the
JSON `"task"` field and the eval-routing key used by `run_eval.py` and
`convert_back/`. Only new `category` and `display` metadata is added per
task, in `multitask/schemas.py`'s `TASKS` dict.

Categories intentionally **not** created:

- **"BRAT-format tasks"** (`ner_re_medication`, `medication_event`,
  `sdoh_event`) — an annotation-tooling/file-format fact about the source
  data, not a task-type distinction. All three are already
  `information_extraction`.
- **"Document-level tasks"** (`DOC_LEVEL_TASKS` in
  `multitask/hf_dataset_stats.py`) — an example-granularity/build-engineering
  property (one example per document vs. per-criterion splitting like
  `cohort_eligibility`), orthogonal to task type.
- **A separate entity-normalization/linking category** — `disorder_normalization`
  bundles NER + CUI-linking, but its output contract
  (`{"entities": [{"text","cui"}]}`) is the same "list of structured spans"
  shape as the other IE tasks.

### Ambiguous placements (open questions, resolved)

Two task tags didn't fall cleanly into one category. Resolved with the
project owner rather than silently defaulted:

- **`qa_yesnomaybe` (PubMedQA) → `Question_Answering`.** Its `decision`
  field (yes/no/maybe) is a closed label, classification-shaped like
  `nli`/`cohort_eligibility`. But the task's overall framing (question +
  context → answer, plus a free-text `long_answer`) is QA-shaped like
  `mcqa`/`mcqa_explain`, and matches PubMedQA's native leaderboard framing.
  Genuinely a hybrid — documented inline in `schemas.py` rather than
  pretending the boundary is clean. If category-level behavior control
  (e.g. category-specific decoding constraints) is ever built on top of this
  taxonomy, `qa_yesnomaybe` may need task-level, not just category-level,
  handling.
- **`abbreviation_disambiguation` (umn_abbreviation) → `Classification`.**
  Word-sense disambiguation: pick the correct sense/expansion given context.
  The candidate expansions aren't enumerated in-prompt the way `nli`'s or
  `multilabel_cls`'s label sets are (open-vocabulary output), but the task
  is fundamentally a labeling task over a fixed (if per-abbreviation)
  candidate set — categorized as `Classification`.

Also noted, non-blocking: `cohort_eligibility` structurally resembles
NLI-over-long-context (premise = notes, hypothesis = one criterion) more
than it resembles `multilabel_cls`, but a single-dataset `entailment`-only
category wasn't judged worth the extra category at this task count — it
stays in `Classification`.

## Tag Format

### Tag string spec

`<<{category_display}: {task_display}>>` — colon + space between category
and task, underscores (not spaces) inside each display name, e.g.:

```
<<Information_Extraction: SDoH>>
<<Information_Extraction: Family_History>>
<<Question_Answering: MCQA>>
<<Classification: NLI>>
```

Underscores were chosen over spaces (`Information Extraction: SDoH`) to keep
each display name a single token-friendly identifier and match the format
the project owner specified directly.

This fully **replaces** the old single-level `<<TASK:tag>>` format — it is
not kept anywhere for backward compatibility. Nothing in the codebase parses
the tag string out of rendered text (see below), no model has been trained
on the old format yet (this is pre-training data prep), and the instruction
text itself changed in this same pass, forcing a full dataset rebuild
regardless — so preserving the old string would add inconsistency risk with
no compatibility benefit.

### Decoupling from the `task` routing field

The tag string is a **model-facing/analysis signal only** — it is never
parsed out of `text`/`messages` by any code in this repo. All routing uses
the JSON `"task"` field directly:

- `eval/run_eval.py`'s `_task_of()` reads `json.loads(f.readline())["task"]`
  from the eval jsonl — never the tag text.
- Every dispatch branch in `run_eval()` (`_run_brat`, `_run_cohort_eligibility`,
  `_run_family_history`, `_run_sts`, `_SIMPLE_TASK_OF_DATASET`,
  `PLAIN_TEXT_TASKS`) is keyed on the unchanged low-level `task` string or
  `dataset` name.
- `convert_back/*` modules each hard-code one fixed JSON schema; which
  module gets imported is decided by `run_eval.py`'s dispatch on `task`, not
  by anything module-internal.

This means the whole two-level-tag change is confined to `schemas.py` (plus
the two small metadata additions in `build_hf_datasets.py` and
`hf_dataset_stats.py` below) and is safe by construction — no eval/routing
code needed to change.

## Prompt Structure: System vs. User Split

### `CATEGORIES` and `TASKS` data structures

`INSTRUCTIONS` (flat `{task: instruction}`) is replaced by two dicts in
`multitask/schemas.py`:

```python
CATEGORIES = {
    "information_extraction": {
        "display": "Information_Extraction",
        "system": "You are performing information extraction from clinical or "
                  "biomedical text: identify the entities, events, relations, or "
                  "attributes described in the task-specific instructions that "
                  "follow, and return them as the JSON object those instructions "
                  "specify.",
    },
    # ... classification, question_answering, similarity_regression, summarization
}

TASKS = {
    "sdoh_event": {
        "category": "information_extraction",
        "display": "SDoH",
        "instruction": (
            'Extract social-determinants-of-health events (e.g. Tobacco, '
            'Alcohol, Drug, Employment, LivingStatus) with their role-filler '
            'entities and attribute values. Return JSON {"events": [...]}.'
        ),
    },
    # ... one entry per task tag, same instruction text as the old
    # INSTRUCTIONS dict, each annotated with category + display.
}
```

`category`'s value is the canonical snake_case key into `CATEGORIES` (single
source of truth for the display string — never duplicated).

### `render()` / `render_messages()` changes

```python
def tag(task: str) -> str:
    """The two-level model-facing tag, e.g. '<<Information_Extraction: SDoH>>'."""
    meta = TASKS[task]
    category = CATEGORIES[meta["category"]]
    return f"<<{category['display']}: {meta['display']}>>"

def render(task, input_text, target) -> str:
    meta = TASKS[task]
    category = CATEGORIES[meta["category"]]
    return (
        f"{tag(task)}\n"
        f"{category['system']}\n"
        f"{meta['instruction']}\n"
        f"INPUT: {input_text}\n"
        f"OUTPUT: {_target_str(task, target)}"
    )

def render_messages(task, input_text, target) -> list:
    meta = TASKS[task]
    category = CATEGORIES[meta["category"]]
    return [
        {"role": "system", "content": f"{tag(task)}\n{category['system']}"},
        {"role": "user", "content": f"{meta['instruction']}\nINPUT: {input_text}"},
        {"role": "assistant", "content": _target_str(task, target)},
    ]
```

`render()` stays a straight concatenation of the same four pieces
`render_messages()` splits across roles, so the flat-text pipeline
(`build_unified_dataset.py`/`token_stats.py`) and the chat-messages pipeline
(`build_hf_datasets.py`) can never drift apart.

The dataset-specific instruction (label sets, JSON schema) is deliberately
restated in the **user** turn on every example, immediately before `INPUT:`,
rather than living only in a memorized system prompt — the output contract
should sit next to the generation site, while the system turn carries only
the small, category-level, reusable "which family of task is this" framing:
the part any decoder model can learn once across every task in the
category.

### Before/after example

`sdoh_event` (Information_Extraction):

Before:
```
system: <<TASK:sdoh_event>>
        Extract social-determinants-of-health events (e.g. Tobacco, Alcohol,
        Drug, Employment, LivingStatus) with their role-filler entities and
        attribute values. Return JSON {"events": [...]}.
user:   <note text>
```

After:
```
system: <<Information_Extraction: SDoH>>
        You are performing information extraction from clinical or biomedical
        text: identify the entities, events, relations, or attributes
        described in the task-specific instructions that follow, and return
        them as the JSON object those instructions specify.
user:   Extract social-determinants-of-health events (e.g. Tobacco, Alcohol,
        Drug, Employment, LivingStatus) with their role-filler entities and
        attribute values. Return JSON {"events": [...]}.
        INPUT: <note text>
```

The same pattern applies uniformly across all 15 tasks: the dataset-specific
sentence moves from the system turn to the top of the user turn, unchanged
in content; only the category-level framing is new.

## Metadata Schema

### `schemas.example()` fields

```python
def example(task, dataset, split, locator, input_text, target):
    return {
        "task": task,
        "category": TASKS[task]["category"],
        "dataset": dataset,
        "split": split,
        "id": locator,
        "text": render(task, input_text, target),
        "messages": render_messages(task, input_text, target),
    }
```

One field added: `category` (canonical snake_case key). Everything else is
unchanged.

### Threading through the pipeline

- **`build_unified_dataset.py`** — no code change. It writes the full
  `example()` dict via `_write_jsonl` for every train/dev/eval row, so
  `category` flows through automatically to `train_unified.jsonl`,
  `dev_unified.jsonl`, and every `multitask/data/eval/<dataset>.jsonl`.
- **`build_hf_datasets.py`** — `_build_split()` previously narrowed each row
  to `{"messages", "task", "dataset", "id"}`; now also includes `"category"`.
- **`hf_dataset_stats.py`** — reads `TASKS.get(task, {}).get("category", "")`
  and `TASKS.get(task, {}).get("instruction", "")` (replacing the old
  `INSTRUCTIONS.get(task, "")` lookup), and includes `category` in both the
  JSON stats report and the regenerated README stats table.

### Confirmed no-op: `run_eval.py` dispatch

`eval/run_eval.py` needed **no changes**. Its dispatch (`_task_of()`, the
`run_eval()` if/elif chain, `_SIMPLE_TASK_OF_DATASET`, `PLAIN_TEXT_TASKS`) is
entirely keyed on the pre-existing low-level `task` string or `dataset`
name — never on `category` or on tag text. Every `convert_back/*` module is
similarly unaffected: each hard-codes one fixed JSON schema and never reads
`category`.

## README Changes

- Fixed the dangling "See the design plan for full background..." reference
  (previously pointing at nothing) to link this file.
- Updated the `Layout` section's description of `multitask/schemas.py`.
- Added a `## Task Taxonomy` section (between `## Per-task evaluation
  status` and `## Data balance`, outside the auto-generated
  `<!-- BEGIN/END: multitask dataset stats -->` block so
  `hf_dataset_stats.py` regeneration never clobbers it): the full
  category → task → dataset table, the two ambiguous-placement notes above,
  and a reconciliation note tying the new taxonomy to the three pre-existing
  informal prose clusters (extraction/regression vs.
  QA/classification/summarization in Data balance; "BRAT datasets" and
  "document-level tasks" in Known limitations / `hf_dataset_stats.py`) so
  the new section supersedes/organizes them rather than contradicting them.

## Migration Plan

1. Rewrite `multitask/schemas.py` — `CATEGORIES` + `TASKS`, updated
   `render()`/`render_messages()`/`example()`. Single file, but changes the
   literal text of every rendered example.
2. Update `multitask/build_hf_datasets.py` (`category` in `_build_split()`).
3. Update `multitask/hf_dataset_stats.py` (`TASKS` import, `category` in
   report + generated table, comment clarifying `DOC_LEVEL_TASKS` is
   orthogonal to the new taxonomy).
4. Update `README.md` (Task Taxonomy section, dangling link fix) and add
   this file.
5. **Rebuild all artifacts** — every already-built file
   (`train_unified.jsonl`, `dev_unified.jsonl`, `multitask/data/eval/*.jsonl`,
   `multitask/huggingface_format/*`) is stale, since instruction text
   changed:
   ```bash
   python -m multitask.build_unified_dataset
   python -m multitask.build_hf_datasets
   python -m multitask.hf_dataset_stats
   python -m multitask.eval.run_eval
   ```
   `run_eval`'s self-check round-trips gold target JSON through
   converters/scorers and never touches prompt/instruction text, so every
   "verified" status in the README's evaluation-status table should
   reproduce identically — this step is a regression check, not an
   expected-diff step.

## Open Questions / Future Work

- **Using `category`/`task` metadata for curriculum or sampling weighting**
  (the README's [Data balance](./README.md#data-balance) section already
  identifies severe example-count and token-volume imbalance across
  datasets) — the new `category` field makes category-stratified
  temperature-scaled resampling straightforward to add to
  `build_unified_dataset.py`, but this is not implemented here.
- **Few-shot template selection keyed on category/task** — e.g. selecting
  in-context examples from the same category, or applying category-specific
  decoding constraints — unblocked by this metadata but out of scope for
  this change.
- **`qa_yesnomaybe`'s hybrid categorization** (see Ambiguous placements
  above) may need revisiting if category-level behavior control is ever
  built on top of this taxonomy.

---

# Proposal: Fixing doc-level training-data scarcity for the BRAT datasets

**Status: proposal for discussion only — not an approved implementation
plan.** No code changes have been made. This is a write-up of options and
supporting data to decide direction, coordinated with a sibling session
doing `track2-mimic-uw-redesign` work (which overlaps the SDOH side below)
before anything here is implemented.

## Context

Training the three BRAT-based datasets (`ner_re_medication`,
`medication_event`, `sdoh_event`) at **document level** (one full note =
one training example) significantly reduces the number of training
examples relative to the number of gold annotations packed inside each
note. This section evaluates three candidate fixes — subtask decomposition,
paragraph-based window chunking, and resampling — with subtask
decomposition checked first (window/section chunking risks damaging
relation extraction) and paragraph-based windowing validated empirically
via an explicit cross-paragraph relation-rate check before being trusted.

## Framing against the README's own documented imbalance

The [Data balance](./README.md#data-balance) section already documents
this same scarcity problem from the other direction (example-count and
token-volume imbalance across the whole joint mixture, not just within the
BRAT cluster), and its numbers are the authoritative baseline for token
costs below:

- **Example-count imbalance, confirmed exact in the README: 2,134:1**
  (pubmedqa 211,269 train examples vs. family_history 99). The two BRAT
  datasets this proposal targets (`ner_re_medication` 303,
  `medication_event` 350) sit at the same tiny end of that spectrum.
- **Real per-dataset train-token totals** (tiktoken cl100k, full rendered
  example including instruction + target JSON, not just raw doc words):
  `ner_re_medication` (2018_n2c2_track2) ≈ **1.7M** tokens (mean 5,463.5
  tok/example × 303), `medication_event` (2022n2c2_track1) ≈ **0.67M**
  tokens (mean 1,902.2 tok/example × 350), against a **~474M-token total**
  corpus. Both are individually under 0.4% of total training volume despite
  being two of the paper's core clinically-rich extraction tasks.
- A rough raw-document-word estimate of these totals undercounts by ~2x —
  the gap is the instruction text plus the target JSON itself (~200
  annotations/doc is not free), which a words-only estimate omits. All
  token-cost figures below use the real per-example totals, not a
  words-only estimate.

## Data gathered

| dataset | train/dev/test docs | avg annotations/doc (T+E lines) | avg doc length |
|---|---|---|---|
| 2018_n2c2_track2 | 303 / – / 202 | ~200 | ~1,900 words / ~3,000-3,500 tok |
| 2022n2c2_track1 | 350 / 50 / 100 | ~26 (avg **18.0** events/doc, measured exactly) | ~615 words / ~1,000-1,200 tok |
| 2022n2c2_track2 (SDOH, mimic) | 1,316 / 188 / 373 | ~12 | ~43 words (already a short snippet, not a full note) |
| 2022n2c2_track2 (SDOH, uw) | 1,751 / 259 / 518 | ~15 | ~40 words |

Additional measurements taken specifically to answer "how many relations
cross paragraphs," over all 505 2018_n2c2_track2 docs (303 train + 202
test), splitting on blank-line paragraph boundaries:

- **59,810 gold relations checked; only 146 (0.24%) have their two arguments
  in different paragraphs.** Paragraph-based windowing is safe for this
  dataset's relation axis — it would lose well under 1% of relations.
- Avg **35.7 paragraphs/doc** (median 32), but most are short/empty
  (headers, blank fields); avg **9.7 paragraphs/doc actually contain ≥1
  entity** (median 9) — i.e. splitting into non-empty paragraph windows
  would turn 303 train docs into roughly **~2,900 training examples** (a
  ~9.7x multiplier), each a much smaller, easier-to-learn target than
  today's ~200-annotation whole-note target.
- For 2022n2c2_track1, avg **18.0 events/doc** (measured over all 500
  train+dev+test docs) — this is the relevant multiplier for the subtask
  decomposition option, not paragraph chunking (see below).

No existing chunking/windowing utility exists anywhere in `multitask/`
(confirmed by grep) — this would be new.

## Options evaluated

### 1. Subtask decomposition (recommended primary lever)

**Key design choice: additive, not a replacement.** Keep the existing full
end-to-end example (whole note in, complete events/relations JSON out) for
every doc exactly as it is today, and *add* extra training examples that
each cover one pipeline stage of that same task on the same doc. This
matters because `run_eval.py`/`convert_back/*.py` score the model on the
full end-to-end task only (see `multitask.eval.run_eval`'s dispatch,
unchanged by this proposal) — if training only ever showed the model
decomposed sub-views, it would never have practiced producing the exact
full-task output shape it's scored on. Training on both means the
full-task skill is preserved while extra, cheaper-to-get-right signal is
layered on top.

**Tier A — doc-level pipeline-stage subtasks (~2-3x, do this first):**
mirror the natural NER→RE / detection→classification pipeline stages as
their own whole-doc examples, added alongside the existing full-task one:
- `2018_n2c2_track2`: (i) full task [unchanged], (ii) NER-only ("extract
  every Drug/Strength/Route/.../ADE span, no relations"), (iii) RE-only
  ("given these already-extracted entities, link each attribute span to its
  Drug"). 303 train docs → ~909 examples (**3x**), and each sub-example is
  a strictly easier target than today's single ~200-annotation blob.
  **RE-only's input contract needs to be pinned down explicitly:** the
  entities fed into this sub-example's prompt would be the *gold* spans
  (that's the only thing training has), but at real pipeline inference
  time (if RE-only were ever chained after a separate NER pass rather than
  scored standalone) the upstream entities would be model-*predicted*,
  with their own errors. Training only on gold-entity inputs risks a
  train/inference mismatch if this is ever used as a literal two-stage
  pipeline rather than as an auxiliary training signal on top of the
  unchanged full end-to-end task. Since the full end-to-end task (which
  never assumes pre-extracted entities) remains the only thing actually
  scored, this is containable — but worth stating as an explicit
  assumption rather than leaving it implicit, and worth revisiting if
  RE-only's own accuracy is ever reported/used standalone rather than just
  as a training-signal booster.
- `2022n2c2_track1`: (i) full task [unchanged], (ii) trigger-detection-only
  ("find every Disposition/NoDisposition/Undetermined mention, no
  attributes"), (iii) attribute-classification-only ("given this note and
  its already-identified triggers, classify each one's 5 attributes"). 350
  train docs → ~1,050 examples (**3x**).

**Tier B — per-event/per-relation subtasks (much bigger multiplier,
optional stretch once Tier A is measured):** instead of one doc-level
"classify all attributes" example, emit one example per event: "given this
note and this one trigger ('Prozac'), classify Action/Certainty/
Temporality/Actor/Negation." Attributes attach directly to the
trigger/event id (`A5 Action E3 Start`), not to a separately-located span
elsewhere in the note, so — unlike relations — there's no spatial
"cross-boundary" failure mode to worry about here; it's a pure re-framing
with zero information loss. Multiplier: **~18x** for track1's
attribute-classification signal alone (measured avg events/doc), on top of
the unchanged 350 full-task and 350 Tier-A examples. The 2018_track2
relation axis could get the same per-relation treatment.

**Risk to flag either way:** Tiers A/B mean the same document's text is
now the context for 3-20+ examples in one epoch. That's fine for gradient
signal but worth watching for (a) token-budget accounting (a doc's text is
no longer "read once per epoch"), and (b) the mixture's relative weight
versus the many much-larger simple-classification datasets already in the
joint training mix — more BRAT examples changes that balance and may need
re-tuning regardless of which tier is used.

**Where this plugs in:** purely a train-split concern — only whichever
builder function assembles the training JSONL/HF dataset needs new
sub-example emission (in addition to, not instead of, today's single
`yield` per doc); dev/test splits, `run_eval.py`, `convert_back/*.py`, and
`brat_scorer.py` all stay untouched since scoring is always against the
full end-to-end task.

**Task-tag accounting:** the README's top-line "16 task tags, 21 datasets"
count would need updating if any Tier A/B sub-task gets its own *task tag*
rather than reusing the parent task's tag. Recommendation: reuse the
existing task tag (`ner_re_medication` for both new
`2018_n2c2_track2_ner_only`/`_re_only` dataset names, `medication_event`
for both new `2022n2c2_track1_trigger_only`/`_attr_only` dataset names) —
these are sub-*views* of the same underlying task category, not new tasks,
so the tag count (16) stays unchanged and only the dataset count (21) grows
by however many new dataset names are added. This should be called out
explicitly in whatever PR implements this, and the README's Data balance
table/dataset count updated alongside the code change — not left to drift
out of sync.

**Plumbing check (researched, not yet built):** `task` in
`multitask/schemas.py`'s `example()` is **not free-form** — `_check_task()`
raises on anything not already in the `TASKS` registry, so each new
sub-task (`ner_only_medication`, `re_only_medication`,
`trigger_detection_only`, `attribute_classification_only`, ...) needs a new
`TASKS` entry (category + display + instruction text). No existing builder
currently yields more than one distinct `task` from one dataset name; the
closest prior art is `multitask/builders/sts_2019_t1.py`, which factors
shared parsing into `_iter_rows()` and exposes two thin top-level builder
functions, each with its **own dataset name** (`2019_n2c2_track1` /
`2019_n2c2_track1_cls`) and its own `REGISTRY` row in both
`build_hf_datasets.py` and `build_unified_dataset.py`. That's the pattern
to reuse here — one new dataset name + REGISTRY row per sub-task (e.g.
`2018_n2c2_track2_ner_only`, `2018_n2c2_track2_re_only`,
`2022n2c2_track1_trigger_only`, `2022n2c2_track1_attr_only`), sharing the
existing parse/entity-extraction helpers already in each dataset's builder
— rather than trying to emit multiple tasks under the same dataset name.

**Important footgun found and avoided by that choice:** `run_eval.py`
determines which scorer to dispatch for a dataset by reading the **task
name of the first line** of that dataset's built eval `.jsonl` (via
`_task_of`, from `build_unified_dataset.py`'s own registry). If sub-task
examples were mixed into the *same* dataset name's train split, that's
fine (dev/test wouldn't be touched), but it's a sharp edge worth stating
explicitly: sub-task examples must never be added to a dataset name that
also has a dev/test entry in `build_unified_dataset.py`'s registry unless
carefully guarded by `if split == "train"`. Giving each sub-task its own
dataset name (as recommended above) sidesteps this entirely — new sub-task
dataset names simply never get a `run_eval.py` dispatch entry or a dev/test
`REGISTRY` row, so there's no way for them to be accidentally scored or to
corrupt an existing dataset's eval dispatch.

**Token cost:** `ner_re_medication`'s real baseline is ~1.7M train tokens.
Tier A's ~3x multiplier projects to **~5.1M** tokens for `ner_re_medication`
and **~2M** for `medication_event` (from its ~0.67M baseline) — a real 3x
increase in training compute/token volume, not just a free example-count
multiplier — worth weighing against Tier B or paragraph chunking (which
shrink the context per example rather than repeating the full doc 3x).
Even at ~5.1M + ~2M combined, this remains under 1.5% of the ~474M-token
total corpus, so the mixture-balance risk flagged above is about
gradient/example-count weighting, not raw token budget.

**Tier A → Tier B is a builder tweak, not new registry surface:** since
Tier B (per-event attribute classification) is the same underlying task as
Tier A's doc-level "classify all attributes," it can reuse the *same* new
dataset name/`TASKS` entry (e.g. `2022n2c2_track1_attr_only`) — the only
difference is whether the builder yields one example per doc (all triggers
given at once) or one example per event (one trigger at a time). So trying
Tier A first and upgrading to Tier B later, if warranted, doesn't require
touching `TASKS`/`REGISTRY` again.

### 2. Paragraph-based window chunking (secondary lever, for entity/relation detection itself)

Only relevant to 2018_n2c2_track2, since that's the one dataset where the
*detection* step itself (not just attribute classification) is choked by
doc-level granularity — SDOH docs are already short snippets (~40-80
words), and track1 docs are much smaller than track2's (~600-1,200 tok vs
~3,000-3,500 tok), so the case for chunking those is weak to nonexistent.

For 2018_n2c2_track2: split each note on blank-line paragraph boundaries,
keep only entities/relations fully inside a paragraph, discard the ~0.24%
of relations that straddle two paragraphs (measured above — negligible
loss). This requires new machinery that doesn't exist yet:
- A chunking helper (new module, e.g. `multitask/chunk_utils.py`) to split
  text into paragraph spans and project entities/relations onto them.
- A **reassembly step** before scoring: `run_eval.py`/`convert_back/*.py`
  currently assume one example id == one whole document
  (`schemas.example(..., locator=txt_path.stem)`, used directly as
  `gold_dir / f"{doc_id}.txt"`); chunked examples would need ids like
  `f"{doc_stem}#{i}"` plus stored offsets, and predictions from all chunks
  of one doc would need to be merged (offset-shifted back to global
  coordinates) into a single reconstructed `.ann` before `brat_scorer.py`
  scores it. `brat_scorer.py` itself needs no changes — only the
  merge-before-score step is new.

### 3. Resampling (weak, complementary only)

Oversampling/upweighting the scarce dataset's examples in the joint
multi-task mixture doesn't add information (no new annotations), just
changes gradient weighting so it isn't drowned out by much larger
simple-task datasets. Worth doing regardless of which of the above is
chosen, but shouldn't be treated as *the* fix.

## Recommendation

1. Add Tier A doc-level subtask examples first (additive, full task kept),
   for 2018_n2c2_track2 and 2022n2c2_track1 — safe, zero information loss,
   no reassembly machinery needed, ~3x each.
2. **Validate with an ablation before treating Tier A as a fix, not just an
   example-count multiplier.** An increase in raw example/token count is
   not evidence of improved model quality by itself — the actual test is
   to train two runs (with vs. without the Tier A additions) on identical
   full-task eval (`run_eval.py`'s existing `ner_re_medication`/
   `medication_event` scoring, unchanged by this proposal) and compare
   held-out F1. Only once that ablation shows a real gain should Tier A be
   considered validated, and only then should Tier B (bigger multiplier,
   bigger redundancy risk) be considered.
3. Measure with a real model before deciding on Tier B (per-event, ~18x) —
   bigger multiplier but bigger redundancy/mixture-balance risk (see
   above), so treat as a follow-up once Tier A's effect is known rather
   than doing both at once.
4. Layer paragraph-based window chunking on top for 2018_n2c2_track2's
   entity/relation *detection* step specifically — now justified by the
   measured 0.24% cross-paragraph relation rate — since that's the one axis
   subtask decomposition alone doesn't fix (the pipeline still needs to
   find the drug/attribute spans in the first place from limited
   whole-note examples).
5. Leave 2022n2c2_track2 (SDOH)/mimic-uw untouched in this proposal — docs
   are already near mention-granularity so chunking has little to offer,
   and a sibling session is actively doing `track2-mimic-uw-redesign` work
   there; touching it now risks colliding with that.
6. Add mixture-level resampling/upweighting as a cheap complement once (1)
   and (4) are decided, independent of raw example count.
7. Update the README's Data balance table and top-line dataset count
   alongside whichever implementation PR lands this (see "Task-tag
   accounting" above) — don't let the documented counts drift out of sync.

## Implementation status

**Tier A built** for `ner_re_medication` (`2018_n2c2_track2_ner_only`/`_re_only`)
and `medication_event` (`2022n2c2_track1_trigger_only`/`_attr_only`) --
4 new task tags (`ner_re_medication_ner_only`, `ner_re_medication_re_only`,
`medication_event_trigger_only`, `medication_event_attr_only`, each with
`"Subtask"` in its rendered tag display so it's unmistakably distinct from
the parent end-to-end tag), tracked via a `SUBTASK_OF` mapping in
`multitask/schemas.py`. Per owner decision, these are registered **only**
in `build_hf_datasets.py` (own `multitask/huggingface_format/<name>/`
directory each, train split only) and deliberately **excluded** from
`build_unified_dataset.py`'s merged `train_unified.jsonl` -- inclusion in a
training run is an explicit opt-in (`concatenate_datasets()` the desired
subtask directories alongside the parent's), not automatic.

`sdoh_event` (2022n2c2_track2) remains intentionally excluded from Tier A
(reasoning unchanged from "Options evaluated" above: docs already near
mention-granularity, and collision risk with the sibling
`track2-mimic-uw-redesign` session). `family_history` was never evaluated
either way and stays out of scope.

**Paragraph-based window chunking built** for `ner_re_medication`
(`2018_n2c2_track2_paragraph`, task tag `ner_re_medication_paragraph`) --
a mechanism distinct from Tier A: paragraphs are genuinely new, distinct
text spans (blank-line split via `multitask/chunk_utils.py`), not a
relabeling of the same whole-note text, tracked via a separate
`PARAGRAPH_OF` mapping (kept apart from `SUBTASK_OF` on purpose). Per
owner decision ("prepare all so we can adjust during experimentation"),
**both `train` and `test` splits are built** (train-only would have
matched Tier A's convention, but the owner explicitly asked for
flexibility to experiment with either eval strategy later) -- `dev` is
then auto-carved from `test` the same way the parent dataset gets its
`dev`. Despite `test` existing as HF data, it is registered **only** in
`build_hf_datasets.py`; `build_unified_dataset.py` and `run_eval.py` are
untouched, so doc-level prediction/scoring remains the sole authoritative
eval for `ner_re_medication` -- the model must still solve (and be scored
on) the whole-document task, matching the owner's explicit goal. This
sidesteps the reassembly-before-scoring machinery ("Open questions" below
originally anticipated) entirely, since nothing produced from this
dataset is ever scored via chunked inference.

If a paragraph-level diagnostic is wanted later, the recommended approach
is a post-hoc slice of the model's existing whole-document predictions:
after the unchanged full-doc `convert_back` reconstructs a predicted
`.ann`, use `chunk_utils.project_doc_onto_span` to slice both gold and
predicted `.ann` into the same paragraphs, then score each slice with the
existing `brat_scorer`. No separate paragraph-scale inference run, no new
scorer, no train/inference mismatch -- not yet built, flagged here as
future work only.

**Still not done: the ablation.** This change makes Tier A's and the
paragraph-window dataset's data buildable and loadable, but does not
itself run or schedule the train-with-vs-without comparison against real
`run_eval.py` F1 that the Recommendation section above lists as the
required gate before either counts as validated rather than just a bigger
example count. Tier B remains unstarted.

**Both levers extended to `concept_assertion_relation` (2010_i2b2_va)**
once its raw data became available on disk. Tier A: `concept_assertion_relation_ner_only`/
`_re_only` (`2010_i2b2_va_ner_only`/`_re_only`, train-only, 170 examples
each), exact same shape/convention as `ner_re_medication`'s. Section-window:
`concept_assertion_relation_section` (`2010_i2b2_va_section`, train+test+dev
built) -- this dataset's raw `.txt` format has **zero blank-line breaks**
(confirmed empirically), so `chunk_utils.paragraph_spans` is degenerate on
it (one span per doc); a new sibling helper, `chunk_utils.section_spans`,
splits on ALL-CAPS `"SECTION HEADER :"` lines instead (e.g. `"HISTORY OF
PRESENT ILLNESS :"`), falling back to one whole-doc span for the 1/426
docs (0.2%) with no detected header line. Measured before implementing:
**0/9413 (0.00%) gold relations cross a section boundary**, ~10.4 non-empty
sections/doc on average (170 train docs -> 1,666 built section examples,
256 test docs -> 2,868). Since i2b2's concepts/relations are plain dicts
(not `BratDoc` dataclasses), the projection logic (`_iter_sections`) lives
in `multitask/builders/i2b2_2010.py` itself rather than reusing
`chunk_utils.project_doc_onto_span` (which is BratDoc-specific) -- only the
span-computation helper is shared. Same eval-strategy reasoning as
`ner_re_medication_paragraph`: `build_unified_dataset.py`/`run_eval.py` are
untouched, doc-level scoring stays authoritative for
`concept_assertion_relation`.

**Corrected 2026-09-23, two bugs in the section-window path fixed:**

1. `_SECTION_HEADER` originally matched ALL-CAPS headers only
   (`^[A-Z][A-Z0-9 /\-,()\.]{2,60}:\s*$`). i2b2 2010 discharge summaries mix
   ALL-CAPS headers with Title/mixed-case ones ("Discharge Date :",
   "Service :", "cc :") that pattern silently skipped -- measured, it
   caught 8,141 of the 14,076 real header lines and left 32/426 docs
   (7.5%) with zero detected headers, collapsing to one whole-doc span for
   each. Case-insensitive matching (`chunk_utils.py`) finds the other 5,935
   and drops the no-header count to 1/426. Numbers above are already the
   post-fix ones (pre-fix: ~7.1 sections/doc, 1,059/1,952 train/test
   examples). Cost: a handful of narrative sentences that happen to end in
   a colon (e.g. "On 2014-04-04, labs were as follows :") now also match
   and create a spurious split -- rare and tolerated, since a wrong split
   only ever drops content at that boundary rather than corrupting a kept
   section.
2. `_iter_sections` reused each concept's **document-level** id (`c0` is
   the first concept in the whole note, not the first concept in the
   section) inside a section-scoped target. Since a section only ever sees
   its own text, 85.9% of section examples had a target whose first
   concept id wasn't `c0` (e.g. `['c13','c14']` with no `c0..c12` in
   sight) -- unrecoverable from the input, so this trained id
   hallucination directly. Fixed by renumbering `c0..cN-1` local to each
   section and remapping relations through the same map;
   `multitask/builders/i2b2_2010.py`'s `_iter_sections` docstring has the
   full rationale for why this is safe (nothing downstream keys off
   section ids matching document-level ones). Relations still resolve
   correctly before and after (0 dangling either way) -- this was an
   id-hallucination-in-training problem, not a broken-relations problem.

Building `2010_i2b2_va`'s own end-to-end dataset for the first time (data
was previously absent from disk) also activated its `run_eval.py`
self-check path for the first time -- added `"2010_i2b2_va"` to
`_self_check_all()`'s dataset list; the gold-fed-back-as-prediction
self-check now runs and passes (concept F1 0.9996, assertion accuracy 1.0,
relation F1 1.0 -- the small concept-F1 gap is a pre-existing
`convert_back/i2b2_2010.py` reanchor ceiling, not something introduced
here).

## Open questions before this becomes a real implementation plan

- Whether attribute/relation subtask examples should reuse the full note as
  context (higher token cost per example) or a trimmed window around the
  target trigger (cheaper, but reintroduces a "how much context" tuning
  question).
- Exact paragraph-chunking boundary rule for 2018_n2c2_track2 (blank-line
  split, as measured above, vs. note-section-header split) — blank-line
  split is what was measured; header-based splitting wasn't tested and may
  behave differently.
- Whether the id/offset/reassembly machinery for paragraph chunking is
  worth building now vs. only after subtask decomposition alone is tried
  and measured against a real model.
- Whether Tier A's ~3x is enough on its own, or whether Tier B's ~18x is
  worth the added redundancy risk — recommend deciding this only after
  Tier A has been tried, not up front.
- How the joint-mixture sampling weights should be re-tuned once these
  datasets have 3x (or more) as many examples as today, so they don't
  crowd out the other simple-task datasets disproportionately.

"""
Separate evaluation pipeline for the 2018 n2c2 Track 2 paragraph-window
dataset (ner_re_medication_paragraph / 2018_n2c2_track2_paragraph -- see
multitask.builders.ner_re_2018_t2 and design.md's paragraph-chunking
section).

Why this needs its own pipeline rather than reusing inference_and_eval.eval.run_eval:
run_eval() assumes one completion id == one whole document (it reads
`gold_dir / f"{doc_id}.txt"` directly). A model trained only on the
paragraph-window task produces one completion PER PARAGRAPH
(id = f"{doc_stem}__p{i}"), so before the existing doc-level scorer can run,
every document's paragraph completions have to be merged back into one
whole-note .ann, offset-shifted from paragraph-local to document-global
coordinates -- exactly the "reassembly step" design.md's paragraph-chunking
section flagged as the missing piece for scoring this dataset at inference
time.

That merge (write_anns_from_paragraphs / _merge_paragraph_anns) lives in
multitask.convert_back.ner_re_2018_t2, next to to_ann(), since it's tightly
coupled to that function's exact .ann line format; this module is just the
thin CLI/entry-point wrapper around it, mirroring inference_and_eval.eval.run_eval's
own CLI conventions.

Once merged, scoring reuses the SAME whole-note GOLD_DIR and brat_scorer as
the doc-level ner_re_medication task (multitask.convert_back.ner_re_2018_t2.
score_dirs) -- so a model trained only at paragraph level is directly
comparable to one trained on the whole-note task; this is not a separate or
weaker metric, it's the same metric with an extra assembly step in front of
it. run_doc_level_eval-style offset-free scoring is intentionally NOT
offered here -- brat_scorer's real, offset-based scorer already works fine
post-merge, unlike the raw per-paragraph JSON (which has no offsets at all
to compare against gold's whole-doc offsets).

Usage:
    from inference_and_eval.eval.paragraph_eval import run_paragraph_eval
    run_paragraph_eval({"100035__p0": '{"drugs": [...], "unlinked": [...]}', ...})

Or from the command line, with completions as a JSON file of
{paragraph_id: completion}:
    python -m inference_and_eval.eval.paragraph_eval completions.json

With no arguments, self-checks by feeding the paragraph dataset's own gold
targets back through the merge+score pipeline (see convert_back.
ner_re_2018_t2._self_check_paragraph) -- exercises the reassembly machinery
end to end without needing a real model.

Getting completions for this dataset: it has no multitask/data/eval/*.jsonl
file (paragraph datasets are HF-only and additive, never wired into
build_unified_dataset.py -- see design.md and multitask/build_hf_datasets.py),
so point inference_and_eval.inference.vllm_structured_infer.run() at the built HF
dataset directly instead of by name (its `dataset` argument accepts an
already-loaded datasets.Dataset, not just an eval-dir jsonl path):

    from datasets import load_from_disk
    from inference_and_eval.inference.vllm_structured_infer import run
    test_split = load_from_disk(
        "multitask/huggingface_format/2018_n2c2_track2_paragraph"
    )["test"]
    completions = run(test_split, model="...", backend="xgrammar")
    run_paragraph_eval(completions)
"""
import json
import sys

from inference_and_eval.convert_back.ner_re_2018_t2 import score_paragraph_predictions


def run_paragraph_eval(completions, split="test"):
    """completions: {paragraph_id: raw completion string or already-parsed
    dict}, paragraph_id = f"{doc_stem}__p{i}". Merges back to whole-note
    .ann and prints the same brat_scorer output the doc-level task uses.
    Returns nothing -- score_paragraph_predictions prints the metrics."""
    score_paragraph_predictions(completions, split)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with open(sys.argv[1], encoding="utf-8") as f:
            run_paragraph_eval(json.load(f))
    else:
        from inference_and_eval.convert_back.ner_re_2018_t2 import _self_check_paragraph
        _self_check_paragraph()

"""
Combined per-dataset eval runner: given a dataset name and real model
completions (raw strings, keyed by the `id` field of that dataset's
multitask/data/eval/<dataset>.jsonl), converts back to native format where
needed and invokes that dataset's real scorer -- the single entry point
described (but not yet built) in the project README.

This module only wires together what already exists in convert_back/ and
eval/*.py; it adds no new scoring logic of its own beyond completion
parsing and dispatch. See README.md's "Running an eval" section and the
per-task status table for what each dataset's scorer actually verifies and
which numbers are an inherent ceiling rather than a converter bug.

Usage:
    from multitask.eval.run_eval import run_eval
    run_eval("2018_n2c2_track2", {"100130": '{"entities": [...], "relations": [...]}', ...})

Or from the command line, with completions as a JSON file of {id: completion}:
    python -m multitask.eval.run_eval 2018_n2c2_track2 completions.json

completions may be a raw string (as real model output would be) for every
task; JSON-shaped tasks are parsed leniently (first {...} object found, via
multitask.eval.simple_tasks.parse_completion) rather than requiring an
exact json.loads -- a real model's output is never guaranteed to be
perfectly-formed JSON on the first try, and the conservative behavior for
anything unparseable is to score it as a miss, not to crash the run.
"""
import json
import sys
import tempfile
from pathlib import Path

from multitask.eval.simple_tasks import DATASETS as SIMPLE_TASK_DATASETS
from multitask.eval.simple_tasks import parse_completion

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "multitask" / "data" / "eval"

# dataset name -> filename stem under multitask/data/eval/ (differs for the
# two names that contain characters not used in filenames on disk)
_FILENAME_OVERRIDES = {"covid (LitCovid)": "covid_LitCovid"}

# task tag -> dataset name(s) that use it, for the simple-scorer group
_SIMPLE_TASK_OF_DATASET = {name: task for task, names in SIMPLE_TASK_DATASETS.items() for name in names}


def _eval_path(dataset):
    return EVAL_DIR / f"{_FILENAME_OVERRIDES.get(dataset, dataset)}.jsonl"


def _task_of(dataset):
    path = _eval_path(dataset)
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())["task"]


def _parsed_target(completion):
    """completion: raw model output string, or already-parsed dict/list
    (self-checks pass gold targets straight through)."""
    if isinstance(completion, (dict, list)):
        return completion
    return parse_completion(completion)


def _run_brat(dataset, completions, module):
    from multitask.io_utils import read_text

    predictions = [
        (doc_id, _parsed_target(completion), read_text(module.GOLD_DIR / f"{doc_id}.txt"))
        for doc_id, completion in completions.items()
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = module.write_anns(predictions, tmp)
        if hasattr(module, "score_dirs"):
            module.score_dirs(module.GOLD_DIR, out_dir)
        else:
            module._run_eval(module.GOLD_DIR, out_dir)


def _run_cohort_eligibility(completions):
    from multitask.convert_back import cohort_eligibility_2018_t1 as m

    predictions = [(locator, _parsed_target(c)) for locator, c in completions.items()]
    with tempfile.TemporaryDirectory() as tmp:
        m.write_xmls(predictions, tmp)
        m._run_eval(m.GOLD_DIR, tmp)


def _run_family_history(completions):
    from multitask.convert_back import family_history_2019_t2 as m

    predictions = [(doc_id, _parsed_target(c)) for doc_id, c in completions.items()]
    with tempfile.TemporaryDirectory() as tmp:
        s1_path, s2_path = m.write_tsvs(predictions, tmp)
        print("--- subtask 1 ---")
        m._run_eval(1, m.GOLD_S1, s1_path)
        print("--- subtask 2 ---")
        m._run_eval(2, m.GOLD_S2, s2_path)


def _run_sts(completions):
    from multitask.convert_back import sts_2019_t1 as m

    n = sum(1 for _ in open(m.GOLD_SCORES, encoding="utf-8"))
    # completion ids are "<test filename>#<line index>" -- reorder into the
    # test file's original line order (correlation-noconfidence.pl compares
    # line-by-line against the unmodified gold gs file); a missing index
    # (model didn't answer that pair) falls back to the scale midpoint, the
    # same conservative default parse_similarity() uses for unparseable text.
    by_index = {int(doc_id.rsplit("#", 1)[1]): c for doc_id, c in completions.items()}
    targets = [by_index.get(i) for i in range(n)]

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(m.to_score_lines(targets))
        system_path = f.name
    m._run_eval(m.GOLD_SCORES, system_path)
    Path(system_path).unlink(missing_ok=True)


def _run_concept_assertion_relation(completions):
    from multitask.convert_back import i2b2_2010 as m

    predictions = {doc_id: _parsed_target(c) for doc_id, c in completions.items()}
    print(json.dumps(m.score_dataset(predictions, split="test"), indent=2))


def run_eval(dataset, completions):
    """dataset: a name from README's status table (e.g. "2018_n2c2_track2",
    "MedNLI", "covid (LitCovid)"). completions: {id: raw completion string}.
    Prints that dataset's real scorer output; returns nothing (the scorers
    themselves print/return metrics in whatever shape their own official
    tool -- or multitask.eval.simple_tasks/discharge_me -- uses)."""
    task = _task_of(dataset)

    if task == "ner_re_medication":
        from multitask.convert_back import ner_re_2018_t2 as m
        return _run_brat(dataset, completions, m)
    if task == "medication_event":
        from multitask.convert_back import medication_event_2022_t1 as m
        return _run_brat(dataset, completions, m)
    if task == "sdoh_event":
        from multitask.convert_back import sdoh_2022_t2 as m
        return _run_brat(dataset, completions, m)
    if task == "cohort_eligibility":
        return _run_cohort_eligibility(completions)
    if task == "family_history":
        return _run_family_history(completions)
    if task == "sts_regression":
        return _run_sts(completions)
    if task == "concept_assertion_relation":
        return _run_concept_assertion_relation(completions)
    if task == "disorder_normalization":
        from multitask.convert_back import disorder_normalization_2015_t14 as m

        predictions = {doc_id: _parsed_target(c) for doc_id, c in completions.items()}
        print(json.dumps(m.score_dataset(predictions, split="test"), indent=2))
        return
    if task == "summarization":
        from multitask.eval.discharge_me import score_dataset
        print(json.dumps(score_dataset(completions, include_bertscore=False), indent=2))
        return
    if task in _SIMPLE_TASK_OF_DATASET.values() or dataset in _SIMPLE_TASK_OF_DATASET:
        from multitask.eval.simple_tasks import score_dataset
        print(json.dumps(score_dataset(_FILENAME_OVERRIDES.get(dataset, dataset), completions), indent=2))
        return
    raise ValueError(f"no scorer wired up for task {task!r} (dataset {dataset!r})")


def _self_check_all():
    """Every dataset, gold fed back as its own "prediction" -- exercises
    the full dispatch table end to end. Reuses each convert_back module's
    own _examples_with_raw_target() (the BRAT/XML/TSV/STS datasets) or reads
    gold straight out of multitask/data/eval/<name>.jsonl (everything else)
    -- the same two patterns each module's own standalone self-check already
    uses, so this adds no new gold-loading logic of its own. Slow (runs
    every dataset's real scorer); this is what CI/a pre-flight check would
    call, not routine use."""
    from multitask.convert_back import (
        cohort_eligibility_2018_t1, disorder_normalization_2015_t14,
        family_history_2019_t2, medication_event_2022_t1, ner_re_2018_t2,
        sdoh_2022_t2, sts_2019_t1,
    )

    raw_target_jobs = [
        ("2018_n2c2_track2", ner_re_2018_t2, "test"),
        ("2022n2c2_track1", medication_event_2022_t1, "dev"),
        ("2022n2c2_track2", sdoh_2022_t2, "dev"),
        ("2019_n2c2_track2", family_history_2019_t2, "test"),
        ("2019_n2c2_track1", sts_2019_t1, "test"),
        ("semeval_2015", disorder_normalization_2015_t14, "test"),
    ]
    for dataset, module, split in raw_target_jobs:
        print(f"\n=== {dataset} ===")
        completions = {ex["id"]: ex["target"] for ex in module._examples_with_raw_target(split)}
        run_eval(dataset, completions)

    print("\n=== 2018_n2c2_track1 ===")
    run_eval("2018_n2c2_track1", dict(cohort_eligibility_2018_t1._examples_with_raw_target("test")))

    for dataset in ("MedNLI", "MedQA-USMLE-US", "medmcqa", "pubmedqa", "covid (LitCovid)", "discharge-me", "umn_abbreviation"):
        print(f"\n=== {dataset} ===")
        completions = {}
        with open(_eval_path(dataset), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                completions[d["id"]] = d["messages"][-1]["content"]
        run_eval(dataset, completions)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        with open(sys.argv[2], encoding="utf-8") as f:
            run_eval(sys.argv[1], json.load(f))
    else:
        _self_check_all()

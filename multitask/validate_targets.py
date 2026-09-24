"""
Gate over the target contract declared in multitask/target_schema.py.

Historically the contract lived in the instruction prose, the builder code,
and convert_back independently, with nothing checking them against each
other. This walks every built example and asserts:

1. **Structure/vocabulary** -- every target validates against its
   `target_schema.py` spec. A builder emitting an unenumerated key or label
   is a hard failure, not silent training signal. This is also what catches
   any vocabulary value present in the data but missing from the spec.
2. **Coverage** -- every task in `SPECS` produced at least one example, and
   every (dataset, split) that produced any produced at least a floor
   count. Without this the gate reads as a clean pass when a corpus is
   missing from disk entirely: the dataset simply vanishes from the table.
3. **Spec not over-permissive** -- vocabulary values the spec *declares*
   but no example ever *uses*. Not a failure (train/test can legitimately
   differ), but each one needlessly widens the exported decoding grammar,
   so it is reported. This is the reverse of check 1, which covers the
   data-not-in-spec direction.
4. **Instruction covers the spec** -- every closed-vocabulary value the
   spec allows is mentioned in the task's instruction text, so the model is
   never expected to emit a label it was never told about. A warning, not a
   failure: a few vocabularies are deliberately open (CUIs, abbreviation
   expansions).

Exits non-zero if (1) or (2) fail.

Deliberately NOT checked here, so nobody mistakes this for full coverage:

- **Canonical key order.** `schemas._target_str()` canonicalizes on the way
  out and `messages` is constructed nowhere else in the repo, so any
  round-trip through a built example is canonical by construction -- an
  idempotence assertion here would be tautological. Key order is instead
  covered by the property tests over `canonicalize()` itself.
- **convert_back.** Nothing in `convert_back/` is imported or invoked, so
  this gate does not prove the third copy of the contract agrees -- only
  that the builders and the spec do. `to_ann` silently ignores keys it does
  not recognize, so a spec/converter divergence would not show up here.

Run with:
    python -m multitask.validate_targets
"""
import json
import sys
from collections import defaultdict

from multitask.build_hf_datasets import REGISTRY
from multitask.schemas import TASKS
from multitask.target_schema import (
    SPECS, instruction_gaps, observed_vocab, validate, vocabularies,
)

# A (dataset, split) that yields fewer than this is almost certainly a
# missing-corpus or bad-path problem rather than a real tiny split. The
# smallest real split in the registry is family_history train at 99.
MIN_EXAMPLES_PER_SPLIT = 10


def _walk():
    """Yields (dataset, split, task, parsed_target)."""
    for name, builder, splits in REGISTRY:
        for split in splits:
            try:
                for ex in builder(split):
                    if ex["task"] not in SPECS:
                        break  # whole dataset is unspecced; skip it cheaply
                    yield name, split, ex["task"], json.loads(ex["messages"][-1]["content"])
            except Exception as exc:
                # Must wrap the ITERATION, not the builder() call: every
                # registry entry is a generator function (the lambdas wrap
                # generator functions too), so builder(split) itself never
                # raises -- a missing corpus surfaces on first next().
                print(f"  ! {name}/{split}: builder raised {type(exc).__name__}: {exc}")


def main():
    violations = defaultdict(list)
    counts = defaultdict(int)
    tasks_seen = set()
    observed = defaultdict(dict)

    for dataset, split, task, target in _walk():
        counts[f"{dataset}/{split}"] += 1
        tasks_seen.add(task)
        for v in validate(task, target):
            violations[f"{dataset}/{split} [{task}]"].append(v)
        observed_vocab(task, target, observed[task])

    print("=== examples validated ===")
    for key in sorted(counts):
        print(f"  {counts[key]:7d}  {key}")

    print("\n=== 1. structural + vocabulary violations ===")
    if violations:
        for key in sorted(violations):
            items = violations[key]
            print(f"  {key}: {len(items)} violation(s)")
            for v in sorted(set(items))[:10]:
                print(f"      {v}")
    else:
        print("  none")

    print("\n=== 2. coverage ===")
    coverage_bad = []
    missing_tasks = sorted(set(SPECS) - tasks_seen)
    if missing_tasks:
        coverage_bad.append(f"tasks in SPECS that produced no examples at all: {missing_tasks}")
    for key, n in sorted(counts.items()):
        if n < MIN_EXAMPLES_PER_SPLIT:
            coverage_bad.append(f"{key}: only {n} example(s), below floor {MIN_EXAMPLES_PER_SPLIT}")
    if coverage_bad:
        for line in coverage_bad:
            print(f"  FAIL {line}")
    else:
        print(f"  all {len(tasks_seen)} specced tasks produced examples; "
              f"every split above the floor of {MIN_EXAMPLES_PER_SPLIT}")

    print("\n=== 3. spec declares values the data never uses (widens the grammar) ===")
    overwide = {}
    for task in sorted(observed):
        for path, values in vocabularies(task).items():
            unused = sorted(set(values) - observed[task].get(path, set()))
            if unused:
                overwide[f"{task} {path}"] = unused
    if overwide:
        for key in sorted(overwide):
            print(f"  {key}: {overwide[key]}")
    else:
        print("  none -- every declared value occurs in the data")

    print("\n=== 4. spec values never mentioned in the instruction (warning) ===")
    gaps_found = False
    for task in sorted(SPECS):
        gaps = instruction_gaps(task, TASKS.get(task, {}).get("instruction", ""))
        if gaps:
            gaps_found = True
            print(f"  {task}:")
            for path, missing in sorted(gaps.items()):
                print(f"      {path}: {list(missing)}")
    if not gaps_found:
        print("  none -- every closed vocabulary is enumerated in its instruction")

    failed = bool(violations or coverage_bad)
    print("\n" + ("FAILED" if failed else "OK"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

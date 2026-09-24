"""Shared helper for datasets that ship no released dev split.

Used by both build_unified_dataset.py and build_hf_datasets.py so the same
deterministic sample is carved regardless of which pipeline runs.
"""
import random

DEV_FRAC = 0.3
DEV_SEED = 13


def carve_dev_from_test(test_examples, frac=DEV_FRAC, seed=DEV_SEED):
    """Deterministically sample `frac` of `test_examples` to stand in as a
    monitoring-only dev set for a dataset with no released dev split.

    The sample overlaps test rather than being removed from it -- the
    original test set is left intact so per-task scoring still runs against
    the full, official test split; only merged/joint-training monitoring
    ever sees this carved dev sample.
    """
    items = list(test_examples)
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    n_dev = round(len(items) * frac)
    dev_idx = sorted(order[:n_dev])
    return [items[i] for i in dev_idx]

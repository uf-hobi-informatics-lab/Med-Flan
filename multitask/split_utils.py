"""Shared helper for datasets that ship no released dev split.

Used by both build_unified_dataset.py and build_hf_datasets.py so the same
deterministic sample is carved regardless of which pipeline runs.
"""
import random

# Fraction of *train* held out as a monitoring-only dev set when a dataset
# ships no released dev split. A 9:1 train:dev ratio -> 10% of train becomes
# dev; those rows are removed from train (disjoint), so unlike the earlier
# test-carve the dev sample no longer overlaps the final eval split.
DEV_FRAC = 0.1
DEV_SEED = 13


def carve_dev_from_train(train_examples, frac=DEV_FRAC, seed=DEV_SEED):
    """Deterministically split `train_examples` into (train, dev) with a
    9:1 ratio for datasets that ship no released dev split.

    The carve is disjoint: the dev rows are removed from train (the returned
    train is the remaining 90%), so train and dev never overlap. The official
    test split is left entirely untouched for per-task scoring.
    """
    items = list(train_examples)
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    n_dev = round(len(items) * frac)
    dev_idx = set(order[:n_dev])
    dev = [items[i] for i in sorted(order[:n_dev])]
    train = [items for k, items in enumerate(items) if k not in dev_idx]
    return train, dev

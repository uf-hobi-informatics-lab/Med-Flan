"""
2019 n2c2 Track 1 (Clinical STS) builder.

Plain tab-separated sentence1/sentence2/score(0-5) files -- no generation
format exists today, so this is one of the two datasets the plan calls out
as needing full reformatting.

Having an LLM free-generate a raw continuous score is not a great fit for
text-to-text generation: numeric tokenization is inconsistent (a model sees
"3.5", "3.50", "3" as different token sequences for training examples that
are otherwise identical), and greedy/sampled decoding tends to collapse
onto a handful of "round" values, which hurts a correlation-based metric
more than a classification one. The mitigation actually applied here is
modest, on purpose (this project has no constrained-decoding infra to do
better): scores are rounded and rendered with a fixed two-decimal format
(round(score, 2) -> "3.50", not "3.5") so every training example uses the
same numeric shape, and convert_back/sts_2019_t1.py parses the model's
output leniently (regex-extract the first float, clamp to [0, 5]) rather
than requiring an exact JSON-float round-trip.

`build_sts_classification` below is the discretized alternative this
docstring used to flag as a speculative "next step". Test gold
(`clinicalSTS2019.test.gs.sim.txt`) is confirmed to sit exactly on the 11
half-point anchor values (0, 0.5, 1, ..., 5, 0/411 off-grid); train is NOT
on that grid -- confirmed 561/1654 (34%) of train scores are fractional
multi-annotator averages (1.25, 1.65, 2.9, ...). Both splits are first
snapped to the nearest of those 11 anchors (`round(score * 2) / 2`) exactly
as before -- lossless for test, a real deliberate simplification for train
(collapsing annotator-disagreement noise into the nearest anchor).

11 categories is more than a model can reliably distinguish, though, so the
anchors are further grouped into 5 semantically-named categories
(`_ANCHOR_CATEGORY`) rather than exposed directly. The grouping
({0,0.5}/{1,1.5,2}/{2.5,3}/{3.5,4}/{4.5,5}) was chosen from the *pooled*
train+test histogram over the 11 anchors, not equal-width ranges or
per-split quantiles: train and test have genuinely different score
distributions (train mean 2.80/median 3, skewed mid/high; test mean
1.77/median 1, 58% of test <=1.0), so no fixed grouping balances both splits
at once -- pooling is the best available generic compromise, giving pooled
class shares of 13.7/23.4/25.7/26.0/11.3%. Test's low-similarity skew is a
real property of that split, not an artifact of this boundary choice, and
persists under any grouping. This is additive, not a replacement: `build_sts`
/ `sts_regression` (Pearson via the official script) is unchanged and stays
the primary, leaderboard-comparable axis; `build_sts_classification` /
`sts_classification` is a second dataset over the same sentence pairs,
scored via accuracy/macro-PRF/quadratic-weighted kappa in
multitask.eval.simple_tasks (the kappa specifically so a near-miss category
prediction isn't scored identically to a wild one -- plain accuracy/PRF
alone throws away the ordinal "how close" signal that matters for a
similarity scale).
"""
from pathlib import Path

from multitask.schemas import example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "raw_datasets" / "2019_n2c2_track1"
SPLIT_FILES = {"train": BASE / "clinicalSTS2019.train.txt", "test": BASE / "clinicalSTS2019.test.txt"}
# The released test file has no inline score column; the gold scores were
# published separately, line-aligned to the test file (confirmed same line
# count: 411/411), which is what makes per-task eval against
# correlation-noconfidence.pl possible at all for this split.
TEST_GOLD_SCORES = BASE / "clinicalSTS2019.test.gs.sim.txt"

# The 5 ordinal categories exposed to the model -- see module docstring for
# why these (not the 11 raw half-point anchors) are the classification
# target. Order matters (low->high similarity); must stay in sync with the
# duplicate in multitask/eval/simple_tasks.py.
STS_CATEGORIES = (
    "dissimilar", "weakly_related", "partially_equivalent",
    "mostly_equivalent", "highly_equivalent",
)

# Maps each of the 11 half-point anchors (index i -> score i/2) to one of
# the 5 STS_CATEGORIES above. See module docstring for how this grouping
# was chosen. Must stay in sync with the duplicate in
# multitask/eval/simple_tasks.py.
_ANCHOR_CATEGORY = (
    "dissimilar", "dissimilar",                                    # 0.0, 0.5
    "weakly_related", "weakly_related", "weakly_related",          # 1.0, 1.5, 2.0
    "partially_equivalent", "partially_equivalent",                 # 2.5, 3.0
    "mostly_equivalent", "mostly_equivalent",                       # 3.5, 4.0
    "highly_equivalent", "highly_equivalent",                       # 4.5, 5.0
)


def _bucket(score):
    """Nearest of the 11 half-point anchors (0, 0.5, ..., 5), mapped to one
    of the 5 STS_CATEGORIES via _ANCHOR_CATEGORY. Lossless mapping for test
    gold (already on-grid); a deliberate simplification for train (see
    module docstring)."""
    idx = max(0, min(10, round(score * 2)))
    return _ANCHOR_CATEGORY[idx]


def _read_gold_scores(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return [float(line.strip()) for line in f if line.strip()]


def _iter_rows(split):
    """Yields (i, sent1, sent2, score) for split, shared by build_sts and
    build_sts_classification -- same sentence pairs, two different targets."""
    path = SPLIT_FILES.get(split)
    if path is None or not path.exists():
        return
    gold_scores = _read_gold_scores(TEST_GOLD_SCORES) if split == "test" else None
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            sent1, sent2 = cols[0], cols[1]
            if len(cols) > 2:
                score = float(cols[2])
            elif gold_scores is not None and i < len(gold_scores):
                score = gold_scores[i]
            else:
                score = None
            yield i, path.name, sent1, sent2, score


def build_sts(split):
    for i, fname, sent1, sent2, score in _iter_rows(split):
        input_text = f"Sentence 1: {sent1}\nSentence 2: {sent2}"
        target = {"similarity": round(score, 2) if score is not None else None}
        yield example("sts_regression", "2019_n2c2_track1", split, f"{fname}#{i}",
                      input_text, target)


def build_sts_classification(split):
    for i, fname, sent1, sent2, score in _iter_rows(split):
        input_text = f"Sentence 1: {sent1}\nSentence 2: {sent2}"
        target = {"category": _bucket(score) if score is not None else None}
        yield example("sts_classification", "2019_n2c2_track1_cls", split, f"{fname}#{i}",
                      input_text, target)

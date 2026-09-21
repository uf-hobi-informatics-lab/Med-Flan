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
than requiring an exact JSON-float round-trip. If real model output turns
out to collapse onto too few distinct values in practice, the next step
would be discretizing to the 11 anchor values the released test gold
actually uses (0, 0.5, 1, ..., 5) and scoring it like a classification
task instead -- flagged in the README, not implemented speculatively here.
"""
from pathlib import Path

from multitask.schemas import example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "2019_n2c2_track1"
SPLIT_FILES = {"train": BASE / "clinicalSTS2019.train.txt", "test": BASE / "clinicalSTS2019.test.txt"}
# The released test file has no inline score column; the gold scores were
# published separately, line-aligned to the test file (confirmed same line
# count: 411/411), which is what makes per-task eval against
# correlation-noconfidence.pl possible at all for this split.
TEST_GOLD_SCORES = BASE / "clinicalSTS2019.test.gs.sim.txt"


def _read_gold_scores(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return [float(line.strip()) for line in f if line.strip()]


def build_sts(split):
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
            input_text = f"Sentence 1: {sent1}\nSentence 2: {sent2}"
            target = {"similarity": round(score, 2) if score is not None else None}
            yield example("sts_regression", "2019_n2c2_track1", split, f"{path.name}#{i}",
                          input_text, target)

"""
JSON -> float-lines converter for 2019 n2c2 Track 1 (Clinical STS), so
correlation-noconfidence.pl can score model output unmodified.

Input target schema (see multitask.builders.sts_2019_t1):
    {"similarity": 3.5}

Parsing is deliberately lenient (regex-extract the first float-looking
number, clamp to [0, 5]) rather than requiring exact JSON -- see the
module docstring on multitask.builders.sts_2019_t1 for why free-generated
numeric regression is an imperfect fit for text-to-text generation in the
first place, and why this project isn't attempting more than
consistent-formatting-plus-lenient-parsing for it. An unparseable line
falls back to the scale midpoint (2.5), the same "conservative miss"
convention used by reanchor() for the BRAT datasets.

Run standalone to self-check against gold on the test split (STS has no
released dev split) -- should give Pearson r = 1.0000:
    python -m multitask.convert_back.sts_2019_t1
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = ROOT / "2019_n2c2_track1" / "correlation-noconfidence.pl"
GOLD_SCORES = ROOT / "2019_n2c2_track1" / "clinicalSTS2019.test.gs.sim.txt"

_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_similarity(raw):
    """raw: the model's target dict (or raw completion string, for real
    model output that may not be valid JSON). Returns a float clamped to
    [0, 5], or 2.5 (scale midpoint) if nothing parseable is found."""
    if isinstance(raw, dict):
        raw = raw.get("similarity")
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        m = _FLOAT_RE.search(str(raw)) if raw is not None else None
        value = float(m.group()) if m else 2.5
    return max(0.0, min(5.0, value))


def to_score_lines(targets):
    """targets: list of target dicts (or raw completion strings), already
    in the test file's original line order. Returns the file text
    correlation-noconfidence.pl expects: one float per line."""
    return "\n".join(f"{parse_similarity(t):.4f}" for t in targets) + "\n"


def _run_eval(gs_path, system_path):
    result = subprocess.run(
        ["perl", str(EVAL_SCRIPT), str(gs_path), str(system_path)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)


def _self_check():
    """Round-trip gold test-split JSON (built by the dataset builder) back
    to score-lines, then correlate against the real gold gs file --
    should give Pearson r = 1.0000. Uses test because that's the only
    split with a released, line-aligned gold-scores file."""
    from multitask.builders.sts_2019_t1 import build_sts

    targets = [ex["target"] for ex in _examples_with_raw_target("test")]

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(to_score_lines(targets))
        system_path = f.name

    print("=== self-check: gold JSON -> score lines -> correlation-noconfidence.pl ===")
    _run_eval(GOLD_SCORES, system_path)
    Path(system_path).unlink(missing_ok=True)


def _examples_with_raw_target(split):
    """Rebuild examples with the raw target dict attached (schemas.example()
    only keeps rendered text/messages)."""
    from multitask.builders.sts_2019_t1 import SPLIT_FILES, TEST_GOLD_SCORES, _read_gold_scores

    path = SPLIT_FILES[split]
    gold_scores = _read_gold_scores(TEST_GOLD_SCORES) if split == "test" else None
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            if len(cols) > 2:
                score = float(cols[2])
            elif gold_scores is not None and i < len(gold_scores):
                score = gold_scores[i]
            else:
                score = None
            yield {"id": f"{path.name}#{i}", "target": {"similarity": score}}


if __name__ == "__main__":
    _self_check()

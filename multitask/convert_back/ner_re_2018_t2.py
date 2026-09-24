"""
JSON -> BRAT .ann converter for 2018 n2c2 Track 2 (Drug-ADE), so
Track2-evaluate-ver4.py can score model output unmodified.

Input target schema (see multitask.builders.ner_re_2018_t2):
    {"drugs": [{"text": "Lipitor", "Strength": ["10 mg"], "Reason": ["pain"]},
               {"text": "Colace"}],
     "unlinked": [{"type": "Reason", "text": "occasional cough"}]}

Every relation is <OtherType>-Drug with Arg2 = the Drug entity, so each
drug's linked attribute mentions are nested directly under it (keyed by
attribute type) instead of a flat entity list + id-based relations list --
no ids are minted by the model at all; fresh T/E-style ids are minted here,
purely internally, only so Track2-evaluate-ver4.py has something to read.

Drug spans are resolved left-to-right with a shared cursor+exclusion list
(multitask.reanchor.reanchor(), same disambiguation reasoning as every
other BRAT builder in this family: drugs are emitted in true document
order). Each attribute filler is resolved with reanchor(..., near=
drug_start) against that SAME shared exclusion list -- unlike
sdoh_2022_t2.py's role fillers (short category values inside short
snippets), this dataset's filler values (e.g. "po", "b.i.d", "10") are
short and repeat often across a FULL clinical note, so a fresh
per-filler exclusion list (sdoh_2022_t2.py's approach) causes wrong
cross-drug collisions -- confirmed empirically: it dropped the self-check
below 1.0 F1 on exactly the short/common attribute types (Route, Form,
Dosage, Frequency).

The one legitimate reason to break exclusivity: the same physical Other
entity can be linked via separate relations to more than one Drug (e.g. one
Reason mention shared by 4 different drugs in
gold_standard_test/105885.ann). So filler resolution is two-tier: try the
shared exclusion list first (correct for the common case of genuinely
repeated-but-distinct short values); only if every occurrence of that text
is already claimed, fall back to a non-exclusive nearest match, treating
that as a signal this is the deliberate-duplicate/fan-out case rather than
dropping the filler.

Known limitation -- this dataset's self-check does NOT reach P=R=F1=1.0,
unlike the other BRAT-based convert_back modules in this family. Root
cause (confirmed by diffing round-tripped .ann against real gold via
Track2-evaluate-ver4.py's own RecordTrack2 parser): full clinical notes
densely repeat short drug/attribute strings (e.g. "oxygen", "platelet",
"po") both as genuine annotated mentions AND as coincidental plain prose
(e.g. "her platelet count" sitting textually before the real "platelet
transfusions" mention) -- reanchor.py's own docstring documents this
decoy ambiguity as inherent to text-only span reconstruction. `_pick_drug_
span()` uses each drug's linked filler text as disambiguating evidence
(the real mention should have its fillers nearby; a decoy usually won't),
which measurably helps, but can't resolve every case -- e.g. a drug
mention with zero fillers, repeated many times, has no signal to use at
all. Several stricter/looser variants of this heuristic were tried
empirically; this window-count version scored best. This ceiling is not
specific to this code -- it reflects what ANY id-free, text-only
reconstruction can recover from these documents, so it's also roughly the
ceiling a real model's predictions would face at inference time, not just
this self-check.

Most of what used to be attributed to that ceiling was in fact a bug,
fixed in brat.align_entity_text() (see its docstring): 1,460 gold
entities carry .ann mention text that is not a substring of the document
(adjacent-fragment spans like "Prednison e" for "Prednisone"), which
reanchor's `\\s+` token join can never match, so 316/10,575 Drug mentions
failed to anchor and each discarded every attribute nested beneath it.
Before that fix this self-check scored tags F1 0.9429 / relations F1
0.9039; it now scores 0.9752 / 0.9609. What remains is genuinely
precision-dominated (FP 1,382 vs FN 279) and largely structural rather
than ambiguity: 768 fillers link to more than one Drug, and because
linking is by nesting with no entity ids, each is emitted once per drug
and to_ann() does not deduplicate the resulting T-lines.

Scored via multitask.brat_scorer (see that module's docstring) rather than
the official Track2-evaluate-ver4.py -- that script's tag/relation matching
is a single iteration-order-dependent pass ("pare down duplicates then
count"), not a real assignment algorithm; brat_scorer's greedy
nearest-distance matcher fixes that. Numbers therefore aren't directly
identical to a leaderboard submission scored by the official script (run
Track2-evaluate-ver4.py separately for that comparison). Current result on
the test split (run this file standalone to reproduce): tags micro
F1=0.9752 strict / 0.9755 lenient, relations micro F1=0.9609 strict /
0.9611 lenient.

Run standalone to self-check against gold (see "Known limitation" above
for why this doesn't reach P=R=F1=1.0):
    python -m multitask.convert_back.ner_re_2018_t2
"""
import re
import tempfile
from pathlib import Path

from multitask import brat_scorer
from multitask.reanchor import _flex_pattern, reanchor
from multitask.target_schema import MEDICATION_ATTR_TYPES

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "2018_n2c2_track2" / "gold_standard_test"

ATTR_TYPES = MEDICATION_ATTR_TYPES


def _find_all(text, doc_text, exclude):
    """Every non-overlapping (start, end) occurrence of text in doc_text not
    overlapping a span in exclude -- same flex-whitespace/case-fallback
    matching as reanchor(), but returning every candidate instead of one."""
    pattern = _flex_pattern(text)
    if pattern is None:
        return []

    def _cands(flags):
        it = pattern.finditer(doc_text) if flags == 0 else re.finditer(pattern.pattern, doc_text, flags)
        return [m.span() for m in it if not any(a < m.end() and m.start() < b for a, b in exclude)]

    return _cands(0) or _cands(re.IGNORECASE)


def _pick_drug_span(drug_text, filler_texts, doc_text, used_spans, floor):
    """Resolve a Drug mention's span among possibly-repeated occurrences of
    the same text. Plain leftmost-from-floor matching alone picks a
    coincidental plain-prose decoy (e.g. "her platelet count") over the real
    annotated mention (e.g. "further platelet transfusions") whenever a
    decoy sits between the previous entity's end and the real occurrence --
    reanchor.py's own docstring documents this ambiguity as inherent to
    text-only reconstruction. Disambiguate using the drug's own linked
    attribute fillers as supporting evidence: the correct occurrence should
    have its fillers' text nearby (within a window), while a decoy's
    nearby text usually won't. This is a best-effort heuristic, not a
    guarantee -- pure text-only reconstruction has an inherent ceiling on
    this dataset (see module docstring's "Known limitation" section);
    several stricter/fancier variants were tried empirically and this
    window-count version scored best."""
    forward = [s for s in _find_all(drug_text, doc_text, used_spans) if s[0] >= floor]
    candidates = forward or _find_all(drug_text, doc_text, used_spans)
    if not candidates:
        return None
    if len(candidates) == 1 or not filler_texts:
        return candidates[0]

    window = 400
    filler_spans = [_find_all(ftext, doc_text, []) for ftext in filler_texts]

    def support(span):
        s, e = span
        lo, hi = s - window, e + window
        return sum(1 for spans in filler_spans if any(lo <= fs and fe <= hi for fs, fe in spans))

    return max(candidates, key=lambda sp: (support(sp), -sp[0]))


def to_ann(target, doc_text):
    """target: the JSON dict (drugs/unlinked). Returns .ann file text.

    A Drug, filler, or unlinked entity whose text can't be found anywhere
    in doc_text is no longer dropped -- it gets a "poison span" (past the
    end of doc_text, monotonically increasing so none collide) instead.
    The poison span is syntactically a normal T-line, so it's still
    counted as a real system prediction by brat_scorer.score_tags's
    `fp = len(sys_) - tp` (and by any other BRAT-format scorer, including
    the official Track2-evaluate-ver4.py cross-check) -- it just can never
    overlap a real gold span, so it can only ever be a false positive,
    never a false true-positive. Previously an unresolvable Drug dropped
    its entire filler block too (the `continue` skipped straight to the
    next drug), discarding fillers that might themselves have been correct
    predictions; they now still get their own reanchor attempt and, if
    resolved, can still register as a true positive on the tags axis even
    though their relation to the (poison-anchored) Drug is necessarily
    wrong."""
    used_spans = []
    cursor = 0
    lines = []
    tid = rid = 0
    poison = len(doc_text) + 1

    def _poison_span():
        nonlocal poison
        span = (poison, poison + 1)
        poison += 1
        return span

    for drug in target.get("drugs", []):
        if not isinstance(drug, dict):
            continue
        drug_text = drug.get("text") or ""
        filler_texts = [t for attr_type in ATTR_TYPES for t in (drug.get(attr_type) or [])]
        d_span = _pick_drug_span(drug_text, filler_texts, doc_text, used_spans, cursor)
        if d_span is None:
            d_span = _poison_span()
        else:
            used_spans.append(d_span)
            cursor = max(cursor, d_span[1])
        d_start, d_end = d_span
        tid += 1
        drug_tid = f"T{tid}"
        lines.append(f"{drug_tid}\tDrug {d_start} {d_end}\t{drug_text}")

        for attr_type in ATTR_TYPES:
            for filler_text in drug.get(attr_type) or []:
                # Exclusive attempt first (correct for distinct-but-repeated
                # short values); if every occurrence is already claimed,
                # fall back to a non-exclusive nearest match -- see module
                # docstring's two-tier resolution rationale.
                f_span = reanchor(filler_text, doc_text, used_spans, near=d_start)
                if f_span is None:
                    f_span = reanchor(filler_text, doc_text, [], near=d_start)
                if f_span is None:
                    f_span = _poison_span()
                f_start, f_end = f_span
                tid += 1
                filler_tid = f"T{tid}"
                lines.append(f"{filler_tid}\t{attr_type} {f_start} {f_end}\t{filler_text}")
                rid += 1
                lines.append(f"R{rid}\t{attr_type}-Drug Arg1:{filler_tid} Arg2:{drug_tid}")

    unlinked_cursor = 0
    for other in target.get("unlinked", []):
        if not isinstance(other, dict):
            continue
        other_text = other.get("text") or ""
        o_span = reanchor(other_text, doc_text, used_spans, min_start=unlinked_cursor)
        if o_span is None:
            o_span = _poison_span()
        else:
            unlinked_cursor = max(unlinked_cursor, o_span[1])
        o_start, o_end = o_span
        tid += 1
        lines.append(f"T{tid}\t{other.get('type')} {o_start} {o_end}\t{other_text}")

    return "\n".join(lines) + ("\n" if lines else "")


def write_anns(predictions, out_dir):
    """predictions: iterable of (doc_id, target, doc_text). Writes one
    <doc_id>.ann per prediction under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, target, doc_text in predictions:
        (out_dir / f"{doc_id}.ann").write_text(to_ann(target, doc_text), encoding="utf-8")
    return out_dir


def score_dirs(gold_dir, system_dir):
    """Score system_dir's .ann predictions against gold_dir using the new
    unified brat_scorer (see module docstring) rather than the official
    Track2-evaluate-ver4.py."""
    brat_scorer.score_dirs(gold_dir, system_dir, brat_scorer.CONFIGS["2018_n2c2_track2"])


def _self_check():
    """Round-trip gold test-split JSON (built by the dataset builder) back
    to .ann, then score against the real gold -- see module docstring's
    "Known limitation": expect tags/relations F1 well below 1.0, not exact."""
    from multitask.io_utils import read_text

    predictions = []
    for ex in _examples_with_raw_target("test"):
        doc_text = read_text(GOLD_DIR / f"{ex['id']}.txt")
        predictions.append((ex["id"], ex["target"], doc_text))

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = write_anns(predictions, tmp)
        print("=== self-check: gold JSON -> .ann -> brat_scorer ===")
        score_dirs(GOLD_DIR, out_dir)


def _examples_with_raw_target(split):
    """Rebuild examples with the raw target dict attached (schemas.example()
    only keeps rendered text/messages).

    Delegates to the builder's own _iter_docs/_build_ner_re_target instead of
    reimplementing them. This was previously a ~40-line copy of the builder
    carrying a "Mirrors the builder exactly" comment -- two implementations
    of one target schema that had to be kept in sync by hand, and which
    silently diverged the moment the builder gained
    brat.align_entity_text()."""
    from multitask.builders.ner_re_2018_t2 import _build_ner_re_target, _iter_docs

    for txt_path, _doc_text, doc in _iter_docs(split):
        yield {"id": txt_path.stem, "target": _build_ner_re_target(doc)}


if __name__ == "__main__":
    _self_check()

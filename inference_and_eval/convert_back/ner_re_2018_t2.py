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
import sys
import tempfile
from pathlib import Path

from inference_and_eval import brat_scorer
from inference_and_eval.reanchor import _flex_pattern, reanchor
from inference_and_eval.target_schema import MEDICATION_ATTR_TYPES

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


# --- Paragraph-window prediction merge (design.md's paragraph-chunking
# section's "reassembly step") -- lets a model trained ONLY on
# ner_re_medication_paragraph (multitask.builders.ner_re_2018_t2.
# build_ner_re_medication_paragraph) still be scored against the SAME
# whole-note gold as the doc-level task, via multitask.eval.paragraph_eval.
# Each paragraph's predicted JSON is converted to .ann via to_ann() above,
# scoped to just that paragraph's own text (the model never saw the rest of
# the note, so confining span search to its own paragraph -- rather than the
# whole document -- avoids picking a same-text mention from a different
# paragraph; see to_ann's/reanchor's decoy-ambiguity discussion), then every
# real span is shifted back into the whole document's coordinates and every
# T/R id is renumbered so paragraphs can be concatenated into one legal .ann
# per document.
_T_LINE = re.compile(r"^(T\d+)\t(\S+) (\d+) (\d+)\t(.*)$")
_R_LINE = re.compile(r"^(R\d+)\t(\S+) Arg1:(T\d+) Arg2:(T\d+)$")


def _paragraph_items_by_doc(split):
    """{doc_id: (doc_text, [(start, end), ...])} for every doc in split --
    reuses the builder's own _iter_docs/_iter_paragraphs (same blank-line
    spans, filtered to non-empty-gold ones) so the i-th surviving span here
    is exactly the paragraph a `{doc_stem}__p{i}` id refers to. Recomputed
    here rather than read back off the built dataset (which stores no
    offsets), since it's fully deterministic from the same .txt/.ann files
    the doc-level gold scorer (GOLD_DIR) already reads."""
    from inference_and_eval.builders.ner_re_2018_t2 import _iter_docs, _iter_paragraphs

    result = {}
    for txt_path, doc_text, doc in _iter_docs(split):
        spans = [(start, end) for start, end, _para_text, _para_doc in _iter_paragraphs(doc_text, doc)]
        result[txt_path.stem] = (doc_text, spans)
    return result


def _merge_paragraph_anns(doc_text, para_items):
    """para_items: ordered [(para_start, para_end, target), ...] for one
    document. Returns whole-document .ann text with every paragraph's
    predictions offset-shifted into global coordinates and every T/R id
    renumbered uniquely. Poison spans (to_ann's fallback for an
    unresolvable mention -- see its docstring -- detected here as
    start >= len(that paragraph's own text)) are reassigned fresh, from a
    single counter starting past the end of the FULL document, so poison
    spans from different paragraphs can never collide with each other or
    with a later paragraph's real (shifted) spans -- same invariant to_ann
    itself relies on, just reapplied once for the whole merged document
    instead of once per paragraph."""
    lines = []
    tid = rid = 0
    poison = len(doc_text) + 1

    def _next_poison():
        nonlocal poison
        span = (poison, poison + 1)
        poison += 1
        return span

    for para_start, para_end, target in para_items:
        para_text = doc_text[para_start:para_end]
        id_map = {}
        for line in to_ann(target, para_text).splitlines():
            m = _T_LINE.match(line)
            if m:
                old_id, etype, start, end, text = m.groups()
                start, end = int(start), int(end)
                new_span = _next_poison() if start >= len(para_text) else (start + para_start, end + para_start)
                tid += 1
                new_id = f"T{tid}"
                id_map[old_id] = new_id
                lines.append(f"{new_id}\t{etype} {new_span[0]} {new_span[1]}\t{text}")
                continue
            m = _R_LINE.match(line)
            if m:
                old_id, rtype, arg1, arg2 = m.groups()
                rid += 1
                lines.append(f"R{rid}\t{rtype} Arg1:{id_map[arg1]} Arg2:{id_map[arg2]}")

    return "\n".join(lines) + ("\n" if lines else "")


def write_anns_from_paragraphs(completions, split="test", out_dir=None):
    """completions: {paragraph_id: raw completion string or already-parsed
    dict}, paragraph_id = f"{doc_stem}__p{i}" (build_ner_re_medication_
    paragraph's own id scheme -- doc stems in this dataset are plain
    numeric, so splitting on the LAST "__p" is unambiguous). Groups by
    document, merges each doc's paragraph predictions back into one
    whole-note .ann (_merge_paragraph_anns), and writes <doc_stem>.ann per
    doc under out_dir (a fresh temp dir if omitted) -- so the existing
    whole-note GOLD_DIR can score it with score_dirs() exactly like the
    doc-level pipeline. A surviving paragraph with no completion at all
    (e.g. --limit cut inference short) is scored as an empty prediction
    (all its gold entities become false negatives) rather than silently
    dropped -- a warning lists which. Returns out_dir."""
    from inference_and_eval.eval.simple_tasks import parse_completion

    doc_index = _paragraph_items_by_doc(split)
    by_doc = {doc_id: {} for doc_id in doc_index}
    for para_id, completion in completions.items():
        doc_id, _sep, idx_str = para_id.rpartition("__p")
        if doc_id not in by_doc:
            print(f"[write_anns_from_paragraphs] skipping unknown paragraph id {para_id!r}", file=sys.stderr)
            continue
        target = completion if isinstance(completion, (dict, list)) else parse_completion(completion)
        by_doc[doc_id][int(idx_str)] = target

    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, (doc_text, spans) in doc_index.items():
        targets_by_idx = by_doc[doc_id]
        missing = [i for i in range(len(spans)) if i not in targets_by_idx]
        if missing:
            print(
                f"[write_anns_from_paragraphs] {doc_id}: no completion for paragraph(s) "
                f"{missing} of {len(spans)} -- scored as empty prediction", file=sys.stderr,
            )
        para_items = [(start, end, targets_by_idx.get(i) or {}) for i, (start, end) in enumerate(spans)]
        ann_text = _merge_paragraph_anns(doc_text, para_items)
        (out_dir / f"{doc_id}.ann").write_text(ann_text, encoding="utf-8")
    return out_dir


def score_paragraph_predictions(completions, split="test"):
    """One-call convenience: merge paragraph-level completions back to doc
    level (write_anns_from_paragraphs) and score them against the SAME
    whole-note GOLD_DIR/score_dirs the doc-level ner_re_medication task
    uses -- so a model trained only on ner_re_medication_paragraph is
    directly comparable to one trained on the whole-note task, not scored
    on some separate/weaker metric. See multitask.eval.paragraph_eval for
    the CLI wrapper around this."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = write_anns_from_paragraphs(completions, split, tmp)
        score_dirs(GOLD_DIR, out_dir)


def score_dirs(gold_dir, system_dir):
    """Score system_dir's .ann predictions against gold_dir using the new
    unified brat_scorer (see module docstring) rather than the official
    Track2-evaluate-ver4.py."""
    brat_scorer.score_dirs(gold_dir, system_dir, brat_scorer.CONFIGS["2018_n2c2_track2"])


def _self_check():
    """Round-trip gold test-split JSON (built by the dataset builder) back
    to .ann, then score against the real gold -- see module docstring's
    "Known limitation": expect tags/relations F1 well below 1.0, not exact."""
    from inference_and_eval.io_utils import read_text

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
    from inference_and_eval.builders.ner_re_2018_t2 import _build_ner_re_target, _iter_docs

    for txt_path, _doc_text, doc in _iter_docs(split):
        yield {"id": txt_path.stem, "target": _build_ner_re_target(doc)}


def _examples_with_raw_target_paragraph(split):
    """Same pattern as _examples_with_raw_target, but one entry per
    surviving paragraph -- yields {"id": "<doc_stem>__p<i>", "target":
    <drugs/unlinked dict>}, same id scheme as
    build_ner_re_medication_paragraph. Used by _self_check_paragraph to
    exercise the merge/reassembly machinery with known-good targets."""
    from inference_and_eval.builders.ner_re_2018_t2 import _build_ner_re_target, _iter_docs, _iter_paragraphs

    for txt_path, doc_text, doc in _iter_docs(split):
        for i, (_start, _end, _para_text, para_doc) in enumerate(_iter_paragraphs(doc_text, doc)):
            yield {"id": f"{txt_path.stem}__p{i}", "target": _build_ner_re_target(para_doc)}


def _self_check_paragraph():
    """Round-trip the paragraph dataset's OWN gold-derived targets (same
    ones build_ner_re_medication_paragraph trains on) through
    write_anns_from_paragraphs -> score against the whole-note GOLD_DIR --
    proves the merge/reassembly machinery itself introduces no bugs, using
    known-good paragraph targets as the "predictions", isolated from
    whatever a real model does. Expect a score in the same ballpark as
    _self_check()'s whole-note round-trip, not exactly 1.0, for the same
    reason (see module docstring's "Known limitation": text-only span
    reconstruction has an inherent ceiling)."""
    completions = {ex["id"]: ex["target"] for ex in _examples_with_raw_target_paragraph("test")}
    print("=== paragraph self-check: gold paragraph JSON -> merged .ann -> brat_scorer ===")
    score_paragraph_predictions(completions, "test")


if __name__ == "__main__":
    _self_check()
    _self_check_paragraph()

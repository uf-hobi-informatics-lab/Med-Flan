"""
Vocab-agnostic character-offset span-resolution primitives, shared by every
`*_seq2seq_grammar.py` module's `parse_and_resolve_with_spans`. Extracted
out of `inference_and_eval/seq2seq_grammar.py` (the first seq2seq grammar,
for `sdoh_event_seq2seq`) once a second and third real consumer needed the
same logic (`ner_re_medication_seq2seq_grammar.py`'s unlinked-mention chain,
`medication_event_seq2seq_grammar.py`'s single-trigger chain) -- pure
extraction, no behavior change from the versions that used to live inline in
`seq2seq_grammar.py`.

These are candidate-list-returning siblings of
`inference_and_eval/reanchor.py`'s single-match `reanchor()`: callers here
need the full candidate list (to build DP-style / floor-search /
nearest-search logic across multiple literals), not one mutate-in-place
match.
"""
import re

from inference_and_eval.reanchor import _flex_pattern


def flex_candidates(text, doc_text):
    """Every occurrence of `text` in `doc_text`, via the same
    whitespace-flexible regex `reanchor()` itself uses. Returns
    `(case_sensitive_spans, case_insensitive_spans)` -- both lists, not just
    a merged/first-nonempty one, so `search_floor`/`search_nearest` can retry
    case-insensitively at the same floor/hint before ever falling back to an
    earlier or less-relevant case-sensitive match (mirrors `reanchor()`'s own
    per-tier fallback order exactly -- see its docstring)."""
    pattern = _flex_pattern(text)
    if pattern is None:
        return [], []
    cs = [(m.start(), m.end()) for m in pattern.finditer(doc_text)]
    ci = [(m.start(), m.end()) for m in re.finditer(pattern.pattern, doc_text, re.IGNORECASE)]
    return cs, ci


def overlaps_any(span, used_spans):
    return any(a < span[1] and span[0] < b for a, b in used_spans)


def search_floor(candidates, used_spans, floor):
    """`candidates`: the `(case_sensitive, case_insensitive)` pair from
    `flex_candidates`. Nearest unclaimed candidate at or after `floor`,
    tried case-sensitive first then case-insensitive at that same floor,
    only falling back to the first unclaimed candidate anywhere (again
    case-sensitive before case-insensitive) if neither list has one past
    `floor`. Mirrors `reanchor()`'s `min_start` mode exactly."""
    for cands in candidates:
        forward = [c for c in cands if c[0] >= floor and not overlaps_any(c, used_spans)]
        if forward:
            return min(forward, key=lambda c: c[0])
    for cands in candidates:
        unclaimed = [c for c in cands if not overlaps_any(c, used_spans)]
        if unclaimed:
            return min(unclaimed, key=lambda c: c[0])
    return None


def search_nearest(candidates, hint):
    """`candidates`: the `(case_sensitive, case_insensitive)` pair from
    `flex_candidates`. Candidate with the smallest distance to `hint`,
    either direction, tried case-sensitive first, only falling back to
    case-insensitive if the case-sensitive list is empty. Mirrors
    `reanchor()`'s `near` mode exactly."""
    for cands in candidates:
        if cands:
            return min(cands, key=lambda c: min(abs(c[0] - hint), abs(c[1] - hint)))
    return None

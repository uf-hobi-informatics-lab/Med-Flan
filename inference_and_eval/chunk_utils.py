"""Paragraph/section-window chunking for a document.

Splits raw document text into smaller spans and projects entities/relations
onto one such span, dropping any relation that straddles two spans (see
design.md's paragraph-chunking section). Two boundary rules are provided,
picked per-dataset by which one the raw text actually supports:

- `paragraph_spans`: blank-line boundaries (2018_n2c2_track2's BRAT .txt
  format). Paired with `project_doc_onto_span`, which is BratDoc-specific
  (multitask/brat.py dataclasses).
- `section_spans`: ALL-CAPS "SECTION HEADER :" line boundaries (2010
  i2b2/VA's format has zero blank lines at all -- confirmed empirically --
  so paragraph_spans is degenerate/useless on it: it returns exactly one
  span covering the whole doc). Projection for this one lives in
  multitask/builders/i2b2_2010.py instead, since its concepts/relations are
  plain dicts, not BratDoc dataclasses.
"""
import re
from dataclasses import replace

_PARA_BREAK = re.compile(r"\n\s*\n")
# Case-insensitive: i2b2 2010 discharge summaries mix ALL-CAPS headers
# ("HISTORY OF PRESENT ILLNESS :") with Title/mixed-case ones ("Discharge
# Date :", "Service :", "cc :") -- an ALL-CAPS-only pattern (the original
# version of this regex) missed roughly half of all real section
# boundaries and left 32/693 docs with zero detected headers at all.
# Case-insensitive matching drops that to 1/693 and finds 14,076 header
# lines vs. 8,141. Cost: a handful of one-off narrative sentences that
# happen to end in a colon also match (e.g. "On 2014-04-04, labs were as
# follows :") and create a spurious section split -- rare (<0.1% of
# matched lines in a manual check) and the existing "drop anything
# straddling a boundary" rule already tolerates wrong splits without
# corrupting a kept section's content, so this is accepted rather than
# solved with a header-name allowlist.
_SECTION_HEADER = re.compile(r"^[A-Za-z][A-Za-z0-9 /\-,()\.]{2,60}:\s*$")


def paragraph_spans(text):
    """Yield (start, end) char offsets covering the full text, split on
    blank-line boundaries, in document order."""
    spans = []
    pos = 0
    for m in _PARA_BREAK.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    return spans


def section_spans(text):
    """Yield (start, end) char offsets covering the full text, split on
    ALL-CAPS "SECTION HEADER :" line boundaries (e.g. "HISTORY OF PRESENT
    ILLNESS :"), in document order. Falls back to one span covering the
    whole text if no header line is found (measured: 32/426 i2b2 2010
    docs have none)."""
    lines = text.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    header_idxs = [i for i, line in enumerate(lines) if _SECTION_HEADER.match(line.strip())]
    if not header_idxs:
        return [(0, len(text))]
    bounds = sorted(set([0] + header_idxs + [len(lines)]))
    spans = []
    for i in range(len(bounds) - 1):
        start_line, end_line = bounds[i], bounds[i + 1]
        start = offsets[start_line]
        end = offsets[end_line] if end_line < len(offsets) else len(text)
        if end > start:
            spans.append((start, end))
    return spans


def project_doc_onto_span(doc, start, end):
    """Return a new BratDoc with entities fully inside [start, end) and
    relations whose both linked entities survive that filter -- entities
    and relations crossing the span boundary are dropped. events/attributes
    pass through unchanged (unused by the paragraph-chunked datasets)."""
    kept_entities = {
        eid: e for eid, e in doc.entities.items() if e.start >= start and e.end <= end
    }
    kept_relations = {
        rid: r for rid, r in doc.relations.items()
        if r.arg1 in kept_entities and r.arg2 in kept_entities
    }
    return replace(doc, entities=kept_entities, relations=kept_relations)

"""
Recover BRAT T-line character offsets from an entity's mention text, since
an LLM can only be expected to generate the text it saw, not exact byte
offsets. Shared by all three BRAT-based convert_back modules
(ner_re_2018_t2, medication_event_2022_t1, sdoh_2022_t2).
"""
import re


def _flex_pattern(text):
    # Escaping literally, but letting any whitespace run in the mention
    # text match any whitespace run in the document -- needed because a
    # handful of gold entities are discontinuous BRAT spans (e.g. a mention
    # split across a line break), where the visible mention text uses a
    # plain space where the document actually has a newline.
    parts = [p for p in re.split(r"\s+", text.strip()) if p]
    if not parts:
        return None
    return re.compile(r"\s+".join(re.escape(p) for p in parts))


def reanchor(entity_text, doc_text, used_spans, min_start=0, near=None):
    """Find an occurrence of entity_text in doc_text whose span does not
    overlap any span already in used_spans (a list of (start, end), mutated
    in place on a successful match).

    Repeated mention text is ambiguous on its own -- e.g. a document can
    contain a plain-prose "p.o" that isn't annotated as an entity, sitting
    *before* the one occurrence that actually is. Text-only left-to-right
    claiming falls for that decoy. Two disambiguation modes, tried
    exclusively of each other:

    - `near` (a hint character position, e.g. an already-resolved trigger's
      start): picks the unclaimed occurrence with the smallest absolute
      distance to `near`, searching in both directions. Use this for role
      fillers / arguments that can sit either side of the entity they
      attach to (e.g. a SHAC role filler can precede or follow its event
      trigger in the sentence), where a forward-only floor would be wrong.
    - `min_start` (default mode, ignored if `near` is given): prefers the
      first unclaimed occurrence at or after that point, falling back to
      the first unclaimed occurrence anywhere only if none exists past
      min_start (e.g. slightly overlapping/nested gold spans). Use this
      when callers supply entities in true left-to-right document order
      (pass the end offset of the previously-placed entity) and matches are
      expected to keep moving forward, e.g. flat entity/trigger lists.

    Falls back to case-insensitive search if an exact-case match isn't
    found anywhere. Returns (start, end), or None if the text can't be
    found at all (the caller should drop the entity -- a miss against gold,
    which is the correct and conservative behavior for scoring)."""
    pattern = _flex_pattern(entity_text)
    if pattern is None:
        return None

    def _matches(flags):
        return pattern.finditer(doc_text) if flags == 0 else re.finditer(pattern.pattern, doc_text, flags)

    def _candidates(flags):
        for m in _matches(flags):
            span = (m.start(), m.end())
            if not any(a < span[1] and span[0] < b for a, b in used_spans):
                yield span

    def _search_floor(flags, floor):
        for span in _candidates(flags):
            if span[0] >= floor:
                return span
        return None

    def _search_nearest(flags, hint):
        best = None
        for span in _candidates(flags):
            dist = min(abs(span[0] - hint), abs(span[1] - hint))
            if best is None or dist < best[0]:
                best = (dist, span)
        return best[1] if best else None

    if near is not None:
        span = _search_nearest(0, near) or _search_nearest(re.IGNORECASE, near)
    else:
        span = (
            _search_floor(0, min_start)
            or _search_floor(re.IGNORECASE, min_start)
            or _search_floor(0, 0)
            or _search_floor(re.IGNORECASE, 0)
        )
    if span is not None:
        used_spans.append(span)
    return span

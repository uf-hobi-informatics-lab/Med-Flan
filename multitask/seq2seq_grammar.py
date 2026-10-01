"""
Microsoft-Nuance-style seq2seq target grammar for `sdoh_event_seq2seq` (see
design.md / the approved plan for this feature). This is a brand-new,
additive sibling to the existing JSON-based `sdoh_event` task -- nothing
here is consumed by that path, and nothing in it is changed.

Ben Abacha, Fan & Romanowski's #1-ranked 2022 n2c2 Track 2 submission
("Microsoft-Nuance at n2c2 2022 Track 2") trained T5 to emit a compact,
non-JSON structured sequence (source tokens re-emitted verbatim inside
typed tag markup) and recovered character spans afterward via a constraint
solver (left-to-right ordering as a hard constraint, "compactness" as a
soft tiebreak). The paper does not publish a formal grammar, so this one is
our own -- built to reuse the exact closed vocabularies (`SHAC_EVENT_TYPES`,
`SHAC_ROLE_NAMES`, `SHAC_FILLER_TYPES`, `SHAC_ATTR_VALUES`) that
multitask/target_schema.py already declares for the JSON path, so the two
paths can never disagree on what counts as a legal event/role/value --
only the string serialization differs.

Grammar:
    event    := "<" EventType ">" trigger_text role*
    role     := "<" RoleName ["!" FillerType] ">" ["[" attr_value "]"] filler_text
    filler_text := literal_text | "=TRIGGER="
    sequence := event (" {AND} " event)*

- Roles are emitted in the fixed canonical `SHAC_ROLE_NAMES` order.
- "[attr_value]" only ever follows a Status/Type role (matches the corpus:
  attribute values only ever attach to those two base role names' fillers,
  via StatusTimeVal/StatusEmployVal/TypeLivingVal -- see
  `_FILLER_TYPE_TO_ATTR_NAME`).
- "=TRIGGER=" replaces a filler's literal text when that filler's text
  equals its own event's trigger text -- the same shared-mention
  convention `convert_back/sdoh_2022_t2.py`'s `to_ann` already documents
  (a role filler can legitimately be the exact same span as its trigger).
- "<RoleName!FillerType>" is an override tag, needed only when a role
  filler's true BRAT entity type disagrees with the type
  `_default_filler_type` would otherwise infer from (event_type,
  role_name) -- keeps the common case compact while staying lossless.

Example:
    <LivingStatus> lives <Status> [current] =TRIGGER= <Type> [with_family] with husband {AND} <Tobacco> smokes <Status> [current] =TRIGGER= <Type> cigarettes <Type2> cigars

Span recovery (`parse_and_resolve_with_spans`) resolves triggers and role
fillers with two different disambiguation modes, matching
`convert_back/sdoh_2022_t2.py`'s own `to_ann` (which calls `reanchor()`
directly) precisely enough to be a drop-in replacement of that mechanism on
this grammar's literals -- not the flat single-chain global DP an earlier
version of this module used. That DP (one forward-only, then one
direction-agnostic-cost variant) was tried and measured against this
module's own gold round-trip self-check first; both scored measurably below
`sdoh_2022_t2.py`'s own numbers on the same gold, for two independent
reasons a single chain can't cleanly separate:
  1. A role filler's literal text routinely precedes its own trigger *in
     the document* (a negation/status cue like "no", "denies", "quit"
     modifies a phrase that follows it -- "no smoking history"), even
     though roles are always emitted after their trigger. A hard
     forward-only chain pushes those onto the next, wrong, occurrence of
     the same word.
  2. Two different EVENTS can have the identical trigger text (e.g. two
     separate "lives" mentions naming two different living situations).
     Nothing in a flat per-literal chain stops two unrelated trigger
     literals from both resolving to the same earlier occurrence once the
     forward-only rule is relaxed to fix (1) -- there's no "already
     claimed" bookkeeping once cost, not hard order, is what decides.
Two tiers instead:
  - **Triggers**: resolved in a single forward-floor chain across the
    whole document, each excluding every span already claimed by an
    earlier trigger in this doc (`used_spans`) -- mirrors `reanchor(...,
    min_start=cursor)`'s exact contract (nearest unclaimed occurrence at or
    after the running cursor, falling back to the first unclaimed
    occurrence anywhere if none exists past the cursor). This is still a
    genuinely doc-wide, order-aware resolution (it's what actually
    implements the paper's "left-to-right" hard constraint) -- it's just
    scoped to triggers only, which really do progress monotonically
    through the document in this corpus's annotation convention (the
    builder sorts events by trigger offset), rather than to every literal
    in emission order.
  - **Role fillers**: each resolved independently, nearest occurrence to
    its own event's resolved trigger end, in either direction -- mirrors
    `reanchor(..., near=t_end)`'s exact contract, with **no** exclusivity
    across fillers (a fresh search every time, not a shared `used_spans`),
    since the same mention is often legitimately referenced as a role
    filler by more than one event.
Candidate spans (both tiers) come from the same whitespace-flexible regex
(`reanchor._flex_pattern`) reanchor() itself uses, case-sensitive first then
case-insensitive fallback. A literal with zero candidates anywhere in the
document gets a poison span past the document end (see
convert_back/sdoh_2022_t2.py's `to_ann` for why an unresolvable prediction
must never simply be dropped).
"""
import re

from multitask.span_solver import (
    flex_candidates as _candidates, overlaps_any as _overlaps_any, search_floor as _search_floor,
    search_nearest as _search_nearest,
)
from multitask.target_schema import SHAC_EVENT_TYPES, SHAC_ROLE_NAMES

AND_SEP = " {AND} "
TRIGGER_SENTINEL = "=TRIGGER="

# Only these three filler types ever carry an attribute value in the real
# corpus (see target_schema.py's SHAC_ATTR_VALUES) -- Status's filler is
# StatusTime (Tobacco/Alcohol/Drug/LivingStatus) or StatusEmploy
# (Employment); LivingStatus's Type filler is TypeLiving.
_FILLER_TYPE_TO_ATTR_NAME = {
    "StatusTime": "StatusTimeVal",
    "StatusEmploy": "StatusEmployVal",
    "TypeLiving": "TypeLivingVal",
}

_EVENT_TYPE_ALTS = "|".join(SHAC_EVENT_TYPES)
_ROLE_NAME_ALTS = "|".join(SHAC_ROLE_NAMES)

# A tag's alternatives are followed by a literal ">" in the pattern, so
# "Type" can never falsely prefix-match "<Type2>" -- the ">" immediately
# after "Type" in the pattern fails against the "2" in the input, and the
# regex engine backtracks to the "Type2" alternative. No explicit
# longest-first ordering needed.
_TAG_RE = re.compile(rf"<(?P<name>{_EVENT_TYPE_ALTS}|{_ROLE_NAME_ALTS})(?:!(?P<override>[A-Za-z]+))?>")
_BRACKET_RE = re.compile(r"^\s*\[(?P<value>[a-z_]+)\]")


def _default_filler_type(event_type, role_name):
    """The filler type `serialize_events` can omit as redundant (i.e. skip
    the "!FillerType" override tag for) -- everything the SHAC schema
    predicts from (event_type, role_name) alone, with no lookup needed at
    parse time."""
    base = re.sub(r"\d+$", "", role_name)
    if base == "Status":
        return "StatusEmploy" if event_type == "Employment" else "StatusTime"
    if base == "Type" and event_type == "LivingStatus":
        return "TypeLiving"
    return base


def serialize_events(events):
    """events: the same `{"events": [...]}`-shaped list the existing
    sdoh_event builder produces (each event: id/type/trigger_text/roles/
    attributes) -- this only changes how it's stringified, not what it
    contains."""
    return AND_SEP.join(_serialize_event(ev) for ev in events)


def _serialize_event(ev):
    attrs_by_role = {a["role"]: a for a in ev.get("attributes", [])}
    tokens = [f"<{ev['type']}>", ev["trigger_text"]]
    for role_name in SHAC_ROLE_NAMES:
        filler = ev.get("roles", {}).get(role_name)
        if filler is None:
            continue
        default_type = _default_filler_type(ev["type"], role_name)
        if filler["type"] == default_type:
            tokens.append(f"<{role_name}>")
        else:
            tokens.append(f"<{role_name}!{filler['type']}>")
        attr = attrs_by_role.get(role_name)
        if attr is not None:
            tokens.append(f"[{attr['value']}]")
        filler_text = filler["text"]
        tokens.append(TRIGGER_SENTINEL if filler_text == ev["trigger_text"] else filler_text)
    return " ".join(tokens)


def _lex_event(chunk):
    """One "<EventType> trigger ..." chunk (already split on AND_SEP) ->
    {"type", "trigger_text", "roles": {role_name: {"type", "text",
    "_sentinel"}}, "attributes"}, or None if it doesn't even start with a
    recognized event tag (a malformed/hallucinated chunk -- dropped, same
    "unparseable is a miss, not a crash" contract run_eval.py already
    documents for the JSON path's lenient parsing)."""
    chunk = chunk.strip()
    m = _TAG_RE.match(chunk)
    if m is None or m.group("name") not in SHAC_EVENT_TYPES:
        return None
    event_type = m.group("name")
    pos = m.end()

    role_tags = [tm for tm in _TAG_RE.finditer(chunk, pos) if tm.group("name") in SHAC_ROLE_NAMES]
    trigger_text = chunk[pos:role_tags[0].start() if role_tags else len(chunk)].strip()

    roles = {}
    attributes = []
    for i, tm in enumerate(role_tags):
        role_name = tm.group("name")
        override = tm.group("override")
        filler_type = override if override else _default_filler_type(event_type, role_name)
        rest_start = tm.end()
        rest_end = role_tags[i + 1].start() if i + 1 < len(role_tags) else len(chunk)
        segment = chunk[rest_start:rest_end]

        value = None
        bm = _BRACKET_RE.match(segment)
        if bm:
            value = bm.group("value")
            segment = segment[bm.end():]

        filler_text = segment.strip()
        is_sentinel = filler_text == TRIGGER_SENTINEL
        roles[role_name] = {
            "type": filler_type,
            "text": trigger_text if is_sentinel else filler_text,
            "_sentinel": is_sentinel,
        }
        if value is not None:
            attr_name = _FILLER_TYPE_TO_ATTR_NAME.get(filler_type)
            if attr_name is not None:
                attributes.append({"role": role_name, "name": attr_name, "value": value})

    return {"type": event_type, "trigger_text": trigger_text, "roles": roles, "attributes": attributes}


def _lex(seq_str):
    seq_str = seq_str.strip()
    if not seq_str:
        return []
    return [ev for ev in (_lex_event(chunk) for chunk in seq_str.split(AND_SEP)) if ev is not None]


def parse_and_resolve(seq_str, doc_text=None):
    """Text-only inverse of `serialize_events` -- reproduces the same
    `{"events": [...]}` shape the builder produces (ignoring spans
    entirely), for the structural round-trip fixture. `doc_text` is unused
    here (kept for signature symmetry with `parse_and_resolve_with_spans`);
    span recovery needs it, plain text reconstruction never does."""
    events = []
    for i, ev in enumerate(_lex(seq_str), start=1):
        roles_out = {name: {"type": f["type"], "text": f["text"]} for name, f in ev["roles"].items()}
        events.append({
            "id": f"E{i}",
            "type": ev["type"],
            "trigger_text": ev["trigger_text"],
            "roles": roles_out,
            "attributes": ev["attributes"],
        })
    return {"events": events}


def parse_and_resolve_with_spans(seq_str, doc_text):
    """Same shape as `parse_and_resolve`, plus a resolved `"span"` on the
    trigger (as `"trigger_span"`) and on every role filler -- consumed
    directly by convert_back/sdoh_2022_t2_seq2seq.py's `to_ann`, which
    calls this and only this to place every T-line. See module docstring
    for the two-tier (trigger-chain / per-role-nearest) resolution this
    uses instead of a flat single-chain solver."""
    events = _lex(seq_str)
    poison = [len(doc_text) + 1]

    def next_poison():
        span = (poison[0], poison[0] + 1)
        poison[0] += 1
        return span

    used_spans = []
    cursor = 0
    out_events = []
    for ei, ev in enumerate(events):
        trig_span = _search_floor(_candidates(ev["trigger_text"], doc_text), used_spans, cursor)
        if trig_span is None:
            trig_span = next_poison()
        else:
            used_spans.append(trig_span)
            cursor = max(cursor, trig_span[1])

        roles_out = {}
        for role_name, filler in ev["roles"].items():
            if filler["_sentinel"]:
                roles_out[role_name] = {"type": filler["type"], "text": filler["text"], "span": trig_span}
                continue
            f_span = _search_nearest(_candidates(filler["text"], doc_text), trig_span[1])
            if f_span is None:
                f_span = next_poison()
            roles_out[role_name] = {"type": filler["type"], "text": filler["text"], "span": f_span}

        out_events.append({
            "id": f"E{ei + 1}",
            "type": ev["type"],
            "trigger_text": ev["trigger_text"],
            "trigger_span": trig_span,
            "roles": roles_out,
            "attributes": ev["attributes"],
        })
    return {"events": out_events}

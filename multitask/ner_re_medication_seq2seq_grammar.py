"""
Microsoft-Nuance-style tag-delimited seq2seq target for ner_re_medication
(2018 n2c2 Track 2), additive sibling of multitask/seq2seq_grammar.py's
sdoh_event_seq2seq grammar -- same `<Tag> text ... {AND} ...` shape, applied
to this task's drugs/unlinked target instead of SDOH's events/roles.

Grammar:
    drug_chunk     := "<Drug>" drug_text role*
    role           := "<" AttrType ">" filler_text      # AttrType may repeat
    unlinked_chunk := "<" AttrType ">" filler_text        # standalone
    sequence       := (drug_chunk | unlinked_chunk) (" {AND} " ...)*

AttrType is one of target_schema.MEDICATION_ATTR_TYPES. Example:
    <Drug> Lipitor <Strength> 10 mg <Reason> high cholesterol {AND} \
<Drug> Colace {AND} <Reason> occasional cough

Deliberately no `parse_and_resolve_with_spans`/span-resolution logic in
this module: resolving character offsets for this corpus needs the
purpose-built disambiguation already proven in
multitask.convert_back.ner_re_2018_t2 (`_pick_drug_span`'s filler-support
heuristic, the two-tier exclusive/non-exclusive filler reanchor, the
unlinked-mention reanchor-with-cursor) -- see that module's docstring for
why plain nearest/floor search under-scores on this dataset's dense
decoy repetition. `parse_and_resolve` here produces the exact same
{"drugs": [...], "unlinked": [...]} shape that module's own `to_ann`
already accepts, so multitask/convert_back/ner_re_2018_t2_seq2seq.py's
`to_ann` is a thin wrapper: `parse_and_resolve` then call that function
directly, rather than re-deriving a second, divergence-prone copy of its
span-resolution logic here.
"""
import re

from multitask.target_schema import MEDICATION_ATTR_TYPES

AND_SEP = " {AND} "

_TAG_ALTS = "|".join(["Drug"] + list(MEDICATION_ATTR_TYPES))
_TAG_RE = re.compile(rf"<(?P<name>{_TAG_ALTS})>")


def serialize_events(target):
    """target: {"drugs": [{"text", "<AttrType>": [...], ...}, ...],
    "unlinked": [{"type", "text"}, ...]} (same shape as the JSON-path
    ner_re_medication target). -> tag-delimited str, or "" if both lists
    are empty."""
    chunks = []
    for drug in target.get("drugs") or []:
        parts = [f"<Drug> {drug.get('text', '')}"]
        for attr_type in MEDICATION_ATTR_TYPES:
            for filler_text in drug.get(attr_type) or []:
                parts.append(f"<{attr_type}> {filler_text}")
        chunks.append(" ".join(parts))
    for other in target.get("unlinked") or []:
        chunks.append(f"<{other.get('type')}> {other.get('text', '')}")
    return AND_SEP.join(chunks)


def _lex_chunk(chunk):
    """One chunk's `<Tag> text <Tag2> text2 ...` -> [(tag_name, text), ...],
    text spanning from just after a tag to just before the next one (or end
    of chunk), stripped."""
    tags = list(_TAG_RE.finditer(chunk))
    items = []
    for i, m in enumerate(tags):
        start = m.end()
        end = tags[i + 1].start() if i + 1 < len(tags) else len(chunk)
        items.append((m.group("name"), chunk[start:end].strip()))
    return items


def parse_and_resolve(seq_str):
    """Inverse of serialize_events -- text-level only, no character spans
    (see module docstring for why span resolution lives in convert_back
    instead). A chunk whose first tag is Drug becomes a drug entry (every
    later tag in that chunk a role filler); any other chunk becomes one or
    more unlinked entries (normally just one, per the grammar, but a
    malformed/multi-tag chunk from a real model degrades gracefully into
    several unlinked entries rather than being dropped)."""
    drugs, unlinked = [], []
    for chunk in (seq_str or "").split(AND_SEP):
        items = _lex_chunk(chunk)
        if not items:
            continue
        if items[0][0] == "Drug":
            drug = {"text": items[0][1]}
            for attr_type, filler_text in items[1:]:
                drug.setdefault(attr_type, []).append(filler_text)
            drugs.append(drug)
        else:
            for attr_type, filler_text in items:
                unlinked.append({"type": attr_type, "text": filler_text})
    return {"drugs": drugs, "unlinked": unlinked}

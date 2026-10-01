"""
Microsoft-Nuance-style tag-delimited seq2seq target for medication_event
(2022 n2c2 Track 1 / CMED), additive sibling of
multitask/seq2seq_grammar.py's sdoh_event_seq2seq grammar -- single-trigger
events with closed-vocabulary attributes only (no filler spans copied from
doc_text), so resolution needs nothing beyond a single reanchor() call per
event (see multitask/convert_back/medication_event_2022_t1_seq2seq.py) --
no shared span_solver dependency at all.

Grammar:
    event    := "<" EventType ">" trigger_text ("<" AttrName ">" "[" attr_value "]")*
    sequence := event (" {AND} " event)*

EventType in CMED_EVENT_TYPES, AttrName in CMED_ATTR_NAMES. Only Disposition
events carry non-empty attributes in the released data, but the grammar
itself doesn't special-case that. Example:
    <Disposition> Prozac <Action> [Start] <Certainty> [Certain] <Temporality> [Present] <Actor> [Physician] <Negation> [NotNegated] {AND} <NoDisposition> barbituate
"""
import re

from inference_and_eval.target_schema import CMED_ATTR_NAMES, CMED_EVENT_TYPES

AND_SEP = " {AND} "

_EVENT_TAG_ALTS = "|".join(CMED_EVENT_TYPES)
_ATTR_TAG_ALTS = "|".join(CMED_ATTR_NAMES)
_TAG_RE = re.compile(rf"<(?P<name>{_EVENT_TAG_ALTS}|{_ATTR_TAG_ALTS})>")
_BRACKET_RE = re.compile(r"^\[(?P<value>.*)\]$")


def serialize_events(target):
    chunks = []
    for ev in target.get("events") or []:
        parts = [f"<{ev['type']}> {ev.get('trigger_text', '')}"]
        for attr_name in CMED_ATTR_NAMES:
            value = (ev.get("attributes") or {}).get(attr_name)
            if value is not None:
                parts.append(f"<{attr_name}> [{value}]")
        chunks.append(" ".join(parts))
    return AND_SEP.join(chunks)


def _lex_chunk(chunk):
    tags = list(_TAG_RE.finditer(chunk))
    items = []
    for i, m in enumerate(tags):
        start = m.end()
        end = tags[i + 1].start() if i + 1 < len(tags) else len(chunk)
        items.append((m.group("name"), chunk[start:end].strip()))
    return items


def parse_and_resolve(seq_str):
    events = []
    for chunk in (seq_str or "").split(AND_SEP):
        items = _lex_chunk(chunk)
        if not items:
            continue
        ev_type, trigger_text = items[0]
        if ev_type not in CMED_EVENT_TYPES:
            continue
        attributes = {}
        for attr_name, filler_text in items[1:]:
            if attr_name not in CMED_ATTR_NAMES:
                continue
            m = _BRACKET_RE.match(filler_text)
            attributes[attr_name] = m.group("value") if m else filler_text
        events.append({"type": ev_type, "trigger_text": trigger_text, "attributes": attributes})
    return {"events": events}

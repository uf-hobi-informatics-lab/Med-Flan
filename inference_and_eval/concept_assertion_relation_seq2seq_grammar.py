"""
Microsoft-Nuance-style tag-delimited seq2seq target for
concept_assertion_relation (2010 i2b2/VA), additive sibling of
multitask/seq2seq_grammar.py's sdoh_event_seq2seq grammar.

i2b2 relations have no trigger/span of their own -- they're typed links
between two already-typed concepts. Concept resolution is delegated
entirely to the existing multitask/convert_back/i2b2_2010.py scorer (its
own reanchor()-based _resolve_predicted_concepts ignores predicted ids and
re-resolves every concept from text), so this module mints fresh local ids
in emission order and never touches doc_text or a span solver. The one new
primitive here is resolve_against_emitted(), a local/doc-text-free lookup
for which already-emitted concept a relation's Arg1/Arg2 refers to when two
emitted concepts share the same (type, text).

Grammar:
    concept  := "<" ConceptType ">" ["[" assertion_value "]"] concept_text
    relation := "<" RelationType ">" "<Arg1!" ConceptType ">" arg1_text "<Arg2!" ConceptType ">" arg2_text
    sequence := (concept | relation) (" {AND} " ...)*   # concepts emitted before any relation referencing them

ConceptType in I2B2_CONCEPT_TYPES, assertion_value in I2B2_ASSERTIONS
(present only for "problem" concepts), RelationType in I2B2_RELATION_TYPES.
<Arg1!Type>/<Arg2!Type> reuse SDOH's <RoleName!FillerType> override-tag
syntax, repurposed to carry the referenced concept's type inline (needed
because resolution matches on (type, text)). Example:
    <problem> [present] chest pain {AND} <treatment> aspirin {AND} <TrAP> <Arg1!treatment> aspirin <Arg2!problem> chest pain
"""
import re

from inference_and_eval.target_schema import I2B2_ASSERTIONS, I2B2_CONCEPT_TYPES, I2B2_RELATION_TYPES

AND_SEP = " {AND} "
UNMATCHED = "__unmatched__"

_CONCEPT_TAG_ALTS = "|".join(I2B2_CONCEPT_TYPES)
_RELATION_TAG_ALTS = "|".join(I2B2_RELATION_TYPES)
_ARG_TAG_ALTS = "|".join(f"Arg1!{t}|Arg2!{t}" for t in I2B2_CONCEPT_TYPES)
_TAG_RE = re.compile(rf"<(?P<name>{_RELATION_TAG_ALTS}|{_CONCEPT_TAG_ALTS}|{_ARG_TAG_ALTS})>")
_ARG_NAME_RE = re.compile(rf"^(?P<argnum>Arg1|Arg2)!(?P<ctype>{_CONCEPT_TAG_ALTS})$")
_BRACKET_RE = re.compile(r"^\[(?P<value>[^\]]*)\]\s*(?P<rest>.*)$", re.DOTALL)


def _order_chunks(concepts, relations):
    """Interleave: emit concepts in document order, and emit each relation
    right after the LATER (by document position) of its two endpoint
    concepts. This keeps a relation's Arg1/Arg2 close to their referent in
    the emission sequence -- batching all concepts before all relations
    (the naive ordering) would mean every same-(type,text) duplicate in
    the whole document is already "emitted" by the time any relation is
    parsed, destroying the positional signal resolve_against_emitted's
    recency/nearest-hint heuristics rely on to disambiguate duplicates."""
    order_by_id = {c["id"]: i for i, c in enumerate(concepts)}
    relations_after = {}
    for r in relations:
        i1, i2 = order_by_id.get(r["arg1"]), order_by_id.get(r["arg2"])
        if i1 is None or i2 is None:
            continue  # endpoint isn't one of this target's own concepts -- drop, conservative
        relations_after.setdefault(max(i1, i2), []).append(r)

    ordered = []
    for i, c in enumerate(concepts):
        ordered.append(("concept", c))
        for r in relations_after.get(i, []):
            ordered.append(("relation", r))
    return ordered


def serialize_target(target):
    concepts = target.get("concepts") or []
    relations = target.get("relations") or []
    id_to_concept = {c["id"]: c for c in concepts}

    chunks = []
    for kind, item in _order_chunks(concepts, relations):
        if kind == "concept":
            parts = [f"<{item['type']}>"]
            if item.get("assertion"):
                parts.append(f"[{item['assertion']}]")
            parts.append(item.get("text", ""))
            chunks.append(" ".join(parts))
        else:
            arg1c, arg2c = id_to_concept[item["arg1"]], id_to_concept[item["arg2"]]
            chunks.append(
                f"<{item['type']}> <Arg1!{arg1c['type']}> {arg1c.get('text', '')} "
                f"<Arg2!{arg2c['type']}> {arg2c.get('text', '')}"
            )
    return AND_SEP.join(chunks)


def _lex_chunk(chunk):
    tags = list(_TAG_RE.finditer(chunk))
    items = []
    for i, m in enumerate(tags):
        start = m.end()
        end = tags[i + 1].start() if i + 1 < len(tags) else len(chunk)
        items.append((m.group("name"), chunk[start:end].strip()))
    return items


def resolve_against_emitted(text, concept_type, emitted_concepts, hint_index=None):
    """emitted_concepts: list of {"id","type","text"} dicts already emitted,
    in emission order. Returns the matching concept's local id, or the
    UNMATCHED sentinel (the same one i2b2_2010.py's own _score_doc already
    uses for an unresolvable relation endpoint) if no (type, text) match
    exists among them."""
    candidates = [
        (i, c) for i, c in enumerate(emitted_concepts)
        if c["type"] == concept_type and c["text"] == text
    ]
    if not candidates:
        return UNMATCHED
    if hint_index is not None:
        # On a tie in absolute distance, prefer the candidate AFTER
        # hint_index over the one before it -- relations typically link a
        # concept to another appearing immediately after it in the same
        # sentence (e.g. a TeRP test result immediately followed by the
        # problem it reveals), so when two same-(type,text) duplicates are
        # equidistant, the later one is the more likely true referent.
        best = min(candidates, key=lambda ic: (abs(ic[0] - hint_index), ic[0] <= hint_index))
    else:
        best = max(candidates, key=lambda ic: ic[0])
    return best[1]["id"]


def parse_and_resolve(seq_str):
    concepts = []
    relations = []
    emitted = []  # {"id","type","text"}, parallel to `concepts`

    for chunk in (seq_str or "").split(AND_SEP):
        items = _lex_chunk(chunk)
        if not items:
            continue
        head_name, head_text = items[0]

        if head_name in I2B2_CONCEPT_TYPES:
            # Only strip a leading "[...]" as the assertion marker when its
            # content is actually a valid assertion value -- concept text
            # itself can legitimately start with a bracket (e.g. lab-value
            # notation like "[Hgb]"), and assertion only applies to
            # "problem" concepts.
            m = _BRACKET_RE.match(head_text)
            if m and head_name == "problem" and m.group("value") in I2B2_ASSERTIONS:
                assertion, concept_text = m.group("value"), m.group("rest").strip()
            else:
                assertion, concept_text = None, head_text.strip()
            cid = f"c{len(concepts)}"
            concepts.append({"id": cid, "type": head_name, "text": concept_text, "assertion": assertion})
            emitted.append({"id": cid, "type": head_name, "text": concept_text})
            continue

        if head_name in I2B2_RELATION_TYPES:
            arg1_type = arg2_type = None
            arg1_text = arg2_text = None
            for arg_name, arg_text in items[1:]:
                am = _ARG_NAME_RE.match(arg_name)
                if am is None:
                    continue
                if am.group("argnum") == "Arg1":
                    arg1_type, arg1_text = am.group("ctype"), arg_text
                else:
                    arg2_type, arg2_text = am.group("ctype"), arg_text

            # This relation was interleaved (see _order_chunks) right after
            # the LATER of its two true arguments, so whichever of
            # Arg1/Arg2 exactly matches the most-recently-emitted concept
            # is unambiguous by construction -- use it to anchor
            # resolution of the other, less pinned-down argument, rather
            # than guessing both from scratch.
            trailing = emitted[-1] if emitted else None
            if trailing and trailing["type"] == arg1_type and trailing["text"] == arg1_text:
                arg1_id, arg1_idx = trailing["id"], len(emitted) - 1
                arg2_id = resolve_against_emitted(arg2_text, arg2_type, emitted, hint_index=arg1_idx)
            elif trailing and trailing["type"] == arg2_type and trailing["text"] == arg2_text:
                arg2_id, arg2_idx = trailing["id"], len(emitted) - 1
                arg1_id = resolve_against_emitted(arg1_text, arg1_type, emitted, hint_index=arg2_idx)
            else:
                arg1_id = resolve_against_emitted(arg1_text, arg1_type, emitted)
                arg1_idx = next((i for i, c in enumerate(emitted) if c["id"] == arg1_id), None)
                arg2_id = resolve_against_emitted(arg2_text, arg2_type, emitted, hint_index=arg1_idx)
            relations.append({"type": head_name, "arg1": arg1_id, "arg2": arg2_id})
            continue

    return {
        "concepts": [{"id": c["id"], "type": c["type"], "text": c["text"], "assertion": c["assertion"]} for c in concepts],
        "relations": relations,
    }

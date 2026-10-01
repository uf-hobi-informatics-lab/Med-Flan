"""
Seq2seq target grammar for `family_history_seq2seq` (see the approved plan's
Phase 1). Additive sibling to the existing JSON-based `family_history` task
-- nothing here is consumed by that path, and nothing in it is changed.

Unlike `seq2seq_grammar.py` (sdoh_event_seq2seq), this task's gold data has
no character offsets anywhere (see builders/family_history_2019_t2.py's
docstring) -- scoring is done via the official 2019 n2c2 Track 2 Subtask 2
TSV/tuple matcher, not BRAT span overlap. So there is no
`parse_and_resolve_with_spans` variant here, no `doc_text` parameter
anywhere in this module, and no span_solver.py dependency: this is a pure
string-format change over the exact same target dict shape
`builders/family_history_2019_t2.py` already produces.

Grammar:
    record   := "<LivingStatus>" member_text "<Side>" "[" side_value "]" "<Score>" "[" score_value "]"
              | "<Observation>" member_text "<Side>" "[" side_value "]" "<Text>" observation_text "<Negation>" "[" negation_value "]"
    sequence := record (" {AND} " record)*

`side_value` in FH_SIDES, `score_value` in FH_LIVING_SCORES, `negation_value`
in FH_NEGATION (target_schema.py's existing vocabs, reused verbatim -- the
JSON path and this path can never disagree on what counts as a legal
value). `member_text`/`observation_text` are free text in the JSON schema
(`Str()`), though empirically drawn from a small closed set in the real
corpus (Father/Mother/Parent/... for member) -- this grammar does not
enforce that at parse time, same "lenient parse, let the scorer judge"
contract every other *_seq2seq_grammar.py module in this family follows.

Example:
    <LivingStatus> Father <Side> [Paternal] <Score> [1] {AND} <Observation> Mother <Side> [Maternal] <Text> breast cancer <Negation> [Non_Negated]
"""
import re

AND_SEP = " {AND} "

#  Single literal space delimiters + greedy captures (not `\s*` boundaries /
# lazy `.*?`) are deliberate: the real corpus has rows that differ *only* in
# leading/trailing whitespace within a field (e.g. one doc's gold has both
# an Observation with text "ADHD" and one with text "ADHD " for the same
# member/side/negation -- confirmed via the raw TSV, matches target_schema
# .py's own note that this corpus has upstream annotation inconsistencies).
# A `\s*`/lazy-`.*?` boundary would silently collapse that distinction and
# break the exact-equality gold round-trip bar. Greedy `.*` backtracks from
# the right to find the single required literal delimiter, so any leading/
# trailing whitespace *inside* a field's own text is captured as part of
# that field, not eaten by the delimiter -- this is exact as long as the
# fixed tag literals never occur as a substring inside member/text (already
# confirmed: 0 occurrences of "{AND}"/tag strings in the real corpus).
_LIVING_RE = re.compile(
    r"^<LivingStatus> (?P<member>.*) <Side> \[(?P<side>[^\]]*)\] <Score> \[(?P<score>[^\]]*)\]$"
)
_OBS_RE = re.compile(
    r"^<Observation> (?P<member>.*) <Side> \[(?P<side>[^\]]*)\] <Text> (?P<text>.*) "
    r"<Negation> \[(?P<negation>[^\]]*)\]$"
)


def serialize_records(target):
    """target: the same `{"living_status": [...], "observations": [...]}`
    shape builders/family_history_2019_t2.py already produces -- this only
    changes how it's stringified, not what it contains. Living-status
    records are emitted before observation records (matches that module's
    own TSV row order); relative order carries no scoring weight either
    way (Subtask 2 is scored as an unordered tuple set)."""
    chunks = []
    for ls in target.get("living_status", []):
        chunks.append(f"<LivingStatus> {ls['member']} <Side> [{ls['side']}] <Score> [{ls['score']}]")
    for ob in target.get("observations", []):
        chunks.append(
            f"<Observation> {ob['member']} <Side> [{ob['side']}] <Text> {ob['text']} "
            f"<Negation> [{ob['negation']}]"
        )
    return AND_SEP.join(chunks)


def parse_and_resolve(seq_str):
    """Text-only inverse of `serialize_records` -- reproduces the same
    `{"living_status": [...], "observations": [...]}` shape the builder
    produces. A chunk that doesn't match either record pattern (a
    malformed/hallucinated chunk) is dropped, same "unparseable is a miss,
    not a crash" contract every other *_seq2seq_grammar.py module in this
    family follows."""
    seq_str = seq_str.strip()
    living_status, observations = [], []
    if not seq_str:
        return {"living_status": living_status, "observations": observations}

    for chunk in seq_str.split(AND_SEP):
        chunk = chunk.strip()
        m = _LIVING_RE.match(chunk)
        if m:
            living_status.append({
                "member": m.group("member"),
                "side": m.group("side"),
                "score": m.group("score"),
            })
            continue
        m = _OBS_RE.match(chunk)
        if m:
            observations.append({
                "member": m.group("member"),
                "side": m.group("side"),
                "text": m.group("text"),
                "negation": m.group("negation"),
            })
    return {"living_status": living_status, "observations": observations}

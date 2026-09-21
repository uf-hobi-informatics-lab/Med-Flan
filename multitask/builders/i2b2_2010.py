"""
2010 i2b2/VA (Concepts, Assertions, and Relations in Clinical Text) builder.

Not BRAT -- concepts/assertions/relations are line-based standoff files
keyed by 1-indexed (line, token) positions into whitespace-split tokens,
not character offsets (multitask/brat.py's parser doesn't apply here).
Confirmed .con/.ast/.rel line syntax against bigbio/n2c2_2010's public
dataloader script (published file-format documentation, not the
DUA-restricted clinical text itself):

    .con:  c="<text>" <sl>:<st> <el>:<et>||t="<problem|test|treatment>"
    .ast:  c="<text>" <sl>:<st> <el>:<et>||t="problem"||a="<assertion>"
    .rel:  c="<text1>" <sl1>:<st1> <el1>:<et1>||r="<type>"||c="<text2>" <sl2>:<st2> <el2>:<et2>

Exact on-disk subdirectory layout varies by release/mirror (nested
per-institution folders, separate txt/concept/ast/rel subfolders, etc.), so
rather than assume one layout this recursively globs for *.txt/*.con/*.ast/
*.rel under the split root and matches them up by filename stem. No real
data has been obtained yet (DUA-gated), so this is untested against a real
archive -- if the actual layout produces stem collisions across
institutions, that's the most likely fixup needed once real data lands.

Concepts, assertions (only ever present on "problem"-type concepts), and
relations are merged into one JSON target per document -- see
multitask.schemas INSTRUCTIONS["concept_assertion_relation"] for the shape.
"""
import re
from pathlib import Path

from multitask.io_utils import read_text
from multitask.schemas import example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "2010_i2b2_va"
SPLIT_DIRS = {"train": BASE / "train", "test": BASE / "test"}

_CON_RE = re.compile(r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|t="(.*?)"$')
_AST_RE = re.compile(r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|t="(.*?)"\|\|a="(.*?)"$')
_REL_RE = re.compile(
    r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|r="(.*?)"\|\|c="(.*?)" (\d+):(\d+) (\d+):(\d+)$'
)


def _index_by_stem(root, suffix):
    return {p.stem: p for p in root.rglob(f"*{suffix}")}


def _line_token_offsets(doc_text):
    """Map every 1-indexed (line, 0-indexed token) position to a character
    (start, end) span into doc_text, by re-deriving the same whitespace
    tokenization the challenge's own line:token offsets are defined over."""
    offsets = {}
    pos = 0
    for line_no, line in enumerate(doc_text.split("\n"), start=1):
        for tok_no, m in enumerate(re.finditer(r"\S+", line)):
            offsets[(line_no, tok_no)] = (pos + m.start(), pos + m.end())
        pos += len(line) + 1  # +1 for the '\n' that str.split("\n") consumed
    return offsets


def _resolve_span(offsets, sline, stok, eline, etok):
    start = offsets.get((sline, stok))
    end = offsets.get((eline, etok))
    if start is None or end is None:
        return None
    return (start[0], end[1])


def _parse_con_file(path):
    concepts = []
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        m = _CON_RE.match(line)
        if not m:
            continue
        text, sl, st, el, et, ctype = m.groups()
        concepts.append({"text": text, "raw_span": (int(sl), int(st), int(el), int(et)), "type": ctype})
    return concepts


def _parse_ast_file(path):
    assertions = {}
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        m = _AST_RE.match(line)
        if not m:
            continue
        _text, sl, st, el, et, _ctype, assertion = m.groups()
        assertions[(int(sl), int(st), int(el), int(et))] = assertion
    return assertions


def _parse_rel_file(path):
    relations = []
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        m = _REL_RE.match(line)
        if not m:
            continue
        _t1, sl1, st1, el1, et1, rtype, _t2, sl2, st2, el2, et2 = m.groups()
        relations.append({
            "type": rtype,
            "raw_span1": (int(sl1), int(st1), int(el1), int(et1)),
            "raw_span2": (int(sl2), int(st2), int(el2), int(et2)),
        })
    return relations


def _iter_records(split):
    """Yields (doc_id, doc_text, concepts, relations) with concepts carrying
    a "_char_span" (exact, derived from the released line:token offsets --
    not reanchored) alongside "id"/"type"/"text"/"assertion". Shared by the
    public builder (which strips "_char_span" for the JSON target) and
    convert_back/i2b2_2010.py's gold loader (which needs it)."""
    split_dir = SPLIT_DIRS.get(split)
    if split_dir is None or not split_dir.exists():
        return
    txt_by_stem = _index_by_stem(split_dir, ".txt")
    con_by_stem = _index_by_stem(split_dir, ".con")
    ast_by_stem = _index_by_stem(split_dir, ".ast")
    rel_by_stem = _index_by_stem(split_dir, ".rel")

    for stem, txt_path in sorted(txt_by_stem.items()):
        con_path = con_by_stem.get(stem)
        if con_path is None:
            continue
        doc_text = read_text(txt_path)
        offsets = _line_token_offsets(doc_text)

        raw_concepts = _parse_con_file(con_path)
        assertions = _parse_ast_file(ast_by_stem[stem]) if stem in ast_by_stem else {}
        raw_relations = _parse_rel_file(rel_by_stem[stem]) if stem in rel_by_stem else []

        resolved = []
        for c in raw_concepts:
            span = _resolve_span(offsets, *c["raw_span"])
            if span is None:
                continue
            resolved.append({"raw_span": c["raw_span"], "char_span": span, "type": c["type"], "text": c["text"]})
        # True document order -- both natural for a generative model and
        # required for convert_back's left-to-right reanchor() matching,
        # same reasoning as ner_re_2018_t2/medication_event_2022_t1.
        resolved.sort(key=lambda c: c["char_span"])

        concepts = []
        raw_span_to_id = {}
        for i, c in enumerate(resolved):
            cid = f"c{i}"
            raw_span_to_id[c["raw_span"]] = cid
            assertion = assertions.get(c["raw_span"]) if c["type"] == "problem" else None
            concepts.append({
                "id": cid, "type": c["type"], "text": c["text"], "assertion": assertion,
                "_char_span": c["char_span"],
            })

        relations = []
        for r in raw_relations:
            arg1 = raw_span_to_id.get(r["raw_span1"])
            arg2 = raw_span_to_id.get(r["raw_span2"])
            if arg1 is None or arg2 is None:
                continue  # one endpoint wasn't a parsed concept -- drop, conservative
            relations.append({"type": r["type"], "arg1": arg1, "arg2": arg2})

        yield stem, doc_text, concepts, relations


def build_concept_assertion_relation(split):
    for doc_id, doc_text, concepts, relations in _iter_records(split):
        target = {
            "concepts": [
                {"id": c["id"], "type": c["type"], "text": c["text"], "assertion": c["assertion"]}
                for c in concepts
            ],
            "relations": relations,
        }
        yield example("concept_assertion_relation", "2010_i2b2_va", split, doc_id, doc_text, target)

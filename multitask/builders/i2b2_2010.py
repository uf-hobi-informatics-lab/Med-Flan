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
*.rel under each split's root(s) and matches them up by filename stem.
Confirmed against the real archive layout: train is
concept_assertion_relation_training_data/{beth,partners}/{txt,concept,ast,rel}/
(plus a partners/unannotated/ of extra unlabeled .txt with no matching
.con -- skipped, since a doc needs a .con to be a record). Test ships as
two separate archives that don't share a root: annotations in
reference_standard_for_test_data/{ast,concepts,rel}/ (dir named "concepts",
files still end in .con) and text in test_data.tar.gz's test_data/*.txt
(extract by hand, gitignored raw data -- see .gitignore's 2010i2b2/ entry)
-- so SPLIT_DIRS entries are lists of roots, merged by stem across all of
them, rather than one root each.

Concepts, assertions (only ever present on "problem"-type concepts), and
relations are merged into one JSON target per document -- see
multitask.schemas TASKS["concept_assertion_relation"]["instruction"] for the shape.
"""
import re
from pathlib import Path

from multitask.chunk_utils import section_spans
from multitask.io_utils import read_text
from multitask.schemas import example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "2010i2b2"
SPLIT_DIRS = {
    "train": [BASE / "concept_assertion_relation_training_data"],
    "test": [BASE / "reference_standard_for_test_data", BASE / "test_data"],
}

_CON_RE = re.compile(r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|t="(.*?)"$')
_AST_RE = re.compile(r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|t="(.*?)"\|\|a="(.*?)"$')
_REL_RE = re.compile(
    r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|r="(.*?)"\|\|c="(.*?)" (\d+):(\d+) (\d+):(\d+)$'
)


def _index_by_stem(roots, suffix):
    return {p.stem: p for root in roots for p in root.rglob(f"*{suffix}")}


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
    split_dirs = SPLIT_DIRS.get(split)
    if not split_dirs:
        return
    roots = [d for d in split_dirs if d.exists()]
    if not roots:
        return
    txt_by_stem = _index_by_stem(roots, ".txt")
    con_by_stem = _index_by_stem(roots, ".con")
    ast_by_stem = _index_by_stem(roots, ".ast")
    rel_by_stem = _index_by_stem(roots, ".rel")

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
            # The .con file's c="..." text is always lowercased by the i2b2
            # 2010 annotation format itself (confirmed empirically: 0/31159
            # concept-text fields in the released test .con files contain an
            # uppercase letter), which does not match the source document's
            # actual casing. Use the resolved span's real slice of doc_text
            # instead -- a generative model can only be expected to
            # reproduce what it actually saw in the input, and
            # convert_back/i2b2_2010.py's case-sensitive reanchor() must be
            # able to recover this exact span from that text.
            resolved.append(
                {"raw_span": c["raw_span"], "char_span": span, "type": c["type"], "text": doc_text[span[0]:span[1]]}
            )
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


def _build_target(concepts, relations):
    """concepts/relations -> the JSON target shape shared by the end-to-end
    task and the re_only/section subtasks -- concepts carry "_char_span" for
    internal use (chunking) but that key is never part of the target."""
    return {
        "concepts": [
            {"id": c["id"], "type": c["type"], "text": c["text"], "assertion": c["assertion"]}
            for c in concepts
        ],
        "relations": relations,
    }


def build_concept_assertion_relation(split):
    for doc_id, doc_text, concepts, relations in _iter_records(split):
        target = _build_target(concepts, relations)
        yield example("concept_assertion_relation", "2010_i2b2_va", split, doc_id, doc_text, target)


def build_concept_assertion_relation_ner_only(split):
    """Tier A subtask: concept spans + types only, no assertion, no
    relations -- see design.md."""
    for doc_id, doc_text, concepts, relations in _iter_records(split):
        entities = [{"type": c["type"], "text": c["text"]} for c in concepts]
        yield example(
            "concept_assertion_relation_ner_only", "2010_i2b2_va_ner_only", split,
            doc_id, doc_text, {"concepts": entities},
        )


def build_concept_assertion_relation_re_only(split):
    """Tier A subtask: concepts are pre-extracted and given in the input
    (EXTRACTED_CONCEPTS block); the target reuses _build_target, same full
    shape as the end-to-end task -- see design.md."""
    for doc_id, doc_text, concepts, relations in _iter_records(split):
        concept_list = "\n".join(f"- {c['type']}: {c['text']}" for c in concepts)
        input_text = f"{doc_text}\n\nEXTRACTED_CONCEPTS:\n{concept_list}"
        target = _build_target(concepts, relations)
        yield example(
            "concept_assertion_relation_re_only", "2010_i2b2_va_re_only", split,
            doc_id, input_text, target,
        )


def _iter_sections(doc_text, concepts, relations):
    """Yields (section_text, section_concepts, section_relations) for every
    section in doc_text, per chunk_utils.section_spans -- this format has no
    blank-line paragraph breaks at all, unlike 2018_n2c2_track2's BRAT .txt
    (confirmed empirically), so it needs a section-header boundary rule
    instead of chunk_utils.paragraph_spans. Sections with no concepts are
    skipped -- no training signal. Concepts/relations straddling a section
    boundary are dropped, same convention as chunk_utils.project_doc_onto_span
    (measured: 0/9413 gold relations cross a section boundary in this
    dataset -- see design.md).

    Concept ids are renumbered c0..cN-1 local to each section, not reused
    from the document-level numbering. Document-level ids reflect a
    concept's position among ALL of a document's concepts, so a section
    partway through a long note gets ids like ['c13','c14'] with no c0..c12
    -- unrecoverable from section-only input (measured: 85.9% of section
    examples had a first id != 'c0' before this fix). Renumbering per
    section is safe because nothing downstream keys off these ids matching
    the document-level ones: this dataset is HF-only and never reaches
    convert_back/run_eval (doc-level scoring stays authoritative, per
    design.md), and relations are remapped through the same local id map
    so they still resolve correctly within the section."""
    for start, end in section_spans(doc_text):
        sec_concepts_doc = [
            c for c in concepts if c["_char_span"][0] >= start and c["_char_span"][1] <= end
        ]
        if not sec_concepts_doc:
            continue
        local_id = {c["id"]: f"c{j}" for j, c in enumerate(sec_concepts_doc)}
        sec_concepts = [{**c, "id": local_id[c["id"]]} for c in sec_concepts_doc]
        sec_relations = [
            {**r, "arg1": local_id[r["arg1"]], "arg2": local_id[r["arg2"]]}
            for r in relations
            if r["arg1"] in local_id and r["arg2"] in local_id
        ]
        yield doc_text[start:end], sec_concepts, sec_relations


def build_concept_assertion_relation_section(split):
    """Section-window subtask (design.md's paragraph-chunking section): same
    concepts+assertions+relations target shape as the parent task, scoped to
    one clinical-note section instead of the whole note. HF-only, additive
    (train+test both built for experimentation flexibility, but not wired
    into the doc-level authoritative eval pipeline)."""
    for doc_id, doc_text, concepts, relations in _iter_records(split):
        for i, (sec_text, sec_concepts, sec_relations) in enumerate(
            _iter_sections(doc_text, concepts, relations)
        ):
            target = _build_target(sec_concepts, sec_relations)
            yield example(
                "concept_assertion_relation_section", "2010_i2b2_va_section", split,
                f"{doc_id}__s{i}", sec_text, target,
            )

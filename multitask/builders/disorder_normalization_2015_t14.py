"""
Builder for SemEval-2015 Task 14 (ShARe corpus) -- disorder-mention
extraction + UMLS/SNOMED-CT CUI normalization from MIMIC-II discharge
summaries / radiology reports.

Raw layout: semeval_2015/<split>/<doc_id>.text (+ .pipe stand-off
annotations; a redundant .jsonl pre-baked prompt file is ignored -- see
plan notes). Offsets in .pipe are relative to the *entire* .text file
content (metadata header line included), confirmed by direct slicing.

Column layout of each .pipe line (19 `|`-delimited fields), only the
first 3 used -- the rest are SemEval-2015 Task 1c template-attribute
fields (negation/subject/course/severity/...), out of scope here:
    0: source filename
    1: char-offset span, possibly discontinuous ("865-873,882-905")
    2: CUI, or the literal string "CUI-less"
"""
from pathlib import Path

from multitask.io_utils import read_text
from multitask.schemas import example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "semeval_2015"


def _parse_span(span_field):
    """"865-873,882-905" -> [(865, 873), (882, 905)]"""
    parts = []
    for chunk in span_field.split(","):
        a, b = chunk.split("-")
        parts.append((int(a), int(b)))
    return parts


def _parse_pipe_line(line, doc_text):
    fields = line.rstrip("\n").split("|")
    spans = _parse_span(fields[1])
    cui = fields[2]
    text = " ".join(doc_text[a:b] for a, b in spans)
    start = spans[0][0]
    return start, {"text": text, "cui": cui}


def _iter_docs(split):
    for pipe_path in sorted((BASE / split).glob("*.pipe")):
        doc_id = pipe_path.stem
        text_path = pipe_path.with_suffix(".text")
        doc_text = read_text(text_path)
        entities = []
        with open(pipe_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                start, entity = _parse_pipe_line(line, doc_text)
                entities.append((start, entity))
        entities.sort(key=lambda pair: pair[0])
        yield doc_id, doc_text, [entity for _, entity in entities]


def build_disorder_normalization(split):
    for doc_id, doc_text, entities in _iter_docs(split):
        target = {"entities": entities}
        yield example("disorder_normalization", "semeval_2015", split, doc_id, doc_text, target)

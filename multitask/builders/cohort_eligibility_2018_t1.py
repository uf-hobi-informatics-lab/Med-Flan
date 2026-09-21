"""
2018 n2c2 Track 1 (cohort eligibility) builder.

Per the user's explicit design change: one patient record is NOT one
13-key JSON example. Each of the 13 criteria becomes its OWN example --
same patient notes as input, one named criterion (with its definition and
a rarity note where applicable, from schemas.COHORT_CRITERIA*) appended to
the prompt, and a single-label {"label": "met"|"not met"} target. This
gives the model a clean single-label signal per criterion instead of
forcing 13 keys out of one generation, at the cost of 13x more examples
for this dataset (288 patients -> 3744 train-ish examples for train+test
combined here; train/test kept separate as usual).
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from multitask.schemas import COHORT_CRITERIA, cohort_criterion_prompt, example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "2018_n2c2_track1"
SPLIT_DIRS = {"train": BASE / "train", "test": BASE / "n2c2-t1_gold_standard_test_data" / "test"}

_CDATA_STRIP_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.S)


def _load_patient(xml_path):
    raw = xml_path.read_text(encoding="utf-8", errors="replace")
    # CDATA can contain characters ElementTree's parser is picky about in some
    # files; extract TEXT via the CDATA marker directly rather than relying on
    # the XML parser's CDATA handling for the note body.
    m = _CDATA_STRIP_RE.search(raw)
    root = ET.fromstring(raw) if m is None else None
    if m is not None:
        text = m.group(1)
        tags_match = re.search(r"<TAGS>(.*?)</TAGS>", raw, re.S)
        tags = {}
        if tags_match:
            for tag_m in re.finditer(r'<([A-Z0-9-]+)\s+met="([^"]+)"\s*/>', tags_match.group(1)):
                tags[tag_m.group(1)] = tag_m.group(2)
        return text, tags
    text_el = root.find("TEXT")
    text = text_el.text or "" if text_el is not None else ""
    tags = {}
    tags_el = root.find("TAGS")
    if tags_el is not None:
        for child in tags_el:
            tags[child.tag] = child.get("met")
    return text, tags


def build_cohort_eligibility(split):
    split_dir = SPLIT_DIRS.get(split)
    if split_dir is None or not split_dir.exists():
        return
    for xml_path in sorted(split_dir.glob("*.xml")):
        note_text, tags = _load_patient(xml_path)
        if not tags:
            continue
        for criterion in COHORT_CRITERIA:
            met = tags.get(criterion)
            if met is None:
                continue
            input_text = f"{cohort_criterion_prompt(criterion)}\n\nPatient notes:\n{note_text}"
            yield example("cohort_eligibility", "2018_n2c2_track1", split,
                          f"{xml_path.stem}#{criterion}", input_text, {"label": met})

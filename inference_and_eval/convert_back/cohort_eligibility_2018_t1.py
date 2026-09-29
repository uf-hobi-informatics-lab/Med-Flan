"""
JSON -> BRAT-style XML <TAGS> converter for 2018 n2c2 Track 1 (cohort
eligibility), so track1_eval_script/track1_eval.py can score model output
unmodified.

Unlike every other convert_back module, one patient here is NOT one
prediction: multitask.builders.cohort_eligibility_2018_t1 splits each
patient into 13 separate single-criterion examples (id
"<patient>#<CRITERION>"), so predictions must first be grouped back by
patient id before a per-patient XML file can be written. A patient is only
written out once all 13 criteria have a prediction (matching the gold
files, which always carry all 13) -- see group_by_patient().

track1_eval.py only ever reads the <TAGS> block (via `.//TAGS/*`), so the
written XML omits <TEXT> entirely; it just needs to be well-formed XML with
the same basename as its gold counterpart ("<patient>.xml").

Input target schema (see multitask.builders.cohort_eligibility_2018_t1):
    {"label": "met"}  # or "not met" -- one such dict per (patient, criterion)

Run standalone to self-check against gold on the test split (this dataset's
only local split besides train) -- should give P=R=F1=1.0 for every
criterion:
    python -m multitask.convert_back.cohort_eligibility_2018_t1
"""
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from inference_and_eval.schemas import COHORT_CRITERIA

ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = ROOT / "2018_n2c2_track1" / "track1_eval_script" / "track1_eval.py"
GOLD_DIR = ROOT / "2018_n2c2_track1" / "n2c2-t1_gold_standard_test_data" / "test"


def group_by_patient(predictions):
    """predictions: iterable of (locator, target) where locator is
    "<patient>#<CRITERION>" and target is {"label": "met"|"not met"}.
    Returns {patient_id: {criterion: label}}, dropping (with a stderr
    warning) any patient missing one or more of the 13 criteria."""
    by_patient = {}
    for locator, target in predictions:
        patient_id, _, criterion = locator.partition("#")
        by_patient.setdefault(patient_id, {})[criterion] = target.get("label")

    complete = {}
    for patient_id, tags in by_patient.items():
        missing = [c for c in COHORT_CRITERIA if c not in tags]
        if missing:
            print(f"warning: {patient_id} missing criteria {missing}, skipping", file=sys.stderr)
            continue
        complete[patient_id] = tags
    return complete


def to_xml(tags):
    """tags: {criterion: "met"|"not met"}, all 13 keys present. Returns the
    XML file text."""
    lines = ['<?xml version="1.0" encoding="UTF-8" ?>', "<PatientMatching>", "<TAGS>"]
    for criterion in COHORT_CRITERIA:
        lines.append(f'<{criterion} met="{escape(tags[criterion])}" />')
    lines += ["</TAGS>", "</PatientMatching>"]
    return "\n".join(lines) + "\n"


def write_xmls(predictions, out_dir):
    """predictions: iterable of (locator, target) -- see group_by_patient.
    Writes one <patient_id>.xml per complete patient under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for patient_id, tags in group_by_patient(predictions).items():
        (out_dir / f"{patient_id}.xml").write_text(to_xml(tags), encoding="utf-8")
    return out_dir


def _run_eval(gold_dir, system_dir):
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), str(gold_dir), str(system_dir)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)


def _self_check():
    """Round-trip gold test-split JSON (built directly from the gold XML)
    back to XML, then score against the real gold -- should be
    P=R=F1=1.0."""
    predictions = list(_examples_with_raw_target("test"))

    with tempfile.TemporaryDirectory() as tmp:
        write_xmls(predictions, tmp)
        print("=== self-check: gold JSON -> XML -> track1_eval.py ===")
        _run_eval(GOLD_DIR, tmp)


def _examples_with_raw_target(split):
    """Yields (locator, target) pairs straight from gold XML -- bypasses
    schemas.example(), which only keeps rendered text/messages, not the
    raw {"label": ...} target dict."""
    from inference_and_eval.builders.cohort_eligibility_2018_t1 import SPLIT_DIRS, _load_patient

    split_dir = SPLIT_DIRS[split]
    for xml_path in sorted(split_dir.glob("*.xml")):
        _, tags = _load_patient(xml_path)
        for criterion, met in tags.items():
            yield f"{xml_path.stem}#{criterion}", {"label": met}


if __name__ == "__main__":
    _self_check()

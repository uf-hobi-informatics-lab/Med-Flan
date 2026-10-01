"""
2019 n2c2 Track 2 (family history) -- seq2seq builder. Additive sibling to
`family_history_2019_t2.py`'s `build_family_history` (task tag
`family_history`, JSON target): reuses that module's exact TSV-loading
helper (`_load_tsv_by_doc`) and split maps (`SPLIT_FILES`/
`SPLIT_TEXT_DIRS`) so the two paths can never disagree on which docs/rows
feed the target -- only the final serialization
(`family_history_seq2seq_grammar.serialize_records` instead of a raw JSON
dict) differs.

No character offsets exist in this task's gold data at all (see that
module's docstring), so there is nothing to redact/chunk the way
sdoh_event_seq2seq's true-negative/curriculum variants do -- no
augmentation variants exist for this task (see the approved plan's Phase 1
"explicit gaps" note).
"""
from multitask.builders.family_history_2019_t2 import SPLIT_FILES, SPLIT_TEXT_DIRS, _load_tsv_by_doc
from multitask.family_history_seq2seq_grammar import serialize_records
from multitask.io_utils import read_text
from multitask.schemas import example


def build_family_history_seq2seq(split):
    files = SPLIT_FILES.get(split)
    text_dir = SPLIT_TEXT_DIRS.get(split)
    if files is None or text_dir is None:
        return
    s2_by_doc = _load_tsv_by_doc(files["subtask2"])
    for doc_id in sorted(s2_by_doc):
        txt_path = text_dir / f"{doc_id}.txt"
        if not txt_path.exists():
            continue
        doc_text = read_text(txt_path)

        living_status, observations = [], []
        for row in s2_by_doc.get(doc_id, []):
            member, side, rtype = row[1], row[2], row[3]
            if rtype == "LivingStatus" and len(row) >= 5:
                living_status.append({"member": member, "side": side, "score": row[4]})
            elif rtype == "Observation" and len(row) >= 6:
                observations.append({"member": member, "side": side, "text": row[4], "negation": row[5]})

        target_str = serialize_records({"living_status": living_status, "observations": observations})
        yield example("family_history_seq2seq", "2019_n2c2_track2_seq2seq", split, doc_id, doc_text, target_str)

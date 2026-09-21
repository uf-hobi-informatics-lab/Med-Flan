"""One-off script used to generate sample_data/ excerpts from the full datasets.
Not part of the pipeline; kept for reproducibility. Run from repo root with the
project venv: python sample_data/_gen_samples.py
"""
import json
import pandas as pd
import datasets

BASE = "."
SD = "sample_data"
N = 3

# discharge-me: proper CSV parsing (text fields contain embedded newlines/commas)
dm_base = f"{BASE}/discharge-me-bionlp-acl24-shared-task-on-streamlining-discharge-documentation-1.3/train"
for name in ["radiology", "diagnosis", "discharge", "triage", "discharge_target", "edstays"]:
    df = pd.read_csv(f"{dm_base}/{name}.csv.gz", nrows=N)
    df.to_csv(f"{SD}/discharge-me/{name}.sample.csv", index=False)

# MMLU-Pro-Health: arrow -> jsonl
ds = datasets.load_from_disk(f"{BASE}/MMLU-Pro-Health")
with open(f"{SD}/MMLU-Pro-Health/sample.jsonl", "w") as f:
    for row in ds.select(range(min(N, len(ds)))):
        f.write(json.dumps(row) + "\n")

# medmcqa: json array -> first N records
with open(f"{BASE}/medmcqa/raw/train.json") as f:
    try:
        data = json.load(f)
        records = data[:N] if isinstance(data, list) else list(data.values())[:N]
    except json.JSONDecodeError:
        f.seek(0)
        records = [json.loads(line) for line in f.readlines()[:N]]
with open(f"{SD}/medmcqa/train.sample.json", "w") as f:
    json.dump(records, f, indent=2)

# pubmedqa: dict-of-dicts keyed by pmid -> first N
with open(f"{BASE}/pubmedqa/raw/ori_pqaa.json") as f:
    data = json.load(f)
sample = {k: data[k] for k in list(data.keys())[:N]}
with open(f"{SD}/pubmedqa/ori_pqaa.sample.json", "w") as f:
    json.dump(sample, f, indent=2)

print("done")

# Dataset report

Qwen3-4B tokenizer available: **True**

## Duplicates skipped (not re-tokenized)
- **medqa** (train/dev/test.jsonl): byte-identical duplicate of MedQA-USMLE-US (question/train,dev,test.jsonl); not re-tokenized

## Per-dataset stats

| dataset | subset | task_type | n | tk_mean | tk_p95 | tk_max | qw_mean | qw_p95 | qw_max |
|---|---|---|---|---|---|---|---|---|---|
| MedNLI | sentence pairs | NLI | 14049 | 34.0 | 71.0 | 348 | 35.2 | 74.0 | 364 |
| MedQA-USMLE-US | question+options | MCQA | 27092 | 207.5 | 350.0 | 929 | 214.0 | 367.0 | 964 |
| MedQA-USMLE-Mainland | question+options (zh) | MCQA | 68504 | 91.7 | 218.0 | 644 | 57.9 | 139.0 | 331 |
| MedQA-USMLE-Taiwan | question+options (zh-TW) | MCQA | 28246 | 118.1 | 281.0 | 1311 | 72.9 | 171.0 | 1330 |
| MedQA-USMLE-textbooks-en | whole textbook | raw corpus (long-form) | 18 | 1120717.1 | 3026825.899999997 | 4812987 | 1156764.3 | 3181856.6999999974 | 4896815 |
| MedQA-USMLE-textbooks-zh | whole textbook (zh) | raw corpus (long-form) | 2 | 18368322.5 | 18372362.15 | 18372811 | 10010759.0 | 10014872.0 | 10015329 |
| MMLU-Pro-Health | question+options+cot | MCQA | 823 | 140.9 | 283.0 | 1048 | 143.9 | 291.9 | 1074 |
| medmcqa | question+options+explanation | MCQA + explanation | 193155 | 142.9 | 390.0 | 5357 | 145.5 | 396.0 | 5374 |
| pubmedqa | question+contexts+long_answer | QA / yes-no-maybe classification | 211819 | 392.3 | 556.0 | 1614 | 406.1 | 583.0 | 1792 |
| covid (LitCovid) | title+abstract | multi-label text classification | 33699 | 309.2 | 543.0 | 2277 | 323.2 | 580.0 | 2379 |
| CORD-19-metadata | title+abstract | raw corpus / IR | 29500 | 299.7 | 539.0 | 35185 | 307.1 | 559.0 | 36565 |
| CORD-19-fulltext | full paper body text | raw corpus / IR | 1677 | 5433.1 | 12235.199999999999 | 131293 | 5668.7 | 12658.799999999977 | 136836 |
| discharge-me | train: discharge note text | summarization | 68785 | 3458.7 | 5736.799999999996 | 17254 | 3599.4 | 5976.0 | 18682 |
| discharge-me | train: radiology report text | summarization (source) | 259304 | 329.8 | 943.0 | 5359 | 332.8 | 954.0 | 5464 |
| discharge-me | train: target (instructions+course) | summarization (target) | 68785 | 804.7 | 1642.0 | 7991 | 811.4 | 1659.0 | 8007 |
| discharge-me | valid: discharge note text | summarization | 14719 | 3457.8 | 5741.0 | 14951 | 3597.3 | 5974.299999999996 | 16157 |
| discharge-me | valid: radiology report text | summarization (source) | 54650 | 331.5 | 941.0 | 4835 | 334.5 | 952.0 | 4930 |
| discharge-me | valid: target (instructions+course) | summarization (target) | 14719 | 803.0 | 1636.0 | 8539 | 809.6 | 1652.199999999997 | 8599 |
| discharge-me | test_phase_1: discharge note text | summarization | 14702 | 3458.7 | 5743.0 | 15433 | 3599.5 | 5986.949999999999 | 15953 |
| discharge-me | test_phase_1: radiology report text | summarization (source) | 54797 | 329.1 | 940.0 | 3481 | 332.1 | 951.0 | 3528 |
| discharge-me | test_phase_1: target (instructions+course) | summarization (target) | 14702 | 804.1 | 1646.0 | 5457 | 810.8 | 1662.949999999999 | 5491 |
| discharge-me | test_phase_2: discharge note text | summarization | 10962 | 3448.9 | 5742.0 | 14393 | 3588.6 | 5980.949999999999 | 14620 |
| discharge-me | test_phase_2: radiology report text | summarization (source) | 40608 | 331.8 | 941.0 | 5363 | 334.9 | 953.0 | 5486 |
| discharge-me | test_phase_2: target (instructions+course) | summarization (target) | 10962 | 802.6 | 1651.0 | 4611 | 809.3 | 1667.0 | 4658 |
| 2018_n2c2_track1 | note <TEXT> | text classification (cohort eligibility) | 288 | 5300.0 | 9472.549999999997 | 15199 | 5620.4 | 10135.599999999997 | 16103 |
| 2018_n2c2_track2 | clinical note text | NER + RE (ADE / medication) | 505 | 3889.9 | 7089.999999999998 | 13666 | 4136.2 | 7573.6 | 14611 |
| 2019_n2c2_track1 | sentence1+sentence2 | semantic textual similarity (regression) | 2066 | 53.9 | 117.0 | 274 | 54.7 | 118.0 | 274 |
| 2019_n2c2_track2 | clinical note text | NER + attribute/relation extraction (family history) | 216 | 341.5 | 619.0 | 1090 | 351.1 | 637.5 | 1109 |
| 2022n2c2_track1 | clinical note text | event extraction / classification (medication disposition) | 400 | 1234.6 | 2751.099999999998 | 5952 | 1308.3 | 2974.9999999999973 | 6227 |
| 2022n2c2_track2 | clinical note text | NER + attribute extraction (social history) | 1504 | 70.2 | 166.8499999999999 | 335 | 73.3 | 175.8499999999999 | 370 |

## Longest document overall

- **tiktoken cl100k_base**: 18372811 tokens — `MedQA-USMLE/textbooks/zh_sentence/all_books.txt` (dataset: MedQA-USMLE-textbooks-zh / whole textbook (zh))
- **Qwen3-4B**: 10015329 tokens — `MedQA-USMLE/textbooks/zh_sentence/all_books.txt` (dataset: MedQA-USMLE-textbooks-zh / whole textbook (zh))

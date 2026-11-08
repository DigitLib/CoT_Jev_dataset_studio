# Exported datasets (`data/exports/`)

Exports from the **📦 Dataset Export & Analytics** and **🎯 Jev** tabs, and from `tools/run_batch.py`, are written
here. They are generated locally and not tracked by git:

* `clearcot_sft_dataset.jsonl`: SFT samples (question, `<think>` reasoning, answer, sequent, λ-term, provenance).
* `clearcot_dpo_pairs.jsonl`: DPO pairs (`chosen` = ClearCoT trace, `rejected` = a generated naive trace).
* `clearcot_dataset.parquet`: the ClearCoT samples as a table.
* `jev_decisions_dataset.jsonl`: Jev scenarios with kernel-decided gold labels and certificates.
* `typed_decisions_dataset.parquet`: `typed-decisions` table (8 fixed columns + 5 per question).
* `open_jev_s1mb_dataset.parquet`: Open-Jev S1MB format (input state, target distributions).

A sample keeps the same train / validation / test split in every file.

## `samples/`

A small, tracked example of each format (about 20 records each), so the formats can be inspected without
running the Studio. Field definitions and annotated records: [`docs/DATASETS_AND_SAMPLES.md`](../../docs/DATASETS_AND_SAMPLES.md).

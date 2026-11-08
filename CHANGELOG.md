# Changelog

## 0.1.0: first public release

### Verification integrity
- Jev: a label that is provable only after state completion added facts the scenario does not state is now
  `PROVEN_UNDER_ASSUMPTIONS` and not `gentzen_verified` (the added facts are listed in `assumed_facts`).
- Jev: certified "undecided" labels (`NO_OPTION_PROVABLE`) stay verified and now carry `certified_undecided: true`.
- Jev: records whose intended label was refuted by the kernel and relabelled to the proven option now have a
  consistent per-question `verification` (`PROVEN_TRUE`, `target_relabelled: true`).
- v4 server: a consistency check that hits its time limit is never treated as "consistent"; answers proved from
  premises compiled heuristically from the criteria text are not `verified`; the shared domain registry is only
  reconfigured when the requested pack changes (thread safe).
- v3 proof checker: the one-premise `Imp_L` rule now requires the antecedent atom in the context.
- Atom renaming can no longer merge two different atoms.
- v3 checkpoints are refused with a clear error (v4 is the only engine).

### Datasets
- A sample has the same train / validation / test split in the SFT JSONL, the DPO pairs and the Parquet file.
- DPO negatives that echo their generation instructions ("To construct a naive thinking trace ...") are
  regenerated once, and never exported.
- Dataset provenance stores the model name, never a local filesystem path, and records the knowledge graph
  (`kg_id`) a sample was generated from.
- Sample ids use 12 hex digits (6 before), so new ids no longer risk overwriting an existing sample.
- Jev scenario validation reports option labels that are phrases rather than labels (`nonstandard_options`).
- Export filters (document, minimum clarity) apply to the exported files, not only to the table.

### Knowledge graphs
- One saved graph per topic: graphs are saved as `data/kg/<kg_id>.json` with an id derived from their documents,
  listed through `data/kg/index.json`. An existing `latest.json` is registered unchanged.
- On startup the Studio asks whether to restore the last graph (`DTSTUDIO_KG_AUTOLOAD=ask|off|last`), and a
  restored graph no longer fills the Library with its chunks unless asked.
- "🧹 New empty graph", graph deletion, and a warning before appending documents of another topic.
- The Jev tab uses exactly the paths found in the Knowledge Graph tab (it used to run its own search, which
  always returned the slider maximum); a Jev-specific re-discovery is opt-in.

### vLLM
- Local models live in `vllm_model/<name>/` (any number); per-model launch defaults in `studio_vllm.json`;
  `--trust-remote-code` detected for checkpoints with custom modeling code; the server is started with
  `--served-model-name <name>` and binds to 127.0.0.1 by default.
- Stopping the vLLM server checks that the recorded PID is a vLLM process.

### Fixes
- The batch "Stop" button stops the batch (queued seeds are cancelled, finished samples are kept).
- Uploaded file names cannot place files outside `data/uploads/`; `.md` files are re-indexed.
- Crashes with a single chunk or path (Streamlit sliders), an empty model list, `"strength": Infinity` in
  extraction output, empty answers, and malformed provenance JSON.
- Knwler failure no longer duplicates edges; the clean-markdown download works; no temp file per graph render.
- Sentence splitting no longer breaks at words ending in "p." / "no." and keeps abbreviations' case.
- Generic text cleanup no longer inserts a fixed, document-specific sentence.
- `tools/run_batch.py`: KG extraction no longer crashes (`max_workers` argument), uses `--model`, and gains
  `--kg`, `--kg-name`, `--overwrite`.

- `tools/run_batch.py` rewritten against the current generator APIs (it could not run end to end before):
  ClearCoT batch over kernel-verified paths, Jev scenarios from KG paths or chunks with counterfactual variants,
  `--concurrency`, `--jev-source`, `--jev-workflow`, `--all-samples`.

### Project
- `tools/audit_exports.py` (release checks for an exports folder) and `tools/make_samples.py` (example set).
- Tests never touch the real `data/` folder (`DTSTUDIO_DATA_DIR`), however they are started.
- `pyproject.toml` (torch optional), GitHub Actions workflow, `CITATION.cff`.

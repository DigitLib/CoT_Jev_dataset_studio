# Contributing

Thanks for helping improve the Studio.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .                 # core (nanoGentzen runs on its NumPy backend)
pip install -e ".[torch]"        # optional: torch backend
streamlit run app.py
```

## Tests

```bash
python tools/run_tests.py                     # whole suite, readable summary
python -m unittest discover -s tests -v       # plain unittest
```

The suite redirects every runtime file (SQLite DB, knowledge graphs, exports, uploads) to a temporary
directory, so it never changes your `data/` folder. Tests that need the vLLM server or optional packages skip
themselves when those are missing.

## Ground rules

- **Verification must stay honest.** A sample may only be `gentzen_verified` when the nanoGentzen v4 kernel
  proved it (with a checked Curry–Howard certificate), or certified that no option applies
  (`NO_OPTION_PROVABLE`). Anything resting on assumed bridges, invented facts or heuristic premises must be
  labelled as such. See [docs/logic_integrity.md](docs/logic_integrity.md).
- **Keep the ClearCoT and Jev pipelines separate.** `nanogentzen/cot/` and `nanogentzen/jev/` are deliberately
  independent; do not merge them or share new helpers between them (`tests/test_pipeline_separation.py`).
- `nanogentzen_v4/` is vendored by `tools/vendor_v4_engine.py`; the Studio's local safety patches are
  covered by `tests/test_logic.py` and must survive re-vendoring.
- Never commit generated data from third-party documents (`data/uploads/`, `data/kg/`, full exports) or model
  folders under `vllm_model/`.
- Add a test with every fix, and an entry in [CHANGELOG.md](CHANGELOG.md).

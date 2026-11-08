# `nanogentzen/` and `nanogentzen_v4/`

The Studio has two prover folders. They do different jobs.

| Folder | What it is | Who edits it |
|---|---|---|
| [`nanogentzen_v4/`](../nanogentzen_v4/) | The **nanoGentzen v4 engine**: G4ip kernel, proof search, Curry–Howard certificates, the independent certificate checker, the decision server. | Nobody by hand: it is copied from the trained model folder by `tools/vendor_v4_engine.py`. |
| `nanogentzen/` (this folder) | The **Studio's logic layer**: the ClearCoT and Jev pipelines, and the adapter that loads the v4 engine. | Studio developers. |

## `nanogentzen_v4/`: the engine

When nanoGentzen is retrained, `tools/vendor_v4_engine.py` copies the engine code from the model folder and
rewrites its imports so they cannot collide with this package. Keeping it separate means a new model can be
dropped in without touching Studio code.

The Studio adds a few safety fixes to the vendored files (consistency-check timeouts, heuristic grounding, the
shared domain registry). Re-vendoring overwrites them; `tests/test_logic.py` fails until they are
re-applied. The weights live in [`model/`](../model/).

## `nanogentzen/`: the Studio's logic layer

```
nanogentzen/
  cot/                ClearCoT pipeline: KG paths -> sequents -> proofs (architect, parser_cot, relation_logic)
  jev/                Jev pipeline: state + policy rules -> decisions (architect, parser_jev)
  engine.py           entry point: always returns the v4 engine (v3 checkpoints are refused)
  engine_v4.py        adapter: loads nanoGentzen v4 from model/, translates formulas both ways
  logical_architect.py, verify.py, logic_syntax.py    shared helpers
  kernel.py, search.py, parser.py, domain_registry.py older (v3) modules, see below
```

**`cot/` and `jev/` are deliberately separate.** Neither imports the other; they only share the common modules
(`logic_syntax`, `kernel`, `search`, `engine` / `engine_v4`, `verify` and the v4 engine). Merging them, or moving
new helpers between them, has caused conflicts that changed the generated datasets.
`tests/test_pipeline_separation.py` enforces the split.

## Why some code appears twice

`kernel.py`, `search.py`, `parser.py` and `domain_registry.py` come from nanoGentzen v3. The v3 *engine* is no
longer used: every proof and every verdict comes from the v4 kernel. But both pipelines still import their
formula types (`Atom`, `Imp`, `Sequent`, ...) and helpers from these modules, and `jev/parser_jev.py` uses
`domain_registry.py` for workflow detection. That is why about twenty functions, and most of
`domain_registry.py`, exist in both folders.

Removing the duplication means moving `cot/` and `jev/` onto the v4 formula types, which changes how both
pipelines build and read formulas. It is planned work, not part of 0.1.0; see [ROADMAP.md](../ROADMAP.md).

## Rules of thumb

- New verification logic belongs in `nanogentzen_v4/` (as a documented patch) or in the pipeline that needs it.
- A result is only `gentzen_verified` when the v4 kernel proved it with a checked certificate (or, for Jev,
  certified that no option applies). See [docs/logic_integrity.md](../docs/logic_integrity.md).
- Don't extend the v3 modules; they stay only as long as the pipelines import from them.

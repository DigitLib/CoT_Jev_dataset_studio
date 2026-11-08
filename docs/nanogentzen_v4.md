# nanoGentzen v4 in the Studio

The Studio runs the **nanoGentzen v4** model in `model/` (v4 is the only supported engine; v3 checkpoints are refused). The engine is loaded directly via `config.json` (`format_version: 4`, `input_format: "normalized_canonical"`).

## What v4 changes for dataset generation

|                           | v3 path                                                                          | v4 path                                                                                                                                                                                                                                                               |
| ------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Who decides               | neural search + constructive "synthesizers" that build a proof when search fails | Dyckhoff G4ip kernel; network only orders choice moves; complete prover as backstop                                                                                                                                                                                   |
| Proof check               | tree verifier                                                                    | strict tree verifier **and** a typed λ-term checked by `nanogentzen_v4/checker.py` (shares no code with the kernel)                                                                                                                                                   |
| Unprovable claim          | sometimes marked verified via a synthesized proof of a different sequent         | never verified; status `INTUITIONISTIC_UNDECIDED` with a checked Kripke countermodel and `missing_facts`                                                                                                                                                              |
| Refuted claim             | —                                                                                | `PROVEN_FALSE` with its own certificate                                                                                                                                                                                                                               |
| Contradictory premises    | not detected                                                                     | `INCONSISTENT_PREMISES` with the minimal conflicting set                                                                                                                                                                                                              |
| Relevance                 | used premises                                                                    | used premises + `minimal_witnesses` (irredundant subset that still proves the goal)                                                                                                                                                                                   |
| Jev state reconciliation  | adds *every* antecedent atom of the target's rule to the state                   | adds only the minimal missing literals that make the target provable **without contradiction**, never overwrites a stated fact, records it in `provenance.nanogentzen.state_completion` (switchable); such a label is `PROVEN_UNDER_ASSUMPTIONS` and **not verified** |
| Jev rules                 | unparsable rules skipped silently                                                | rejected and listed in `provenance.nanogentzen.rejected_rules`; plain sentences (no `=>`) are rejected because the parser would otherwise turn them into facts                                                                                                        |
| Jev probabilities         | heuristic 0.85–0.91                                                              | kernel-proven label gets 1.0 (optional label smoothing in the sidebar)                                                                                                                                                                                                |
| Offline fallback scenario | hard-coded `PROVEN_TRUE`, fixed 0.85                                             | decided by the kernel like every other scenario                                                                                                                                                                                                                       |
| Negation / falsum         | `0`, `FALSUM`, `BOT` atoms                                                       | dedicated ⊥ node; strictly intuitionistic (excluded middle is *not* provable)                                                                                                                                                                                         |

Statuses you can now see in the Logic Tree and Jev tabs: `PROVEN_TRUE`, `PROVEN_FALSE`,
`INTUITIONISTIC_UNDECIDED`, `INCONSISTENT_PREMISES`, `SEARCH_BUDGET_EXHAUSTED`, and for Jev also
`MULTIPLE_PROVEN`, `TARGET_REFUTED_BY_PROOF` (the facts prove a different option than `target_decision`; the
record is relabelled to the proven option), `NO_OPTION_PROVABLE` and `PROVEN_UNDER_ASSUMPTIONS`.
`gentzen_verified = true` only for `PROVEN_TRUE` with a checked certificate and, for Jev, for
`NO_OPTION_PROVABLE` (certified "undecided", flagged `certified_undecided`).

## Layout

```
model/                    v4 weights + config.json + vocab.json, report.json / gates.json
nanogentzen_v4/           the v4 engine (kernel, search, certificates, checker, parser, jev_server ...)
  backend.py              torch / NumPy glue                         (Studio file)
  numpy_model.py          torch-free forward pass of the network      (Studio file)
  everything else         vendored v4 engine modules
nanogentzen/engine_v4.py  NanoGentzenV4Engine: v4 interface + prove_sequent / decide / system_one
nanogentzen/engine.py     load_engine(): loads NanoGentzenV4Engine from model/
tools/vendor_v4_engine.py re-vendors a newly trained model folder
tests/test_logic.py       kernel, engine (soundness, certificates, statuses, invariance, torch/NumPy parity), architects
```

The v4 package is kept separate from `nanogentzen/` on purpose: the original v4 files import each other
as top-level modules and re-point `sys.modules["nanogentzen.*"]` at themselves, which inside the Studio
would silently swap the v3 kernel for the v4 one. The vendoring tool rewrites those imports and removes
the aliasing; a test checks that nothing leaks.

## Backends

Sidebar → *v4 Network Backend*:

* `auto` – torch if `torch` + `transformers` import, otherwise NumPy
* `torch` – GPU/CPU, ~2 ms per System 1 call
* `numpy` – no torch needed; ~5–20 ms per call on CPU (a 9-hop chain proves and certifies in ~15 ms)
* `symbolic` – no network at all (fixed-order G4ip)

Verdicts are identical in all four: the network can only make search faster or slower. The test
`test_network_never_changes_a_verdict` checks this on random sequents, and `test_parity` compares the
NumPy and torch forward passes when torch is installed.

## Dataset fields added

* ClearCoT SFT / DPO / Parquet: `provenance.nanogentzen` = engine, status, ASCII sequent, sequent source,
  `certificate` (context, type, term, checked), `minimal_witnesses`, `missing_facts`, `conflicting_premises`.
  `proof_tree_json` is the raw v4 tree, re-verifiable with `nanogentzen_v4.kernel.verify_proof_tree_strict`.
* Jev: each `gold[question]` gains `verification_v4`, `certificate`, `certificate_checked`, `relevance`,
  `epistemic` (4-class System 1), and when relevant `missing_facts`, `proven_options`, `conflicting_premises`,
  `countermodel`, `target_mismatch`. `provenance.nanogentzen` holds the engine, sequent, predicates used to
  bind rule atoms to nested state fields, state completion and rejected rules.

Re-check a certificate offline:

```python
from nanogentzen_v4.checker import check_certificate
ok, err = check_certificate(record["provenance"]["nanogentzen"]["certificate"])
```

## Things to know

* **Nested state keys.** v4 grounding names nested booleans by path (`attributes.kyc` → `attributes_kyc`),
  while scenario rules use the leaf key (`kyc`). The Jev architect binds each rule atom to its field with a
  v4 *predicate*, so LLM scenarios keep working unchanged.
* **Option names** are matched to rule atoms case-insensitively (`Compatible` ↔ `compatible`).
* **Reserved identifiers** (`or`, `not`, `and`, `implies`, `iff`, `therefore`, `FALSUM`) are renamed for the
  kernel and restored in all outputs.
* **System 1 probabilities** are a ranking on unfamiliar structures; the kernel status is the label.
* **Model window:** 256 tokens / 32 premises. Larger sequents are decided by the kernel without the network.
* The shipped checkpoint: 15 training epochs, best epoch 7, `s1_temperature` 1.24,
  `search_confidence_threshold` 0.3; all four gates pass (`model/gates.json`).
* The natural-language → proposition step (KG paths, `text_to_sequent`) is unchanged and is still the
  weakest link: a proof is only as meaningful as the atoms it is about.

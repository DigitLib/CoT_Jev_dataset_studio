# Logic integrity: what a "verified" sample means

A proof is only as meaningful as the sequent it proves. These rules decide which sequent a sample gets
and when the Studio calls it verified.

## KG relations → logic (`nanogentzen/cot/relation_logic.py`)

| Relation (canonical) | Logical reading | Examples |
|---|---|---|
| entails | `A ⇒ B` | implies, requires, depends on, is a kind of, proves, is defined as, causes |
| excludes | `A ⇒ ¬B` | lacks, excludes, precludes, prevents, contradicts, is incompatible with |
| none | no rule | contrasts with, models, formulates, enables, argued, any **negated** entailment ("does not require") |

Predicates keep their polarity: `does not require` → `not requires` (no rule), `lacks` → `excludes`.
"Does not require" denies an implication; it does not assert `A ⇒ ¬B`.

## KG paths → sequents (`nanogentzen/cot/parser_cot.py: path_to_sequent`)

Each hop adds only the rule its relation licenses, **in the edge's own direction**. A path that walks an
edge backwards (the undirected fallback in path search) keeps the edge as `B ⇒ A`, so
"Rain causes Wet Street, Sprinkler causes Wet Street" no longer proves *Rain ⊢ Sprinkler*.

Every path carries `edges` (direction + logic of each hop) and `logic_basis`; *Only entailment chains*
in the graph tab filters out the rest before any proof work.

## `logic_basis` and verification

| logic_basis | Meaning | Can be verified |
|---|---|---|
| `symbolic` | sequent given explicitly | yes |
| `conditional_premises` | premises contain their own if/then; atoms are aligned so the proof uses them | yes |
| `kg_entailment_chain` | every hop is a forward inference | yes |
| `assumed_bridge` | paragraph seed: `(P1 ∧ … ∧ Pk) ⇒ C` is **added** | no → `BRIDGE_ASSUMED` (proof kept as CoT guide, `proof_valid=true`) |
| `kg_relational` | a hop is backwards / non-inferential / follows an exclusion | no (decided honestly, normally `INTUITIONISTIC_UNDECIDED`) |
| `kg_unlabelled` | path without relation labels | no → `RELATIONS_UNLABELLED` |

The old "synthesizers" that built a proof of a plain chain when the real sequent failed are not used with
the v4 engine (the only engine). Their v3 code remains for the shared modules, but a synthesized proof now only
counts if it concludes exactly the target sequent.

## Dataset gating

* Batch generation stores a sample as `accepted` only if the proof is verified, the reasoning is grounded
  in the logic tree, and the LLM actually produced it (no template fallback). Otherwise `draft`, with
  `provenance.generation.status_reasons`.
* `tokens_saved_pct` is measured against a plain step-by-step answer from the same model
  (`CoTGenerator.generate_standard_cot`), or 0 when it could not be measured. Negative = ClearCoT longer.
* DPO export uses accepted samples only and skips pairs whose rejected side is the fixed template.
* `provenance.generation` records model, fallback, grounding, logic basis, whether the naive trace is synthetic,
  and whether token savings were measured.

Tests: `tests/test_clearcot.py`, `tests/test_datasets.py`.

## Knowledge-graph data (item 3)

**Stable chunk IDs.** `chunk_id` = first 12 hex of SHA-1(document, page, section, normalised text);
identical text in the same place gets `-2`, `-3`… Re-chunking a document with the same settings reproduces
the IDs, so KG edges and saved samples keep pointing at their source. (Changing chunk settings changes the
text and therefore the IDs; that is intended.)

**Evidence grounding** (`core/evidence.py`). Every extracted edge's evidence is checked against its chunk:

| status | meaning |
|---|---|
| `exact` | the quote is in the chunk (case, whitespace, curly quotes, dashes and line-break hyphens normalised) |
| `fuzzy` | ≥ 60 % of the quote's word trigrams are in the chunk |
| `located` | no usable quote, but one sentence mentions both subject and object; that sentence becomes the evidence |
| `unsupported` | none of these: **dropped** by default (`KnowledgeGraphBuilder.evidence_policy = "keep"` keeps and flags it) |

`located` checks the two concepts, not the relation, so it is weaker than a quote. Edges also record
`extractor` (`llm`, `heuristic`, `knwler`, `manual`); the graph tab shows the evidence mix and how many
chunks fell back to the regex extractor. Paths carry `evidence_statuses` and `evidence_supported`.
The chunk prefix is no longer used as stand-in evidence.

**Saved graphs.** After every build the graph, its chunks and the communities are written (atomically) to
`data/kg/<kg_id>.json`, with the id derived from the source documents (`core/kg_store.py`); a different
document set never overwrites another graph. On startup the Studio offers to restore the last graph
(`KG_AUTOLOAD` / `DTSTUDIO_KG_AUTOLOAD`: `ask`, `off`, `last`). Every generated sample records `kg_id` in its
provenance. The graph tab's *Saved graphs* panel saves, loads and deletes graphs.

Tests: `tests/test_knowledge_graph.py`, `tests/test_kg_store.py`.

## Jev labels, inputs and splits (item 4)

**Counterfactual variants** (`JevScenarioGenerator.make_counterfactuals`, nanoGentzen v4). Starting from a
kernel-certified scenario, each fact in a proof's `minimal_witnesses` (an irredundant set, so every fact is
needed) is removed from the state, or negated, and the scenario is decided again with no target. A variant
is kept only when its labels are certified and differ from the parent's, typically:

| edit | typical new label |
|---|---|
| remove a required fact | the question's *undecided*-type option (`NO_OPTION_PROVABLE`) |
| negate a required fact | *undecided*, or another option when a rule concludes it from the negated fact |

`NO_OPTION_PROVABLE` is a certified label: the complete G4ip prover showed that no option follows. It is
only used when the question has an undecided-type option (undecided, insufficient…, unknown, inconclusive…).
Variants drop the free-text `observations` (they describe the original facts) and record the edit in
`provenance.counterfactual` (`of` = parent sample id). The Jev builder and batch generator save up to four
variants per accepted scenario (checkboxes, on by default).

**Batch Jev acceptance.** Scenarios are saved as `accepted` only with kernel-certified labels; the rest are `draft`.

**S1MB input.** `input.state_json` is now `{"state", "decisions", "policy_rules"}`: the decisions to make
(id, type, instructions, options, criteria) and the rules that determine the labels are part of the input.
`include_decisions=False, include_rules=False` restores the old state-only input.

**typed-decisions columns.** `column_mode="positional"` writes `q1__id`, `q1__label`, `q1__confidence`,
`q1__probabilities`, `q1__verification` … instead of one column set per LLM-chosen question id
(the default `question_id` mode is unchanged, plus a `__verification` column).

**Splits** (`core/dataset_splits.py`). All Jev and ClearCoT exports (SFT JSONL / Parquet, DPO, typed-decisions,
S1MB, Jev JSONL) assign train / validation / test (80/10/10) by group, so no group is split:
`document` → `path` (KG path / source chunks) → `family` (scenario + its counterfactuals); `auto` uses the
first level with at least 3 groups, `stored` keeps the saved split. Assignment is deterministic. S1MB
`group_id` follows the same grouping.

Tests: `tests/test_jev.py`, `tests/test_datasets.py`.

## Cross-document KG paths

Paths across PDFs are valuable when the PDFs are about the same topic. `KnowledgeGraphBuilder.doc_topic_overlap()`
scores each document pair by the overlap coefficient of the specific (non-generic) concepts they contribute,
|A ∩ B| / min(|A|, |B|). A cross-document path is kept only if **every** document hand-over on it joins documents
with overlap ≥ `min_topic_overlap` (graph-tab slider, default 0.05) and the bridging concept is not generic
("Model", "Framework" …). Two same-topic papers typically score 0.05–0.3; an unrelated paper that shares one
term with a 150-concept paper scores below 0.01. Each path records `doc_topic_overlap`.

Path search runs one depth-first search per source (instead of one per source/target pair), ranks candidates
(entailment chain > supported evidence > more distinct chunks > edge strength) and runs proofs only for the
paths it returns. The composite claim now includes every hop.

Splits: documents joined by any cross-document sample form one group (union-find), so a pair of same-topic
PDFs is never divided between train and test.

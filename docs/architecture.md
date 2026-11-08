# System Architecture & Core Design

**nanoGentzen ClearCoT & Jev Decision Studio** is designed as a decoupled, local-first neuro-symbolic dataset manufacturing pipeline. Every component is built for deterministic reproducibility, low latency, and zero cloud dependency.

---

## 🏗️ High-Level Component Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Streamlit Frontend (gui/)                         │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬───────────┐  │
│  │ Tab 1: Lib   │ Tab 2: KG    │ Tab 3: CoT   │ Tab 4: Jev   │ Tab 5: Exp│  │
│  └──────────────┴──────────────┴──────────────┴──────────────┴───────────┘  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                            Core Pipeline (core/)                            │
│  ┌────────────────────────┐  ┌────────────────────────┐  ┌───────────────┐  │
│  │ DocumentParser         │  │ DocumentChunker        │  │ KnowledgeGraph│  │
│  │ (Noise & Layout Clean) │  │ (Semantic / Paragraph) │  │ (NetworkX)    │  │
│  └────────────────────────┘  └────────────────────────┘  └───────────────┘  │
│  ┌────────────────────────┐  ┌────────────────────────┐  ┌───────────────┐  │
│  │ CoTGenerator           │  │ JevScenarioGenerator   │  │ BatchGenerator│  │
│  │ (Reasoning & SFT/DPO)  │  │ (State & Jev Decision) │  │ (Concurrency) │  │
│  └────────────────────────┘  └────────────────────────┘  └───────────────┘  │
└──────────────────┬───────────────────────────────────┬──────────────────────┘
                   │                                   │
┌──────────────────▼───────────────┐   ┌───────────────▼──────────────────────┐
│  nanoGentzen v4 Prover Engine    │   │  LLM Backend Layer (llm_backend/)    │
│  (nanogentzen/, nanogentzen_v4/) │   │  ┌─────────────────────────────────┐ │
│  • Intuitionistic LI & G4ip      │   │  │ LocalLLMClient                  │ │
│  • v4 neural search ordering     │   │  │ (vLLM / Ollama / OpenAI-compat) │ │
│  • 3-Class Kripke Epistemic Head │   │  └─────────────────────────────────┘ │
│  • Curry-Howard λ-Term Compiler  │   │  ┌─────────────────────────────────┐ │
│  • Relevance Logic Causal Factor │   │  │ vLLM Subprocess Server Manager  │ │
│  • Dyckhoff G4ip Loop-Free Rules │   │  └─────────────────────────────────┘ │
└──────────────────┬───────────────┘   └───────────────┬──────────────────────┘
                   │                                   │
                   └─────────────────┬─────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────────┐
│                        Data Storage Layer (data/)                           │
│  • data/uploads/: Raw source literature (PDFs, DOCX, TXT)                   │
│  • data/clearcot.db: Local SQLite database (`samples` and `jev_samples`)    │
│  • data/exports/: SFT JSONL, DPO Pairs, typed-decisions & S1MB Parquet      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Module Breakdown

### 1. `core/` — Document Understanding & Generation Engine
* **`document_parser.py`**:
  * PyMuPDF layout analysis with geometric bounding-box detection.
  * Unicode ligature normalization (`ﬁ` $\to$ `fi`, `ﬂ` $\to$ `fl`, soft hyphens `\xad`).
  * Structural noise suppression: Table of Contents (leader dot chains `\.{5,}`), Bibliography/References state machine, author frontmatter, and vector chart/axis tick debris.
* **`chunker.py`**:
  * Multi-strategy chunking: Natural paragraph reconstruction, Semantic boundary clustering via embedding distance, and Fixed window buffering.
  * Abbreviations protection (`e.g.`, `i.e.`, `et al.`, `vol.`, `p.`).
  * Deterministic lexical vectorizer fallback (character 3-grams + word unigrams).
* **`kg_builder.py`**:
  * Relational triple extraction conforming to canonical schemas.
  * Incremental entity linking, plural normalization, and alias deduplication.
  * Scoped multi-hop path finder (default intra-document constraint + cross-document generic stop-concept suppression).
* **`community_detector.py`**:
  * NetworkX modularity clustering (Louvain algorithm) to identify concept communities and high-centrality hub concepts.
* **`question_generator.py`**:
  * Graph-aware question and premise synthesizer.
  * Converts multi-hop relational paths into academic hypotheses and formal antecedents.
* **`cot_generator.py`**:
  * Teacher model orchestrator.
  * Generates load-bearing reasoning traces (`<think>`) and rich answers (`<answer>`).
  * Evaluates naive verbosity vs. ClearCoT compactness to compute token savings.
* **`jev_generator.py`**:
  * Neuro-symbolic decision scenario generator.
  * Grounds unstructured context into structured `state` JSON, multiple-choice `questions` with mutually exclusive `criteria`, and formal `policy_rules`.
  * Runs Dyckhoff $\mathbf{G4ip}$ proof search, extracts Saul Kripke model status (`PROVEN_TRUE`), Curry-Howard typed $\lambda$-terms, and isolates causal `factors` via Relevance Logic.
* **`jev_dataset_manager.py`**:
  * SQLite persistence manager for Jev decision records.
  * Serializes datasets to `typed-decisions` Parquet (8 fixed columns + 5 per question) and `Open-Jev S1MB` Parquet schemas.
* **`batch_generator.py`**:
  * Multi-threaded continuous batch generation with thread-safe database commits and main-thread progress reporting.
* **`dataset_manager.py`**:
  * SQLite persistence manager, export serializers (SFT JSONL, DPO JSONL, Parquet).

---

### 2a. Two front ends, one verification core

ClearCoT reasoning and Jev decisions turn different inputs into logic, so each has its own front end.
They share only the engine.

```
ClearCoT  KG path / NL claims ──► nanogentzen/cot/parser_cot.py ──┐
                                  (relation_logic, missing_links)  │
                                                                   ├──► nanogentzen/verify.py ──► kernel / search /
Jev       state JSON + rules ───► nanogentzen/jev/parser_jev.py ──┘     engine_v4 / nanogentzen_v4 (checker)
                                  (validate_scenario, coverage)
                     shared syntax: nanogentzen/logic_syntax.py (formula / sequent parser)
```

| Module | Pipeline | Role |
|---|---|---|
| `nanogentzen/cot/parser_cot.py` | ClearCoT | `text_to_sequent`, `path_to_sequent`, `path_logic_report`, `path_missing_links`, `is_degenerate_concept` |
| `nanogentzen/cot/relation_logic.py` | ClearCoT | which KG relations license an inference |
| `nanogentzen/cot/architect.py` | ClearCoT | `CotLogicalArchitect`, `proof_record` |
| `nanogentzen/jev/parser_jev.py` | Jev | state grounding, rule compilation, predicate binding, `validate_scenario`, `uncovered_options` |
| `nanogentzen/jev/architect.py` | Jev | `JevLogicalArchitect` (gold labels, factors, counterfactual labelling rule) |
| `nanogentzen/verify.py` | shared | `prove_sequent`, `decide`, `missing_facts`, `check_certificate` |
| `nanogentzen/logic_syntax.py` | shared | `FormulaParser`, `parse_formula`, `parse_sequent`, `parse_symbolic_sequent` |

`nanogentzen/parser.py`, `architect_cot.py`, `architect_jev.py`, `logical_architect.py` and `relation_logic.py`
are kept as compatibility aliases. `tests/test_pipeline_separation.py` fails if `cot/` imports `jev/` (or the
reverse), or if either imports the old mixed modules.

**Meaningful concepts** (`core/concepts.py`, shared by both pipelines because they read the same graph):
* **Names are never cut mid-phrase.** Older builds kept the first 4 words ("Respect For Practical And").
  * Names with a truncation, pronoun, citation or numeric issue are rejected at insertion.
  * Clauses and generic words are kept but get a low `concept_quality`.
  * Path discovery skips nodes below `min_concept_quality` (default 0.5).
* **One name per concept.**
  * `concept_key` merges word-order, plural and possessive variants.
  * The LLM is shown the existing concepts that occur in each chunk, and parallel extraction runs in waves.
* **Evidence gate.** An inference predicate (`requires`, `causes`, `excludes` …) is kept only if its own evidence states that relation:
  * the cue must come from the same family, not hedged by a modal or a conditional;
  * the cue sentence must mention the subject or object, on its own side of the cue;
  * if the sentence states it the other way round, the edge is flipped.
  * Otherwise the edge becomes `relates to`, with `claimed_predicate` kept for review.
* **`repair_concepts()` / Graph tab → 🧠 Concept Quality.** Repairs an existing graph without re-extraction. It is idempotent, and the previous save is kept as `before_concept_repair.json`. It:
  * completes truncated names from the evidence;
  * moves leading prepositions into the predicate;
  * merges variants and named-entity aliases;
  * re-reads relations strictly (`is based on` → `depends on`, `is necessary for` → reversed `requires`);
  * adds `is a kind of` taxonomy edges from head-noun structure. These skip named entities and vague heads. With an embedding model they are sense-checked against how the text uses each concept: the similarity threshold is 0.45, calibrated on `paraphrase-multilingual`. Rejected pairs are remembered.

**nanoGentzen-guided path discovery** (`KnowledgeGraphBuilder.find_multihop_paths`):
* `logic_first=True` (the default) enumerates chains of inference edges from *every* node. The general search starts only at the `max_sources` highest-degree nodes.
* An `excludes` edge may be walked backwards as the last hop: `A ⇒ B`, `C excludes B` ⊢ `¬C`. Exclusion is symmetric, intuitionistically too. An entailment edge walked backwards still breaks the chain.
* `verified_only=True` proves every entailment chain before `limit` is applied and returns only PROVEN_TRUE paths. It is exposed as "Only kernel-verified paths" in the Graph tab (default off) and the Logic Tree tab (default on).
* `taxonomy_policy` decides where an `is a kind of` hop may stand: `any` (default; Graph and Jev tabs), `last` (at most one, as the final hop; the Logic Tree tab default) or `none`. A taxonomy hop first (`X Consciousness is a kind of Consciousness`, `Consciousness requires Y`) copies one claim about a hub onto every subtype, which gives near-duplicate samples that over-generalise.
* `max_paths_per_hub` caps how many returned paths pass *through* one concept (Logic Tree tab: 8; 0 = no cap).
* The evidence gate (`core/concepts.py: check_inference`) also checks the cue's grammatical subject. It rejects:
  * a subject that is another noun or a demonstrative ("this is implied by many theories");
  * an unresolved pronoun;
  * a cue that introduces a `that`-clause neither endpoint names;
  * a claim that is only mentioned ("the proposal that X depends on Y").

  It reverses passive cues ("X is required for Y" means Y requires X). Repair re-applies the gate to every edge and drops taxonomy edges that no longer qualify (`taxonomy_removed`).

Each pipeline also has its own data-quality gates:
* **ClearCoT**
  * KG paths default to entailment chains (`cot_require_entailment`, set in the Logic Tree tab).
  * Paths through degenerate concepts (`Both`, `Claim`, `Reader` ...) are `kg_relational`.
  * Relational paths record `missing_links`.
  * Exports honour *Verified Only* and drop rows with `tokens_saved_pct < 0`.
* **Jev**
  * Rule atoms written as phrases are normalised to snake_case before compiling. A scenario whose rules all fail to compile is `no_rules` and is never decided by `jev_server`'s `criteria_heuristic`, which invents `ev_*`/`warrant_*` atoms; any answer from that heuristic is `HEURISTIC_UNVERIFIED`.
  * Scenarios are validated before proof search: an empty rule set, the prompt's example rule, or contradictory rules (same premises concluding two options) mean one retry, then `SCENARIO_REJECTED`.
  * `undecided` is a certified label (`NO_OPTION_PROVABLE`) only when every other option has a rule.
  * Unverified rows are exported with `label: null` and `intended_label`.
  * KG paths are not ranked by the ClearCoT architect (`jev_require_entailment`, default off).

### 2. `nanogentzen/` — Studio front end of the prover (ClearCoT `cot/` and Jev `jev/` pipelines)
The decisions are made by the vendored v4 engine in `nanogentzen_v4/` (G4ip kernel, certificate generation and
an independent certificate checker); `nanogentzen/` holds the Studio's two pipelines and shared helpers:

* **`kernel.py`** (shared data types and helpers):
  * Sequent calculus kernel for **Intuitionistic Propositional Logic ($LI$)** and Dyckhoff's loop-free **$\mathbf{G4ip}$**.
  * Implements introduction rules, axioms, Relevance Logic witness extraction (`extract_relevance_witnesses`), and Curry-Howard proof term translation (`proof_tree_to_curry_howard`).
* **`search.py`**:
  * Goal-directed backward proof search engine with Kripke countermodel extraction on unprovable branches.
  * Explores derivation trees to find closed, verified proofs ($\Gamma \vdash \Delta$).
* **`logical_architect.py`**:
  * Translates natural language relational paths into formal sequents.
  * Linearizes proof trees into execution instructions for ClearCoT generation.
* **`engine.py` / `engine_v4.py`**:
  * Load the nanoGentzen v4 model from `model/` (torch, NumPy or symbolic backend; verdicts are identical in
    every backend, the network only orders the search). v3 checkpoints are refused.
  * Statuses: `PROVEN_TRUE`, `PROVEN_FALSE`, `INTUITIONISTIC_UNDECIDED` (with Kripke countermodel),
    `INCONSISTENT_PREMISES`; see [nanogentzen_v4.md](nanogentzen_v4.md).

---

### 3. `llm_backend/` — Inference & Model Abstraction
* **`client.py` (`LocalLLMClient`)**:
  * High-performance unified HTTP client for any OpenAI-compatible API endpoint.
  * Supports streaming, custom stop tokens (`</think>`, `</reasoning>`), and batch embedding generation (`/v1/embeddings`).
* **`vllm_server.py`**:
  * Subprocess supervisor to start, stop, and monitor local vLLM instances directly from the Streamlit UI.

---

## 🗄️ Database Schema (`clearcot.db`)

All generated samples are stored in SQLite table `samples`:

```sql
CREATE TABLE IF NOT EXISTS samples (
    sample_id TEXT PRIMARY KEY,
    doc_name TEXT,
    question TEXT,
    reasoning TEXT,
    answer TEXT,
    naive_cot TEXT,
    status TEXT,
    gentzen_verified INTEGER,
    gentzen_sequent TEXT,
    proof_tree_json TEXT,
    provability_score REAL,
    tokens_saved_pct INTEGER,
    clarity_score REAL,
    provenance_json TEXT,
    created_at TEXT,
    curry_howard_term TEXT DEFAULT '',
    epistemic_status TEXT DEFAULT 'PROVEN_TRUE',
    grounding_score REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS jev_samples (
    sample_id TEXT PRIMARY KEY,
    doc_name TEXT,
    workflow TEXT,
    split TEXT,
    state_json TEXT,
    questions_json TEXT,
    gold_json TEXT,
    factors_json TEXT,
    policy_rules_json TEXT,
    gentzen_verified INTEGER,
    gentzen_sequent TEXT,
    curry_howard_term TEXT,
    epistemic_status TEXT,
    proof_tree_json TEXT,
    provenance_json TEXT,
    context_text TEXT,
    status TEXT,
    created_at TEXT
);
```

### Provenance Object Specification
Every sample stores full grounding metadata in `provenance_json`:
```json
{
  "doc_name": "bayesian-models-of-the-mind.pdf",
  "documents": ["bayesian-models-of-the-mind.pdf"],
  "is_cross_document": false,
  "pages": [26, 47, 58],
  "path_repr": "(Credal State) <--[relates to]-- (Mental Representation) --[understood through]--> (Bayesian Model) --[rely on]--> (Postulated Prior)",
  "path_nodes": ["Credal State", "Mental Representation", "Bayesian Model", "Postulated Prior"],
  "chunk_ids": ["fa05c3a9", "1e45c418", "8901209e"]
}
```

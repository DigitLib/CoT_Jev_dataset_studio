# Complete Pipeline Walkthrough

This document provides a step-by-step operational walkthrough of **nanoGentzen ClearCoT & Jev Decision Studio** across all five functional tabs.

---

## 📑 Tab 1: Document Library & Semantic Ingestion

The Document Library is where raw literature enters the studio. High quality datasets begin with **noise-free chunking**.

```
[PDF / Book Upload] ➔ [PyMuPDF Layout Parsing] ➔ [Structural Noise Filter] ➔ [Semantic Chunker] ➔ [Session Chunks Pool]
```

### 1. Layout-Aware Parsing & Normalization
* **Ligature De-aliasing**: Converts typography ligatures (`ﬁ`, `ﬂ`, `ﬀ`, `ﬃ`, `ﬄ`) to clean ASCII equivalents and removes soft hyphens (`\xad`).
* **Header & Margin Watermark Filtering**: Strips out arXiv stamps, journal margin notices, running page headers ($y_0 \le 48$), and running page numbers.

### 2. Structural Noise Suppression
Unlike standard RAG chunkers that blindly cut text every $N$ characters, the studio employs intelligent structural suppression:
1. **Title Page Frontmatter**: Detects author affiliation markers (`*`, `†`, `‡`, `♠`, `♡`), email links, repo addresses, and publishing dates, marking them as `frontmatter`.
2. **Vector Graphic & Chart Debris**: Calculates geometric bounding boxes above figure captions (`Figure \d+`), filtering out axis numbers, metric tables, and subfigure tags (`(a)`, `(c)`).
3. **Table of Contents (TOC) Suppression**: Detects leader dot chains (`. . . . . . 16`) and section outline indexes, ensuring TOC entries are never parsed as body text.
4. **Bibliography / References Section Filtering**: Uses a state machine to track `References`, `Bibliography`, or `Works Cited` sections, suppressing dense citation lists.

### 3. Segmentation Modes
* **🧠 Semantic Boundary (Embedding Distance)** *(Recommended)*:
  * Computes sentence-level embeddings via your local embedding client (e.g. `snowflake-arctic-embed2` on Ollama).
  * Measures cosine distances between adjacent sentences.
  * Splits chunks dynamically at topical shift percentiles with min/max token guards.
  * Automatically falls back to deterministic character 3-gram lexical vectors if the embedding server is offline.
* **📄 Natural Paragraph-Level**:
  * Stitches fragmented text blocks and cross-page hyphenated sentences into coherent narrative units.
* **📏 Fixed-Size Window**:
  * Buffers text blocks up to a specified character limit.

---

## 🕸️ Tab 2: Thematic Knowledge Graph & Multi-Hop Discovery

Once clean chunks are ingested, Tab 2 builds a structured concept graph to uncover latent, multi-step deductive paths.

```
[Clean Chunks] ➔ [Triple Extraction LLM/Regex] ➔ [NetworkX MultiDiGraph] ➔ [Scoped Path Discovery]
```

### 1. Canonical Relational Extraction
* Chunks are processed via the local LLM using a typed extraction schema (Orbifold/Knwler architecture):
  ```json
  {
    "subject": "Entity A",
    "predicate": "relation",
    "object": "Entity B",
    "statement": "Natural language claim connecting them",
    "evidence": "Exact supporting excerpt from text",
    "strength": 8
  }
  ```
* **Entity Canonicalization**: Normalizes plurals, trims leading articles (`the`, `a`), resolves aliases, and standardizes verbs (`contrasts with`, `enables`, `implies`, `requires`).

### 2. Community Detection
* Runs the **Louvain modularity algorithm** to partition the graph into thematic concept clusters.
* Displays high-centrality hub nodes and isolated dyadic pairs.

### 3. Scoped Multi-Hop Reasoning Discovery
Discovers multi-hop deductive chains ($\ge 2$ hops) connecting distant propositions:
* **🔒 Intra-Document Restriction (Default)**:
  * Enforces that every proposition in a multi-hop path originates from the **same document**.
  * Prevents "wormhole hallucinations" across disparate literature domains.
* **🌐 Cross-Document Synthesis (Opt-In)**:
  * Allows cross-document paths across thematically aligned books/papers.
  * **Generic Stop-Concept Guard**: Prohibits generic words (`"Model"`, `"Framework"`, `"System"`, `"Analysis"`, `"Generalization"`, `"Performance"`) from acting as cross-document bridges.
  * **Provenance Tracking**: Preserves full multi-document lists (`documents: [doc1, doc2]`) and page numbers.

---

## 🔬 Tab 3: Logic Tree & ClearCoT Workspace

Tab 3 is where formal logic meets neural generation for generative Chain-of-Thought datasets.

```
[Discovered KG Path] ➔ [nanoGentzen v4 Prover] ➔ [Closed Derivation Tree] ➔ [Teacher LLM] ➔ [ClearCoT Sample (Q-R-A)]
```

### 1. Translating Paths to Sequents
The relational path $(A \to B \to C \vdash C)$ is converted into an intuitionistic logic sequent:
$$\Gamma \vdash \Delta \quad \equiv \quad A,\; (A \to B),\; (B \to C) \vdash C$$

### 2. nanoGentzen v4 Prover
* Searches for a closed proof tree in **Intuitionistic Sequent Calculus ($LI$)** using Dyckhoff's loop-free **$\mathbf{G4ip}$** calculus.
* Search is ordered by the v4 neural policy (`model/model.safetensors`); verdicts do not depend on it.
* If closed, computes the **Proof Depth**, **Node Count**, and **Provability Score (1.00)**.
* Renders an interactive ASCII Proof Tree and linear deduction steps.

### 3. ClearCoT Generation
The Teacher LLM is prompted with:
1. Grounded context snippets from the source document(s).
2. The formulated question.
3. The nanoGentzen-verified deductive steps to follow.

The model generates:
* `<think>`: Lean, load-bearing deductive reasoning conditioned on the formal logic tree, stripped of conversational fluff and robotic ordinals (*"First, Second, Third"*).
* `<answer>`: A rich, articulate, comprehensive final synthesis answer.
* **Interactive Quality Monitor & Batch Generator**: Review individual samples, re-verify logic trees, or trigger multi-threaded batch generation.

---

## 🎯 Tab 4: Jev Decision Builder

Tab 4 converts document context and relational paths into structured, neuro-symbolic decision scenarios conforming to the **Jev format** (`typed-decisions` and `S1MB` Open-Jev schemas).

```
[Evidence Context / KG Path] ➔ [Jev Scenario Extraction] ➔ [Dyckhoff G4ip Verification] ➔ [Curry-Howard & Relevance Logic] ➔ [Jev Dataset]
```

### 1. Structured Scenario Modeling
* **State (`state`)**: A clean, valid JSON representation capturing entities, verified properties, boolean flags, and operational parameters.
* **Decision Questions (`questions`)**: Formulates multiple-choice questions with 2 to 4 mutually exclusive `criteria` definitions.
* **Grounding Policy Rules (`policy_rules`)**: Propositional implications connecting the state facts to candidate outcomes (e.g. `(is_scientific_theory & uses_processes) => compatible`).

### 2. Formal Proof & Kripke Epistemic Certification
* Evaluates candidate criteria options as succedents against the grounded state facts and policy rules using **Dyckhoff's $\mathbf{G4ip}$ sequent calculus**.
* Certifies the **Saul Kripke Status**:
  * `PROVEN_TRUE`: Constructively proven under intuitionistic logic.
  * `INTUITIONISTIC_UNDECIDED`: Epistemically open (unproven counter-options).
  * `PROVEN_FALSE`: Constructively refuted.
* **Curry-Howard Typed $\lambda$-Term**: Translates the derivation tree into computational proof terms (e.g. `apply(pair(is_scientific_theory, uses_processes))`).
* **Relevance Logic Causal Factors**: Isolates the exact minimal subset of `state` properties that causally witness the decision, discarding irrelevant variables.
* **Probabilities**: a kernel-proven label gets 1.0 (or $1-\varepsilon$ with the *label smoothing* slider, the rest spread over the other options); unproven questions keep the System 1 estimate and are not verified.

### 3. Curation & Bulk Generation
* **Single Scenario Builder**: 3-column interactive layout for live verification and one-click curation (`✅ Accept`, `❌ Reject`).
* **Batch Complete Jev Generator**: Multi-threaded bulk generation across all chunks and discovered multi-hop paths.

---

## 📦 Tab 5: Multi-Format Dataset Export

Tab 5 serializes verified samples into standard training and benchmarking formats ready for fine-tuning or evaluation.

### Supported Export Schemas

1. **ClearCoT SFT JSONL (`clearcot_sft_dataset.jsonl`)**:
   ```json
   {
     "id": "S-B1CBEA",
     "document": "consciousness_study.pdf",
     "question": "How do global workspace theory and neuroscientific theories relate?",
     "reasoning": "Global workspace theory posited information broadcasting...",
     "answer": "Both theories align with computational functionalism...",
     "gentzen_verified": true,
     "sequent": "GWT, (GWT → Info_Integration)... ⊢ Functionalism",
     "tokens_saved_pct": 65,
     "clarity_score": 0.87,
     "provenance": { ... }
   }
   ```
2. **ClearCoT Contrastive DPO Pairs JSONL (`clearcot_dpo_pairs.jsonl`)**:
   * **`chosen`**: High-clarity, nanoGentzen-structured ClearCoT reasoning + rich answer.
   * **`rejected`**: Unstructured, naive flat reasoning or verbose repetitive canopy.
3. **Canonical Jev Typed Decisions Parquet (`typed_decisions_dataset.parquet`)**:
   * Flattened Parquet schema (8 fixed columns + 5 per question; positional `q1__…` or per question id) matching reference benchmarks (`agent_trace_observability`, `security_incidents`, etc.):
   * `id`, `workflow`, `split`, `state` (JSON string), `questions` (JSON string), `gold` (JSON string), `factors` (JSON string), `n_questions`, plus dynamic target columns (`{q}__label`, `{q}__confidence`, `{q}__probabilities`).
4. **Open-Jev S1MB Parquet (`open_jev_s1mb_dataset.parquet`)**:
   * Open-Jev schema with struct `input: {'state_json': ...}` and struct array `targets: [{decision_id, kind: 'judgment_distribution', annotation_kind: 'hard_label', ids, probabilities, label}]`, `case_id`, `input_hash`.
5. **Jev Decisions Multi-Task JSONL (`jev_decisions_dataset.jsonl`)**:
   * Complete multi-task records with proof trees, Curry-Howard terms, and Relevance Logic causal factors.
6. **ClearCoT Columnar Parquet (`clearcot_dataset.parquet`)**:
   * Columnar storage of all verified SFT samples with provability scores, clarity metrics, and Curry-Howard terms.

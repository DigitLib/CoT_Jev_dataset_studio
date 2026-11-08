# nanoGentzen ClearCoT & Jev Decision Studio

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Inference: vLLM | Ollama | LM Studio](https://img.shields.io/badge/Inference-vLLM%20%7C%20Ollama%20%7C%20LM%20Studio-orange.svg)](docs/llm_providers.md)
[![Logic: nanoGentzen v4](https://img.shields.io/badge/Logic-Intuitionistic%20G4ip%20%7C%20nanoGentzen%20v4-purple.svg)](docs/nanogentzen_v4.md)

A local, neuro-symbolic **dataset studio**. It turns documents (PDF, DOCX, TXT, Markdown) into training and evaluation datasets whose logical claims are checked by a theorem prover:

1. **ClearCoT datasets**: question / reasoning / answer samples for SFT, and preference pairs for DPO. Each 
   sample is generated from a knowledge-graph path whose inference chain is proved by the nanoGentzen v4
   kernel (intuitionistic propositional logic, Dyckhoff's G4ip calculus) and carries a Curry–Howard
   certificate checked by an independent type checker.
2. **Jev decision datasets**: structured decision scenarios (state JSON, typed questions, policy rules, gold
   distribution) in the `typed-decisions` and `Open-Jev S1MB` Parquet formats. Each gold label is decided by
   the kernel.

Everything runs locally: the LLM through any OpenAI-compatible server (vLLM, Ollama, LM Studio), the prover in-process (torch, NumPy or purely symbolic backend).

> **References**: ClearCoT paper, [doi:10.5281/zenodo.19409889](https://doi.org/10.5281/zenodo.19409889) ·
> Occam CCoT reasoner, [`Sagicc/Occam-2B-CCoT`](https://huggingface.co/Sagicc/Occam-2B-CCoT) ·
> earlier prover release, [`Sagicc/nanoGentzen-v3`](https://huggingface.co/Sagicc/nanoGentzen-v3)
> (the v4 weights ship in [`model/`](model/)).

---

## What "verified" means

A sample is `gentzen_verified` only when the nanoGentzen v4 kernel proved it and its Curry–Howard certificate type-checked, or (Jev) when the complete search certified that no option applies (`NO_OPTION_PROVABLE`, marked `certified_undecided`). Everything else is labelled honestly:

| Status                                    | Meaning                                                                                        |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `PROVEN_TRUE`                             | closed G4ip proof, certificate checked                                                         |
| `NO_OPTION_PROVABLE`                      | Jev: every option has a rule and the complete search proves none, so "undecided" is certified  |
| `PROVEN_UNDER_ASSUMPTIONS`                | Jev: provable only after adding facts the scenario does not state; **not verified**            |
| `INTUITIONISTIC_UNDECIDED`                | not provable; a Kripke countermodel and the missing facts are reported                         |
| `PROVEN_FALSE` / `INCONSISTENT_PREMISES`  | the claim is refuted, or the premises contradict each other                                    |
| `BRIDGE_ASSUMED` / `RELATIONS_UNLABELLED` | ClearCoT: the proof rests on an added bridge rule or on unlabelled relations; **not verified** |

**Scope.** The proof certifies the *knowledge-graph entailment chain* behind a sample (for example `A requires B, B implies C ⊢ C`), not every sentence of the free-text reasoning the LLM writes around it.
Details: [docs/logic_integrity.md](docs/logic_integrity.md).

---

## Pipeline

```
Documents (data/uploads/)
   ↓  layout-aware parsing, noise filtering (TOC dot chains, references, figure debris)
   ↓  chunking (paragraph / semantic / fixed)
Knowledge graph  (one saved graph per topic: data/kg/<kg_id>.json)
   ↓  LLM triple extraction, evidence gating, concept repair, communities (Louvain)
Multi-hop paths  (A → B → C, intra-document by default)
   ├──────────────────────────────┬───────────────────────────────┐
   ▼                              ▼                               │
ClearCoT (nanogentzen/cot/)     Jev (nanogentzen/jev/)              │  two independent
 • path → sequent Γ ⊢ C          • state + policy rules → formulas  │  pipelines
 • G4ip proof + certificate      • G4ip decision per option        │
 • lean <think> trace + answer   • gold distribution, factors      │
 • naive / standard CoT baselines• counterfactual variants         │
   ▼                              ▼                               │
Exports (data/exports/)  ◄────────────────────────────────────────┘
 clearcot_sft_dataset.jsonl · clearcot_dpo_pairs.jsonl · clearcot_dataset.parquet
 typed_decisions_dataset.parquet · open_jev_s1mb_dataset.parquet · jev_decisions_dataset.jsonl
```

Splits (80/10/10) are grouped by document, then by KG path or scenario family, and are the same for a sample in
every export file. Token savings are measured against an unprompted step-by-step answer from the same model.

---

## Quick start

```bash
git clone <repo-url> nanogentzen-studio
cd nanogentzen-studio
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .                 # core; nanoGentzen runs on its NumPy backend
pip install -e ".[torch]"        # optional: torch backend (install CUDA torch from pytorch.org first)
streamlit run app.py             # http://localhost:8501
```

`pip install -r requirements.txt` installs the full desktop set instead. Check an installation with
`python tools/check_desktop.py`.

### A teacher LLM

The Studio needs an OpenAI-compatible endpoint for question, trace and scenario generation:

* **vLLM** (Linux + NVIDIA): put models in [`vllm_model/<name>/`](vllm_model/README.md), e.g.
  `huggingface-cli download Sagicc/Occam-2B-CCoT --local-dir vllm_model/Occam-2B-CCoT`, then start one from the
  sidebar (**🚀 Launch vLLM Local Server**). Each folder can carry launch defaults in `studio_vllm.json`.
* **Ollama**: `ollama run qwen2.5:7b`, endpoint `http://localhost:11434/v1`.
* **LM Studio**: start its local server, endpoint `http://localhost:1234/v1`.

Without an LLM, extraction and generation fall back to heuristics (useful for trying the UI, not for datasets).
More: [docs/llm_providers.md](docs/llm_providers.md).

---

## The five tabs

1. **📚 Document Library**: upload and parse documents (PyMuPDF; Surya layout engine optional), inspect layout
   blocks, chunk.
2. **🕸️ Knowledge Graph & Paths**: extract triples, repair concepts, detect communities, discover multi-hop
   paths (limit, document scope, entailment-only and kernel-verified filters). Graphs are saved per topic;
   **🧹 New empty graph** starts a clean one, and appending documents of another topic asks first.
3. **🔬 Logic Tree & ClearCoT**: proof tree, sequent and certificate for a path; generate, review and batch-run
   ClearCoT samples.
4. **🎯 Jev Decision Builder**: scenarios from the paths found in tab 2 (or from paragraphs), kernel-decided gold
   labels, counterfactual variants for label balance, batch generation.
5. **📦 Export & Analytics**: dataset statistics, split preview, all exports.

Walkthrough: [docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md).

---

## Dataset formats

Full field definitions and annotated samples: [docs/DATASETS_AND_SAMPLES.md](docs/DATASETS_AND_SAMPLES.md).
Small example exports: [`data/exports/samples/`](data/exports/README.md).

* **SFT JSONL**: `id, split, document, question, reasoning, answer, conversations, gentzen_verified, sequent,
  curry_howard_term, epistemic_status, tokens_saved_pct, clarity_score, grounding_score, provenance`.
* **DPO JSONL**: `id, split, prompt, chosen, rejected, metadata`. `chosen` is the ClearCoT trace; `rejected` is a
  genuinely generated naive trace (template or instruction-echoing negatives are never exported).
* **ClearCoT Parquet**: the SFT columns plus `status, created_at`.
* **typed-decisions Parquet**: `id, workflow, split, state, questions, gold, factors, n_questions` plus five
  columns per question (`<q>__id/label/confidence/probabilities/verification`), named by position (`q1__…`,
  default) or by question id.
* **Open-Jev S1MB Parquet**: `input{state_json}, targets[{decision_id, kind, annotation_kind, ids, probabilities,
  label}], split, case_id, group_id, language, input_hash`.
* **Jev JSONL**: state, questions, gold (with kernel status, certificate, relevance), factors, policy rules,
  sequent, λ-term and provenance.

Every record's provenance names its source document, pages, KG path, generating model and knowledge graph
(`kg_id`).

---

## Command line

```bash
# headless batch run: ingest data/uploads/, build a graph, generate and export both dataset families
python tools/run_batch.py --base-url http://localhost:8000/v1 --model Occam-2B-CCoT --limit 50
python tools/run_batch.py --kg <kg_id> --mode jev          # reuse a saved graph

python tools/audit_dataset.py            # audit the ClearCoT samples in the database
python tools/audit_exports.py            # release checks on data/exports/ (splits, leaks, verified flags, paths)
python tools/make_samples.py data/exports  # cut the small example set into data/exports/samples/
python tools/run_tests.py                # test suite with a readable summary
python tools/check_desktop.py            # installation / GPU / end-to-end check
```

Configuration through the environment: `DTSTUDIO_DATA_DIR` (runtime data), `DTSTUDIO_VLLM_MODELS_DIR`,
`DTSTUDIO_DEFAULT_VLLM_MODEL`, `DTSTUDIO_KG_AUTOLOAD=ask|off|last`.

---

## Repository layout

```
app.py, config.py         Streamlit entry point and central configuration
core/                     parsing, chunking, knowledge graph, generators, dataset managers, splits
nanogentzen/              Studio front end of the prover
  cot/                    ClearCoT pipeline: KG paths → sequents → proofs
  jev/                    Jev pipeline: state + rules → decisions
nanogentzen_v4/           vendored nanoGentzen v4 engine (kernel, certificates, checker, decision server)
model/                    nanoGentzen v4 weights (~19 MB, safetensors)
llm_backend/              OpenAI-compatible client, vLLM process manager
gui/                      the five tabs
vllm_model/               local models for the vLLM server (contents not tracked)
data/                     runtime data: uploads, knowledge graphs, SQLite DB, exports (not tracked)
docs/                     documentation
tests/, tools/            test suite and command-line tools
```

Why there are two prover folders (`nanogentzen/` and `nanogentzen_v4/`): [nanogentzen/README.md](nanogentzen/README.md).

---

## Documentation

* [docs/architecture.md](docs/architecture.md): components and data flow
* [docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md): the tabs step by step
* [docs/clearcot_and_nanogentzen.md](docs/clearcot_and_nanogentzen.md): ClearCoT and the logic behind it
* [docs/nanogentzen_v4.md](docs/nanogentzen_v4.md): the v4 engine, its statuses and retraining
* [docs/logic_integrity.md](docs/logic_integrity.md): what counts as verified
* [docs/DATASETS_AND_SAMPLES.md](docs/DATASETS_AND_SAMPLES.md): dataset schemas
* [docs/llm_providers.md](docs/llm_providers.md): vLLM, Ollama, LM Studio
* [docs/DESKTOP_TESTING.md](docs/DESKTOP_TESTING.md): testing on a desktop (incl. Windows)
* [ROADMAP.md](ROADMAP.md) · [CHANGELOG.md](CHANGELOG.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Limitations

* Verification covers the knowledge-graph inference chain, not the full natural-language reasoning.
* Dataset quality depends on the teacher LLM and on the source documents; review samples before training.
* Knowledge-graph extraction and Jev scenario writing are LLM steps; their errors are filtered (evidence gating,
  scenario validation) but not eliminated.
* Datasets you generate from third-party documents inherit those documents' licences.

---

## License & citation

MIT. See [LICENSE](LICENSE). If you use the Studio or ClearCoT, please cite the paper
([CITATION.cff](CITATION.cff)):

```bibtex
@software{clearcot2025,
  title     = {ClearCoT: Towards Clear, Coherent and Efficient Chain-of-Thought via Formal Logic Verification},
  author    = {Andrija Sagic},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19409889},
  url       = {https://doi.org/10.5281/zenodo.19409889}
}
```

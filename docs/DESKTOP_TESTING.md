# Testing the Studio on a desktop (incl. Windows)

All commands run from the repository root.

## 1. Setup (once)

Requirements: Python 3.10 or newer; about 3 GB of disk if you install torch.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell / cmd)
source .venv/bin/activate          # Linux / macOS
pip install -e .                   # core: nanoGentzen v4 runs on its NumPy backend
# optional, NVIDIA GPU: install the CUDA build of torch that matches your driver (see pytorch.org), then
pip install --index-url https://download.pytorch.org/whl/cu124 torch
pip install -e ".[torch]"
```

torch is optional: without it nanoGentzen v4 runs on the NumPy backend with identical verdicts.
`vllm` is not installable on Windows; use Ollama or LM Studio (or vLLM inside WSL2), see `requirements-optional.txt`.

## 2. Automatic checks

```bash
python tools/check_desktop.py
python tools/run_tests.py
```

**check_desktop.py** (exit code 0 = OK) reports:

| Section | What it checks |
|---|---|
| Environment | Python, every package (with the pip command if missing), CUDA device, `model/` files |
| Engine | loads v4 on NumPy and torch; torch vs NumPy forward pass (max diff < 1e-3); 5 known verdicts (incl. no excluded middle); 200 random sequents: network-guided vs symbolic search give the same verdict and every proof's certificate is accepted by the independent checker; System 1 latency |
| Pipeline (offline, no LLM) | creates two same-topic PDFs → parser (and that the file is not left locked) → stable chunk ids → KG with evidence check and a negated relation → topic-gated cross-document paths (verified ones must be entailment chains; overlap 0.99 must block them) → ClearCoT sample stored as draft because the CoT is a template → SFT JSONL/Parquet with split → Jev scenario + counterfactual variants → S1MB / typed-decisions exports → KG save/load |

With a running LLM backend: `python tools\check_desktop.py --llm http://localhost:11434/v1 --model qwen2.5:7b`.
`--quick` skips the 200-sequent sweep; `--verbose` prints tracebacks.

**run_tests.py** runs all test modules and separates failures caused by a missing package (with the pip
command) from real failures. With every requirement installed, all tests should pass.

Expected numbers on a desktop CPU (NumPy backend): ~15 ms per decided sequent, ~7 ms per System 1 call.
On a GPU the torch backend's System 1 call is ~2 ms.

## 3. Manual test in the UI

```bash
streamlit run app.py
```

Open http://localhost:8501 and use 2–3 PDFs on **one topic** (cross-document paths need shared concepts).

1. **Sidebar** – engine shows `nanoGentzen-v4: torch on cuda` (or `numpy on cpu`) and the gates G1–G4 ✔.
   Connect your LLM backend (Ollama / LM Studio) and pick the generation model.
2. **Document Library** – upload the PDFs; check the layout overlay; chunk (paragraph strategy).
   Re-chunking the same file gives the same chunk ids.
3. **Knowledge Graph** – build. Check the evidence line under the stats (most edges `exact`/`fuzzy`),
   the "dropped" note, and that the graph was saved (`data/kg/<kg_id>.json`, see 💾 *Saved graphs*). Restart the
   app: it offers to restore the graph (or start empty). *🧹 New empty graph* starts a clean graph for another topic.
4. **Multi-hop paths** – untick *Restrict to same document*. Open *Document topic overlap*: same-topic pairs
   should be well above the slider (default 0.05). Tick *Only entailment chains* to see the paths a proof can
   be about. A path through "contrasts with" / "models" or walked against an edge must not be verified.
5. **Logic Tree & ClearCoT** – generate one sample from a cross-document path and one from a paragraph.
   The paragraph sample shows `BRIDGE_ASSUMED`; the path sample shows the certificate, minimal witnesses and
   the logic basis. Accept one, reject one.
6. **Batch generation** – run ~10 samples; the summary shows accepted vs draft with reasons.
7. **Jev Decision Builder** – build a scenario from a path; accept with *counterfactual variants* ticked;
   run a small batch. In the export tab use *Split by: auto* and *positional* columns and export all three formats.
8. **Export** – SFT / DPO / Parquet. Check that each row has `split`, that documents joined by a cross-document
   sample are in the same split, and that the DPO file contains only accepted samples.

## 4. If something fails

| Symptom | Fix |
|---|---|
| `torch backend unavailable` in the log | torch/transformers missing or CPU-only build: the NumPy backend is used; install the CUDA build for GPU |
| `pymupdf4llm ... has no to_json()` | `pip install -U pymupdf4llm` (PDFs are still parsed by the plain PyMuPDF fallback) |
| PDF cannot be deleted / replaced while the app runs | should not happen any more (handles are closed); report the file |
| No cross-document paths | the PDFs share too few concepts: lower the topic-overlap slider, or check the overlap table |
| Everything is `draft` after batch generation | LLM backend offline (template CoT) or proofs not verified: see `provenance.generation.status_reasons` |
| `... is a nanoGentzen v3 checkpoint` | v3 checkpoints are no longer supported; use the v4 model in `model/` |


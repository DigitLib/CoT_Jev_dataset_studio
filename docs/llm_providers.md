# LLM Providers & Inference Backends

**nanoGentzen ClearCoT & Jev Decision Studio** is designed to be **backend-agnostic**. Any engine that exposes an OpenAI-compatible HTTP API can be used as the Teacher LLM or Embedding Provider for both ClearCoT reasoning and Jev decision synthesis.

---

## ⚡ Supported Providers Matrix

| Provider | Purpose | Default URL | Setup Complexity |
| :--- | :--- | :--- | :--- |
| **Local vLLM** | Teacher LLM & Batch Dataset Generation (Continuous Batching) | `http://localhost:8000/v1` | Low (Built-in Manager) |
| **Ollama** | Local Embeddings, Logic & Reasoning Models | `http://localhost:11434/v1` | Minimal (`ollama serve`) |
| **LM Studio** | Desktop GUI Inference & GGUF Models | `http://localhost:1234/v1` | Minimal (1-click server) |
| **Zero-Dependency Fallback** | Offline / Testing | Internal Python Engine | None (Automatic) Deterministic Heuristic Kernels |

---

## 🚀 1. Local vLLM Engine (Built-In Server Manager)

vLLM provides state-of-the-art PagedAttention and continuous batching, making it the optimal choice for high-throughput batch dataset generation.

### Integrated server manager and `vllm_model/`
The sidebar's **🚀 Launch vLLM Local Server** box (shown while no backend answers) starts and stops a vLLM server:
* Put each local model in its own folder under [`vllm_model/`](../vllm_model/README.md) (any folder with a
  `config.json`); the box lists them. A Hugging Face id or any path can be entered via *Custom*.
* Optional `vllm_model/<name>/studio_vllm.json` gives that model's launch defaults (`max_model_len`,
  `gpu_memory_utilization`, `quantization`, `trust_remote_code`, `extra_args`). `--trust-remote-code` is
  pre-selected for checkpoints that ship their own modeling code (e.g. Spark-X).
* The server is started with `--served-model-name <folder name>` and binds to `127.0.0.1`; the port is taken
  from the endpoint URL in the sidebar. Log and PID: `data/vllm_server.log`, `data/vllm_server.pid`.
* One server runs per port: to switch models, stop it and start it with another model.
* `DTSTUDIO_VLLM_MODELS_DIR` moves the model folder (e.g. to a larger disk), `DTSTUDIO_DEFAULT_VLLM_MODEL`
  changes the preselected model.

### CLI launch
You can also launch vLLM in a terminal:
```bash
vllm serve vllm_model/Occam-2B-CCoT \
  --served-model-name Occam-2B-CCoT \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.60
```

---

## 🦙 2. Ollama Integration (Embeddings & Inference)

Ollama is ideal for local embeddings and convenient local GGUF models.

### A. Semantic Boundary Embeddings
To enable **Semantic Boundary Chunking** in Tab 1, pull an embedding model:
```bash
ollama pull snowflake-arctic-embed2
```
In the studio:
* Set **Base URL**: `http://localhost:11434/v1`
* In Tab 1 (Document Library), select **Semantic Boundary Mode**, specify model `snowflake-arctic-embed2:latest`, and click **Parse & Ingest**.

### B. Chat & Reasoning Models
Any chat model served by Ollama works as the teacher model, for example:
```bash
ollama pull qwen2.5:7b
ollama pull deepseek-r1:14b      # a reasoning model (emits its own thinking tokens)
```
Set the teacher model in the sidebar (or in the Knowledge Graph tab for extraction) to the pulled name.

---

## 🧠 3. Reasoning Models & Internal CoT Token Budgeting

Reasoning models (such as **Occam-2B-CCoT**, **Spark-X** and **DeepSeek-R1**) generate internal chain-of-thought tokens before producing their final response.

### How the Studio Handles Reasoning Models:
1. **Separation of Concerns**: Thinking tokens arrive in `message.reasoning` / `reasoning_content` (Ollama, vLLM) or inside `<think>` tags (`llm_backend/adapters.py`). The studio captures these tokens into `<think>...</think>` blocks while keeping final answers distinct and structured.
2. **Knowledge Graph Extraction Budget (`max_tokens=600`)**: For Tab 2 (Knowledge Graph Extraction), the studio allocates 600 tokens per chunk. This ensures reasoning models have enough headroom to complete their internal thinking (~150–200 tokens) and emit a complete, valid JSON proposition array.
3. **Resilient JSON Truncation Recovery**: If a model hits a token limit, the built-in JSON parser automatically salvages all completed proposition triples up to the last closing brace `}`, preventing chunks from discarding valid data into regex heuristics.
4. **Parallelism Guidelines**:
   * **vLLM**: Set extraction workers to **4–16** (vLLM's continuous batching processes them concurrently).
   * **Ollama**: Keep extraction workers at **1–2** unless you have configured `OLLAMA_NUM_PARALLEL` in your environment.

---

## 🌐 4. LM Studio / OpenAI-Compatible Endpoints

To use LM Studio, Jan, or any custom local server:
1. Start the local server in your application (e.g. LM Studio on port `1234`).
2. In the studio sidebar:
   * **API Base URL**: `http://localhost:1234/v1`
   * **API Key**: local servers need none. For a hosted OpenAI-compatible API, set the environment variable
     `DTSTUDIO_LLM_API_KEY` before starting the Studio.
   * **Model Name**: Enter the model identifier loaded in your server.

---

## 🛡️ 5. Zero-Dependency Heuristic Fallback Engine

nanoGentzen ClearCoT & Jev Decision Studio does **not** fail if external LLM or embedding servers are offline. It contains robust, deterministic fallback engines:

1. **Deterministic Lexical Vectorizer (`_fast_lexical_vectors`)**:
   * Uses word unigrams and character 3-grams with cosine similarity math.
   * Automatically computes semantic boundary shifts if Ollama/vLLM is unreachable.
2. **Rule-Based Triple Extractor (`_heuristic_proposition_extract`)**:
   * Extracts academic causal triples using regex verb dependency trees (`implies`, `enables`, `requires`, `contrasts with`).
3. **nanoGentzen Python Proof Engine**:
   * Fully self-contained pure-Python sequent calculus solver with zero external requirements.

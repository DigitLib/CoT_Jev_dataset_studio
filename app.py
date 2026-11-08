"""
nanoGentzen ClearCoT & Jev Decision Studio
100% Local Dataset Factory: Unstructured Text ➔ Knowledge Graph ➔ nanoGentzen Logic Tree ➔ LLM ClearCoT (Q-R-A) & Jev Decision Parquet.
"""

import streamlit as st
from pathlib import Path
import logging
import time

from config import (
    MODEL_DIR, DB_PATH, DEFAULT_MODEL_PATH, KG_AUTOLOAD,
    LLMBackendConfig, GentzenConfig
)
from llm_backend.client import LocalLLMClient, clean_model_display_name
from core.document_parser import DocumentParser
from core.chunker import DocumentChunker
from gui.tab_library import render_library_tab
from nanogentzen.engine import NanoGentzenEngine, load_engine, detect_model_version
from nanogentzen.logical_architect import LogicalArchitect
from core.kg_builder import KnowledgeGraphBuilder
from core.community_detector import CommunityDetector
from core.question_generator import QuestionGenerator
from core.cot_generator import CoTGenerator
from core.dataset_manager import DatasetManager
from core.batch_generator import BatchDatasetGenerator
from gui.tab_graph import render_graph_tab
from gui.tab_review import render_review_tab
from gui.tab_export import render_export_tab
from core.jev_generator import JevScenarioGenerator
from core.jev_dataset_manager import JevDatasetManager
from gui.tab_jev import render_jev_tab
from core import kg_store
from gui._kg_session import restore_graph

logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="nanoGentzen ClearCoT & Jev Decision Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Sidebar: Local Engine & Model Settings
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Engine Settings")
    st.caption("100% Local Pipeline (Ollama / LM Studio / vLLM + nanoGentzen v4 / v3)")

    st.subheader("1. Local LLM Backend")
    backend_choice = st.selectbox(
        "Generation Backend Engine",
        ["vLLM", "LM Studio", "Ollama", "Custom URL"],
        index=0
    )

    default_urls = {
        "vLLM": "http://localhost:8000/v1",
        "LM Studio": "http://localhost:1234/v1",
        "Ollama": "http://localhost:11434/v1",
        "Custom URL": "http://localhost:8000/v1",
    }
    col_be1, col_be2 = st.columns([3, 1])
    with col_be1:
        base_url = st.text_input("Generation Endpoint URL", value=default_urls[backend_choice])
    with col_be2:
        st.write("")
        st.write("")
        if st.button("🔄", help="Force refresh backend connection and model discovery"):
            LocalLLMClient.invalidate_cache()
            st.rerun()

    llm_client = LocalLLMClient(base_url=base_url)
    is_online = llm_client.health_check()

    if is_online:
        st.success(f"Connected to {backend_choice}")
        available_models = llm_client.list_models()
        if backend_choice == "vLLM":
            from llm_backend.vllm_server import get_active_vllm_pid, stop_vllm_server
            v_pid = get_active_vllm_pid()
            if v_pid:
                st.caption(f"vLLM Local Server Process: `PID {v_pid}`")
                if st.button("⏹️ Stop vLLM Server"):
                    stop_vllm_server()
                    st.rerun()
    else:
        st.warning(f"Backend offline at {base_url}.")
        available_models = ["Sagicc/Occam-2B-CCoT", "qwen2.5:7b", "llama3.1:8b"]
        if backend_choice == "vLLM":
            from urllib.parse import urlparse
            from config import VLLM_MODELS_DIR
            from llm_backend.vllm_server import (discover_local_models, is_vllm_installed, start_vllm_server,
                                                 wait_for_vllm_ready)
            v_installed = is_vllm_installed()
            local_models = discover_local_models()
            if local_models:
                available_models = [m["name"] for m in local_models] + available_models
            with st.expander("🚀 Launch vLLM Local Server", expanded=True):
                if not v_installed:
                    st.info("💡 To launch from Python, install vLLM: `pip install vllm`")
                st.caption(f"Model folder: `{VLLM_MODELS_DIR.name}/` - {len(local_models)} local model(s) found"
                           + ("" if local_models else ". Put each model in its own sub-folder (see vllm_model/README.md)."))
                custom_label = "Custom (Hugging Face ID or path)…"
                options = [m["name"] for m in local_models] + [custom_label]
                v_choice = st.selectbox("Model", options, index=0, key="vllm_model_choice",
                                        format_func=lambda n: n + ("  (old location, move to vllm_model/)"
                                                                   if any(m["name"] == n and m["legacy"] for m in local_models) else ""))
                if v_choice == custom_label:
                    v_model = st.text_input("Hugging Face ID or local path", value="Sagicc/Occam-2B-CCoT",
                                            key="vllm_model_custom")
                    v_settings = {}
                else:
                    v_model = v_choice
                    v_settings = next(m["settings"] for m in local_models if m["name"] == v_choice)
                col_vq, col_vm = st.columns(2)
                quant_opts = ["none", "awq", "gptq", "bitsandbytes"]
                with col_vq:
                    v_quant = st.selectbox("Quantization", quant_opts,
                                           index=quant_opts.index(str(v_settings.get("quantization") or "none").lower())
                                           if str(v_settings.get("quantization") or "none").lower() in quant_opts else 0,
                                           key=f"vllm_quant_{v_choice}")
                with col_vm:
                    v_gpu_mem = st.slider(
                        "GPU Mem Ratio", 0.40, 0.85, float(v_settings.get("gpu_memory_utilization", 0.60)), step=0.05,
                        key=f"vllm_gpu_{v_choice}",
                        help="0.60 allocates ~14.4GB VRAM on RTX 4090, keeping 9.6GB free for nanoGentzen and embeddings."
                    )
                len_opts = [2048, 4096, 8192, 16384]
                want_len = int(v_settings.get("max_model_len", 4096))
                v_max_len = st.selectbox("Max Context Length", len_opts,
                                         index=len_opts.index(want_len) if want_len in len_opts else 1,
                                         key=f"vllm_len_{v_choice}")
                v_trust = st.checkbox("Trust remote code", value=bool(v_settings.get("trust_remote_code", False)),
                                      key=f"vllm_trust_{v_choice}",
                                      help="Needed for checkpoints that ship their own modeling code (e.g. Spark-X). "
                                           "Only enable for models you trust: their Python code is executed.")
                v_port = urlparse(base_url).port or 8000
                if st.button("▶️ Start vLLM Server", disabled=not v_installed):
                    with st.spinner("Starting vLLM server process in background..."):
                        pid = start_vllm_server(
                            model=v_model,
                            port=v_port,
                            gpu_memory_utilization=v_gpu_mem,
                            max_model_len=v_max_len,
                            quantization=v_quant if v_quant != "none" else None,
                            trust_remote_code=v_trust,
                        )
                        st.info(f"Spawned vLLM process (PID {pid}). Waiting for model initialization...")
                        ready = wait_for_vllm_ready(base_url=base_url, timeout_seconds=90)
                        if ready:
                            st.success("vLLM server is ONLINE!")
                            st.rerun()
                        else:
                            st.warning("Server is initializing in background. Check data/vllm_server.log for progress.")

    # Filter generation models (exclude dedicated embedding models)
    gen_models = [m for m in available_models if not LLMBackendConfig.is_embedding_model(m)] or available_models
    if not gen_models:
        # backend reachable but it lists no models: never send model=None to it
        st.warning("The backend lists no models. Type the model name it serves.")
        gen_models = [st.text_input("Model name", value="Sagicc/Occam-2B-CCoT", key="manual_gen_model")
                      or "Sagicc/Occam-2B-CCoT"]
    default_gen_idx = 0
    for i, m in enumerate(gen_models):
        if any(k in m.lower() for k in ("occam", "ccot", "qwen", "gemma", "llama")):
            default_gen_idx = i
            break

    selected_model = st.selectbox(
        "🎯 Text Generation Model",
        gen_models,
        index=default_gen_idx,
        format_func=clean_model_display_name,
        help="The local model used for question formulation and verbalizing the nanoGentzen Logic Tree into ClearCoT."
    )

    # -------------------------------------------------------------
    # Embedding Model Selection
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("##### 🧬 Text Embedding Model")
    use_separate_embed = st.checkbox(
        "Use separate endpoint for Embedding",
        value=False,
        help="Enable if your embedding model is running on another port or backend (e.g., Ollama on 11434 while vLLM is on 8000)."
    )

    if use_separate_embed:
        embed_backend_choice = st.selectbox(
            "Embedding Backend",
            ["Ollama", "LM Studio", "vLLM", "Custom URL"],
            index=0,
            key="emb_backend_select"
        )
        embed_base_url = st.text_input(
            "Embedding Endpoint URL",
            value=default_urls[embed_backend_choice],
            key="emb_url_input"
        )
        embed_client = LocalLLMClient(base_url=embed_base_url)
        if embed_client.health_check():
            st.success(f"Connected to Embedding Backend ({embed_backend_choice})")
            embed_models_raw = embed_client.list_models()
        else:
            st.warning(f"Embedding backend offline at {embed_base_url}.")
            embed_models_raw = ["snowflake-arctic-embed2:latest", "text-embedding-nomic-embed-text-v1.5"]
    else:
        embed_client = llm_client
        embed_models_raw = available_models

    # Sort embedding models first
    emb_preferred = [m for m in embed_models_raw if LLMBackendConfig.is_embedding_model(m)]
    emb_others = [m for m in embed_models_raw if m not in emb_preferred]
    emb_options = emb_preferred + emb_others
    if not emb_options:
        emb_options = ["text-embedding-nomic-embed-text-v1.5", "snowflake-arctic-embed2:latest", "bge-m3"]

    selected_embedding_model = st.selectbox(
        "Selected Embedding Model",
        emb_options,
        index=0,
        help="Local model used to generate vector embeddings for semantic retrieval, similarity, and graph grounding."
    )

    st.divider()

    st.subheader("2. nanoGentzen Logic Engine")
    model_path = st.text_input(
        "nanoGentzen Model Path", value=str(DEFAULT_MODEL_PATH),
        help="Folder or model.safetensors. v4 checkpoints (config.json format_version 4) are detected automatically.")
    device_mode = st.selectbox("Inference Device", ["auto", "cuda", "cpu"], index=0)
    gz_defaults = GentzenConfig()
    model_version = detect_model_version(model_path)
    v4_opts = {}
    if model_version >= 4:
        v4_backend = st.selectbox(
            "v4 Network Backend", ["auto", "torch", "numpy", "symbolic"], index=0,
            help="auto: torch if installed, else NumPy. symbolic: no network, fixed-order G4ip. "
                 "Verdicts are identical in every mode; the network only orders proof search.")
        v4_budget = st.number_input("Search budget (nodes)", 8, 4096, gz_defaults.search_budget, step=8)
        v4_network = st.checkbox("Network-guided search", value=gz_defaults.network)
        v4_opts = {"backend": v4_backend, "search_budget": int(v4_budget), "network": bool(v4_network)}
        st.session_state.jev_allow_completion = st.checkbox(
            "Jev: complete state with minimal missing facts", value=gz_defaults.allow_state_completion,
            help="If the target decision is not provable, add only the minimal facts that make it provable "
                 "without contradiction (recorded in provenance). Such records are PROVEN_UNDER_ASSUMPTIONS and "
                 "not verified, because the added facts are not from the document. Off: unprovable targets stay unproven.")
        st.session_state.jev_label_smoothing = st.slider(
            "Jev: label smoothing on proven labels", 0.0, 0.3, gz_defaults.label_smoothing, 0.01,
            help="0.0 keeps the kernel's degenerate distribution (1.0 on the proven option).")

    engine_key = (str(model_path), device_mode, model_version, tuple(sorted(v4_opts.items())))
    if st.session_state.get("gentzen_engine_key") != engine_key or "gentzen_engine" not in st.session_state:
        try:
            st.session_state.gentzen_engine = load_engine(model_path=model_path, device=device_mode, **v4_opts)
        except Exception as e:
            st.error(f"Could not load nanoGentzen from `{model_path}`: {e}")
            st.session_state.gentzen_engine = NanoGentzenEngine(model_path=model_path, device=device_mode)
        st.session_state.gentzen_engine_key = engine_key

    engine = st.session_state.gentzen_engine

    if st.button("Reload nanoGentzen"):
        st.session_state.pop("gentzen_engine_key", None)
        st.rerun()

    if getattr(engine, "version", 3) >= 4:
        s = engine.status()
        gates = s.get("gates") or {}
        gate_txt = " ".join(f"{k.split('_')[0]}{'✔' if v else '✘'}" for k, v in gates.items())
        msg = f"nanoGentzen-v4: {s['backend'].replace('v4_', '')} on {s['device']}"
        (st.success if engine.is_safetensors_loaded else st.info)(msg)
        st.caption(f"best epoch {s.get('best_epoch')} · s1_temperature {s.get('s1_temperature')} · gates {gate_txt}")
    else:
        st.info("nanoGentzen: Symbolic Heuristic Mode (place the nanoGentzen v4 weights in `model/` for neural inference)")

    st.divider()
    st.caption("nanoGentzen Logic Tree Engine • 100% Local Inference")


# ----------------------------------------------------------------------
# Initialize Core Services (Preserved in session_state across reruns)
# ----------------------------------------------------------------------
if "parser" not in st.session_state:
    st.session_state.parser = DocumentParser()
parser = st.session_state.parser

if "chunker" not in st.session_state:
    st.session_state.chunker = DocumentChunker()
chunker = st.session_state.chunker

if "architect" not in st.session_state:
    st.session_state.architect = LogicalArchitect(engine=engine)
architect = st.session_state.architect
architect.update_engine(engine)

if "kg_builder" not in st.session_state:
    st.session_state.kg_builder = KnowledgeGraphBuilder(llm_client=llm_client, architect=architect)
kg_builder = st.session_state.kg_builder
kg_builder.llm_client = llm_client
kg_builder.architect = architect

# Startup policy for the last active knowledge graph (config.KG_AUTOLOAD: "ask" / "off" / "last")
if not st.session_state.get("kg_autoload_done"):
    _last_kg = None
    if KG_AUTOLOAD != "off" and kg_builder.graph.number_of_nodes() == 0:
        try:
            _last_kg = kg_store.last_active()
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not read the knowledge-graph index: {e}")
    if _last_kg is None:
        st.session_state.kg_autoload_done = True
    elif KG_AUTOLOAD == "last":
        st.session_state.kg_autoload_done = True
        try:
            _r = restore_graph(st.session_state, kg_builder, _last_kg["kg_id"],
                               restore_chunks=not st.session_state.get("chunks"))
            st.toast(f"Restored knowledge graph '{_last_kg['kg_id']}' "
                     f"({_r['num_nodes']} entities, {_r['num_edges']} relations)", icon="📂")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not restore saved knowledge graph: {e}")
    else:  # "ask"
        _docs = _last_kg.get("documents") or []
        _doc_txt = ", ".join(_docs[:3]) + (f" +{len(_docs) - 3} more" if len(_docs) > 3 else "")
        with st.container(border=True):
            st.markdown(f"📂 **Last knowledge graph:** `{_last_kg['kg_id']}` - {_last_kg.get('num_nodes', 0)} entities, "
                        f"{_last_kg.get('num_edges', 0)} relations, {len(_docs)} document(s)"
                        + (f" ({_doc_txt})" if _doc_txt else ""))
            _a1, _a2, _a3 = st.columns([1, 1, 2])
            with _a3:
                _with_chunks = st.checkbox("Also restore its chunks into the Library", value=False,
                                           key="kg_autoload_with_chunks")
            with _a1:
                if st.button("Restore", type="primary", key="kg_autoload_restore"):
                    st.session_state.kg_autoload_done = True
                    try:
                        restore_graph(st.session_state, kg_builder, _last_kg["kg_id"], restore_chunks=_with_chunks)
                    except Exception as e:
                        st.session_state.kg_autoload_error = str(e)
                    st.rerun()
            with _a2:
                if st.button("Start empty", key="kg_autoload_skip"):
                    st.session_state.kg_autoload_done = True
                    st.rerun()
if st.session_state.get("kg_autoload_error"):
    st.warning(f"Could not restore the saved knowledge graph: {st.session_state.pop('kg_autoload_error')}")

if "comm_detector" not in st.session_state:
    st.session_state.comm_detector = CommunityDetector()
comm_detector = st.session_state.comm_detector

if "q_generator" not in st.session_state:
    st.session_state.q_generator = QuestionGenerator(llm_client=llm_client)
q_generator = st.session_state.q_generator
q_generator.llm_client = llm_client

if "cot_generator" not in st.session_state:
    st.session_state.cot_generator = CoTGenerator(llm_client=llm_client)
cot_generator = st.session_state.cot_generator
cot_generator.llm_client = llm_client

if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatasetManager(db_path=str(DB_PATH))
db_manager = st.session_state.db_manager

if "jev_manager" not in st.session_state:
    st.session_state.jev_manager = JevDatasetManager(db_path=str(DB_PATH))
jev_manager = st.session_state.jev_manager
# new samples record which knowledge graph they came from (provenance.kg_id)
db_manager.kg_id = jev_manager.kg_id = st.session_state.get("kg_id")

if "jev_generator" not in st.session_state:
    st.session_state.jev_generator = JevScenarioGenerator(llm_client=llm_client, engine=engine)
jev_generator = st.session_state.jev_generator
jev_generator.llm_client = llm_client
jev_generator.update_engine(engine)
jev_generator.architect.allow_state_completion = st.session_state.get("jev_allow_completion", True)
jev_generator.architect.label_smoothing = float(st.session_state.get("jev_label_smoothing", 0.0))

if "batch_generator" not in st.session_state:
    st.session_state.batch_generator = BatchDatasetGenerator(
        q_generator=q_generator,
        architect=architect,
        cot_generator=cot_generator,
        db_manager=db_manager
    )
batch_generator = st.session_state.batch_generator


# ----------------------------------------------------------------------
# Main Application Tabs
# ----------------------------------------------------------------------
st.title("nanoGentzen ClearCoT & Jev Decision Studio")
st.markdown("##### *From Unstructured Text ➔ Knowledge Graph ➔ nanoGentzen Logic Proofs ➔ ClearCoT Reasoning & Jev Decision Datasets*")

tab_lib, tab_kg, tab_rev, tab_jev, tab_exp = st.tabs([
    "📚 Document Library",
    "🕸️ Knowledge Graph & Paths",
    "🔬 Logic Tree & ClearCoT Workspace",
    "🎯 Jev Decision Builder",
    "📦 Dataset Export & Stats"
])

with tab_lib:
    render_library_tab(
        parser=parser,
        chunker=chunker,
        embed_client=embed_client,
        embedding_model=selected_embedding_model
    )

with tab_kg:
    render_graph_tab(
        kg_builder=kg_builder,
        comm_detector=comm_detector,
        teacher_model=selected_model,
        default_base_url=base_url,
        default_backend_name=backend_choice,
        embed_client=embed_client,
        embedding_model=selected_embedding_model,
    )

with tab_rev:
    render_review_tab(
        q_generator=q_generator,
        architect=architect,
        cot_generator=cot_generator,
        db_manager=db_manager,
        selected_model=selected_model,
        batch_generator=batch_generator,
        kg_builder=kg_builder
    )

with tab_jev:
    render_jev_tab(
        chunks=st.session_state.get("chunks", []),
        selected_path=st.session_state.get("selected_path", None),
        jev_generator=jev_generator,
        jev_manager=jev_manager,
        selected_model=selected_model,
        kg_builder=kg_builder,
        available_models=gen_models,
    )

with tab_exp:
    render_export_tab(db_manager=db_manager)
"""
Thematic Knowledge Graph Tab for ClearCoT Studio.
Visualizes entities, relations, communities, and discovers multi-hop reasoning paths.
"""

import streamlit as st
import json
import streamlit.components.v1 as components
import pandas as pd
from typing import Dict, Any, List
import tempfile
from pathlib import Path

from core.kg_builder import KnowledgeGraphBuilder
from core import kg_store
from core.community_detector import CommunityDetector
from core.chunker import Chunk
from gui._paths import publish_discovered_paths
from gui._widgets import int_slider
from gui._kg_session import KG_ID_KEY, reset_graph, restore_graph, save_current

def _graph_chunks(kg_builder: KnowledgeGraphBuilder, fallback: List[Chunk]) -> List[Chunk]:
    """Library chunks of the graph's own documents (never other topics' chunks), else `fallback`."""
    docs = kg_builder.graph_documents()
    lib = st.session_state.get("chunks") or []
    own = [c for c in lib if c.doc_name in docs] if docs else []
    return own or fallback


def _embed_fn(embed_client: Any, embedding_model: Any):
    if embed_client is None or not embedding_model:
        return None
    return lambda texts: embed_client.embed(texts, model=embedding_model)


def _run_concept_repair(kg_builder: KnowledgeGraphBuilder, comm_detector: CommunityDetector, embed_fn: Any,
                        sense_threshold: float, chunks: List[Chunk]) -> Dict[str, Any]:
    """Repair the graph's concepts, recompute communities, back up the previous save and save the result."""
    if st.session_state.get(KG_ID_KEY):
        kg_store.backup_before_repair(st.session_state[KG_ID_KEY])
    rep = kg_builder.repair_concepts(chunks=chunks or None, embed_fn=embed_fn, sense_threshold=sense_threshold)
    comm_res = comm_detector.detect_communities(kg_builder.graph)
    st.session_state.communities = comm_res["communities"]
    try:
        save_current(st.session_state, kg_builder, chunks=chunks)
    except Exception as e:
        st.warning(f"Graph could not be saved: {e}")
    st.session_state.kg_repair_report = rep
    return rep


def _render_concept_quality(kg_builder: KnowledgeGraphBuilder, comm_detector: CommunityDetector,
                            embed_client: Any, embedding_model: Any):
    """Concept quality panel: what the nodes are worth as concepts, and the repair pass (core/concepts.py)."""
    summ = kg_builder.concept_summary()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Meaningful concepts", f"{summ['meaningful_share']:.0%}", help="Nodes with concept_quality >= 0.5")
    m2.metric("Inference edges", summ["inference_edges"],
              help="Edges whose predicate licenses an inference AND whose evidence states it (evidence gate)")
    m3.metric("Taxonomy edges", summ["taxonomy_edges"], help="'X is a kind of Y' derived from concept names")
    m4.metric("Disconnected pieces", summ["components"])
    if summ["issues"]:
        st.caption("Issues: " + " · ".join(f"{k} {v}" for k, v in summ["issues"].items()))

    G = kg_builder.graph
    rows = []
    for n, a in G.nodes(data=True):
        q = a.get("concept_quality", 1.0)
        if q < 0.5:
            rows.append({"Concept": n, "Quality": q, "Issues": ", ".join(a.get("concept_issues", [])),
                         "Edges": G.degree(n)})
    if rows:
        with st.expander(f"Concepts excluded from path discovery ({len(rows)})"):
            st.dataframe(sorted(rows, key=lambda r: (r["Quality"], -r["Edges"]))[:300], hide_index=True, width="stretch")

    st.markdown("##### 🛠️ Repair & consolidate concepts")
    st.caption("Completes truncated names from the evidence, merges spelling variants, re-reads relations "
               "('is based on' → depends on, 'is necessary for' → requires, reversed), keeps an inference predicate only "
               "if its evidence states it (and flips swapped edges), and adds 'is a kind of' edges between concepts. "
               "No re-extraction; the previous save is kept as <graph>.before_repair.json.")
    c1, c2 = st.columns(2)
    with c1:
        use_sense = st.checkbox("Sense-check taxonomy with embeddings", value=bool(embedding_model),
                                disabled=not embedding_model, key="kg_repair_sense",
                                help="Rejects homonyms ('Marketing Agency' is not a kind of moral 'Agency') by comparing "
                                     "how the text uses each concept. Uses the sidebar's embedding model.")
    with c2:
        thr = st.slider("Sense threshold", 0.2, 0.8, 0.45, 0.01, key="kg_repair_thr", disabled=not use_sense,
                        help="Calibrated on paraphrase-multilingual (homonyms ~0.38, valid kinds >= 0.5). Other "
                             "embedding models use a different scale; check the rejected list after a run.")
    if st.button("🛠️ Repair & consolidate concepts", key="btn_kg_repair"):
        with st.spinner("Repairing concepts ..."):
            rep = _run_concept_repair(kg_builder, comm_detector,
                                      _embed_fn(embed_client, embedding_model) if use_sense else None, thr,
                                      _graph_chunks(kg_builder, []))
        st.success(f"Inference edges {rep['inference_edges_before']} → {rep['inference_edges_after']}, "
                   f"nodes {rep['nodes_before']} → {rep['nodes_after']}.")
    rep = st.session_state.get("kg_repair_report")
    if rep:
        st.markdown("**Last repair**")
        st.caption(f"renamed {len(rep.get('renamed', {}))} · merged {len(rep.get('merged', {}))} · aliases "
                   f"{len(rep.get('aliases_merged', {}))} · predicates upgraded {rep.get('predicates_upgraded', 0)} · "
                   f"inference labels demoted {len(rep.get('inference_demoted', []))} · flipped "
                   f"{len(rep.get('inference_flipped', []))} · taxonomy {len(rep.get('taxonomy_added', []))} "
                   f"(rejected by sense {len(rep.get('taxonomy_rejected_sense', []))})")
        for title, key, cols in [
            ("Renamed / completed names", "renamed", ["Before", "After"]),
            ("Merged variants and aliases", "merged", ["Merged", "Into"]),
        ]:
            items = list((rep.get(key) or {}).items()) + (list((rep.get("aliases_merged") or {}).items()) if key == "merged" else [])
            if items:
                with st.expander(f"{title} ({len(items)})"):
                    st.dataframe([dict(zip(cols, it)) for it in items], hide_index=True, width="stretch")
        if rep.get("inference_flipped"):
            with st.expander(f"Edges flipped to match the evidence ({len(rep['inference_flipped'])})"):
                st.dataframe([{"Subject": a, "Predicate": b, "Object": c} for a, b, c in rep["inference_flipped"]],
                             hide_index=True, width="stretch")
        if rep.get("inference_demoted"):
            with st.expander(f"Inference labels the evidence does not support ({len(rep['inference_demoted'])})"):
                st.dataframe([{"Subject": a, "Claimed": b, "Object": c, "Why": w} for a, b, c, w in rep["inference_demoted"]],
                             hide_index=True, width="stretch")
        if rep.get("taxonomy_added"):
            with st.expander(f"Taxonomy edges added ({len(rep['taxonomy_added'])})"):
                st.dataframe([{"Specific": a, "is a kind of": b} for a, b in rep["taxonomy_added"]], hide_index=True,
                             width="stretch")
        if rep.get("taxonomy_rejected_sense"):
            with st.expander(f"Taxonomy rejected by the sense check ({len(rep['taxonomy_rejected_sense'])})"):
                st.dataframe([{"Specific": a, "General": b, "Similarity": x} for a, b, x in rep["taxonomy_rejected_sense"]],
                             hide_index=True, width="stretch")


def render_graph_tab(
    kg_builder: KnowledgeGraphBuilder,
    comm_detector: CommunityDetector,
    teacher_model: str,
    default_base_url: str = "http://localhost:1234/v1",
    default_backend_name: str = "Sidebar Active Backend",
    embed_client: Any = None,
    embedding_model: Any = None,
):
    st.header("🕸️ Thematic Knowledge Graph & Multi-hop Discovery")
    st.caption("Construct connected concept graphs from document chunks and extract multi-step reasoning paths.")

    all_chunks: List[Chunk] = st.session_state.get("chunks", [])
    doc_options = ["All Documents"] + sorted(list(set(c.doc_name for c in all_chunks if c.doc_name)))

    col_ctrl, col_stats = st.columns([2, 1])
    with col_ctrl:
        selected_doc = st.selectbox("Source Document for KG", doc_options, index=0)
        if selected_doc != "All Documents":
            chunks = [c for c in all_chunks if c.doc_name == selected_doc]
        else:
            chunks = all_chunks

        total_chunks_count = len(chunks)
        st.write(f"Knowledge Chunks available ({selected_doc}): **{total_chunks_count}**")

        selected_chunks = []
        if total_chunks_count > 0:
            sel_mode = st.radio(
                "Chunk Selection Mode",
                ["📌 Specific Range (e.g. 20–60)", "🎲 Random Sample", "📑 First N", "🌐 All Chunks"],
                horizontal=True
            )

            if "Specific Range" in sel_mode:
                default_end = min(30, total_chunks_count)
                c_start, c_end = int_slider(
                    "Select Chunk Range (1-indexed)",
                    min_value=1,
                    max_value=total_chunks_count,
                    value=(1, default_end),
                    step=1,
                    help="Choose any specific range of chunks (e.g. from 20 to 60 of 180)"
                )
                selected_chunks = chunks[c_start - 1 : c_end]
                st.info(f"Targeting **{len(selected_chunks)}** chunks: #{c_start} to #{c_end} (out of {total_chunks_count})")

            elif "Random Sample" in sel_mode:
                sample_size = int_slider(
                    "Number of Random Chunks",
                    min_value=1,
                    max_value=total_chunks_count,
                    value=min(20, total_chunks_count),
                    step=1,
                    help="Randomly sample chunks across the entire document"
                )
                import random
                col_s1, col_s2 = st.columns([1, 1])
                with col_s1:
                    use_seed = st.checkbox("Reproducible Seed", value=True)
                with col_s2:
                    seed_val = st.number_input("Seed", min_value=0, value=42, step=1, disabled=not use_seed)

                rng = random.Random(seed_val if use_seed else None)
                sample_indices = sorted(rng.sample(range(total_chunks_count), sample_size))
                selected_chunks = [chunks[i] for i in sample_indices]
                preview_indices = ", ".join(f"#{i+1}" for i in sample_indices[:6])
                if len(sample_indices) > 6:
                    preview_indices += f", ... (+{len(sample_indices)-6} more)"
                st.info(f"Randomly selected **{len(selected_chunks)}** chunks: {preview_indices}")

            elif "First N" in sel_mode:
                first_n = int_slider(
                    "Number of Chunks from Start",
                    min_value=1,
                    max_value=total_chunks_count,
                    value=min(30, total_chunks_count),
                    step=1
                )
                selected_chunks = chunks[:first_n]
                st.info(f"Targeting first **{len(selected_chunks)}** chunks (1 to {first_n})")

            else:  # All Chunks
                selected_chunks = chunks
                st.info(f"Targeting all **{len(selected_chunks)}** available chunks")
        else:
            st.warning("No chunks available for this document. Please upload or ingest a document in Library tab.")

        kg_engine = st.radio(
            "KG Construction Engine",
            ["🌟 Knwler (Orbifold Native)", "⚡ Direct Pipeline"],
            index=0,
            horizontal=True,
            help="Knwler applies typed schema extraction, entity linking, and cross-chunk graph consolidation. Direct Pipeline extracts rich proposition triples directly."
        )

        backend_presets = {
            "Sidebar Active Backend": default_base_url,
            "vLLM (http://localhost:8000/v1)": "http://localhost:8000/v1",
            "LM Studio (http://localhost:1234/v1)": "http://localhost:1234/v1",
            "Ollama (http://localhost:11434/v1)": "http://localhost:11434/v1",
            "Custom URL": ""
        }

        # Initialize endpoint in session_state if not present
        if "kg_endpoint_input" not in st.session_state:
            st.session_state.kg_endpoint_input = default_base_url

        def _on_kg_backend_change():
            chosen = st.session_state.get("kg_backend_select")
            if chosen == "Sidebar Active Backend":
                st.session_state.kg_endpoint_input = default_base_url
            elif chosen and chosen in backend_presets and chosen != "Custom URL":
                st.session_state.kg_endpoint_input = backend_presets[chosen]

        # Auto-sync if sidebar active backend changed while "Sidebar Active Backend" is selected
        if st.session_state.get("kg_backend_select") in (None, "Sidebar Active Backend"):
            if st.session_state.get("prev_sidebar_base_url") != default_base_url:
                st.session_state.kg_endpoint_input = default_base_url
                st.session_state.prev_sidebar_base_url = default_base_url

        col_be, col_ep = st.columns([1, 1])
        with col_be:
            default_be_idx = 0
            if "vLLM" in default_backend_name or ":8000" in default_base_url:
                default_be_idx = 1
            elif "LM Studio" in default_backend_name or ":1234" in default_base_url:
                default_be_idx = 2
            elif "Ollama" in default_backend_name or ":11434" in default_base_url:
                default_be_idx = 3

            kg_backend = st.selectbox(
                "KG Extraction Backend",
                list(backend_presets.keys()),
                index=default_be_idx,
                key="kg_backend_select",
                on_change=_on_kg_backend_change,
                help="Select target LLM endpoint for Knowledge Graph extraction. Selecting a preset updates the URL automatically."
            )

        with col_ep:
            kg_endpoint = st.text_input(
                "Backend Endpoint URL",
                key="kg_endpoint_input",
                help="Target API endpoint. Automatically updates when changing presets above, or edit directly."
            )

        # Discover models from the selected endpoint
        from llm_backend.client import LocalLLMClient
        from config import LLMBackendConfig

        probe_client = LocalLLMClient(base_url=kg_endpoint)
        is_kg_online = probe_client.health_check()

        col_mod, col_conc = st.columns([1, 1])
        with col_mod:
            if is_kg_online:
                live_models = probe_client.list_models()
                gen_models = [m for m in live_models if not LLMBackendConfig.is_embedding_model(m)] or live_models
                if not gen_models:   # online but no models listed: fall back to the sidebar model, never None
                    gen_models = [teacher_model] if teacher_model else ["Sagicc/Occam-2B-CCoT"]
                m_idx = gen_models.index(teacher_model) if teacher_model in gen_models else 0
                kg_model = st.selectbox(
                    "Extraction Model",
                    gen_models,
                    index=m_idx,
                    help=f"Discovered {len(gen_models)} models live from {kg_endpoint}."
                )
            else:
                fallback_models = [
                    teacher_model, "occam-2b", "qwen3.5-9b", "qwen3.5-2b",
                    "qwen3-0.6b-sr-instruct", "qwen2.5:7b", "llama3.1:8b"
                ]
                seen_m = set()
                fallback_models = [m for m in fallback_models if not (m in seen_m or seen_m.add(m))]
                kg_model = st.selectbox(
                    "Extraction Model",
                    fallback_models,
                    index=0,
                    help=f"Backend offline at {kg_endpoint}. Showing fallback models."
                )

        with col_conc:
            if "8000" in kg_endpoint or "vllm" in kg_backend.lower():
                suggested_workers = 8
            else:
                suggested_workers = 1

            kg_concurrency = st.slider(
                "⚡ Parallel Extraction Workers",
                min_value=1,
                max_value=16,
                value=suggested_workers,
                help="Set to 4–8 for vLLM continuous batching; keep at 1 for single-GPU LM Studio or Ollama."
            )

        repair_after = st.checkbox("🛠️ Repair & consolidate concepts after extraction", value=True, key="kg_repair_after",
                                   help="Runs the concept repair of the Concept Quality tab right after extraction.")
        has_existing_kg = kg_builder.graph.number_of_nodes() > 0
        graph_docs = kg_builder.graph_documents()
        sel_docs = {c.doc_name for c in selected_chunks if c.doc_name}
        new_docs = sel_docs - graph_docs
        mixing = bool(has_existing_kg and graph_docs and new_docs)
        current_kg_id = st.session_state.get(KG_ID_KEY)
        if has_existing_kg:
            st.caption(f"Active graph: `{current_kg_id or 'unsaved'}` - {len(graph_docs)} document(s)")
        if mixing:
            overlaps = kg_builder.doc_topic_overlap() if len(graph_docs) > 1 else {}
            st.warning(
                f"⚠️ The selected chunks come from {len(new_docs)} document(s) that are not in the current graph "
                f"({', '.join(sorted(new_docs)[:3])}{' ...' if len(new_docs) > 3 else ''}). Appending mixes their "
                "topics into this graph and into the Jev / ClearCoT paths. Usually you want **🧹 New empty graph** "
                "first, then build; the current graph stays saved.")
            allow_mix = st.checkbox("Append anyway (mixed topics; saved as a new graph, the current one is kept)",
                                    value=False, key="kg_allow_mixed_append")
        else:
            allow_mix = True
        if has_existing_kg:
            b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
            with b_col1:
                rebuild_clicked = st.button("🔨 Rebuild Entire KG", type="secondary", disabled=(len(selected_chunks) == 0), key="btn_kg_rebuild")
            with b_col2:
                append_clicked = st.button("➕ Append Chunks to KG", type="primary",
                                           disabled=(len(selected_chunks) == 0 or not allow_mix), key="btn_kg_append",
                                           help="Extracts only chunks not yet in the graph, preserving existing nodes and relations.")
            with b_col3:
                if st.button("🧹 New empty graph", key="btn_kg_new_empty",
                             help="Empty the in-memory graph so a new topic starts clean. Saved graphs are kept."):
                    st.session_state.kg_confirm_reset = True
            if st.session_state.get("kg_confirm_reset"):
                with st.container(border=True):
                    st.markdown("Empty the current graph? Saved graph files are **not** deleted; "
                                "you can load this one again from 💾 Saved graphs.")
                    clear_lib = st.checkbox("Also clear the Library chunks", value=False, key="kg_reset_clear_chunks")
                    r1, r2 = st.columns(2)
                    if r1.button("Yes, start empty", type="primary", key="btn_kg_reset_yes"):
                        reset_graph(st.session_state, kg_builder, clear_chunks=clear_lib)
                        st.session_state.kg_confirm_reset = False
                        st.rerun()
                    if r2.button("Cancel", key="btn_kg_reset_no"):
                        st.session_state.kg_confirm_reset = False
                        st.rerun()
        else:
            rebuild_clicked = st.button("🔨 Extract Triples & Build KG", type="primary", disabled=(len(selected_chunks) == 0), key="btn_kg_rebuild")
            append_clicked = False

        if rebuild_clicked or append_clicked:
            # Which file the result is saved to: rebuilding the same documents overwrites this graph's own file;
            # a different document set, or an append that adds documents, gets a new id and keeps the old file.
            if rebuild_clicked:
                if current_kg_id and graph_docs and sel_docs == graph_docs:
                    target_kg_id = current_kg_id
                else:
                    target_kg_id = kg_store.unique_kg_id(kg_store.kg_id_for(sel_docs))
            elif new_docs or not current_kg_id:
                target_kg_id = kg_store.unique_kg_id(kg_store.kg_id_for(graph_docs | sel_docs))
            else:
                target_kg_id = current_kg_id
            st.session_state[KG_ID_KEY] = target_kg_id
            pbar = st.progress(0.0)
            status_text = st.empty()

            def progress_cb(curr, total, msg):
                pbar.progress(curr / total)
                status_text.text(msg)

            if rebuild_clicked:
                kg_builder.clear()
                if "Knwler" in kg_engine:
                    kg_builder.build_from_chunks_knwler(
                        chunks=selected_chunks,
                        base_url=kg_endpoint,
                        model=kg_model,
                        max_chunks=None,
                        concurrency=kg_concurrency,
                        progress_callback=progress_cb
                    )
                else:
                    kg_builder.build_from_chunks(
                        chunks=selected_chunks,
                        model=kg_model,
                        max_chunks=None,
                        concurrency=kg_concurrency,
                        progress_callback=progress_cb,
                        base_url=kg_endpoint
                    )
            else:
                res = kg_builder.append_from_chunks(
                    chunks=selected_chunks,
                    model=kg_model,
                    concurrency=kg_concurrency,
                    progress_callback=progress_cb,
                    base_url=kg_endpoint
                )
                st.toast(f"Appended {res.get('novel_chunks', 0)} new chunk(s) to knowledge graph!", icon="➕")

            # Detect communities
            comm_res = comm_detector.detect_communities(kg_builder.graph)
            st.session_state.communities = comm_res["communities"]
            st.session_state.kg_built = True
            st.success(f"Graph active with {kg_builder.graph.number_of_nodes()} entities and {kg_builder.graph.number_of_edges()} relations!")
            dropped = kg_builder.extraction_stats.get("dropped_unsupported", 0)
            if dropped:
                st.info(f"🔎 {dropped} extracted relation(s) dropped: their evidence was not found in the source chunk.")
            demoted = kg_builder.extraction_stats.get("inference_demoted", 0)
            if demoted:
                st.info(f"🧮 {demoted} inference label(s) (requires / causes / excludes ...) not stated by their evidence "
                        "were kept as plain relations.")
            if repair_after:
                with st.spinner("Repairing concepts ..."):
                    rep = _run_concept_repair(kg_builder, comm_detector,
                                              _embed_fn(embed_client, embedding_model),
                                              float(st.session_state.get("kg_repair_thr", 0.45)),
                                              _graph_chunks(kg_builder, selected_chunks))
                st.caption(f"🛠️ Concepts repaired: inference edges {rep['inference_edges_before']} → "
                           f"{rep['inference_edges_after']}, taxonomy edges {len(rep['taxonomy_added'])}.")
            try:
                kid = save_current(st.session_state, kg_builder, chunks=_graph_chunks(kg_builder, selected_chunks),
                                   kg_id=target_kg_id)
                st.caption(f"💾 Saved as `{kid}.json` in the graph store (see 💾 Saved graphs).")
            except Exception as e:
                st.warning(f"Graph could not be saved: {e}")

    with col_stats:
        stats = kg_builder.get_stats()
        st.metric("Total Entities", stats["num_nodes"])
        st.metric("Total Relations", stats["num_edges"])
        st.metric("Graph Density", stats["density"])
        ev = stats.get("evidence", {})
        if stats["num_edges"]:
            st.caption("Evidence: " + " · ".join(f"{k} {v}" for k, v in ev.items() if v))
        ex = stats.get("extraction", {})
        if ex.get("chunks_heuristic_fallback"):
            st.caption(f"⚠️ {ex['chunks_heuristic_fallback']} chunk(s) used the regex fallback (LLM output unusable)")
        _render_kg_persistence(kg_builder)

    st.divider()

    if kg_builder.graph.number_of_nodes() > 0:
        tab_vis, tab_comm, tab_paths, tab_quality = st.tabs(
            ["Interactive Visualizer", "Thematic Communities", "Multi-hop Paths", "🧠 Concept Quality"])

        with tab_quality:
            _render_concept_quality(kg_builder, comm_detector, embed_client, embedding_model)

        with tab_vis:
            st.subheader("Interactive Graph Exploration")
            _render_visualizer_on_demand(kg_builder)

        with tab_comm:
            st.subheader("Detected Thematic Communities")
            communities = st.session_state.get("communities", [])
            if communities:
                primary_comms = [c for c in communities if c.get("is_primary", True)]
                dyadic_comms = [c for c in communities if not c.get("is_primary", True)]

                if primary_comms:
                    st.markdown(f"##### 🏛️ Primary Thematic Communities ({len(primary_comms)})")
                    for c in primary_comms:
                        with st.expander(f"{c['label']} ({c['size']} entities)", expanded=(c['size'] >= 4)):
                            if c.get("description"):
                                st.info(c["description"])
                            st.markdown(f"**Central Hub Concept:** `{c['hub_node']}`")
                            st.markdown(f"**Members:** {', '.join(c['members'][:25])}")

                if dyadic_comms:
                    st.markdown(f"##### 🔗 Dyadic & Emerging Concepts ({len(dyadic_comms)})")
                    with st.expander(f"View {len(dyadic_comms)} Dyadic Relational Pairs"):
                        for c in dyadic_comms:
                            other_members = [m for m in c.get('members', []) if m != c.get('hub_node')]
                            if other_members:
                                st.markdown(f"• **{c['hub_node']}** ⟷ {', '.join(other_members)}")
                            else:
                                st.markdown(f"• **{c['hub_node']}** *(isolated concept)*")
            else:
                st.info("No communities detected yet.")

        with tab_paths:
            st.subheader("Discovered Reasoning Paths & Relational Chains")
            path_tab_multi, path_tab_direct = st.tabs([
                "🔗 Multi-Hop Chains (≥ 2 Hops)",
                "📍 Direct Grounded Propositions (1 Hop)"
            ])

            with path_tab_multi:
                distinct_docs_in_graph = set()
                for _, _, ed in kg_builder.graph.edges(data=True):
                    if ed.get("doc_name"):
                        distinct_docs_in_graph.add(ed["doc_name"])
                has_multi_docs = len(distinct_docs_in_graph) > 1

                if "intra_doc_only" not in st.session_state:
                    # When multiple documents exist in graph, default to cross-document synthesis enabled!
                    st.session_state.intra_doc_only = not has_multi_docs
                if "kg_path_limit" not in st.session_state:
                    st.session_state.kg_path_limit = 100

                col_scope1, col_scope2 = st.columns([3, 1])
                with col_scope1:
                    intra_doc_only = st.checkbox(
                        "🔒 Restrict Multi-Hop Paths to Same Document (Prevents Cross-Document Hallucinations)",
                        value=st.session_state.get("intra_doc_only", not has_multi_docs),
                        key="kg_intra_doc_checkbox",
                        help="When checked, reasoning paths are strictly contained within one document. Uncheck only if you wish to allow inter-document synthesis across thematically coherent files."
                    )
                    st.session_state.intra_doc_only = intra_doc_only
                with col_scope2:
                    if not intra_doc_only:
                        st.info("🌐 Cross-doc synthesis (topic-gated)")
                    else:
                        st.success("✅ Intra-doc isolation active")
                if not intra_doc_only:
                    st.session_state.kg_min_topic_overlap = st.slider(
                        "Minimum topic overlap between linked documents", 0.0, 0.5,
                        float(st.session_state.get("kg_min_topic_overlap", 0.05)), 0.01, key="kg_topic_overlap_slider",
                        help="A cross-document path is kept only if every pair of documents it connects shares enough "
                             "specific concepts (shared / smaller document's concepts). Same-topic papers typically "
                             "score 0.05-0.3; unrelated papers that share one term score near 0.")
                    overlaps = kg_builder.doc_topic_overlap()
                    pairs = sorted({tuple(sorted(k)) for k in overlaps})
                    if pairs:
                        with st.expander(f"Document topic overlap ({len(pairs)} pairs)"):
                            thr = st.session_state.kg_min_topic_overlap
                            st.dataframe([{"Document A": a, "Document B": b, "Overlap": overlaps[(a, b)],
                                           "Linked": "✅" if overlaps[(a, b)] >= thr else "—"} for a, b in pairs],
                                         hide_index=True, width="stretch")

                col_lim1, col_lim2 = st.columns([2, 2])
                with col_lim1:
                    path_limit = st.slider(
                        "Path Discovery Limit",
                        min_value=20,
                        max_value=300,
                        value=st.session_state.get("kg_path_limit", 100),
                        step=20,
                        key="kg_path_limit_slider",
                        help="Maximum multi-hop reasoning chains to discover and index across all graph nodes."
                    )
                    st.session_state.kg_path_limit = path_limit
                with col_lim2:
                    st.session_state.kg_require_entailment = st.checkbox(
                        "🧮 Only entailment chains",
                        value=st.session_state.get("kg_require_entailment", False),
                        key="kg_require_entailment_cb",
                        help="Keep only paths whose every hop is walked forwards and licenses an inference "
                             "(implies / requires / depends on / is a kind of / causes / excludes). Other paths "
                             "describe relations, not arguments, and cannot be formally verified.")
                    st.session_state.kg_verified_only = st.checkbox(
                        "🛡️ Only kernel-verified paths",
                        value=st.session_state.get("kg_verified_only", False),
                        key="kg_verified_only_cb",
                        help="Prove every entailment chain with nanoGentzen and show only PROVEN_TRUE paths. "
                             "The discovery limit then counts verified paths.")

                discovery_params = dict(
                    verified_only=st.session_state.get("kg_verified_only", False),
                    require_entailment=st.session_state.get("kg_require_entailment", False),
                    min_topic_overlap=float(st.session_state.get("kg_min_topic_overlap", 0.05)),
                    max_hops=3,
                    min_hops=2,
                    limit=path_limit,
                    intra_document_only=intra_doc_only,
                    scope_doc_name=selected_doc if selected_doc != "All Documents" else None,
                )
                paths = kg_builder.find_multihop_paths(
                    **discovery_params, architect=st.session_state.get("architect")
                )
                # The Jev tab reuses exactly this list (gui/_paths.py) instead of its own search
                publish_discovered_paths(st.session_state, kg_builder, paths, discovery_params)

                cross_paths = [p for p in paths if p.get("is_cross_document")]
                intra_paths = [p for p in paths if not p.get("is_cross_document")]

                if paths:
                    if not intra_doc_only and len(cross_paths) > 0:
                        st.info(
                            f"Discovered **{len(paths)}** multi-hop reasoning chains (≥ 2 hops): "
                            f"**{len(cross_paths)} cross-document** (inter-document synthesis) and "
                            f"**{len(intra_paths)} intra-document**.\n\n"
                            "Select an individual chain below for single-sample quality monitoring, "
                            "or head to the **Logic Tree & ClearCoT Workspace** to generate the complete dataset in batch."
                        )
                        path_filter = st.radio(
                            "Filter Displayed Paths",
                            [f"All Paths ({len(paths)})", f"🌐 Cross-Doc Only ({len(cross_paths)})", f"📄 Intra-Doc Only ({len(intra_paths)})"],
                            horizontal=True,
                            key="tab_graph_filter_mode"
                        )
                        if "Cross-Doc" in path_filter:
                            display_paths = cross_paths
                        elif "Intra-Doc" in path_filter:
                            display_paths = intra_paths
                        else:
                            display_paths = paths
                    else:
                        st.info(
                            f"Discovered **{len(paths)}** multi-hop reasoning chains (≥ 2 hops). "
                            "Select an individual chain below for single-sample quality monitoring, "
                            "or head to the **Logic Tree & ClearCoT Workspace** to generate the complete dataset in batch."
                        )
                        display_paths = paths

                    for idx, p in enumerate(display_paths, 1):
                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**Path {idx}** ({p['hops']} hops): `{p['path_repr']}`")
                                if p.get("composite_claim"):
                                    st.markdown(f"**Grounded Synthesis:** *{p['composite_claim']}*")
                                if p.get("propositions"):
                                    with st.expander("View Deductive Step Propositions", expanded=False):
                                        for s_idx, prop in enumerate(p["propositions"], 1):
                                            doc_tag = f" ({prop['doc_name']})" if prop.get("doc_name") and p.get("is_cross_document") else ""
                                            st.markdown(f"{s_idx}. **{prop.get('statement', '')}** (p.{prop.get('page_number', 1)}{doc_tag})")
                                            if prop.get("evidence"):
                                                st.caption(f"   Evidence: \"{prop['evidence'][:140]}...\"")
                                v_badge = "🛡️ **nanoGentzen:** Verified (PROVEN_TRUE) | " if p.get("gentzen_verified") else ""
                                if p.get("is_cross_document"):
                                    st.caption(f"{v_badge}🌐 **Cross-Doc Path:** `{', '.join(p.get('provenance_docs', []))}` | Source: {p['source']} ➔ Target: {p['target']} | Pages: {p['provenance_pages']}")
                                else:
                                    doc_lbl = p.get("doc_name") or (p.get("provenance_docs", [""])[0] if p.get("provenance_docs") else "")
                                    doc_str = f"Document: `{doc_lbl}` | " if doc_lbl else ""
                                    st.caption(f"{v_badge}{doc_str}Source: {p['source']} ➔ Target: {p['target']} | Pages: {p['provenance_pages']}")
                            with c2:
                                if st.button(f"⚡ Monitor Sample", key=f"gen_seed_{idx}"):
                                    st.session_state.selected_path = p
                                    st.session_state.current_sample = None
                                    st.session_state.pop("sel_review_kg_path", None)
                                    st.session_state["monitor_seed_mode"] = "Multi-hop KG Path"
                                    st.session_state.card_version = st.session_state.get("card_version", 0) + 1
                                    st.toast(f"Path {idx} selected for quality monitor! Head to the Workspace tab.", icon="🎯")
                else:
                    st.info("Graph is currently too sparse for multi-hop reasoning chains (≥ 2 hops). Ingest additional chunks or check the 'Direct Grounded Propositions' tab to monitor individual extracted relations.")

            with path_tab_direct:
                direct_edges = []
                for u, v, d in kg_builder.graph.edges(data=True):
                    stmt = d.get("statement", f"{u} {d.get('predicate', 'relates to')} {v}.")
                    ev = d.get("evidence", "")
                    cid = d.get("chunk_id", "")
                    p_num = d.get("page_number", 1)
                    direct_edges.append({
                        "source": u,
                        "target": v,
                        "hops": 1,
                        "path_nodes": [u, v],
                        "path_repr": f"({u}) --[{d.get('predicate', 'relates to')}]--> ({v})",
                        "statement": stmt,
                        "evidence": ev,
                        "chunk_ids": [cid] if cid else [],
                        "provenance_pages": [p_num],
                        "composite_claim": stmt,
                        "propositions": [{
                            "subject": u,
                            "predicate": d.get("predicate", "relates to"),
                            "object": v,
                            "statement": stmt,
                            "evidence": ev,
                            "page_number": p_num,
                            "chunk_id": cid
                        }]
                    })
                if direct_edges:
                    st.info(f"Found **{len(direct_edges)}** direct relational propositions in the knowledge graph.")
                    for idx, p in enumerate(direct_edges[:40], 1):
                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**Proposition {idx}**: `{p['path_repr']}`")
                                st.markdown(f"**Statement:** *{p['statement']}*")
                                if p.get("evidence"):
                                    st.caption(f"Evidence: \"{p['evidence'][:160]}...\" (p.{p['provenance_pages'][0]})")
                                else:
                                    st.caption(f"Source: {p['source']} ➔ Target: {p['target']} | Provenance Pages: {p['provenance_pages']}")
                            with c2:
                                if st.button(f"⚡ Monitor Sample", key=f"gen_direct_{idx}"):
                                    st.session_state.selected_path = p
                                    st.session_state.current_sample = None
                                    st.session_state.pop("sel_review_kg_path", None)
                                    st.session_state["monitor_seed_mode"] = "Multi-hop KG Path"
                                    st.session_state.card_version = st.session_state.get("card_version", 0) + 1
                                    st.toast(f"Proposition {idx} selected! Head to the Workspace tab.", icon="🎯")
                else:
                    st.info("No propositions extracted yet.")
    else:
        st.info("Knowledge Graph is empty. Ingest documents and click 'Extract Triples & Build KG' above.")


def _render_visualizer_on_demand(kg_builder: KnowledgeGraphBuilder) -> None:
    """
    The interactive view is drawn only when asked for. Streamlit renders every tab on every rerun, so an
    always-on view re-sent the whole network (1000+ nodes) to the browser and restarted its layout physics
    after each click anywhere in the app.
    """
    n, m = kg_builder.graph.number_of_nodes(), kg_builder.graph.number_of_edges()
    shown = st.session_state.get("kg_vis_on", False)
    c1, c2 = st.columns([1, 3])
    with c1:
        if not shown:
            if st.button("🎨 Draw graph", key="kg_vis_draw", use_container_width=True):
                st.session_state.kg_vis_on = True
                st.rerun()
        elif st.button("✖ Hide graph", key="kg_vis_hide", use_container_width=True):
            st.session_state.kg_vis_on = False
            st.session_state.pop("_pyvis_html", None)       # free the cached page as well
            st.session_state.pop("_pyvis_sig", None)
            st.rerun()
    with c2:
        st.caption(f"{n} entities · {m} relations. The view is drawn on request"
                   + (" and slows every page refresh while it is shown; hide it when done." if n > 300 else "."))
    if shown:
        _render_pyvis_graph(kg_builder)


def _render_pyvis_graph(kg_builder: KnowledgeGraphBuilder):
    """Render PyVis interactive network graph inside an iframe with session caching."""
    num_nodes = kg_builder.graph.number_of_nodes()
    num_edges = kg_builder.graph.number_of_edges()
    sig = (num_nodes, num_edges)

    cached_sig = st.session_state.get("_pyvis_sig")
    cached_html = st.session_state.get("_pyvis_html")

    if cached_sig == sig and cached_html:
        components.html(cached_html, height=550)
        return

    try:
        from pyvis.network import Network

        net = Network(height="520px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="white")
        # vis-network's improvedLayout (Kamada-Kawai pre-positioning) gives up on large graphs ("This network
        # could not be positioned by this version of the improved layout algorithm"): turn it off and let the
        # force-atlas physics settle in a bounded number of steps, then stop.
        large = num_nodes > 300
        very_large = num_nodes > 1000
        net.set_options(json.dumps({
            "layout": {"improvedLayout": False},
            "physics": {
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {"gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100,
                                     "springConstant": 0.08, "avoidOverlap": 0.2 if large else 0},
                "stabilization": {"enabled": True, "iterations": 150 if very_large else (300 if large else 1000), "updateInterval": 50,
                                  "fit": True},
                "minVelocity": 0.75,
            },
            "edges": {"smooth": False if large else {"type": "dynamic"},
                      "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}},
            "interaction": {"hideEdgesOnDrag": large, "hideNodesOnDrag": very_large, "hover": not very_large, "tooltipDelay": 150},
        }))

        display_nodes = set()
        if very_large:
            # Level-of-detail (LOD): select top 800 nodes by degree to preserve graph topology without browser lag
            sorted_by_degree = sorted(kg_builder.graph.nodes(), key=lambda n: kg_builder.graph.degree(n), reverse=True)
            display_nodes = set(sorted_by_degree[:800])
            st.info(f"⚡ LOD Optimization: displaying 800 most connected entities (out of {num_nodes} total) for fast interactive rendering.")
        else:
            display_nodes = set(kg_builder.graph.nodes())

        for node in display_nodes:
            data = kg_builder.graph.nodes[node]
            color = data.get("color", "#4A90E2")
            net.add_node(node, label=node, color=color, title=f"Entity: {node}")

        for u, v, data in kg_builder.graph.edges(data=True):
            if u in display_nodes and v in display_nodes:
                pred = data.get("predicate", "")
                # edge labels on 1000+ edges are unreadable and slow; the predicate stays in the hover title
                extra = {} if large else {"label": pred}
                net.add_edge(u, v, title=f"Relation: {pred}", color="#aaaaaa", **extra)

        # Save to temp html file
        with tempfile.TemporaryDirectory() as tmp_dir:   # removed afterwards (was one leaked file per render)
            tmp_html = Path(tmp_dir) / "graph.html"
            net.save_graph(str(tmp_html))
            html_content = tmp_html.read_text(encoding="utf-8")

        st.session_state["_pyvis_sig"] = sig
        st.session_state["_pyvis_html"] = html_content
        components.html(html_content, height=550)
    except Exception as e:
        st.warning(f"Could not render interactive PyVis visualizer: {e}")
        # Fallback table
        nodes_df = pd.DataFrame([{"Entity": n, "Degree": kg_builder.graph.degree(n)} for n in kg_builder.graph.nodes()])
        st.dataframe(nodes_df.head(25), width="stretch")


def _render_kg_persistence(kg_builder: KnowledgeGraphBuilder):
    """Save / load the knowledge graph together with its chunks and communities."""
    with st.expander("💾 Saved graphs"):
        has_graph = kg_builder.graph.number_of_nodes() > 0
        default_name = st.session_state.get(KG_ID_KEY) or (
            kg_store.kg_id_for(kg_builder.graph_documents()) if has_graph else "")
        name = st.text_input("Save as", value=default_name, key="kg_save_name",
                             help="Graph id, derived from the source documents by default. Saved as "
                                  "data/kg/<id>.json; an existing graph with that id is overwritten.")
        if st.button("Save", key="kg_save_btn", disabled=not has_graph):
            kid = save_current(st.session_state, kg_builder, chunks=_graph_chunks(kg_builder, []),
                               kg_id=kg_store.sanitize_kg_id(name or default_name or "untitled-graph"))
            st.success(f"Saved `{kid}.json`")

        graphs = kg_store.list_graphs()
        if not graphs:
            st.caption("No saved graphs yet.")
            return
        ids = [g["kg_id"] for g in graphs]
        by_id = {g["kg_id"]: g for g in graphs}
        cur = st.session_state.get(KG_ID_KEY)
        choice = st.selectbox("Saved graph", ids, index=ids.index(cur) if cur in ids else 0, key="kg_load_choice",
                              format_func=lambda k: f"{k}{'  (active)' if k == cur else ''}")
        info = by_id[choice]
        docs = info.get("documents") or []
        st.caption(f"{info.get('num_nodes')} entities · {info.get('num_edges')} relations · "
                   f"{info.get('num_chunks')} chunks · saved {str(info.get('saved_at') or '')[:19]}")
        if docs:
            st.caption("Documents: " + ", ".join(docs[:5]) + (f" +{len(docs) - 5} more" if len(docs) > 5 else ""))
        with_chunks = st.checkbox("Also load its chunks into the Library", value=True, key="kg_load_with_chunks")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load", key="kg_load_btn"):
                restore_graph(st.session_state, kg_builder, choice, restore_chunks=with_chunks)
                st.rerun()
        with c2:
            if st.button("Delete…", key="kg_delete_btn"):
                st.session_state.kg_confirm_delete = choice
        if st.session_state.get("kg_confirm_delete") == choice:
            st.warning(f"Permanently delete `{info['file']}`? This cannot be undone.")
            d1, d2 = st.columns(2)
            if d1.button("Yes, delete", key="kg_delete_yes"):
                kg_store.delete_graph(choice)
                if cur == choice:
                    st.session_state.pop(KG_ID_KEY, None)  # in-memory graph stays; it is now unsaved
                st.session_state.kg_confirm_delete = None
                st.rerun()
            if d2.button("Cancel", key="kg_delete_no"):
                st.session_state.kg_confirm_delete = None
                st.rerun()


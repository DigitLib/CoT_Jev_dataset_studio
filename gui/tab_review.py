"""
Review & Curation Workspace Tab for ClearCoT Studio.
Three-column layout:
- Left: Source context & grounded evidence spans
- Center: nanoGentzen Best Logic Tree Generator (formal proof tree & deductive nodes)
- Right: Generated CoT (Q-R-A) built directly on the Logic Tree + Curation Decisions
"""

import streamlit as st
import json
import logging
from typing import List, Dict, Any, Optional

from config import EXPORTS_DIR
from core.chunker import Chunk, is_boilerplate
from core.question_generator import QuestionGenerator, _sanitize_question

from core.cot_generator import CoTGenerator
from core.dataset_manager import DatasetManager, DatasetSample
from core.batch_generator import BatchDatasetGenerator
from nanogentzen.logical_architect import LogicalArchitect
from nanogentzen.cot.architect import proof_record, with_proof_record
from llm_backend.client import clean_model_display_name, provenance_model_name
from gui._widgets import int_slider

def render_review_tab(
    q_generator: QuestionGenerator,
    architect: LogicalArchitect,
    cot_generator: CoTGenerator,
    db_manager: DatasetManager,
    selected_model: str,
    batch_generator: Optional[BatchDatasetGenerator] = None,
    kg_builder: Optional[Any] = None,
):
    st.header("🔬 Logic Tree & ClearCoT Workspace")
    st.caption("Monitor individual sample quality, then launch complete dataset generation across all Knowledge Graph paths and chunks.")

    modes = list(CoTGenerator.REASONING_MODES)
    cur_mode = st.session_state.get("cot_reasoning_mode", getattr(cot_generator, "reasoning_mode", "prompted"))
    st.session_state.cot_reasoning_mode = st.radio(
        "🧠 ClearCoT reasoning trace", modes, index=modes.index(cur_mode) if cur_mode in modes else 0,
        horizontal=True, key="cot_reasoning_mode_radio",
        help="prompted: the model writes the trace as text with thinking off. native: the model's own thinking "
             "channel (enable_thinking) is the trace, under the same constraints. Only the SFT/chosen trace changes; "
             "the DPO rejected trace and the token-savings baseline stay prompted. Knowledge-graph extraction and "
             "Jev scenarios never use thinking.")
    if hasattr(cot_generator, "reasoning_mode"):
        cot_generator.reasoning_mode = st.session_state.cot_reasoning_mode

    chunks: List[Chunk] = st.session_state.get("chunks", [])
    selected_path = st.session_state.get("selected_path", None)
    intra_doc_only = st.session_state.get("intra_doc_only", True)
    path_limit = st.session_state.get("kg_path_limit", 100)

    st.session_state.cot_require_entailment = st.checkbox(
        "🧮 ClearCoT: only entailment-chain paths",
        value=st.session_state.get("cot_require_entailment", True),
        key="cot_require_entailment_cb",
        help="Seed ClearCoT samples only from KG paths whose every hop licenses an inference "
             "(requires / implies / is a kind of / causes / excludes ...). Other paths describe relations, "
             "not arguments: nanoGentzen marks them INTUITIONISTIC_UNDECIDED. Independent of the Jev tab.")
    st.session_state.cot_verified_only = st.checkbox(
        "🛡️ ClearCoT: only kernel-verified paths",
        value=st.session_state.get("cot_verified_only", True),
        key="cot_verified_only_cb",
        help="Prove every candidate chain with nanoGentzen first and offer only PROVEN_TRUE paths as seeds, "
             "so the path limit counts usable paths.")
    tc1, tc2 = st.columns(2)
    with tc1:
        opts = ["any", "last", "none"]
        st.session_state.cot_taxonomy_policy = st.selectbox(
            "🌳 ClearCoT: 'is a kind of' hops", opts,
            index=opts.index(st.session_state.get("cot_taxonomy_policy", "any")),
            key="cot_taxonomy_policy_sb",
            help="any: no restriction, discovers all multi-hop reasoning chains. "
                 "last: at most one taxonomy hop, as the final step ('A requires B, B is a kind of C'). "
                 "none: no taxonomy hops.")
    with tc2:
        st.session_state.cot_max_paths_per_hub = int(st.number_input(
            "ClearCoT: max paths through one concept", min_value=0, max_value=500,
            value=int(st.session_state.get("cot_max_paths_per_hub", 8)), step=1, key="cot_max_paths_per_hub_ni",
            help="Paths are taken in rank order; a path is skipped once one of its intermediate concepts already "
                 "carries this many paths, so one hub cannot fill the dataset. 0 = no cap."))

    # Discover available KG paths for seed options
    available_kg_paths: List[Dict[str, Any]] = []
    if kg_builder and kg_builder.graph.number_of_nodes() > 1:
        available_kg_paths = kg_builder.find_multihop_paths(
            require_entailment=st.session_state.get("cot_require_entailment", True),
            verified_only=st.session_state.get("cot_verified_only", True),
            **_cot_path_options(),
            min_topic_overlap=float(st.session_state.get("kg_min_topic_overlap", 0.05)),
            max_hops=3,
            min_hops=2,
            limit=path_limit,
            intra_document_only=intra_doc_only,
            architect=architect,
            verify_with_gentzen=True,
        )
        if not available_kg_paths:
            available_kg_paths = kg_builder.find_multihop_paths(
                require_entailment=st.session_state.get("cot_require_entailment", True),
                verified_only=st.session_state.get("cot_verified_only", True),
                **_cot_path_options(),
                min_topic_overlap=float(st.session_state.get("kg_min_topic_overlap", 0.05)),
                max_hops=3,
                min_hops=1,
                limit=path_limit,
                intra_document_only=intra_doc_only,
                architect=architect,
                verify_with_gentzen=True,
            )

    # Protect selected path across tab navigation
    if selected_path and not any(p.get("path_repr") == selected_path.get("path_repr") for p in available_kg_paths):
        available_kg_paths.insert(0, selected_path)

    has_kg_paths = len(available_kg_paths) > 0
    if not selected_path and has_kg_paths:
        selected_path = available_kg_paths[0]
        st.session_state.selected_path = selected_path

    # Tabs: Single-Sample Quality Monitor vs. Batch Complete Dataset Generator
    subtab_monitor, subtab_batch = st.tabs([
        "🔬 Quality Monitor (1-by-1 Sandbox)",
        "🚀 Complete Dataset Generator (Batch Mode)"
    ])

    # ==================================================================
    # SUBTAB 1: Single-Sample Quality Monitor (Verification Sandbox)
    # ==================================================================
    with subtab_monitor:
        st.markdown("##### *Verify prompts, logic tree, and model reasoning on an individual sample before full generation.*")

        seed_options = ["Paragraph Seed", "Multi-hop KG Path"] if has_kg_paths else ["Paragraph Seed"]
        prev_seed_mode = st.session_state.get("monitor_seed_mode", "Paragraph Seed")
        default_mode_idx = seed_options.index(prev_seed_mode) if prev_seed_mode in seed_options else 0

        ctrl_col1, ctrl_col2 = st.columns([2, 1])
        with ctrl_col1:
            mode = st.radio(
                "Seed Source",
                seed_options,
                index=default_mode_idx,
                horizontal=True,
                key="monitor_seed_mode"
            )

            prev_mode = st.session_state.get("last_previewed_mode", mode)
            if mode != prev_mode:
                st.session_state.last_previewed_mode = mode
                st.session_state.card_version = st.session_state.get("card_version", 0) + 1
                st.session_state.current_sample = None

            if mode == "Multi-hop KG Path" and available_kg_paths:
                cross_in_review = [p for p in available_kg_paths if p.get("is_cross_document")]
                intra_in_review = [p for p in available_kg_paths if not p.get("is_cross_document")]

                if not intra_doc_only and len(cross_in_review) > 0:
                    scope_filter = st.radio(
                        "Path Scope Filter",
                        [f"All Available ({len(available_kg_paths)})", f"🌐 Cross-Doc Only ({len(cross_in_review)})", f"📄 Single-Doc Only ({len(intra_in_review)})"],
                        horizontal=True,
                        key="monitor_kg_scope_filter"
                    )
                    if "Cross-Doc" in scope_filter:
                        active_paths = cross_in_review
                    elif "Single-Doc" in scope_filter:
                        active_paths = intra_in_review
                    else:
                        active_paths = available_kg_paths
                else:
                    active_paths = available_kg_paths

                def _format_path_label(p: Dict[str, Any], idx: int) -> str:
                    hops = p.get("hops", 2)
                    repr_str = p.get("path_repr", "")
                    if p.get("is_cross_document"):
                        docs_str = " + ".join(p.get("provenance_docs", []))
                        return f"🌐 [Cross-Doc: {docs_str}] Path #{idx+1} ({hops} hops): {repr_str}"
                    else:
                        doc_str = p.get("doc_name") or (p.get("provenance_docs", [""])[0] if p.get("provenance_docs") else "")
                        doc_prefix = f"📄 [{doc_str}] " if doc_str else ""
                        return f"{doc_prefix}Path #{idx+1} ({hops} hops): {repr_str}"

                current_p_idx = 0
                if selected_path:
                    for i, p in enumerate(active_paths):
                        if p.get("path_repr") == selected_path.get("path_repr"):
                            current_p_idx = i
                            break

                # Protect against out-of-range index when changing scope filter
                if "sel_review_kg_path" in st.session_state:
                    # the stored value can be stale (another path list) or not an index at all (e.g. a label
                    # string left by an older widget with this key); anything unusable falls back to current_p_idx
                    stored = st.session_state["sel_review_kg_path"]
                    if not isinstance(stored, int) or isinstance(stored, bool) or not 0 <= stored < len(active_paths):
                        st.session_state["sel_review_kg_path"] = min(current_p_idx, max(0, len(active_paths) - 1))

                chosen_p_idx = st.selectbox(
                    "Selected Multi-Hop KG Path",
                    range(len(active_paths)),
                    index=current_p_idx,
                    format_func=lambda i: _format_path_label(active_paths[i], i),
                    key="sel_review_kg_path"
                )
                selected_path = active_paths[chosen_p_idx]
                st.session_state.selected_path = selected_path

                # Detect when selected path changes to invalidate cached widgets & stale samples
                curr_path_repr = selected_path.get("path_repr", "")
                prev_path_repr = st.session_state.get("active_preview_path_repr", "")
                if curr_path_repr != prev_path_repr:
                    st.session_state.active_preview_path_repr = curr_path_repr
                    st.session_state.card_version = st.session_state.get("card_version", 0) + 1
                    old_sample = st.session_state.get("current_sample")
                    if old_sample:
                        old_sample_path = old_sample.get("provenance", {}).get("path_repr", "")
                        if old_sample_path != curr_path_repr:
                            st.session_state.current_sample = None

                if selected_path.get("composite_claim"):
                    st.caption(f"🎯 **Grounded Claim:** *{selected_path['composite_claim']}*")
            elif not has_kg_paths:
                st.caption("💡 Ingest chunks and build a Knowledge Graph in Tab 2 to enable Multi-hop KG paths.")

        with ctrl_col2:
            st.write("")
            btn_label = "⚡ Synthesize ClearCoT (Tree Pre-Verified)" if (mode == "Multi-hop KG Path" and selected_path and selected_path.get("gentzen_verified")) else "⚡ Generate Best Logic Tree & ClearCoT"
            if st.button(btn_label, type="primary", width="stretch", key="btn_gen_single"):
                _build_candidate_sample(mode, chunks, selected_path, q_generator, architect, cot_generator, selected_model)

        current_sample = st.session_state.get("current_sample", None)

        if not current_sample:
            if mode == "Multi-hop KG Path" and selected_path:
                logic_tree_display = selected_path.get("logic_tree_text", "")
                sample_preview = {
                    "question": f"How do the concepts in '{selected_path['source']}' connect to '{selected_path['target']}'?",
                    "reasoning": "",
                    "answer": selected_path.get("composite_claim", ""),
                    "target_claim": selected_path.get("composite_claim", ""),
                    "key_premises": [p.get("statement", "") for p in selected_path.get("propositions", [])],
                    "path_nodes": selected_path.get("path_nodes", []),
                    "provenance": {
                        "doc_name": selected_path.get("doc_name", ""),
                        "documents": selected_path.get("provenance_docs", []),
                        "is_cross_document": selected_path.get("is_cross_document", False),
                        "pages": selected_path.get("provenance_pages", []),
                        "path_repr": selected_path.get("path_repr", "")
                    },
                    "context_text": "\n\n".join(f"[{p.get('doc_name', '')} p.{p.get('page_number', 1)}] {p.get('statement', '')} Evidence: {p.get('evidence', '')}" for p in selected_path.get("propositions", [])),
                    "gentzen_verified": selected_path.get("gentzen_verified", True),
                    "gentzen_sequent": selected_path.get("gentzen_sequent", ""),
                    "proof_tree": selected_path.get("proof_tree", None),
                    "logic_tree_text": logic_tree_display,
                    "provability_score": selected_path.get("provability_score", 1.0),
                    "tree_depth": selected_path.get("tree_depth", selected_path.get("hops", 2)),
                    "node_count": selected_path.get("node_count", len(selected_path.get("path_nodes", []))),
                    "curry_howard_term": selected_path.get("curry_howard_term", ""),
                    "relevance": selected_path.get("relevance", {}),
                    "epistemic_status": selected_path.get("epistemic_status", "PROVEN_TRUE"),
                    "is_precomputed_preview": True,
                }
                st.divider()
                if selected_path.get("gentzen_verified"):
                    st.info("💡 **Pre-computed nanoGentzen Logic Tree Loaded**: This Multi-Hop KG chain has been formally verified. Click **⚡ Synthesize ClearCoT (Tree Pre-Verified)** above to generate the full Q–R–A reasoning trace along this verified proof tree.")
                else:
                    st.info("💡 **Multi-Hop KG Chain Selected**: Click **⚡ Generate Best Logic Tree & ClearCoT** above to construct a formal intuitionistic logic tree and generate the complete reasoning trace.")
                _render_single_sample_workspace(sample_preview, architect, cot_generator, db_manager, selected_model)
            else:
                st.info("No active candidate in monitor. Select a seed source and click '⚡ Generate Best Logic Tree & ClearCoT' above.")
        else:
            st.divider()
            _render_single_sample_workspace(current_sample, architect, cot_generator, db_manager, selected_model)

    # ==================================================================
    # SUBTAB 2: Complete Dataset Generator (Batch Engine)
    # ==================================================================
    with subtab_batch:
        _render_batch_generator_tab(
            chunks=chunks,
            kg_builder=kg_builder,
            batch_generator=batch_generator,
            selected_model=selected_model,
            db_manager=db_manager
        )


def _render_single_sample_workspace(
    current_sample: Dict[str, Any],
    architect: LogicalArchitect,
    cot_generator: CoTGenerator,
    db_manager: DatasetManager,
    selected_model: str
):
    notice = st.session_state.pop("review_notice", None)
    if notice:
        if notice.get("type") == "warning":
            st.warning(notice.get("message", ""))
        elif notice.get("type") == "info":
            st.info(notice.get("message", ""))
        else:
            st.toast(notice.get("message", "ClearCoT updated!"), icon="✨")

    card_ver = st.session_state.get("card_version", 0)

    # Three-column layout
    left_col, center_col, right_col = st.columns([1.0, 1.2, 1.3], gap="medium")

    # ==================================================================
    # LEFT COLUMN: Source Context & Grounded Claim
    # ==================================================================
    with left_col:
        st.subheader("📖 Grounded Source")
        prov = current_sample.get("provenance", {})
        if prov.get("is_cross_document"):
            st.markdown(f"**Documents (Cross-Doc):** `{', '.join(prov.get('documents', []))}`")
        else:
            st.markdown(f"**Document:** `{prov.get('doc_name', 'Unknown')}`")
        st.markdown(f"**Pages:** `{prov.get('pages', [1])}`")
        if "section" in prov:
            st.markdown(f"**Section:** `{prov.get('section')}`")
        if "path_repr" in prov:
            st.markdown(f"**KG Path:** `{prov.get('path_repr')}`")

        st.markdown("**Evidence Context:**")
        st.text_area(
            "Source Text",
            value=current_sample.get("context_text", ""),
            height=260,
            disabled=True,
            key=f"source_context_view_{card_ver}"
        )

        st.markdown("**Target Proposition:**")
        st.info(current_sample.get("target_claim", ""))

    # ==================================================================
    # CENTER COLUMN: nanoGentzen Best Logic Tree
    # ==================================================================
    with center_col:
        st.subheader("🌳 nanoGentzen Logic Tree")

        is_verified = current_sample.get("gentzen_verified", False)
        score = current_sample.get("provability_score", 1.0)
        depth = current_sample.get("tree_depth", 1)
        node_count = current_sample.get("node_count", 1)
        epistemic_status = current_sample.get("epistemic_status", "PROVEN_TRUE" if is_verified else "INTUITIONISTIC_UNDECIDED")
        ch_term = current_sample.get("curry_howard_term", "")
        relevance = current_sample.get("relevance", {})

        if epistemic_status == "PROVEN_TRUE":
            st.success("✅ Saul Kripke Status: PROVEN_TRUE (Closed Derivation)")
        elif epistemic_status == "PROVEN_FALSE":
            st.error("❌ Status: PROVEN_FALSE (the negation of the claim is provable)")
        elif epistemic_status == "BRIDGE_ASSUMED":
            st.warning("🌉 Status: BRIDGE_ASSUMED (the rule premises ⇒ conclusion was added; the proof does not "
                       "show the text supports the claim). Review the reasoning yourself before accepting.")
        elif epistemic_status == "INCONSISTENT_PREMISES":
            st.error("⛔ Status: INCONSISTENT_PREMISES (premises contradict each other; do not use as training data)")
        else:
            st.warning(f"⚠️ Status: {epistemic_status} (no proof of the claim from these premises)")

        ng = current_sample.get("nanogentzen") or {}
        if ng:
            eng = ng.get("engine") or {}
            cert_txt = "certificate checked ✔" if ng.get("certificate_checked") else "certificate not checked"
            st.caption(f"Engine: {eng.get('engine', 'nanoGentzen-v4')} · {eng.get('backend', '')} · {cert_txt}")
            if ng.get("minimal_witnesses"):
                st.caption("Minimal witnesses: " + ", ".join(f"`{w}`" for w in ng["minimal_witnesses"]))
            if ng.get("missing_facts"):
                st.caption("Missing facts that would make it provable: " + ", ".join(f"`{w}`" for w in ng["missing_facts"]))
            if ng.get("conflicting_premises"):
                st.caption("Conflicting premises: " + ", ".join(f"`{w}`" for w in ng["conflicting_premises"]))

        m_t1, m_t2, m_t3, m_t4 = st.columns(4)
        m_t1.metric("Tree Depth", depth)
        m_t2.metric("Nodes", node_count)
        m_t3.metric("Provability", f"{score:.2f}")
        rel_ratio = relevance.get("relevance_ratio", 1.0) if relevance else 1.0
        m_t4.metric("Relevance", f"{rel_ratio * 100:.0f}%")

        if relevance and relevance.get("unused_premises"):
            st.caption(f"🛡️ Distractor Pruning: {len(relevance['unused_premises'])} non-causal premises pruned from reasoning.")

        st.markdown(f"**Sequent:** `{current_sample.get('gentzen_sequent', '')}`")

        # Visual Logic Tree Display
        with st.container(border=True):
            st.markdown(current_sample.get("logic_tree_text", ""))

        if ch_term:
            with st.expander("Curry-Howard Typed λ-Term"):
                st.code(ch_term, language="text")

        if current_sample.get("kripke_countermodel"):
            with st.expander("Saul Kripke Countermodel"):
                st.json(current_sample["kripke_countermodel"])

        if ng.get("certificate"):
            with st.expander("Curry-Howard Certificate (re-checkable with nanogentzen_v4.checker)"):
                st.json(ng["certificate"])

        proof_tree = current_sample.get("proof_tree", None)
        if proof_tree:
            with st.expander("ASCII Proof Tree"):
                st.code(_format_ascii_proof_tree(proof_tree), language="text")

        if st.button("🔄 Re-search Best Logic Tree", width="stretch", key="btn_research_tree"):
            _research_logic_tree(current_sample, architect, cot_generator, selected_model)

    # ==================================================================
    # RIGHT COLUMN: Generated ClearCoT (Built along Logic Tree)
    # ==================================================================
    with right_col:
        st.subheader("✍️ Generated ClearCoT (Q–R–A)")

        q_val = st.text_area(
            "Question",
            value=current_sample.get("question", ""),
            height=80,
            key=f"card_q_{card_ver}"
        )

        r_val = st.text_area(
            "Reasoning `<think>` (Built on nanoGentzen Tree)",
            value=current_sample.get("reasoning", ""),
            height=180,
            key=f"card_r_{card_ver}"
        )

        a_val = st.text_area(
            "Answer (Rich Final Synthesis)",
            value=current_sample.get("answer", ""),
            height=120,
            key=f"card_a_{card_ver}"
        )

        # Sync edits
        current_sample["question"] = q_val
        current_sample["reasoning"] = r_val
        current_sample["answer"] = a_val

        # Token metrics
        r_words = len(r_val.split())
        r_tokens = int(r_words * 1.3)
        gr_info = current_sample.get("grounding", {})
        gr_score = current_sample.get("grounding_score", gr_info.get("grounding_score", 1.0))
        is_gr = current_sample.get("is_grounded", gr_info.get("is_grounded", True))

        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Reasoning Tokens", r_tokens)
        cm2.metric("Clarity Score", current_sample.get("clarity_score", 0.9))
        cm3.metric("Tree Grounding", f"{int(gr_score * 100)}%", delta="Grounded" if is_gr else "Review")

        if gr_info:
            with st.expander("🌳 Logic Tree Grounding Breakdown", expanded=False):
                cov = gr_info.get("covered_nodes", [])
                missing = gr_info.get("missing_nodes", [])
                if cov:
                    st.markdown(f"**Covered Logic Nodes:** {', '.join(f'`{n}`' for n in cov)}")
                if missing:
                    st.caption(f"⚠️ Unmentioned Nodes: {', '.join(f'`{n}`' for n in missing)}")
                st.caption(
                    f"Alignment: {int(gr_info.get('conclusion_alignment', 1.0)*100)}% | "
                    f"Distractor Isolation: {int(gr_info.get('relevance_isolation', 1.0)*100)}% | "
                    f"Epistemic: {int(gr_info.get('epistemic_consistency', 1.0)*100)}%"
                )

        has_reasoning = bool(current_sample.get("reasoning", "").strip())
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔄 Re-generate ClearCoT from Tree", width="stretch", key="btn_regen_cot"):
                _regenerate_cot_only(current_sample, cot_generator, selected_model)
        with b2:
            if st.button("⚖️ Save as DPO Pair", width="stretch", key="btn_save_dpo", disabled=not has_reasoning):
                _save_sample(current_sample, "accepted", db_manager)
                st.toast("Saved as DPO Pair!", icon="✨")

        st.divider()
        st.markdown("**Curation Decisions**")
        dec1, dec2 = st.columns(2)
        with dec1:
            if st.button("✅ Accept Sample", type="primary", width="stretch", key="btn_accept_sample", disabled=not has_reasoning):
                _save_sample(current_sample, "accepted", db_manager)
                st.toast("Sample accepted into dataset!", icon="🎉")
        with dec2:
            if st.button("❌ Reject Sample", width="stretch", key="btn_reject_sample", disabled=not has_reasoning):
                _save_sample(current_sample, "rejected", db_manager)
                st.toast("Sample rejected.", icon="🗑️")


def _build_candidate_sample(
    mode: str,
    chunks: List[Chunk],
    selected_path: Optional[Dict[str, Any]],
    q_gen: QuestionGenerator,
    architect: LogicalArchitect,
    cot_gen: CoTGenerator,
    model: str
):
    substantive_chunks = [c for c in chunks if not is_boilerplate(c.text)] or chunks

    with st.spinner("1/2 Generating Question & Premises..."):
        if "Multi-hop" in mode and selected_path:
            chunk_map = {c.chunk_id: c for c in substantive_chunks}
            exact_cids = selected_path.get("chunk_ids", [])
            exact_supporting = [chunk_map[cid] for cid in exact_cids if cid in chunk_map]

            if exact_supporting:
                supporting = exact_supporting[:3]
            else:
                prov_pages = set(selected_path.get("provenance_pages", []))
                prov_chunks = [c for c in substantive_chunks if c.page_number in prov_pages]
                path_entities = [e for e in selected_path.get("path_nodes", [selected_path.get("source", ""), selected_path.get("target", "")]) if e]

                target_pool = prov_chunks if prov_chunks else substantive_chunks
                scored_chunks = []
                for c in target_pool:
                    score = sum(1 for e in path_entities if any(w.lower() in c.text.lower() for w in e.split() if len(w) > 3))
                    scored_chunks.append((score, c))
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                supporting = [c for _, c in scored_chunks[:3]] or target_pool[:3] or substantive_chunks[:3]

            q_res = q_gen.generate_from_multihop_path(selected_path, supporting, model=model)
        else:
            if not substantive_chunks:
                st.error("No substantive document chunks available. Upload a file in the Library tab.")
                return
            import random
            chosen = random.choice(substantive_chunks)
            q_res = q_gen.generate_from_chunk(chosen, model=model)


    has_precomputed_tree = bool("Multi-hop" in mode and selected_path and selected_path.get("logic_tree_res"))

    premises = q_res.get("key_premises", [q_res.get("question", "")])
    target_claim = q_res.get("target_claim", q_res.get("seed_answer", ""))
    path_nodes = q_res.get("path_nodes", None)

    if has_precomputed_tree:
        tree_res = selected_path["logic_tree_res"]
    else:
        with st.spinner("2/3 nanoGentzen: Constructing Best Logic Tree..."):
            tree_res = architect.generate_best_logic_tree(
                premises=premises,
                conclusion=target_claim,
                path_nodes=path_nodes,
                path_repr=selected_path.get("path_repr") if selected_path else None
            )

    cot_step_lbl = "2/2" if has_precomputed_tree else "3/3"
    with st.spinner(f"{cot_step_lbl} {model}: Synthesizing ClearCoT along Logic Tree..."):
        cot_res = cot_gen.build_cot_from_logic_tree(
            question=q_res["question"],
            target_claim=target_claim,
            logic_tree_text=tree_res["logic_tree_text"],
            context_text=q_res.get("context_text", ""),
            model=model,
            relevance_info=tree_res.get("relevance", None),
            curry_howard_term=tree_res.get("curry_howard_term", ""),
            path_nodes=path_nodes,
            key_premises=premises,
            is_verified=tree_res["is_verified"],
            epistemic_status=tree_res.get("epistemic_status", "PROVEN_TRUE"),
        )

    if cot_res.get("is_fallback", False):
        err = cot_res.get("error")
        if err:
            st.warning(f"⚠️ Model backend warning: `{err[:180]}`. Showing structured fallback ClearCoT.")
        else:
            st.toast("Generated structured fallback ClearCoT (verify backend connection/logs if expecting model output)", icon="ℹ️")
    else:
        st.toast(f"Real ClearCoT generated by {model}!", icon="✨")

    final_q = _sanitize_question(cot_res.get("question", q_res["question"]))
    naive_meta = cot_gen.generate_naive_cot(
        question=final_q,
        target_claim=target_claim,
        context_text=q_res.get("context_text", ""),
        model=model,
        return_meta=True,
    )
    naive_trace = naive_meta["text"]
    with st.spinner("Measuring token savings against a plain step-by-step answer..."):
        baseline = cot_gen.generate_standard_cot(final_q, q_res.get("context_text", ""), model=model)
    measured_saving = CoTGenerator.measure_token_savings(cot_res.get("reasoning", ""), baseline)

    st.session_state.current_sample = {
        "question": _sanitize_question(cot_res.get("question", q_res["question"])),
        "reasoning": cot_res.get("reasoning", ""),
        "answer": cot_res.get("answer", target_claim),
        "naive_cot": naive_trace,
        "target_claim": target_claim,
        "key_premises": premises,
        "path_nodes": path_nodes,
        "provenance": q_res.get("provenance", {}),
        "context_text": q_res.get("context_text", ""),
        "gentzen_verified": tree_res["is_verified"],
        "gentzen_sequent": tree_res["sequent"],
        "proof_tree": tree_res["proof_tree"],
        "logic_tree_text": tree_res["logic_tree_text"],
        "provability_score": tree_res["provability_score"],
        "tree_depth": tree_res.get("depth", 1),
        "node_count": tree_res.get("node_count", 1),
        "clarity_score": cot_res.get("clarity_score", 0.92),
        "curry_howard_term": tree_res.get("curry_howard_term", ""),
        "epistemic_status": tree_res.get("epistemic_status", "PROVEN_TRUE" if tree_res["is_verified"] else "INTUITIONISTIC_UNDECIDED"),
        "kripke_countermodel": tree_res.get("kripke_countermodel", None),
        "relevance": tree_res.get("relevance", {}),
        "nanogentzen": proof_record(tree_res),
        "grounding": cot_res.get("grounding", {}),
        "grounding_score": cot_res.get("grounding_score", 1.0),
        "is_grounded": cot_res.get("is_grounded", tree_res["is_verified"]),
        "tokens_saved_pct": measured_saving if measured_saving is not None else 0,
        "generation": {
            "model": provenance_model_name(model),
            "is_fallback": bool(cot_res.get("is_fallback", False)),
            "is_grounded": bool(cot_res.get("is_grounded", False)),
            "grounding_score": cot_res.get("grounding_score"),
            "logic_basis": tree_res.get("logic_basis"),
            "naive_cot_synthetic": bool(naive_meta.get("synthetic")),
            "token_savings_measured": measured_saving is not None,
            "token_savings_baseline": "standard_cot" if measured_saving is not None else None,
        },
    }
    st.session_state.card_version = st.session_state.get("card_version", 0) + 1


def _research_logic_tree(sample: Dict[str, Any], architect: LogicalArchitect, cot_gen: CoTGenerator, model: str):
    with st.spinner("nanoGentzen: Re-searching Logic Tree with beam exploration..."):
        tree_res = architect.generate_best_logic_tree(
            premises=sample.get("key_premises", [sample.get("target_claim", sample["answer"])]),
            conclusion=sample.get("target_claim", sample["answer"]),
            path_nodes=sample.get("path_nodes", None),
            path_repr=(sample.get("provenance") or {}).get("path_repr") or sample.get("path_repr"),
        )
        sample["gentzen_verified"] = tree_res["is_verified"]
        sample["gentzen_sequent"] = tree_res["sequent"]
        sample["proof_tree"] = tree_res["proof_tree"]
        sample["logic_tree_text"] = tree_res["logic_tree_text"]
        sample["provability_score"] = tree_res["provability_score"]
        sample["tree_depth"] = tree_res.get("depth", 1)
        sample["node_count"] = tree_res.get("node_count", 1)
        sample["curry_howard_term"] = tree_res.get("curry_howard_term", "")
        sample["epistemic_status"] = tree_res.get("epistemic_status", "PROVEN_TRUE" if tree_res["is_verified"] else "INTUITIONISTIC_UNDECIDED")
        sample["kripke_countermodel"] = tree_res.get("kripke_countermodel", None)
        sample["relevance"] = tree_res.get("relevance", {})
        sample["nanogentzen"] = proof_record(tree_res)
        st.session_state.current_sample = sample
        st.rerun()


def _regenerate_cot_only(sample: Dict[str, Any], cot_gen: CoTGenerator, model: str):
    with st.spinner(f"Re-generating ClearCoT with {model} along Logic Tree..."):
        cot_res = cot_gen.build_cot_from_logic_tree(
            question=sample["question"],
            target_claim=sample.get("target_claim", sample["answer"]),
            logic_tree_text=sample.get("logic_tree_text", ""),
            context_text=sample.get("context_text", ""),
            model=model,
            temperature=0.35,
            relevance_info=sample.get("relevance", None),
            curry_howard_term=sample.get("curry_howard_term", ""),
            path_nodes=sample.get("path_nodes"),
            key_premises=sample.get("key_premises"),
            is_verified=sample.get("gentzen_verified", False),
            epistemic_status=sample.get("epistemic_status", "PROVEN_TRUE"),
        )
        sample["reasoning"] = cot_res.get("reasoning", sample["reasoning"])
        sample["answer"] = cot_res.get("answer", sample["answer"])
        if "question" in cot_res and cot_res["question"]:
            sample["question"] = _sanitize_question(cot_res["question"])
        sample["clarity_score"] = cot_res.get("clarity_score", sample.get("clarity_score", 0.92))
        sample["grounding"] = cot_res.get("grounding", {})
        sample["grounding_score"] = cot_res.get("grounding_score", 1.0)
        sample["is_grounded"] = cot_res.get("is_grounded", sample.get("gentzen_verified", False))

        st.session_state.current_sample = sample
        st.session_state.card_version = st.session_state.get("card_version", 0) + 1

        if cot_res.get("is_fallback", False):
            err = cot_res.get("error")
            if err:
                st.session_state["review_notice"] = {
                    "type": "warning",
                    "message": f"⚠️ Model backend warning: `{err[:180]}`. Showing structured fallback ClearCoT."
                }
            else:
                st.session_state["review_notice"] = {
                    "type": "info",
                    "message": "Generated structured fallback ClearCoT."
                }
        else:
            st.session_state["review_notice"] = {
                "type": "toast",
                "message": f"Real ClearCoT re-generated by {model}!"
            }
        st.rerun()


def _save_sample(sample_dict: Dict[str, Any], status: str, db: DatasetManager):
    sample = DatasetSample(
        sample_id="",
        doc_name=sample_dict.get("provenance", {}).get("doc_name", "manual"),
        question=sample_dict["question"],
        reasoning=sample_dict["reasoning"],
        answer=sample_dict["answer"],
        naive_cot=sample_dict.get("naive_cot", ""),
        status=status,
        gentzen_verified=sample_dict.get("gentzen_verified", False),
        gentzen_sequent=sample_dict.get("gentzen_sequent", ""),
        proof_tree_json=json.dumps(sample_dict.get("proof_tree", {})),
        provability_score=sample_dict.get("provability_score", 0.0),
        tokens_saved_pct=int(sample_dict.get("tokens_saved_pct", 0) or 0),
        clarity_score=sample_dict.get("clarity_score", 0.0),
        provenance_json=json.dumps(
            {**sample_dict.get("provenance", {}),
             **({"nanogentzen": sample_dict["nanogentzen"]} if sample_dict.get("nanogentzen") else {}),
             **({"generation": {**sample_dict["generation"], "status_set_by": "reviewer"}} if sample_dict.get("generation") else {})},
            default=str),
        curry_howard_term=sample_dict.get("curry_howard_term", ""),
        epistemic_status=sample_dict.get("epistemic_status", "PROVEN_TRUE" if sample_dict.get("gentzen_verified", False) else "INTUITIONISTIC_UNDECIDED"),
        grounding_score=sample_dict.get("grounding_score", 1.0),
    )
    db.save_sample(sample)


def _format_ascii_proof_tree(node: Dict[str, Any], prefix: str = "", is_last: bool = True) -> str:
    if not node or not isinstance(node, dict):
        return ""
    lines = []
    rule = node.get("rule", "Rule")
    conc = node.get("conclusion") or node.get("sequent", "")
    score = node.get("score", node.get("provability_score", 1.0))

    connector = "└── " if is_last else "├── "
    lines.append(f"{prefix}{connector}[{rule}] {conc} (val: {score})")

    children = node.get("children") or node.get("branches", [])
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            continue
        sub_prefix = prefix + ("    " if is_last else "│   ")
        is_sub_last = (idx == len(children) - 1)
        lines.append(_format_ascii_proof_tree(child, sub_prefix, is_sub_last))

    return "\n".join(lines)


def _render_batch_generator_tab(
    chunks: List[Chunk],
    kg_builder: Optional[Any],
    batch_generator: Optional[BatchDatasetGenerator],
    selected_model: str,
    db_manager: DatasetManager
):
    st.subheader("🚀 Complete Dataset Generator")
    st.caption("Batch process all discovered Knowledge Graph paths and document chunks to construct a complete Q–R–A dataset.")

    if not batch_generator:
        st.error("Batch Generator service is not initialized.")
        return

    # Check available seeds using shared session state
    intra_doc_only = st.session_state.get("intra_doc_only", True)
    path_limit = st.session_state.get("kg_path_limit", 100)

    kg_paths = []
    if kg_builder and kg_builder.graph.number_of_nodes() > 1:
        kg_paths = kg_builder.find_multihop_paths(
            require_entailment=st.session_state.get("cot_require_entailment", True),
            verified_only=st.session_state.get("cot_verified_only", True),
            **_cot_path_options(),
            min_topic_overlap=float(st.session_state.get("kg_min_topic_overlap", 0.05)),
            max_hops=3,
            min_hops=2,
            limit=path_limit,
            intra_document_only=intra_doc_only,
            architect=st.session_state.get("architect"),
            verify_with_gentzen=True,
        )
        if not kg_paths:
            kg_paths = kg_builder.find_multihop_paths(
                require_entailment=st.session_state.get("cot_require_entailment", True),
                verified_only=st.session_state.get("cot_verified_only", True),
                **_cot_path_options(),
                min_topic_overlap=float(st.session_state.get("kg_min_topic_overlap", 0.05)),
                max_hops=3,
                min_hops=1,
                limit=path_limit,
                intra_document_only=intra_doc_only,
                architect=st.session_state.get("architect"),
                verify_with_gentzen=True,
            )

    num_paths = len(kg_paths)
    cross_paths = [p for p in kg_paths if p.get("is_cross_document")]
    intra_paths = [p for p in kg_paths if not p.get("is_cross_document")]
    num_chunks = len(chunks)

    # Seed availability metrics
    sm1, sm2, sm3 = st.columns(3)
    if cross_paths:
        sm1.metric(
            "KG Multi-hop Paths",
            num_paths,
            delta=f"{len(cross_paths)} Cross-Doc",
            delta_color="normal",
            help=f"Discovered {num_paths} paths: {len(cross_paths)} cross-document synthesis, {len(intra_paths)} intra-document"
        )
    else:
        sm1.metric("KG Multi-hop Paths", num_paths, help="Discovered 2-to-3 hop relational concept chains")
    sm2.metric("Document Chunks", num_chunks, help="Available paragraph chunks from uploaded documents")
    sm3.metric("Selected LLM", clean_model_display_name(selected_model), help=f"Active LLM: {selected_model}")

    if num_paths == 0 and num_chunks == 0:
        st.warning("No Knowledge Graph paths or document chunks available. Ingest documents in the **Library tab** and build the graph in the **Knowledge Graph tab** first.")
        return

    st.markdown("#### Configure Batch Run")
    cfg_col1, cfg_col2 = st.columns([2, 1])

    with cfg_col1:
        source_options = []
        if num_paths > 0:
            source_options.append(f"All Multi-Hop KG Paths ({num_paths} paths)")
            if len(cross_paths) > 0 and len(intra_paths) > 0:
                source_options.append(f"🌐 Cross-Document KG Paths Only ({len(cross_paths)} paths)")
                source_options.append(f"📄 Intra-Document KG Paths Only ({len(intra_paths)} paths)")
        if num_chunks > 0:
            source_options.append(f"All Document Chunks ({num_chunks} chunks)")
        if num_paths > 0 and num_chunks > 0:
            source_options.append(f"Combined: All KG Paths + Chunks ({num_paths + num_chunks} total)")

        chosen_source = st.radio("Batch Source Seed", source_options, index=0)

        target_kg_paths = kg_paths
        if "Cross-Document KG Paths Only" in chosen_source:
            target_kg_paths = cross_paths
            source_mode = "kg_paths"
            total_available = len(cross_paths)
        elif "Intra-Document KG Paths Only" in chosen_source:
            target_kg_paths = intra_paths
            source_mode = "kg_paths"
            total_available = len(intra_paths)
        elif "KG Paths" in chosen_source and "Combined" not in chosen_source:
            target_kg_paths = kg_paths
            source_mode = "kg_paths"
            total_available = num_paths
        elif "Chunks" in chosen_source and "Combined" not in chosen_source:
            target_kg_paths = []
            source_mode = "chunks"
            total_available = num_chunks
        else:
            target_kg_paths = kg_paths
            source_mode = "combined"
            total_available = num_paths + num_chunks

    with cfg_col2:
        gen_all = st.checkbox("Generate All Available", value=True)
        if gen_all:
            max_limit = total_available
            st.info(f"Target count: **{max_limit}** samples")
        else:
            max_limit = int_slider("Sample Limit", 1, total_available, min(10, total_available))

        target_status = st.selectbox("Initial Dataset Status", ["accepted", "verified", "draft"], index=0)

        default_conc = 1
        client_url = ""
        if hasattr(batch_generator, 'cot_generator') and hasattr(batch_generator.cot_generator, 'llm_client'):
            client_url = getattr(batch_generator.cot_generator.llm_client, 'base_url', '')
            if ':8000' in client_url or 'vllm' in client_url.lower():
                default_conc = 8

        batch_concurrency = st.slider(
            "⚡ Generation Concurrency",
            min_value=1,
            max_value=16,
            value=default_conc,
            help="Parallel concurrent synthesis workers. Recommended: 4–8 for vLLM (continuous batching); keep at 1 for LM Studio / Ollama single GPU."
        )

    st.divider()

    # Run controls
    col_btn, col_stop = st.columns([1, 1])
    with col_btn:
        start_btn = st.button("▶️ Launch Complete Dataset Generation", type="primary", width="stretch")
    with col_stop:
        if st.button("⏹️ Stop Generation", width="stretch"):
            st.session_state.stop_batch = True
            if batch_generator is not None:
                batch_generator.request_stop()   # thread-safe: reaches the workers of the interrupted run
            st.warning("Stop requested: queued seeds are cancelled; samples already being generated finish and are saved.")

    if start_btn:
        st.session_state.stop_batch = False
        pbar = st.progress(0.0)
        status_ticker = st.empty()
        metrics_container = st.empty()
        preview_table = st.empty()

        batch_results_display = []

        def progress_cb(curr, total, sample_dict, msg):
            pct = min(1.0, curr / max(1, total))
            pbar.progress(pct)
            status_ticker.markdown(f"⏳ **[{curr}/{total}]** {msg}")
            if sample_dict:
                batch_results_display.append(sample_dict)
                with metrics_container.container():
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Generated", f"{len(batch_results_display)}/{total}")
                    verified_so_far = sum(1 for s in batch_results_display if s.get("verified"))
                    mc2.metric("Formally Verified", f"{int((verified_so_far/len(batch_results_display))*100)}%")
                    avg_c = round(sum(s.get("clarity_score", 0.9) for s in batch_results_display) / len(batch_results_display), 2)
                    mc3.metric("Avg Clarity", avg_c)

                # Show table of latest generated
                preview_table.dataframe(
                    [
                        {
                            "ID": s.get("sample_id", ""),
                            "Question": s.get("question", "")[:60] + "...",
                            "Answer": s.get("answer", "")[:50] + "...",
                            "Verified": "✅ Yes" if s.get("verified") else "⚠️ Heuristic",
                            "Clarity": s.get("clarity_score", 0.0),
                        }
                        for s in batch_results_display[-6:]
                    ],
                    width="stretch"
                )

        with st.spinner("Batch generation active. Processing Knowledge Graph seeds..."):
            results = batch_generator.generate_batch(
                source_mode=source_mode,
                kg_paths=target_kg_paths,
                chunks=chunks,
                model=selected_model,
                max_samples=max_limit,
                concurrency=batch_concurrency,
                progress_callback=progress_cb,
                stop_signal=lambda: st.session_state.get("stop_batch", False),
                status=target_status
            )

        # Synchronize Parquet, JSONL & DPO preference exports immediately (matching tab_jev behavior)
        try:
            db_manager.export_jsonl(str(EXPORTS_DIR / "clearcot_sft_dataset.jsonl"), only_accepted=True, only_verified=True)
            db_manager.export_dpo_pairs(str(EXPORTS_DIR / "clearcot_dpo_pairs.jsonl"), only_accepted=True, skip_synthetic_rejected=True, only_verified=True)
            db_manager.export_parquet(str(EXPORTS_DIR / "clearcot_dataset.parquet"), only_accepted=True, only_verified=True)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Auto-sync ClearCoT export failed: {e}")

        st.success(
            f"🎉 Completed! Generated **{results['generated_count']}** new samples "
            f"({results['verified_count']} formally verified, {int(results['verification_rate']*100)}% pass rate, "
            f"average clarity score: {results['avg_clarity']}). All exports (JSONL, DPO, Parquet) updated."
        )
        st.info("Head to the **📦 Dataset Export & Stats** tab to inspect the full dataset and download your JSONL / DPO files.")


def _cot_path_options() -> Dict[str, Any]:
    """Taxonomy-hop policy and hub cap for ClearCoT path discovery (Review tab controls)."""
    cap = int(st.session_state.get("cot_max_paths_per_hub", 8) or 0)
    return {"taxonomy_policy": st.session_state.get("cot_taxonomy_policy", "any"),
            "max_paths_per_hub": cap or None}


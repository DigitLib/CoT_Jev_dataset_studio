"""
Jev Decision Datasets Workspace Tab for ClearCoT Studio (gui/tab_jev.py).
Generates, formally verifies, and curates Jev Decision Scenarios (state JSON + criteria + gold distribution)
from the exact same document source material used for ClearCoT.
Supports exports to canonical `typed-decisions` and `S1MB` Parquet schemas.
"""

import streamlit as st
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.chunker import Chunk, is_boilerplate
from core.jev_generator import JevScenarioGenerator
from core.jev_dataset_manager import JevDatasetManager, JevSample
from config import EXPORTS_DIR
from gui._paths import get_jev_paths, graph_signature

try:
    from llm_backend.client import clean_model_display_name
except ImportError:
    def clean_model_display_name(m: Any) -> str:
        if not m or not isinstance(m, str):
            return "Default Model"
        s = m.strip().rstrip("/\\")
        if "/" in s or "\\" in s:
            parts = [p for p in s.replace("\\", "/").split("/") if p]
            if parts:
                last = parts[-1]
                if last.lower() in ("v1", "models", "checkpoint") and len(parts) > 1:
                    return parts[-2]
                return last
        return s

logger = logging.getLogger(__name__)


def render_jev_tab(
    chunks: List[Chunk],
    selected_path: Optional[Dict[str, Any]],
    jev_generator: JevScenarioGenerator,
    jev_manager: JevDatasetManager,
    selected_model: str,
    kg_builder: Optional[Any] = None,
    available_models: Optional[List[str]] = None,
):
    st.header("🎯 Jev Decision Dataset Generator")
    st.caption("Transform document knowledge into formally verified Jev Typed Decision Scenarios (`typed-decisions` & `S1MB` Parquet schemas).")

    subtab_builder, subtab_batch, subtab_export = st.tabs([
        "🔬 Single Scenario Builder",
        "🚀 Batch Jev Generator",
        "📦 Jev Dataset Analytics & Exports",
    ])

    kg_builder = kg_builder or st.session_state.get("kg_builder")
    available_kg_paths: List[Dict[str, Any]] = []
    paths_source = "no knowledge graph"
    if graph_signature(kg_builder) is not None:
        pc1, pc2 = st.columns([2, 2])
        with pc1:
            rediscover = st.checkbox(
                "🔁 Re-discover relational paths for Jev",
                value=st.session_state.get("jev_rediscover_paths", False),
                key="jev_rediscover_paths",
                help="Off: use exactly the paths found in the Knowledge Graph tab (same limit, document, "
                     "entailment and verification filters). On: run a separate relational search for Jev "
                     "(Jev logic comes from the scenario rules, so paths need not be kernel-verified).")
        with pc2:
            if rediscover:
                st.checkbox("🧮 Only entailment chains (Jev)",
                            value=st.session_state.get("jev_require_entailment", False),
                            key="jev_require_entailment")
        available_kg_paths, paths_source = get_jev_paths(st.session_state, kg_builder, rediscover=rediscover)
        st.caption(f"🔗 **{len(available_kg_paths)}** KG paths ({paths_source})")
    if selected_path is not None and available_kg_paths and \
            selected_path.get("path_repr") not in {p.get("path_repr") for p in available_kg_paths}:
        selected_path = None  # selected in another tab or for an older graph; not one of the offered paths

    has_kg_paths = len(available_kg_paths) > 0 or (selected_path is not None)
    if not selected_path and available_kg_paths:
        selected_path = available_kg_paths[0]
        st.session_state.selected_path = selected_path

    # ==================================================================
    # SUBTAB 1: Single Scenario Builder & Live Verifier
    # ==================================================================
    with subtab_builder:
        st.markdown("##### *Formulate and formally verify a Jev decision scenario from document evidence.*")

        jev_seed_options = ["Paragraph Seed", "Multi-hop KG Path"] if has_kg_paths else ["Paragraph Seed"]
        prev_jev_mode = st.session_state.get("jev_seed_mode", "Paragraph Seed")
        default_jev_idx = jev_seed_options.index(prev_jev_mode) if prev_jev_mode in jev_seed_options else 0

        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            seed_mode = st.radio(
                "Evidence Source",
                jev_seed_options,
                index=default_jev_idx,
                horizontal=True,
                key="jev_seed_mode"
            )
            if seed_mode == "Multi-hop KG Path" and available_kg_paths:
                path_labels = [
                    f"Path #{i+1} ({p['hops']} hops): {p['path_repr']}"
                    for i, p in enumerate(available_kg_paths)
                ]
                current_p_idx = 0
                if selected_path:
                    for i, p in enumerate(available_kg_paths):
                        if p.get("path_repr") == selected_path.get("path_repr"):
                            current_p_idx = i
                            break
                chosen_p_idx = st.selectbox(
                    "Selected KG Path",
                    range(len(available_kg_paths)),
                    index=current_p_idx,
                    format_func=lambda i: path_labels[i],
                    key="sel_jev_kg_path"
                )
                selected_path = available_kg_paths[chosen_p_idx]
                st.session_state.selected_path = selected_path
            elif not has_kg_paths:
                st.caption("💡 Build a Knowledge Graph in Tab 2 to enable Multi-hop KG paths.")
        with col2:
            # Dynamically infer workflow domain from selected path or document evidence
            inferred_wf = jev_generator.infer_workflow(
                doc_name=selected_path.get("doc_name", "") if selected_path else (chunks[0].doc_name if chunks else ""),
                path_nodes=selected_path.get("path_nodes", []) if selected_path else None,
                context_text=selected_path.get("composite_claim", "") if selected_path else (chunks[0].text if chunks else "")
            )
            current_path_key = str(selected_path.get("path_repr")) if selected_path else str(seed_mode)
            if st.session_state.get("last_jev_path_key") != current_path_key:
                st.session_state["last_jev_path_key"] = current_path_key
                st.session_state["jev_workflow"] = inferred_wf

            workflow_name = st.text_input(
                "Workflow Domain",
                value=st.session_state.get("jev_workflow", inferred_wf),
                help="Domain workflow slug (e.g. educational_autonomy, scientific_theories, cultural_policy, security_incidents)",
                key="input_jev_workflow"
            )
            st.session_state["jev_workflow"] = workflow_name

        with col3:
            st.write("")
            st.write("")
            if st.button("⚡ Generate Jev Decision Scenario", type="primary", width="stretch", key="btn_gen_jev"):
                _build_candidate_jev_scenario(
                    mode=seed_mode,
                    chunks=chunks,
                    selected_path=selected_path,
                    generator=jev_generator,
                    workflow=workflow_name,
                    model=selected_model,
                )
            st.caption(f"🤖 Model: `{clean_model_display_name(selected_model)}`", help=f"Active Generation Model: {selected_model}")

        current_jev = st.session_state.get("current_jev_scenario", None)

        if not current_jev:
            st.info("No active scenario. Select an evidence source and click '⚡ Generate Jev Decision Scenario' above.")
        else:
            st.divider()
            _render_single_jev_workspace(current_jev, jev_generator, jev_manager, selected_model)

    # ==================================================================
    # SUBTAB 2: Batch Complete Generator
    # ==================================================================
    with subtab_batch:
        _render_batch_jev_generator(
            chunks=chunks,
            generator=jev_generator,
            manager=jev_manager,
            model=selected_model,
            kg_builder=kg_builder,
            available_kg_paths=available_kg_paths,
            available_models=available_models,
        )

    # ==================================================================
    # SUBTAB 3: Exports & Analytics
    # ==================================================================
    with subtab_export:
        _render_jev_export_tab(jev_manager)


def _render_single_jev_workspace(
    scenario: Dict[str, Any],
    generator: JevScenarioGenerator,
    manager: JevDatasetManager,
    selected_model: str,
):
    col_left, col_center, col_right = st.columns([1.1, 1.3, 1.2], gap="medium")

    # ------------------------------------------------------------------
    # LEFT COLUMN: Evidence & State JSON
    # ------------------------------------------------------------------
    with col_left:
        st.subheader("📖 Evidence & State JSON")
        prov = scenario.get("provenance", {})
        st.markdown(f"**Document:** `{prov.get('doc_name', 'Unknown')}` | **Pages:** `{prov.get('pages', [1])}`")
        if "path_repr" in prov:
            st.markdown(f"**KG Path:** `{prov.get('path_repr')}`")

        with st.expander("Grounded Source Evidence", expanded=False):
            st.text_area("Evidence Text", value=scenario.get("context_text", ""), height=180, disabled=True)

        st.markdown("**Structured `state` (JSON):**")
        st.json(scenario.get("state", {}), expanded=True)

    # ------------------------------------------------------------------
    # CENTER COLUMN: Decision Questions & nanoGentzen Logic Tree
    # ------------------------------------------------------------------
    with col_center:
        st.subheader("🌳 Questions & nanoGentzen Proof")
        ep_status = scenario.get("epistemic_status", "PROVEN_TRUE")
        is_verified = scenario.get("gentzen_verified", False)

        if ep_status == "PROVEN_TRUE":
            st.success("✅ Saul Kripke Status: PROVEN_TRUE (Closed Derivation)")
        elif ep_status == "PROVEN_FALSE":
            st.error("❌ Saul Kripke Status: PROVEN_FALSE (Refuted Option)")
        elif ep_status == "NO_OPTION_PROVABLE":
            st.success("✅ Status: NO_OPTION_PROVABLE (certified undecided: every option has a rule and the "
                       "complete search proves none of them)")
        elif ep_status == "PROVEN_UNDER_ASSUMPTIONS":
            st.warning("🧩 Status: PROVEN_UNDER_ASSUMPTIONS (provable only after adding facts the scenario does "
                       "not state; not verified and not exported as verified)")
        elif ep_status in ("INCONSISTENT_PREMISES", "TARGET_REFUTED_BY_PROOF", "MULTIPLE_PROVEN"):
            st.error(f"⛔ Status: {ep_status} (the facts and rules do not single out the intended decision)")
        else:
            st.warning(f"⚠️ Status: {ep_status} (no proof of the intended decision)")

        ng = (scenario.get("provenance") or {}).get("nanogentzen") or {}
        if ng:
            eng = ng.get("engine") or {}
            st.caption(f"Engine: {eng.get('engine', 'nanoGentzen-v4')} · {eng.get('backend', '')}"
                       + (" · offline heuristic scenario" if ng.get("heuristic_fallback") else ""))
            comp = ng.get("state_completion") or {}
            if comp.get("applied"):
                st.info("State completed with the minimal facts needed for the target decision: "
                        + ", ".join(f"`{k}={v}`" for k, v in comp.get("added_facts", {}).items()))
            elif comp.get("reason"):
                st.caption(f"State completion not applied: {comp['reason']}")
            if ng.get("rejected_rules"):
                with st.expander(f"⚠️ {len(ng['rejected_rules'])} rejected policy rule(s)"):
                    for rr in ng["rejected_rules"]:
                        st.markdown(f"- `{rr.get('rule')}` — {rr.get('error')}")

        st.markdown(f"**Sequent:** `{scenario.get('gentzen_sequent', '')}`")

        ch_term = scenario.get("curry_howard_term", "")
        if ch_term:
            with st.expander("Curry-Howard Typed λ-Term"):
                st.code(ch_term, language="text")

        questions = scenario.get("questions", {})
        for q_id, q_data in questions.items():
            st.markdown(f"**Question: `{q_id}`**")
            st.markdown(f"*{q_data.get('instructions', '')}*")
            criteria = q_data.get("criteria", {})
            for opt_key, crit_desc in criteria.items():
                st.markdown(f"- **`{opt_key}`**: {crit_desc}")

        rules = scenario.get("policy_rules", [])
        if rules:
            with st.expander("Grounding Policy Rules"):
                for r in rules:
                    st.code(r, language="text")

    # ------------------------------------------------------------------
    # RIGHT COLUMN: Gold Decisions, Probabilities & Factors
    # ------------------------------------------------------------------
    with col_right:
        st.subheader("🎯 Verified Gold Decision")
        gold = scenario.get("gold", {})

        for q_id, g_info in gold.items():
            win_label = g_info.get("label", "")
            conf = g_info.get("confidence", 1.0)
            conf = 0.0 if conf is None else float(conf)
            st.markdown(f"### Winning Choice: `{win_label}`")
            st.metric(f"Calibrated Confidence (`{q_id}`)", f"{conf * 100:.1f}%")
            if g_info.get("verification"):
                st.caption(f"Verification: `{g_info['verification']}`"
                           + (" · certificate checked ✔" if g_info.get("certificate_checked") else ""))
            rel = g_info.get("relevance") or {}
            if rel.get("minimal_witnesses"):
                st.caption("Minimal witnesses: " + ", ".join(f"`{w}`" for w in rel["minimal_witnesses"]))
            if g_info.get("missing_facts"):
                st.caption(f"Missing facts: `{g_info['missing_facts']}`")

            probs = g_info.get("probabilities", {})
            if probs:
                st.markdown("**Probability Distribution:**")
                for opt, p_val in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    st.progress(float(p_val), text=f"`{opt}`: {p_val * 100:.1f}%")

        st.markdown("**Causal `factors` (Relevance Logic):**")
        st.json(scenario.get("factors", {}), expanded=True)

        st.divider()
        add_cf = st.checkbox(
            "Also save counterfactual variants", value=True, key="jev_single_cf",
            help="Remove or negate one required fact at a time and let the kernel re-decide. Variants with a "
                 "different certified label (e.g. 'undecided') balance the dataset; they share the parent's split.")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("✅ Accept Jev Record", type="primary", width="stretch", key="btn_accept_jev"):
                parent_id = _save_jev_scenario(scenario, status="accepted", manager=manager)
                n_cf = _save_counterfactuals(generator, scenario, parent_id, manager) if add_cf else 0
                st.toast("Accepted into Jev Dataset!" + (f" (+{n_cf} counterfactual variants)" if n_cf else ""), icon="🎉")
        with b_col2:
            if st.button("❌ Reject Record", width="stretch", key="btn_reject_jev"):
                _save_jev_scenario(scenario, status="rejected", manager=manager)
                st.toast("Scenario rejected.", icon="🗑️")


def _build_candidate_jev_scenario(
    mode: str,
    chunks: List[Chunk],
    selected_path: Optional[Dict[str, Any]],
    generator: JevScenarioGenerator,
    workflow: str,
    model: str,
):
    substantive_chunks = [c for c in chunks if not is_boilerplate(c.text)] or chunks
    if not substantive_chunks and not selected_path:
        st.error("No substantive chunks available. Please upload a document in the Library tab first.")
        return

    with st.spinner("1/2 nanoGentzen: Extracting Structured State & Formulating Policy Questions..."):
        if "Multi-hop" in mode and selected_path:
            chunk_map = {c.chunk_id: c for c in substantive_chunks}
            exact_cids = selected_path.get("chunk_ids", [])
            supporting = [chunk_map[cid] for cid in exact_cids if cid in chunk_map] or substantive_chunks[:3]
            scenario = generator.generate_from_multihop_path(
                path_info=selected_path,
                supporting_chunks=supporting,
                workflow=workflow,
                model=model,
            )
        else:
            import random
            chosen = random.choice(substantive_chunks)
            scenario = generator.generate_from_chunk(
                chunk=chosen,
                workflow=workflow,
                model=model,
            )

    st.session_state.current_jev_scenario = scenario
    st.toast("Jev scenario generated and formally verified!", icon="✨")
    st.rerun()


def _save_jev_scenario(scenario: Dict[str, Any], status: str, manager: JevDatasetManager):
    return manager.save_scenario(scenario, status)


def _save_counterfactuals(generator: JevScenarioGenerator, scenario: Dict[str, Any], parent_id: str,
                          manager: JevDatasetManager, max_variants: int = 2) -> int:
    """Save kernel-certified counterfactual variants of an accepted scenario, linked to the parent."""
    try:
        variants = generator.make_counterfactuals(scenario, max_variants=max_variants)
    except Exception as e:
        logger.warning(f"Counterfactual generation failed: {e}")
        return 0
    for v in variants:
        v["provenance"]["counterfactual"]["of"] = parent_id
        _save_jev_scenario(v, status="accepted" if v.get("gentzen_verified") else "draft", manager=manager)
    return len(variants)


def _render_batch_jev_generator(
    chunks: List[Chunk],
    generator: JevScenarioGenerator,
    manager: JevDatasetManager,
    model: str,
    kg_builder: Optional[Any] = None,
    available_kg_paths: Optional[List[Dict[str, Any]]] = None,
    available_models: Optional[List[str]] = None,
):
    st.subheader("🚀 Complete Jev Batch Generator")
    st.caption("Generate formally verified Jev decision records across Multi-Hop KG reasoning chains or substantive document chunks.")

    substantive = [c for c in chunks if not is_boilerplate(c.text)] or chunks
    # Same list as the Single Scenario Builder (gui/_paths.py); no separate search here
    paths = available_kg_paths or []

    num_paths = len(paths)
    num_chunks = len(substantive)

    active_model_str = (
        st.session_state.get("batch_jev_model_select")
        or st.session_state.get("batch_jev_model_input")
        or model
    )
    clean_model = clean_model_display_name(active_model_str)

    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("Available KG Paths", num_paths,
               help="The KG paths offered above: by default exactly those discovered in the Knowledge Graph tab "
                    "(its limit and filters), or the Jev re-discovery when that option is on.")
    sm2.metric("Substantive Chunks", num_chunks, help="Substantive document paragraph chunks")
    sm3.metric("Selected Model", clean_model, help=f"Active Generation Model: {active_model_str}")

    if num_paths == 0 and num_chunks == 0:
        st.warning("⚠️ No document chunks or Knowledge Graph paths available. Ingest documents in the Library tab first.")
        return

    st.markdown("##### 1. Select Batch Evidence Source")
    source_options = []
    if num_paths > 0:
        source_options.append(f"Multi-hop KG Paths ({num_paths} paths)")
    if num_chunks > 0:
        source_options.append(f"Substantive Document Chunks ({num_chunks} chunks)")
    if num_paths > 0 and num_chunks > 0:
        source_options.append(f"Combined ({num_paths + num_chunks} total)")

    prev_source = st.session_state.get("batch_jev_source_mode")
    default_src_idx = 0
    if prev_source:
        for idx, opt in enumerate(source_options):
            if prev_source in opt:
                default_src_idx = idx
                break

    chosen_source = st.radio(
        "Batch Evidence Source",
        source_options,
        index=default_src_idx,
        horizontal=True,
        key="radio_batch_jev_source",
    )
    is_kg_mode = "KG Paths" in chosen_source and "Combined" not in chosen_source
    is_chunk_mode = "Chunks" in chosen_source and "Combined" not in chosen_source
    is_combined_mode = "Combined" in chosen_source

    total_available = num_paths if is_kg_mode else (num_chunks if is_chunk_mode else (num_paths + num_chunks))

    st.markdown("##### 2. Workflow, Model & Sample Limits")
    b_col1, b_col2, b_col3 = st.columns([1.5, 1.5, 1])
    with b_col1:
        if is_kg_mode and paths:
            default_batch_wf = generator.infer_workflow(
                doc_name=paths[0].get("doc_name", ""),
                path_nodes=paths[0].get("path_nodes", []),
                context_text=paths[0].get("composite_claim", ""),
            )
        elif substantive:
            default_batch_wf = generator.infer_workflow(
                doc_name=substantive[0].doc_name,
                context_text=substantive[0].text,
            )
        else:
            default_batch_wf = "auto"

        batch_wf = st.text_input(
            "Batch Workflow Domain",
            value=default_batch_wf,
            help="Domain slug (e.g. educational_autonomy, scientific_theories, cultural_policy, security_incidents, or 'auto')",
            key="batch_jev_wf",
        )
    with b_col2:
        model_options = list(available_models) if available_models else ([model] if model else [])
        if model and model not in model_options:
            model_options.insert(0, model)

        if len(model_options) > 1:
            def_m_idx = model_options.index(model) if model in model_options else 0
            batch_model = st.selectbox(
                "Generation Model",
                options=model_options,
                index=def_m_idx,
                format_func=clean_model_display_name,
                help=f"Active Generation Model: {model}",
                key="batch_jev_model_select",
            )
        else:
            batch_model = st.text_input(
                "Generation Model",
                value=model,
                help=f"Model identifier or path for verbalizing scenarios: {model}",
                key="batch_jev_model_input",
            )
    with b_col3:
        max_samples = st.number_input(
            "Max Seeds to Process",
            min_value=1,
            max_value=max(1, total_available),
            value=min(50, max(1, total_available)),
            key="batch_jev_max_samples",
            help="Number of seed paths/chunks to process. When 'Add counterfactual variants' is enabled, "
                 "additional balanced contrastive records are generated for each verified seed."
        )

    batch_cf = st.checkbox("Add counterfactual variants (balanced labels)", value=True, key="batch_jev_cf",
                           help="For every verified scenario, up to 4 variants with one required fact removed or "
                                "negated, re-decided by the kernel.")
    if st.button("⚡ Start Batch Jev Generation", type="primary", width="stretch", key="btn_start_batch_jev"):
        prog = st.progress(0, text="Starting batch Jev generation...")
        generated_count = accepted_count = cf_count = 0
        vstats0 = dict(getattr(generator, "validation_stats", {}))
        chunk_map = {c.chunk_id: c for c in substantive}
        active_gen_model = batch_model or model

        items_to_process = []
        if is_kg_mode or is_combined_mode:
            items_to_process.extend([("kg", p) for p in paths])
        if is_chunk_mode or is_combined_mode:
            items_to_process.extend([("chunk", c) for c in substantive])

        targets = items_to_process[:max_samples]
        total_targets = len(targets)

        for i, (item_type, item_data) in enumerate(targets):
            if item_type == "kg":
                exact_cids = item_data.get("chunk_ids", [])
                supporting = [chunk_map[cid] for cid in exact_cids if cid in chunk_map] or substantive[:3]
                wf = batch_wf if batch_wf and batch_wf != "auto" else generator.infer_workflow(
                    doc_name=item_data.get("doc_name", ""),
                    path_nodes=item_data.get("path_nodes", []),
                    context_text=item_data.get("composite_claim", ""),
                )
                scenario = generator.generate_from_multihop_path(
                    path_info=item_data,
                    supporting_chunks=supporting,
                    workflow=wf,
                    model=active_gen_model,
                )
            else:
                wf = batch_wf if batch_wf and batch_wf != "auto" else generator.infer_workflow(
                    doc_name=item_data.doc_name,
                    path_nodes=[item_data.section_title] if item_data.section_title else None,
                    context_text=item_data.text,
                )
                scenario = generator.generate_from_chunk(
                    chunk=item_data,
                    workflow=wf,
                    model=active_gen_model,
                )

            # accepted only with a kernel-certified label; everything else waits for review
            verified = bool(scenario.get("gentzen_verified"))
            parent_id = _save_jev_scenario(scenario, status="accepted" if verified else "draft", manager=manager)
            generated_count += 1
            if verified:
                accepted_count += 1
                if batch_cf:
                    cf_count += _save_counterfactuals(generator, scenario, parent_id, manager)
            prog.progress(
                (i + 1) / total_targets,
                text=f"Processed seed {i + 1}/{total_targets} ({generated_count} base + {cf_count} counterfactual variants)..."
            )

        # Synchronize Parquet & JSONL export files immediately
        try:
            manager.export_typed_decisions_parquet(str(EXPORTS_DIR / "typed_decisions_dataset.parquet"), column_mode="positional", only_verified=True)
            manager.export_open_jev_parquet(str(EXPORTS_DIR / "open_jev_s1mb_dataset.parquet"), only_verified=True)
            manager.export_jev_jsonl(str(EXPORTS_DIR / "jev_decisions_dataset.jsonl"), only_verified=True)
        except Exception as e:
            logger.warning(f"Auto-sync export failed: {e}")

        total_saved = generated_count + cf_count
        st.success(f"Batch generation complete: {generated_count} base scenarios ({accepted_count} kernel-verified, "
                   f"{generated_count - accepted_count} draft) + {cf_count} counterfactual variants "
                   f"= **{total_saved} total records** (all synchronized to `data/exports`).")
        vstats = {k: v - vstats0.get(k, 0) for k, v in getattr(generator, "validation_stats", {}).items()}
        if any(vstats.values()):
            st.info(f"Scenario validation: {vstats.get('retried', 0)} regenerated after a rule problem "
                    f"({vstats.get('fixed_by_retry', 0)} fixed), {vstats.get('rejected', 0)} rejected as "
                    "SCENARIO_REJECTED (copied example rule, contradictory rules or no rules).")
        st.rerun()


def _render_jev_export_tab(manager: JevDatasetManager):
    st.subheader("📦 Jev Dataset Analytics & File Exports")
    stats = manager.get_stats()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Jev Samples", stats["total_samples"])
    m2.metric("Formally Verified", stats["verified_count"], delta=f"{int(stats['verification_rate'] * 100)}% pass")
    m3.metric("Unique Workflows", stats["unique_workflows"])

    st.divider()

    all_samples = manager.get_samples()
    workflows = ["All Workflows"] + sorted(list(set(s.workflow for s in all_samples if s.workflow)))

    f1, f2 = st.columns(2)
    with f1:
        sel_wf = st.selectbox("Workflow Filter", workflows, key="jev_filter_wf")
    with f2:
        sel_status = st.selectbox("Status", ["All", "accepted", "draft", "rejected"], key="jev_filter_stat")

    samples = manager.get_samples(
        status=None if sel_status == "All" else sel_status,
        workflow=sel_wf,
    )

    if samples:
        table_rows = []
        for s in samples:
            g_dict = json.loads(s.gold_json) if s.gold_json else {}
            first_q = list(g_dict.keys())[0] if g_dict else ""
            winning_lbl = g_dict.get(first_q, {}).get("label", s.target_decision)
            conf = g_dict.get(first_q, {}).get("confidence", 1.0)

            table_rows.append({
                "ID": s.sample_id,
                "Document": s.doc_name,
                "Workflow": s.workflow,
                "Question": first_q,
                "Gold Decision": winning_lbl,
                "Confidence": f"{conf * 100:.0f}%",
                "Kripke Status": s.epistemic_status,
                "Verified": "✅ Yes" if s.gentzen_verified else "❌ No",
            })

        st.dataframe(table_rows, width="stretch", hide_index=True)

        st.divider()
        st.subheader("Generate & Download Jev Datasets")
        o1, o2, o3 = st.columns(3)
        with o1:
            split_strategy = st.selectbox(
                "Split by", ["auto", "document", "path", "family", "stored"], key="jev_split_strategy",
                help="Groups never cross splits (80/10/10). auto: by document if there are at least 3, else by KG "
                     "path/chunk, else by scenario family. A scenario and its counterfactuals are always together.")
        with o2:
            column_mode = st.selectbox(
                "typed-decisions columns", ["positional", "question_id"], key="jev_column_mode",
                help="positional: q1__label, q1__id ... instead of one column set per LLM-chosen question id.")
        with o3:
            only_verified = st.checkbox("Only kernel-verified labels", value=True, key="jev_only_verified")
        n_qids = len({q for s in samples for q in (json.loads(s.gold_json) if s.gold_json else {})})
        if column_mode == "question_id" and n_qids > 10:
            st.caption(f"⚠️ {n_qids} distinct question ids: question_id columns will be sparse; consider positional.")
        exp_kw = {"only_accepted": (sel_status == "accepted"), "split_strategy": split_strategy,
                  "only_verified": only_verified}

        exp1, exp2, exp3 = st.columns(3)

        with exp1:
            td_path = EXPORTS_DIR / "typed_decisions_dataset.parquet"
            if st.button("📊 Generate typed-decisions Parquet", type="primary", width="stretch"):
                c = manager.export_typed_decisions_parquet(str(td_path), column_mode=column_mode, **exp_kw)
                _show_split(manager)
                st.success(f"Exported {c} records to {td_path.name}")

            if td_path.exists():
                with open(td_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download typed_decisions.parquet",
                        data=f.read(),
                        file_name="typed_decisions_dataset.parquet",
                        mime="application/octet-stream",
                        width="stretch"
                    )

        with exp2:
            s1mb_path = EXPORTS_DIR / "open_jev_s1mb_dataset.parquet"
            if st.button("📊 Generate Open-Jev (S1MB) Parquet", width="stretch"):
                c = manager.export_open_jev_parquet(str(s1mb_path), **exp_kw)
                _show_split(manager)
                st.success(f"Exported {c} records to {s1mb_path.name}")

            if s1mb_path.exists():
                with open(s1mb_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download open_jev_s1mb.parquet",
                        data=f.read(),
                        file_name="open_jev_s1mb_dataset.parquet",
                        mime="application/octet-stream",
                        width="stretch"
                    )

        with exp3:
            jsonl_path = EXPORTS_DIR / "jev_decisions_dataset.jsonl"
            if st.button("💾 Generate Jev JSONL", width="stretch"):
                c = manager.export_jev_jsonl(str(jsonl_path), **exp_kw)
                _show_split(manager)
                st.success(f"Exported {c} records to {jsonl_path.name}")

            if jsonl_path.exists():
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="⬇️ Download jev_dataset.jsonl",
                        data=f.read(),
                        file_name="jev_decisions_dataset.jsonl",
                        mime="application/jsonl",
                        width="stretch"
                    )

        st.divider()
        st.markdown("##### 🧹 Dataset Maintenance & Legacy Cleanup")
        pur1, pur2 = st.columns(2)
        with pur1:
            if st.button("🧹 Purge Legacy / Placeholder Records", help="Removes records with System_Architecture, empty observations, or document_compliance defaults", width="stretch", key="btn_purge_legacy_jev"):
                purged = manager.purge_legacy_samples()
                manager.export_typed_decisions_parquet(str(EXPORTS_DIR / "typed_decisions_dataset.parquet"), column_mode="positional", only_verified=True)
                manager.export_open_jev_parquet(str(EXPORTS_DIR / "open_jev_s1mb_dataset.parquet"), only_verified=True)
                manager.export_jev_jsonl(str(EXPORTS_DIR / "jev_decisions_dataset.jsonl"), only_verified=True)
                st.toast(f"Purged {purged} legacy records and refreshed exports!", icon="🧹")
                st.rerun()
        with pur2:
            if st.button("🗑️ Clear Entire Jev Database", help="Deletes all Jev samples", width="stretch", key="btn_clear_all_jev"):
                manager.clear_all()
                manager.export_typed_decisions_parquet(str(EXPORTS_DIR / "typed_decisions_dataset.parquet"), column_mode="positional", only_verified=True)
                manager.export_open_jev_parquet(str(EXPORTS_DIR / "open_jev_s1mb_dataset.parquet"), only_verified=True)
                manager.export_jev_jsonl(str(EXPORTS_DIR / "jev_decisions_dataset.jsonl"), only_verified=True)
                st.toast("Jev database cleared.", icon="🗑️")
                st.rerun()

    else:
        st.info("No Jev decision records found. Generate a scenario in the builder tab above.")


def _show_split(manager) -> None:
    """Show how the last export was split (and warn when a split is far below its share)."""
    import pandas as pd
    from core.dataset_splits import describe_split
    info = getattr(manager, "last_split_info", None)
    line, warning = describe_split(info)
    if line:
        st.caption(line)
    if warning:
        st.warning(warning)
    if info and isinstance(info.get("counts"), dict):
        counts = info["counts"]
        total = sum(counts.values()) or 1
        df = pd.DataFrame([
            {"Split": k.capitalize(), "Samples": v, "Share": f"{(v / total) * 100:.1f}%"}
            for k, v in counts.items()
        ])
        st.dataframe(df, hide_index=True, use_container_width=True)


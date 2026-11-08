"""
Dataset Export & Analytics Tab for ClearCoT Studio.
Displays dataset statistics, sample filtering, and multi-format download capabilities.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json

from core.dataset_manager import DatasetManager
from config import EXPORTS_DIR

def render_export_tab(db_manager: DatasetManager):
    st.header("📦 Dataset Analytics & Export")
    st.caption("Inspect curated samples, filter by verification status, and export for SFT / DPO training.")

    stats = db_manager.get_stats()

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Samples", stats["total_samples"])
    m2.metric("Accepted", stats["accepted_count"], delta=f"{int(stats['acceptance_rate']*100)}% rate")
    m3.metric("Gentzen Verified", stats["verified_count"], delta=f"{int(stats['verification_rate']*100)}% pass")
    m4.metric("Avg Tokens Saved", f"{stats['avg_tokens_saved_pct']}%")
    m5.metric("Avg Clarity Score", stats["avg_clarity"])

    st.divider()

    # Filter controls
    all_raw_samples = db_manager.get_samples()
    doc_options = ["All Documents"] + sorted(list(set(s.doc_name for s in all_raw_samples if s.doc_name)))

    f0, f1, f2, f3 = st.columns([1.5, 1, 1, 1.5])
    with f0:
        doc_filter = st.selectbox("Document", doc_options)
    with f1:
        status_filter = st.selectbox("Status Filter", ["All", "accepted", "draft", "rejected"])
    with f2:
        verified_only = st.checkbox("Verified Only", value=True,
                                    help="Applies to the table AND the exports: only kernel-verified (PROVEN_TRUE) samples.")
    with f3:
        min_clarity = st.slider("Min Clarity Score", 0.0, 1.0, 0.0, step=0.05)

    samples = all_raw_samples
    if status_filter != "All":
        samples = [s for s in samples if s.status == status_filter]
    if doc_filter != "All Documents":
        samples = [s for s in samples if s.doc_name == doc_filter]
    if verified_only:
        samples = [s for s in samples if s.gentzen_verified]
    if min_clarity > 0:
        samples = [s for s in samples if s.clarity_score >= min_clarity]

    # Database Maintenance Controls
    with st.expander("⚙️ Database Management (Reset / Clear DB)"):
        c_del1, c_del2 = st.columns([2, 1])
        with c_del1:
            confirm_clear = st.checkbox(f"Confirm deleting samples from '{doc_filter}'?", value=False, key="confirm_clear_db")
        with c_del2:
            if st.button("🗑️ Clear Database", type="secondary", disabled=not confirm_clear, width="stretch"):
                db_manager.clear_all(doc_name=None if doc_filter == "All Documents" else doc_filter)
                st.success(f"Cleared samples for: {doc_filter}")
                st.rerun()

    st.subheader(f"Curated Samples ({len(samples)})")

    if samples:
        table_data = [
            {
                "ID": s.sample_id,
                "Document": s.doc_name,
                "Question": (s.question[:65] + "...") if len(s.question) > 65 else s.question,
                "Status": s.status,
                "Kripke Status": s.epistemic_status,
                "Verified": "✅ Yes" if s.gentzen_verified else "❌ No",
                "Saved Tokens": f"{s.tokens_saved_pct}%",
                "Clarity": s.clarity_score,
            }
            for s in samples
        ]
        df_preview = pd.DataFrame(table_data)
        st.dataframe(df_preview, width="stretch", hide_index=True)

        # Inspector for selected sample
        selected_id = st.selectbox("Select Sample ID to Inspect", [s.sample_id for s in samples])
        sample_obj = next((s for s in samples if s.sample_id == selected_id), None)

        if sample_obj:
            with st.expander(f"Inspect {sample_obj.sample_id}", expanded=True):
                st.markdown(f"**Question:**\n{sample_obj.question}")
                st.markdown(f"**Reasoning (`<think>`):**\n```\n{sample_obj.reasoning}\n```")
                st.markdown(f"**Answer:**\n{sample_obj.answer}")
                st.markdown(f"**Saul Kripke Status:** `{sample_obj.epistemic_status}`")
                st.markdown(f"**Gentzen Sequent:** `{sample_obj.gentzen_sequent}`")
                if sample_obj.curry_howard_term:
                    st.markdown(f"**Curry-Howard Typed λ-Term:**\n```text\n{sample_obj.curry_howard_term}\n```")

        st.divider()

        # Export Actions
        st.subheader("Generate & Download Files")
        split_strategy = st.selectbox(
            "Train / validation / test split by", ["auto", "document", "path", "family"], key="sft_split_strategy",
            help="80/10/10, and no document (or KG path / chunk) is split across train and test. auto: by document "
                 "if there are at least 3, otherwise by path / chunk.")
        exp1, exp2, exp3 = st.columns(3)

        with exp1:
            jsonl_path = EXPORTS_DIR / "clearcot_sft_dataset.jsonl"
            if st.button("💾 Generate SFT Dataset (JSONL)", type="primary", use_container_width=True):
                count = db_manager.export_jsonl(str(jsonl_path), only_accepted=(status_filter == "accepted"), split_strategy=split_strategy,
                                                only_verified=verified_only,
                                                doc_name=(None if doc_filter == "All Documents" else doc_filter), min_clarity=min_clarity)
                st.success(f"Exported {count} samples to {jsonl_path.name}")
                _show_split(db_manager)

            if jsonl_path.exists():
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="⬇️ Download SFT (JSONL)",
                        data=f.read(),
                        file_name="clearcot_sft_dataset.jsonl",
                        mime="application/jsonl",
                        use_container_width=True
                    )

        with exp2:
            dpo_path = EXPORTS_DIR / "clearcot_dpo_pairs.jsonl"
            if st.button("⚖️ Generate DPO Pairs", use_container_width=True):
                count = db_manager.export_dpo_pairs(str(dpo_path), only_accepted=True, skip_synthetic_rejected=True,
                                                    split_strategy=split_strategy, only_verified=verified_only,
                                                doc_name=(None if doc_filter == "All Documents" else doc_filter), min_clarity=min_clarity)
                st.success(f"Exported {count} preference pairs to {dpo_path.name} "
                           "(accepted samples only; pairs with template negatives skipped)")
                _show_split(db_manager)

            if dpo_path.exists():
                with open(dpo_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="⬇️ Download DPO Pairs",
                        data=f.read(),
                        file_name="clearcot_dpo_pairs.jsonl",
                        mime="application/jsonl",
                        use_container_width=True
                    )

        with exp3:
            parquet_path = EXPORTS_DIR / "clearcot_dataset.parquet"
            if st.button("📊 Generate Parquet Table", use_container_width=True):
                count = db_manager.export_parquet(str(parquet_path), only_accepted=(status_filter == "accepted"), split_strategy=split_strategy,
                                                  only_verified=verified_only,
                                                doc_name=(None if doc_filter == "All Documents" else doc_filter), min_clarity=min_clarity)
                st.success(f"Exported {count} samples to {parquet_path.name}")
                _show_split(db_manager)

            if parquet_path.exists():
                with open(parquet_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Parquet",
                        data=f.read(),
                        file_name="clearcot_dataset.parquet",
                        mime="application/octet-stream",
                        use_container_width=True
                    )

    else:
        st.info("No samples match the selected filters.")

    # -------------------------------------------------------------
    # Structured Chunks & Layout Exports Section
    # -------------------------------------------------------------
    st.divider()
    st.subheader("📑 Structured Chunks & Clean Layout Exports")
    st.caption("Export extracted noise-free knowledge chunks, clean reconstructed markdown, and layout blocks.")

    current_chunks = st.session_state.get("chunks", [])
    parsed_pages_dict = st.session_state.get("parsed_pages", {})

    c_exp1, c_exp2, c_exp3 = st.columns(3)

    with c_exp1:
        st.markdown(f"**Knowledge Chunks Parquet** (`{len(current_chunks)} chunks`)")
        chk_parquet_path = EXPORTS_DIR / "knowledge_chunks.parquet"
        if st.button("📦 Export Chunks to Parquet", use_container_width=True, disabled=not current_chunks):
            count = db_manager.export_chunks_parquet(current_chunks, str(chk_parquet_path))
            st.success(f"Exported {count} chunks with embeddings to {chk_parquet_path.name}")

        if chk_parquet_path.exists():
            with open(chk_parquet_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download knowledge_chunks.parquet",
                    data=f.read(),
                    file_name="knowledge_chunks.parquet",
                    mime="application/octet-stream",
                    use_container_width=True
                )

    with c_exp2:
        st.markdown(f"**Clean Markdown** (`{len(parsed_pages_dict)} document(s)`)")
        md_path = EXPORTS_DIR / "document_clean.md"
        if st.button("📝 Export Clean Markdown", use_container_width=True, disabled=not parsed_pages_dict):
            total_chars = 0
            for doc_name, pages in parsed_pages_dict.items():
                target_md = EXPORTS_DIR / f"{Path(doc_name).stem}.clean.md"
                chars = db_manager.export_clean_markdown(pages, str(target_md))
                total_chars += chars
            # one combined file for the download button below (per-document files stay next to it)
            combined = [f"<!-- {doc_name} -->\n\n" + (EXPORTS_DIR / f"{Path(doc_name).stem}.clean.md").read_text(encoding="utf-8")
                        for doc_name in parsed_pages_dict]
            md_path.write_text("\n\n---\n\n".join(combined), encoding="utf-8")
            st.success(f"Exported clean markdown for {len(parsed_pages_dict)} file(s)!")

        if md_path.exists():
            with open(md_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="⬇️ Download Clean Markdown",
                    data=f.read(),
                    file_name="document_clean.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    with c_exp3:
        st.markdown("**Structured Layout Blocks (JSONL)**")
        blocks_jsonl_path = EXPORTS_DIR / "layout_blocks.jsonl"
        all_blocks = []
        for doc_name, pages in parsed_pages_dict.items():
            for p in pages:
                all_blocks.extend(getattr(p, "blocks", []))

        if st.button("📐 Export Layout Blocks JSONL", use_container_width=True, disabled=not all_blocks):
            count = db_manager.export_layout_blocks_jsonl(all_blocks, str(blocks_jsonl_path))
            st.success(f"Exported {count} layout blocks to {blocks_jsonl_path.name}")

        if blocks_jsonl_path.exists():
            with open(blocks_jsonl_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="⬇️ Download layout_blocks.jsonl",
                    data=f.read(),
                    file_name="layout_blocks.jsonl",
                    mime="application/jsonl",
                    use_container_width=True
                )


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


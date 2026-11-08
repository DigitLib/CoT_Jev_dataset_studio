"""
Library & Ingestion Tab for ClearCoT Studio.
Handles document uploads, layout-aware parsing, chunk extraction, visual layout inspection, and preview.
"""

import streamlit as st
from pathlib import Path
import pandas as pd
from typing import List, Optional, Set, Tuple
import json

from core.document_parser import DocumentParser
from core.chunker import DocumentChunker, Chunk
from core.layout_engine import (
    get_layout_engine,
    PyMuPDFLayoutEngine,
    SuryaLayoutEngine,
    DEFAULT_EXCLUDED_LABELS,
    DEFAULT_EXCLUDED_SECTIONS,
    LABEL_PAGE_HEADER,
    LABEL_PAGE_FOOTER,
    LABEL_FOOTNOTE,
    LABEL_REFERENCE,
    LABEL_ACKNOWLEDGMENT,
    LABEL_FIGURE,
    LABEL_CAPTION,
    LABEL_TABLE,
    LABEL_TEXT,
    LABEL_SECTION_HEADER,
    LABEL_LIST_ITEM,
)
from llm_backend.client import LocalLLMClient
from config import UPLOADS_DIR
from gui._widgets import int_slider


def safe_upload_name(name: str):
    """Bare file name of an upload (no directories, no '..'), or None if nothing usable is left.
    The browser-supplied name must never place a file outside data/uploads/."""
    base = Path(str(name or "").replace("\\", "/")).name.strip()
    if not base or base in (".", "..") or base.startswith("."):
        return None
    target = (UPLOADS_DIR / base).resolve()
    return base if target.parent == Path(UPLOADS_DIR).resolve() else None


@st.cache_data(show_spinner=False)
def _cached_render_overlay(
    backend: str,
    pdf_path_str: str,
    page_number: int,
    excluded_labels_tuple: Tuple[str, ...],
    dpi: int = 140
):
    engine = get_layout_engine(backend=backend)
    return engine.render_page_overlay(
        pdf_path_str,
        page_number=page_number,
        excluded_labels=set(excluded_labels_tuple),
        dpi=dpi
    )


@st.cache_data(show_spinner=False)
def _cached_extract_blocks(
    backend: str,
    pdf_path_str: str,
    page_number: int,
):
    engine = get_layout_engine(backend=backend)
    return engine.extract_blocks(
        pdf_path_str,
        pages=[page_number - 1],
        excluded_labels=set(),
        excluded_sections=set(),
    )


def render_library_tab(
    parser: DocumentParser,
    chunker: DocumentChunker,
    embed_client: Optional[LocalLLMClient] = None,
    embedding_model: Optional[str] = None,
):
    st.header("📚 Document Ingestion & Layout Studio")
    st.caption("Upload PDFs or technical documents for vision/layout analysis, semantic chunking, and text vectorization.")

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("1. Document Ingestion")
        uploaded_files = st.file_uploader(
            "Drop PDFs or text files here",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Files will be parsed locally with layout analysis and full provenance tracking."
        )

        with st.expander("👁️ Layout Analysis & Noise Filtering", expanded=True):
            engine_choice = st.selectbox(
                "Layout Analysis Engine",
                ["PyMuPDF Layout (Fast CPU)", "Surya Layout (Vision Model)"],
                index=0,
                help="PyMuPDF Layout analyzes document geometry and fonts locally. Surya runs deep visual layout detection."
            )
            active_engine_backend = "surya" if "Surya" in engine_choice else "pymupdf"

            all_labels = [
                LABEL_PAGE_HEADER,
                LABEL_PAGE_FOOTER,
                LABEL_FOOTNOTE,
                LABEL_REFERENCE,
                LABEL_ACKNOWLEDGMENT,
                LABEL_FIGURE,
                LABEL_CAPTION,
            ]
            selected_excluded_labels = st.multiselect(
                "Visual Noise Tags to Exclude",
                options=all_labels,
                default=[l for l in all_labels if l in DEFAULT_EXCLUDED_LABELS],
                help="Blocks with these layout labels are dropped before chunking to eliminate noise."
            )

            all_sections = [
                "acknowledgements",
                "acknowledgments",
                "funding",
                "authors",
                "details",
                "conflicts of interest",
                "references",
                "bibliography",
                "table of contents",
                "contents",
                "preface",
            ]
            selected_excluded_sections = st.multiselect(
                "Sections to Drop Entirely",
                options=all_sections,
                default=[
                    "acknowledgements", "acknowledgments", "funding", "authors",
                    "details", "conflicts of interest", "references", "bibliography"
                ],
                help="When a Section-header matching any of these is encountered, all subsequent content in that section is omitted."
            )

        with st.expander("⚙️ Chunking Strategy", expanded=False):
            strategy_choice = st.radio(
                "Segmentation Mode",
                [
                    "📄 Paragraph Level (Natural)",
                    "🧠 Semantic Boundary (Embedding Distance)",
                    "📏 Fixed Window (Character Buffer)"
                ],
                index=0,
                help="Paragraph Level segments text by natural document paragraphs (recommended for deductive, philosophical, and legal literature). Semantic Boundary uses sentence-level embedding distances to create cohesive propositional units."
            )
            if "Semantic" in strategy_choice:
                selected_strategy = "semantic"
            elif "Paragraph" in strategy_choice:
                selected_strategy = "paragraph"
            else:
                selected_strategy = "fixed_size"

            percentile_thresh = 85.0
            min_toks = 50
            max_toks = 400

            if selected_strategy == "paragraph":
                st.info("💡 **Recommended for Deductive Literature**: Paragraph boundaries naturally respect an author's complete propositional thought, avoiding arbitrary cuts across qualifying clauses.")
                min_chars = st.slider(
                    "Min Paragraph Length (chars)",
                    min_value=40,
                    max_value=250,
                    value=80,
                    step=10,
                    help="Filters out short noise, headers, footnotes, or page numbers."
                )
                max_chars = 1500
            elif selected_strategy == "semantic":
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    percentile_thresh = st.slider(
                        "Split Threshold (Percentile)",
                        min_value=60.0,
                        max_value=95.0,
                        value=85.0,
                        step=5.0,
                        help="Sentences with distance above this percentile trigger a semantic chunk boundary."
                    )
                with col_s2:
                    min_toks = st.slider(
                        "Min Chunk Tokens",
                        min_value=20,
                        max_value=150,
                        value=50,
                        step=10,
                        help="Guards against undersized propositional fragments."
                    )
                max_toks = st.slider(
                    "Max Chunk Tokens",
                    min_value=100,
                    max_value=800,
                    value=400,
                    step=50,
                    help="Upper token guard to keep chunks within teacher LLM context windows."
                )
                min_chars = max(60, int(min_toks * 3.0))
                max_chars = 1500
                if embedding_model:
                    st.caption(f"🧠 *Using active embedding model:* `{embedding_model}` *(fallback: deterministic lexical vectorizer)*")
                else:
                    st.caption("⚡ *Using deterministic lexical vectors (connect local embedding model in sidebar for neural vector distance)*")
            else:
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    min_chars = st.number_input("Min Chars", min_value=50, max_value=500, value=100, step=50)
                with col_c2:
                    max_chars = st.number_input("Max Chars", min_value=500, max_value=3000, value=1500, step=100)

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                clean_citations = st.checkbox("Clean Inline Citations ([19])", value=True, help="Strips bracketed reference numbers like [19] or [19,28].")
            with col_opt2:
                strip_tables = st.checkbox("Strip Table Pointers", value=False, help="Removes sentences like '...illustrated in Table 1'. Keep unchecked if preserving table citations.")

        # Ingestion execution
        active_layout_engine = get_layout_engine(backend=active_engine_backend)
        parser.layout_engine = active_layout_engine
        parser.excluded_labels = set(selected_excluded_labels)
        parser.excluded_sections = set(selected_excluded_sections)

        chunker.min_chunk_chars = min_chars
        chunker.max_chunk_chars = max_chars
        chunker.strip_citations = clean_citations
        chunker.strip_table_pointers = strip_tables
        chunker.percentile_threshold = percentile_thresh
        chunker.min_tokens = min_toks
        chunker.max_tokens = max_toks
        chunker.embed_client = embed_client
        chunker.embed_model = embedding_model

        # Action 1: Ingest newly uploaded files
        if uploaded_files:
            if st.button("📥 Parse & Ingest Uploaded Files", type="primary", use_container_width=True):
                with st.spinner(f"Running {engine_choice} & extracting {selected_strategy} chunks..."):
                    new_chunks: List[Chunk] = []
                    docs_meta = st.session_state.get("docs_meta", {})
                    all_parsed_pages = st.session_state.get("parsed_pages", {})

                    ingested_names = set()
                    for uf in uploaded_files:
                        doc_name = safe_upload_name(uf.name)
                        if doc_name is None:
                            st.error(f"Skipped {uf.name!r}: not a valid file name.")
                            continue
                        ingested_names.add(doc_name)
                        save_path = UPLOADS_DIR / doc_name
                        with open(save_path, "wb") as f:
                            f.write(uf.getbuffer())

                        try:
                            pages = parser.parse_file(
                                str(save_path),
                                excluded_labels=set(selected_excluded_labels),
                                excluded_sections=set(selected_excluded_sections),
                            )
                            all_parsed_pages[doc_name] = pages
                            doc_chunks = chunker.chunk_pages(
                                doc_name,
                                pages,
                                strategy=selected_strategy,
                                percentile_threshold=percentile_thresh,
                                min_tokens=min_toks,
                                max_tokens=max_toks,
                                embed_client=embed_client,
                                embed_model=embedding_model,
                            )
                            new_chunks.extend(doc_chunks)
                            docs_meta[doc_name] = {
                                "pages": len(pages),
                                "chunks": len(doc_chunks),
                                "path": str(save_path),
                                "strategy": selected_strategy,
                            }
                        except Exception as e:
                            st.error(f"Error parsing {doc_name}: {e}")

                    # Replace existing chunks for these docs
                    doc_names = ingested_names
                    existing_chunks = [c for c in st.session_state.get("chunks", []) if c.doc_name not in doc_names]
                    st.session_state.chunks = existing_chunks + new_chunks
                    st.session_state.docs_meta = docs_meta
                    st.session_state.parsed_pages = all_parsed_pages
                    st.success(f"Extracted {len(new_chunks)} noise-free chunks ({selected_strategy} mode)!")
                    st.rerun()

        # Action 2: Re-parse existing files in data/uploads/
        existing_upload_files = [f for ext in ("pdf", "docx", "txt", "md") for f in sorted(UPLOADS_DIR.glob(f"*.{ext}"))]
        if existing_upload_files:
            if st.button(f"⚡ Re-index Server Files ({len(existing_upload_files)} file(s))", use_container_width=True):
                with st.spinner("Re-parsing with layout analysis & noise filters..."):
                    fresh_chunks: List[Chunk] = []
                    fresh_docs_meta = {}
                    all_parsed_pages = {}

                    for p in existing_upload_files:
                        try:
                            pages = parser.parse_file(
                                str(p),
                                excluded_labels=set(selected_excluded_labels),
                                excluded_sections=set(selected_excluded_sections),
                            )
                            all_parsed_pages[p.name] = pages
                            doc_chunks = chunker.chunk_pages(
                                p.name,
                                pages,
                                strategy=selected_strategy,
                                percentile_threshold=percentile_thresh,
                                min_tokens=min_toks,
                                max_tokens=max_toks,
                                embed_client=embed_client,
                                embed_model=embedding_model,
                            )
                            fresh_chunks.extend(doc_chunks)
                            fresh_docs_meta[p.name] = {
                                "pages": len(pages),
                                "chunks": len(doc_chunks),
                                "path": str(p),
                                "strategy": selected_strategy,
                            }
                        except Exception as e:
                            st.error(f"Error parsing {p.name}: {e}")

                    st.session_state.chunks = fresh_chunks
                    st.session_state.docs_meta = fresh_docs_meta
                    st.session_state.parsed_pages = all_parsed_pages
                    st.success(f"Extracted {len(fresh_chunks)} clean chunks from {len(existing_upload_files)} file(s)!")
                    st.rerun()

    with col2:
        tab_chunks_view, tab_layout_view = st.tabs([
            "🧠 Ingested Knowledge & Chunks",
            "👁️ Layout Analysis & BBox Preview"
        ])

        with tab_chunks_view:
            top_c1, top_c2 = st.columns([2, 1])
            with top_c1:
                st.subheader("Ingested Documents")
            with top_c2:
                if st.session_state.get("chunks") or st.session_state.get("docs_meta"):
                    if st.button("🗑️ Clear Library", help="Reset all chunks and document metadata"):
                        st.session_state.chunks = []
                        st.session_state.docs_meta = {}
                        st.session_state.parsed_pages = {}
                        st.rerun()

            docs_meta = st.session_state.get("docs_meta", {})
            chunks = st.session_state.get("chunks", [])

            if docs_meta or chunks:
                if docs_meta:
                    df_docs = pd.DataFrame([
                        {
                            "Document": doc,
                            "Pages": data["pages"],
                            "Chunks": data["chunks"],
                            "Strategy": data.get("strategy", "semantic"),
                        }
                        for doc, data in docs_meta.items()
                    ])
                    st.dataframe(df_docs, use_container_width=True, hide_index=True)

                m1, m2 = st.columns(2)
                m1.metric("Noise-Free Chunks Available", len(chunks))
                embedded_count = sum(1 for c in chunks if c.embedding is not None)
                m2.metric("Vectorized Chunks", f"{embedded_count}/{len(chunks)}")

                if embed_client and embedding_model:
                    if st.button(f"🧬 Generate Vector Embeddings ({embedding_model})", use_container_width=True):
                        with st.spinner(f"Computing embeddings using {embedding_model}..."):
                            try:
                                texts = [c.text for c in chunks]
                                all_embs = []
                                batch_size = 16
                                prog_bar = st.progress(0)
                                for i in range(0, len(texts), batch_size):
                                    batch = texts[i:i + batch_size]
                                    res_embs = embed_client.embed(batch, model=embedding_model)
                                    all_embs.extend(res_embs)
                                    prog_bar.progress(min(1.0, (i + len(batch)) / len(texts)))

                                for c, emb in zip(chunks, all_embs):
                                    c.embedding = emb
                                st.session_state.chunks = chunks
                                dim = len(all_embs[0]) if all_embs else 0
                                st.success(f"Successfully vectorized {len(chunks)} chunks! Vector dimension: {dim}d")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to generate embeddings: {e}")

                st.divider()
                st.subheader("Chunk Inspector")
                if chunks:
                    selected_idx = int_slider("Select Chunk", 0, len(chunks) - 1, 0)
                    c = chunks[selected_idx]
                    with st.container(border=True):
                        st.markdown(f"**Chunk ID:** `{c.chunk_id}` | **Document:** `{c.doc_name}` | **Page:** {c.page_number} | **Type:** `{c.chunk_type}`")
                        st.markdown(f"**Section:** `{c.section_title}` | **Length:** {len(c.text)} chars ({len(c.text.split())} words, ~{c.token_estimate} tokens)")
                        if c.embedding:
                            st.caption(f"🧬 **Vector Embedded:** {len(c.embedding)} dimensions")
                        else:
                            st.caption("⚪ **Vector Embedded:** No (click 'Generate Vector Embeddings' above)")
                        st.text_area("Chunk Content", value=c.text, height=180, disabled=True)
            else:
                st.info("No documents ingested yet. Upload a PDF or TXT to get started.")

        with tab_layout_view:
            st.subheader("👁️ Visual Layout Inspector & Bounding Boxes")
            st.caption("Inspect how the layout analysis engine detects headers, tables, body text, and eliminates visual noise.")

            available_pdfs = list(UPLOADS_DIR.glob("*.pdf"))
            if not available_pdfs:
                st.info("No PDF files available in `data/uploads/` to inspect.")
            else:
                pdf_names = [p.name for p in available_pdfs]
                sel_pdf_name = st.selectbox("Select PDF to Inspect", pdf_names, key="sel_layout_pdf")
                sel_pdf_path = UPLOADS_DIR / sel_pdf_name

                import pymupdf
                with pymupdf.open(str(sel_pdf_path)) as doc:
                    total_pages = len(doc)

                p_col1, p_col2 = st.columns([1, 2])
                with p_col1:
                    sel_page = st.number_input(
                        "Page Number",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        step=1,
                        key="num_layout_page"
                    )

                # Visual Legend
                st.markdown(
                    """
                    <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; font-size: 13px;">
                        <span>🟩 <b>Text / List</b></span>
                        <span>🟦 <b>Section Header</b></span>
                        <span>🟨 <b>Table</b></span>
                        <span>🟥 <b>Excluded Noise (Header / Footer / Citations / Ack)</b></span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Render page with overlay
                try:
                    overlay_img = _cached_render_overlay(
                        active_engine_backend,
                        str(sel_pdf_path),
                        page_number=int(sel_page),
                        excluded_labels_tuple=tuple(sorted(selected_excluded_labels)),
                        dpi=140
                    )
                    st.image(overlay_img, caption=f"{sel_pdf_name} - Page {sel_page} (Layout Overlay)", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to render layout overlay: {e}")

                # Display detected blocks table for this page
                st.markdown(f"##### Detected Layout Blocks (Page {sel_page})")
                try:
                    page_blocks = _cached_extract_blocks(
                        active_engine_backend,
                        str(sel_pdf_path),
                        page_number=int(sel_page)
                    )
                    block_rows = []
                    for b in page_blocks:
                        is_dropped = (b.label.lower() in selected_excluded_labels)
                        status_badge = "❌ Dropped (Noise)" if is_dropped else "✅ Included"
                        block_rows.append({
                            "Block ID": b.block_id,
                            "Label": b.label.upper(),
                            "Status": status_badge,
                            "Header Level": b.header_level,
                            "BBox [x0, y0, x1, y1]": f"[{b.bbox[0]:.0f}, {b.bbox[1]:.0f}, {b.bbox[2]:.0f}, {b.bbox[3]:.0f}]",
                            "Text Preview": (b.text[:80] + "...") if len(b.text) > 80 else b.text,
                        })
                    if block_rows:
                        st.dataframe(pd.DataFrame(block_rows), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No blocks detected on this page.")
                except Exception as e:
                    st.warning(f"Could not load block details table: {e}")

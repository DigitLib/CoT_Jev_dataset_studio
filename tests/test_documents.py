"""
Document ingestion: layout engine, parsing (PDF / DOCX), noise filtering and chunking.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests  # noqa: F401  (redirects all runtime data to a temp dir; see tests/__init__.py)

from core.chunker import Chunk, DocumentChunker, _fast_lexical_vectors, clean_paragraph_text, is_boilerplate, split_into_paragraphs, split_into_sentences
from core.document_parser import DocumentBlock, DocumentPage, DocumentParser, is_likely_heading, is_toc_block
from core.layout_engine import LABEL_FOOTNOTE, LABEL_LIST_ITEM, LABEL_SECTION_HEADER, LABEL_TABLE, LABEL_TEXT, PyMuPDFLayoutEngine, is_footnote_or_citation_block, is_noise_outline_title


_possible_paths = [
    Path(__file__).resolve().parent.parent / "data" / "uploads" / "clearcot-1.pdf",
    Path("data/uploads/clearcot-1.pdf"),
]

SAMPLE_PDF = str(next((p for p in _possible_paths if p.exists()), _possible_paths[0]))

TEXT_P1 = ("Intentionality requires mental represen- tation in every cognitive system that we know of today.\n\n"
           "Brentano argued that mental phenomena exhibit “intentionality” as their defining mark here.")

TEXT_P2 = ("Mental representation implies a representational vehicle, according to most current accounts of it.\n\n"
           "Mental representation implies a representational vehicle, according to most current accounts of it.")


def _pages():
    return [DocumentPage(page_number=1, text=TEXT_P1, blocks=[]),
            DocumentPage(page_number=2, text=TEXT_P2, blocks=[])]


class TestLayoutEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PyMuPDFLayoutEngine()

    def test_extract_blocks_clearcot(self):
        if not Path(SAMPLE_PDF).exists():
            self.skipTest(f"Sample PDF {SAMPLE_PDF} not found")

        blocks = self.engine.extract_blocks(SAMPLE_PDF, pages=[0, 1])
        self.assertGreater(len(blocks), 0)

        labels = {b.label for b in blocks}
        self.assertIn(LABEL_SECTION_HEADER, labels)
        self.assertIn(LABEL_TEXT, labels)

        # Ensure no headers, footers, or footnotes in clean blocks
        for b in blocks:
            self.assertNotIn(b.label, {"page-header", "page-footer", "footnote"})

    def test_table_preservation(self):
        if not Path(SAMPLE_PDF).exists():
            self.skipTest(f"Sample PDF {SAMPLE_PDF} not found")

        # Page 16 contains a comparison table
        blocks = self.engine.extract_blocks(SAMPLE_PDF, pages=[15])
        labels = [b.label for b in blocks]
        self.assertIn(LABEL_TABLE, labels)

    def test_render_page_overlay(self):
        if not Path(SAMPLE_PDF).exists():
            self.skipTest(f"Sample PDF {SAMPLE_PDF} not found")

        img = self.engine.render_page_overlay(SAMPLE_PDF, page_number=1)
        self.assertIsNotNone(img)
        self.assertGreater(img.width, 100)
        self.assertGreater(img.height, 100)

    def test_document_parser_with_layout_engine(self):
        if not Path(SAMPLE_PDF).exists():
            self.skipTest(f"Sample PDF {SAMPLE_PDF} not found")

        parser = DocumentParser(layout_engine=self.engine)
        pages = parser.parse_file(SAMPLE_PDF)
        self.assertGreater(len(pages), 0)

        # Page 1 must have heading and paragraphs
        p1 = pages[0]
        block_types = [b.block_type for b in p1.blocks]
        self.assertIn("heading", block_types)
        self.assertIn("paragraph", block_types)

    def test_footnote_citation_detection_in_layout_engine(self):
        """Verify is_footnote_or_citation_block identifies academic citations and footnote lines."""
        # Footnotes & citations
        self.assertTrue(is_footnote_or_citation_block("13 Oshana, 21–46."))
        self.assertTrue(is_footnote_or_citation_block("17 Frankfurt, “Freedom of the Will and the Concept of a Person”; Dworkin, The Theory and Practice of Autonomy."))
        self.assertTrue(is_footnote_or_citation_block("Consider 13 Oshana, 21–46."))
        self.assertTrue(is_footnote_or_citation_block("See also Buss and Westlund, “Personal Autonomy,” sec. 2."))
        self.assertTrue(is_footnote_or_citation_block("14 Buss and Westlund, “Personal Autonomy,” sec. 2."))

        # Normal list items & body text
        self.assertFalse(is_footnote_or_citation_block("1. Each agent acts according to reasoned intentions."))
        self.assertFalse(is_footnote_or_citation_block("Autonomy requires that a person have the capacities to form and develop values."))
        self.assertFalse(is_footnote_or_citation_block("Consider your visual experience as you look at this page. Pinch the back of your hand", y0=246.0, page_height=792.0))
        self.assertFalse(is_footnote_or_citation_block("Compare this with the alternative hypothesis that states consciousness is epiphenomenal.", y0=200.0, page_height=792.0))

    def test_normalize_label_reclassifies_list_items_with_citations(self):
        """Verify PyMuPDFLayoutEngine reclassifies pymupdf list-items containing citations as footnotes."""
        # Citation list items must become footnotes
        lbl1 = self.engine._normalize_label("list-item", "13 Oshana, 21–46.")
        self.assertEqual(lbl1, LABEL_FOOTNOTE)

        lbl2 = self.engine._normalize_label(
            "list-item",
            "17 Frankfurt, “Freedom of the Will and the Concept of a Person”; Dworkin, The Theory and Practice of Autonomy."
        )
        self.assertEqual(lbl2, LABEL_FOOTNOTE)

        # Standard list items must remain list items
        lbl_list = self.engine._normalize_label("list-item", "1. Each agent acts according to reasoned intentions.")
        self.assertEqual(lbl_list, LABEL_LIST_ITEM)


class TestDocumentParsing(unittest.TestCase):
    def test_outline_filter_matches_whole_titles(self):
        for t in ("Contents", "Table of Contents", "3 References", "Index", "Acknowledgements", "Cover",
                  "Chapter 7: Bibliography", "About the Authors", "List of Figures"):
            self.assertTrue(is_noise_outline_title(t), t)
        for t in ("Discovering the Self", "Indexical Reference", "Recovering Agency", "Contents of Consciousness",
                  "References to the Divine in Plato", "Coverage Analysis", "The Index Theorem"):
            self.assertFalse(is_noise_outline_title(t), t)

    def test_legacy_doc_has_a_clear_message(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "old.doc"
            p.write_bytes(b"\xd0\xcf\x11\xe0")
            parser = DocumentParser.__new__(DocumentParser)
            parser.excluded_labels = parser.excluded_sections = None
            with self.assertRaises(ValueError) as cm:
                parser.parse_file(str(p))
            self.assertIn(".docx", str(cm.exception))

    def test_docx_tables_are_kept_in_order(self):
        try:
            import docx
        except ImportError:
            self.skipTest("python-docx not installed")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.docx"
            doc = docx.Document()
            doc.add_heading("Results", level=1)
            doc.add_paragraph("Before the table.")
            tbl = doc.add_table(rows=2, cols=2)
            tbl.cell(0, 0).text, tbl.cell(0, 1).text = "Model", "Accuracy"
            tbl.cell(1, 0).text, tbl.cell(1, 1).text = "v4", "0.84"
            doc.add_paragraph("After the table.")
            doc.save(str(p))
            parser = DocumentParser.__new__(DocumentParser)
            parser.excluded_labels = parser.excluded_sections = None
            blocks = parser.parse_file(str(p))[0].blocks
        self.assertEqual([b.block_type for b in blocks], ["heading", "paragraph", "table", "paragraph"])
        self.assertEqual(blocks[2].text, "Model | Accuracy\nv4 | 0.84")

    def test_references_skipped_in_both_chunking_modes(self):
        pages = [DocumentPage(page_number=1, text="", blocks=[
            DocumentBlock(block_type="heading", text="Introduction", bbox=[0, 0, 100, 10]),
            DocumentBlock(block_type="paragraph", text="Minds represent the world through content-bearing states of the brain.", bbox=[0, 20, 100, 30]),
            DocumentBlock(block_type="paragraph", text="References", bbox=[0, 40, 100, 50]),
            DocumentBlock(block_type="heading", text="References", bbox=[0, 60, 100, 70]),
            DocumentBlock(block_type="paragraph", text="Smith, J. A long reference entry about something unrelated to this study.", bbox=[0, 80, 100, 90]),
        ])]
        for strategy in ("paragraph", "semantic"):
            chunks = DocumentChunker(min_chunk_chars=20, strategy=strategy).chunk_pages("d.pdf", pages)
            self.assertTrue(chunks, strategy)
            self.assertFalse(any("Smith" in c.text for c in chunks), strategy)

    def test_bbox_of_source_blocks_is_not_mutated(self):
        b1 = DocumentBlock(block_type="paragraph", text="A sentence without an end", bbox=[0, 0, 100, 10])
        b2 = DocumentBlock(block_type="paragraph", text="continues here and finishes properly.", bbox=[0, 12, 100, 40])
        DocumentChunker(min_chunk_chars=10).chunk_pages("d.pdf", [DocumentPage(page_number=1, text="", blocks=[b1, b2])])
        self.assertEqual(b1.bbox, [0, 0, 100, 10])


class TestStructuralNoiseFiltering(unittest.TestCase):
    def test_is_toc_block_detection(self):
        # Leader dot chains with trailing page numbers
        toc_line1 = "A.1 Prompt for Reasoning Data . . . . . . . . . .\n16"
        toc_line2 = "G.1 Breaking Ties in QwenRerank . . . . . . . . . . . . . . . . . . . . . . . . . .\n25"
        toc_outline = "24 G.1 Breaking Ties in QwenRerank. 25 G.2 Rank1 vs QwenRerank. 25 G.3 Candidates. 25 G.4 Generalization. 26"
        toc_title = "Table of Contents"

        self.assertTrue(is_toc_block(toc_line1))
        self.assertTrue(is_toc_block(toc_line2))
        self.assertTrue(is_toc_block(toc_outline))
        self.assertTrue(is_toc_block(toc_title))

        # Math formulas with 3 dots should NOT be flagged as TOC
        math_formula = "Let x1; x2; . . . ; xn be elements of set R."
        self.assertFalse(is_toc_block(math_formula))

        # Regular prose should NOT be flagged as TOC
        prose = "We present REASONIR-8B, the first retriever specifically trained for general reasoning tasks."
        self.assertFalse(is_toc_block(prose))

    def test_is_likely_heading_excludes_toc(self):
        toc_line = "A.1 Prompt for Reasoning-intensive Document-to-query Data Generation (HQ) . . . . . . . . . .\n16"
        self.assertFalse(is_likely_heading(toc_line))

    def test_is_boilerplate_excludes_toc(self):
        toc_outline = "24 G.1 Breaking Ties in QwenRerank. 25 G.2 Rank1 vs QwenRerank. 25 G.3 Candidates. 25 G.4 Generalization. 26"
        self.assertTrue(is_boilerplate(toc_outline))


class TestChunker(unittest.TestCase):
    def test_clean_paragraph_text_hyphenation(self):
        raw = "The computa-\ntional theory of mind explains mental states."
        cleaned = clean_paragraph_text(raw)
        self.assertIn("computational", cleaned)
        self.assertNotIn("computa-", cleaned)

    def test_clean_paragraph_text_citations(self):
        raw = "Recent studies demonstrate logic trees improve inference [12, 14]."
        cleaned = clean_paragraph_text(raw, strip_citations=True)
        self.assertNotIn("[12, 14]", cleaned)
        self.assertIn("improve inference", cleaned)

    def test_is_boilerplate(self):
        frontmatter = "Received in revised form 15 April 2023; Available online 1 May 2023. Elsevier Ltd."
        self.assertTrue(is_boilerplate(frontmatter))

        copyright_text = "Copyright © 2024. All rights reserved. creativecommons.org/licenses"
        self.assertTrue(is_boilerplate(copyright_text))

        keywords = "Keywords: cognitive architecture, intuitionistic logic, deduction"
        self.assertTrue(is_boilerplate(keywords))

        issn_cover = "ISSN 2516-418X (online) ISSN 2516-4171 (print) Koji Tanaka and Alexander Sandgren The Many Faces of Impossibility."
        self.assertTrue(is_boilerplate(issn_cover))

        cip_data = "Library of Congress Cataloging-in-Publication Data names: Roreitner, Robert, author. title: Aristotle on perception"
        self.assertTrue(is_boilerplate(cip_data))

        toc = "Contents Preface 7 I. Distinctions 1. 11 Only limits of rigor equilibrate extremities of pain. 2. 19"
        self.assertTrue(is_boilerplate(toc))

        substantive_text = (
            "Intuitionistic logic rejects the principle of excluded middle for non-finite domains, "
            "demanding constructive witnesses for existential assertions."
        )
        self.assertFalse(is_boilerplate(substantive_text))

    def test_chunk_token_estimation(self):
        chunk = Chunk(
            chunk_id="chk_001",
            doc_name="test.pdf",
            page_number=1,
            section_title="Introduction",
            text="This is a test paragraph with ten distinct words here.",
            chunk_type="paragraph"
        )
        self.assertGreater(chunk.token_estimate, 0)
        self.assertEqual(chunk.doc_name, "test.pdf")

    def test_split_into_paragraphs(self):
        multiparagraph = "First complete thought.\n\nSecond complete thought."
        paragraphs = split_into_paragraphs(multiparagraph)
        self.assertEqual(len(paragraphs), 2)
        self.assertEqual(paragraphs[0], "First complete thought.")
        self.assertEqual(paragraphs[1], "Second complete thought.")

    def test_split_into_sentences_abbreviations_and_initials(self):
        text = (
            "According to Tanaka, K. and Sandgren, A. (2020), e.g., the theorem holds. "
            "However, as shown by J. L. Mackie, this is not true in all models! "
            "Is that really so? Yes, cf. vol. 2, pp. 45-50. Value is approx. 3.14."
        )
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 5)
        self.assertIn("Tanaka, K. and Sandgren, A. (2020), e.g., the theorem holds.", sentences[0])
        self.assertIn("J. L. Mackie", sentences[1])
        self.assertEqual(sentences[2], "Is that really so?")
        self.assertIn("vol. 2, pp. 45-50", sentences[3])
        self.assertIn("approx. 3.14", sentences[4])

    def test_fast_lexical_vectors(self):
        sents = [
            "Formal logic studies valid deduction and mathematical reasoning.",
            "Mathematical deduction requires sound axioms and inference rules.",
            "Photosynthesis in plants converts solar energy into chemical glucose."
        ]
        vecs = _fast_lexical_vectors(sents)
        self.assertEqual(len(vecs), 3)
        import numpy as np
        cos_01 = np.dot(vecs[0], vecs[1])
        cos_12 = np.dot(vecs[1], vecs[2])
        # Logically related sentences share more subwords/terms than biology sentence
        self.assertGreater(cos_01, cos_12)

    def test_semantic_chunker_lexical_fallback(self):
        blocks = [
            DocumentBlock(
                text="1. Logic and Modality",
                block_type="heading",
                bbox=[10, 10, 200, 25]
            ),
            DocumentBlock(
                text="The concept of possibility is foundational in philosophical logic. "
                     "Possible worlds semantics allows us to model necessity and contingency rigorously. "
                     "Every proposition is evaluated across an index of accessibility relations.",
                block_type="paragraph",
                bbox=[10, 30, 400, 100]
            ),
            DocumentBlock(
                text="On a completely different note, deep learning architectures rely on stochastic backpropagation. "
                     "Convolutional layers capture spatial hierarchies in grid-like input structures. "
                     "Gradient descent minimizes the empirical empirical risk function over the training corpus.",
                block_type="paragraph",
                bbox=[10, 110, 400, 180]
            ),
        ]
        page = DocumentPage(page_number=1, text="", blocks=blocks)
        chunker = DocumentChunker(
            min_chunk_chars=40,
            percentile_threshold=50.0,
            min_tokens=15,
            max_tokens=100
        )
        chunks = chunker.chunk_pages(
            doc_name="modal_logic.pdf",
            pages=[page],
            strategy="semantic"
        )
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertEqual(c.chunk_type, "semantic")
            self.assertEqual(c.doc_name, "modal_logic.pdf")
            self.assertIn("Logic and Modality", c.section_title)

    def test_semantic_chunker_mock_embeddings(self):
        class MockEmbedClient:
            def embed(self, texts, model="test-model"):
                res = []
                for t in texts:
                    if "logic" in t.lower() or "possible" in t.lower():
                        res.append([1.0, 0.0])
                    else:
                        res.append([0.0, 1.0])
                return res

        blocks = [
            DocumentBlock(
                text="Section 1",
                block_type="heading",
                bbox=[10, 10, 200, 25]
            ),
            DocumentBlock(
                text="Possible worlds clarify modal logic statements. "
                     "Logical necessity holds in every accessible world. "
                     "Deep neural networks optimize parameters with gradient descent.",
                block_type="paragraph",
                bbox=[10, 30, 400, 100]
            ),
        ]
        page = DocumentPage(page_number=1, text="", blocks=blocks)
        chunker = DocumentChunker(
            min_chunk_chars=30,
            percentile_threshold=50.0,
            min_tokens=5,
            max_tokens=50,
            embed_client=MockEmbedClient(),
            embed_model="test-embed"
        )
        chunks = chunker.chunk_pages("test.pdf", [page], strategy="semantic")
        self.assertEqual(len(chunks), 2)
        self.assertIn("Possible worlds", chunks[0].text)
        self.assertIn("gradient descent", chunks[1].text)

    def test_is_boilerplate_frontmatter_and_figure_debris(self):
        # Frontmatter / Correspondence / URLs
        self.assertTrue(is_boilerplate("Correspondence: rulin@meta.com, rui.qiao@smart.mit.edu"))
        self.assertTrue(is_boilerplate("Date: April 30, 2025"))
        self.assertTrue(is_boilerplate("Code & Data: https://github.com/facebookresearch/ReasonIR"))
        self.assertTrue(is_boilerplate("Model: https://huggingface.co/reasonir/ReasonIR-8B"))
        self.assertTrue(is_boilerplate("arXiv:2504.20595v1 [cs.AI] 29 Apr 2025"))

        # Author list and affiliations with footnote symbols
        author_block = "Rulin Shao∗,♠,♡, Rui Qiao∗,†,‡, Varsha Kishore♣,♡, Niklas Muennighoff♢, Luke Zettlemoyer♠,♡"
        self.assertTrue(is_boilerplate(author_block))
        affil_block = "♠FAIR at Meta, ♡University of Washington, †National University of Singapore ∗Joint first author"
        self.assertTrue(is_boilerplate(affil_block))

        # Figure diagram debris
        self.assertTrue(is_boilerplate("0 200 400 600 800 1000 Query Lengths"))
        self.assertTrue(is_boilerplate("(c) Difficulty Measure"))
        self.assertTrue(is_boilerplate("BM25 Error\n53.0\n37.3\n25.1"))
        self.assertTrue(is_boilerplate("Varied-Length Query (167 tokens)"))
        self.assertTrue(is_boilerplate("{query, positive document}"))

    def test_frontmatter_and_figure_suppression_in_chunk_pages(self):
        blocks = [
            DocumentBlock(text="ReasonIR: Training Retrievers for Reasoning Tasks", block_type="heading"),
            DocumentBlock(text="Rulin Shao∗,♠,♡, Rui Qiao∗,†,‡, Varsha Kishore♣,♡", block_type="frontmatter"),
            DocumentBlock(text="♠FAIR at Meta, ♡University of Washington", block_type="frontmatter"),
            DocumentBlock(text="We present REASONIR-8B, the first retriever specifically trained for general reasoning tasks.", block_type="paragraph"),
            DocumentBlock(text="Date: April 30, 2025", block_type="frontmatter"),
            DocumentBlock(text="Correspondence: rulin@meta.com", block_type="frontmatter"),
            DocumentBlock(text="0 200 400 Query Lengths BM25 Error 53.0", block_type="figure_content"),
            DocumentBlock(text="Figure 1 Performance of retriever models", block_type="figure_caption"),
            DocumentBlock(text="1 Introduction", block_type="heading"),
            DocumentBlock(text="Retrieval-augmented generation has been widely used in knowledge-seeking tasks.", block_type="paragraph"),
        ]
        page = DocumentPage(page_number=1, text="", blocks=blocks)
        chunker = DocumentChunker(min_chunk_chars=30)
        chunks = chunker.chunk_pages("reasonir.pdf", [page], strategy="semantic")

        # Must have exactly 2 clean chunks: the abstract and the introduction!
        self.assertEqual(len(chunks), 2)
        self.assertNotIn("Rulin Shao", chunks[0].text)
        self.assertNotIn("FAIR at Meta", chunks[0].text)
        self.assertIn("We present REASONIR-8B", chunks[0].text)
        self.assertNotIn("Correspondence:", chunks[0].text)
        self.assertNotIn("Query Lengths", chunks[0].text)
        self.assertIn("Retrieval-augmented generation", chunks[1].text)

    def test_draft_watermarks_and_callout_filtering(self):
        """Verifies that pre-print draft watermarks and reviewer notes are caught by is_boilerplate."""
        self.assertTrue(is_boilerplate("DRAFT comments welcome, treasured, longed-for"))
        self.assertTrue(is_boilerplate("DRAFT - Comments welcome"))
        self.assertTrue(is_boilerplate("Pre-print draft: Do not cite or distribute"))
        self.assertTrue(is_boilerplate("for Cambridge Elements series"))
        # Substantive text must not be flagged
        self.assertFalse(is_boilerplate("This chapter explores the relationship between computational systems and subjective experience."))

    def test_min_tokens_strictly_enforced_in_semantic_mode(self):
        """Verifies that semantic mode does not emit chunks below min_tokens."""
        blocks = [
            DocumentBlock(text="Introductory Chapter", block_type="heading"),
            DocumentBlock(text="This is a brief two-sentence note. It has very few words.", block_type="paragraph"),
            DocumentBlock(text="Main Philosophical Section", block_type="heading"),
            DocumentBlock(
                text="Consciousness represents a fundamental explanatory puzzle in contemporary cognitive science and philosophy of mind. "
                     "Standard functionalist models explain how informational states transition across computational layers, but they fail "
                     "to account for why such state transitions are accompanied by qualitative subjective experience.",
                block_type="paragraph"
            ),
        ]
        page = DocumentPage(page_number=1, text="", blocks=blocks)
        chunker = DocumentChunker()
        chunks = chunker.chunk_pages("doc.pdf", [page], strategy="semantic", min_tokens=50)

        # The short 11-word intro should be stitched or not emitted as an isolated sub-50 chunk
        for c in chunks:
            self.assertGreaterEqual(c.token_estimate, 45)

    def test_footnote_number_clearing_and_safety_guards(self):
        """Verifies that footnote numbers (e.g. 56, 57, 13, 14) are safely purged while preserving valid numerals."""
        from core.chunker import clean_paragraph_text

        # 1. Brentano text with 56 and 57
        raw_brentano = (
            "We can, therefore, define mental phenomena by saying that they are those phenomena "
            "which contain an object intentionally within themselves. 56 Brentano’s argument is conceptual. "
            "Michael Tye’s later argument that all mental states have “representational” content has a similar structure. 57 "
            "The success of such arguments depends on the nonexistence of counterexamples."
        )
        cleaned_brentano = clean_paragraph_text(raw_brentano)
        self.assertNotIn(" 56 ", cleaned_brentano)
        self.assertNotIn(" 57 ", cleaned_brentano)
        self.assertIn("themselves. Brentano’s argument", cleaned_brentano)
        self.assertIn("structure. The success", cleaned_brentano)

        # 2. Nagel text with mid-text 13 and trailing 14.
        raw_nagel = (
            "To be conscious is for there to be “something it’s like” to be you right now. 13 "
            "It is to undergo states or processes with a “qualitative character”. "
            "The main scientific challenge lies not in defining consciousness but in developing robust methods to study it. 14."
        )
        cleaned_nagel = clean_paragraph_text(raw_nagel)
        self.assertNotIn(" 13 ", cleaned_nagel)
        self.assertNotIn(" 14", cleaned_nagel)
        self.assertIn("right now. It is to undergo", cleaned_nagel)
        self.assertTrue(cleaned_nagel.endswith("to study it."))

        # 3. Safety tests: Legitimate numbers must NOT be removed
        # Year test
        text_year = "The war concluded in 1945. 1995 was a peaceful year."
        self.assertEqual(clean_paragraph_text(text_year), text_year)

        # Numbered list test
        text_list = "We consider two conditions. 1. The first condition is valid. 2. The second condition holds."
        self.assertEqual(clean_paragraph_text(text_list), text_list)

        # Lowercase quantity test
        text_qty = "The trial finished. 50 participants were selected."
        self.assertEqual(clean_paragraph_text(text_qty), text_qty)

        # Decimal test
        text_dec = "The measurement was 3.14. 2.71 was Euler constant."
        self.assertEqual(clean_paragraph_text(text_dec), text_dec)


class TestStableChunkIds(unittest.TestCase):
    def test_same_input_same_ids(self):
        a = DocumentChunker(min_chunk_chars=20).chunk_pages("doc.pdf", _pages())
        b = DocumentChunker(min_chunk_chars=20).chunk_pages("doc.pdf", _pages())
        self.assertEqual([c.chunk_id for c in a], [c.chunk_id for c in b])
        self.assertEqual(len({c.chunk_id for c in a}), len(a), "IDs must be unique within a document")

    def test_ids_depend_on_content_and_document(self):
        a = DocumentChunker(min_chunk_chars=20).chunk_pages("doc.pdf", _pages())
        b = DocumentChunker(min_chunk_chars=20).chunk_pages("other.pdf", _pages())
        self.assertNotEqual(a[0].chunk_id, b[0].chunk_id)

    def test_duplicate_text_gets_suffix(self):
        ids = [c.chunk_id for c in DocumentChunker(min_chunk_chars=20).chunk_pages("doc.pdf", _pages())
               if c.page_number == 2]
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[1], ids[0] + "-2")


if __name__ == "__main__":
    unittest.main()

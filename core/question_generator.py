"""
Graph-aware and paragraph-level Question Generator.
Synthesizes questions and draft target claims from knowledge graph paths,
thematic clusters, and paragraph seeds for ClearCoT SFT dataset creation.
"""

from typing import Dict, Any, List, Optional
import json
import re
import logging

from .chunker import Chunk
from llm_backend.client import LocalLLMClient

logger = logging.getLogger(__name__)

PARAGRAPH_Q_SYSTEM_PROMPT = """You are an expert scientific dataset creator for advanced reasoning models adhering to the ClearCoT methodology.
Your task is to generate a rigorous, non-trivial reasoning question and definitive target answer grounded strictly in the domain concepts and arguments.

STRICT CRITERIA:
1. ZERO META-REFERENCES & SELF-CONTAINED PREMISE GROUNDING:
   - Ask about substantive scientific, historical, philosophical, or technical concepts, relationships, or arguments.
   - The question must be COMPLETELY SELF-CONTAINED: introduce the foundational premise, theory, mechanism, or definition directly.
   - NEVER use meta-phrases like "According to the text...", "Based on the provided research...", "in the document", or "within this context".
   - NEVER refer to "the text", "the author", "the excerpt", or "the passage". Ask about the subject matter itself.
   - NEVER start the question, target claim, or answer with transitional relics (e.g. "For example,", "Third,", "Furthermore,").
2. PROHIBITED: NEVER ask about publication metadata (dates, received/accepted timeline, journal names, DOI), copyright licenses (CC BY, open access), author affiliations/addresses, or raw keyword lists.
3. COMPLETENESS & NATURAL PHRASING: The question must be a complete, well-formed, natural grammatical inquiry ending with '?'.
4. TARGET CLAIM & ANSWER: Must be complete, substantive, and authoritative statements answering the question directly without introductory discourse debris.

Return ONLY a valid JSON object matching this structure:
{
  "question": "A complete, self-contained, premise-grounded reasoning question",
  "target_claim": "The exact core conclusion that must be deduced (full authoritative sentence)",
  "key_premises": ["Premise 1 stated in domain context", "Premise 2 stated in domain context"],
  "seed_answer": "Complete, authoritative factual answer sentence"
}
Do NOT include conversational filler or markdown explanations outside the JSON.
"""

MULTIHOP_Q_SYSTEM_PROMPT = """You are an expert dataset creator for multi-hop reasoning models adhering to the ClearCoT methodology.
You are given a relational chain connecting concepts from domain research and supporting evidence.
Formulate a natural, self-contained domain inquiry exploring how these concepts relate conceptually.

STRICT CRITERIA:
1. ZERO META-REFERENCES & AUTONOMOUS DOMAIN INQUIRY:
   - Ground the question directly in the foundational premise, framework, theory, or definition.
   - NEVER use meta-phrases like "According to the text...", "Based on the provided research...", "in the text", "in the document", or formulaic templates like "{Sentence}. How does X relate to Y within this context?".
   - NEVER mention "the text", "the document", "the author", "the excerpt", or "the passage". Ask directly about the concepts as an academic inquiry.
   - Formulate a cohesive, natural, flowing academic inquiry (e.g. "Under physicalist theories of mind, how does the identification of consciousness with brain activity reconcile with subjective epistemic gaps?").
   - NEVER start questions, claims, or answers with transitional markers like "For example,", "Third,", "Furthermore,".
2. PROHIBITED: Do not ask about editorial metadata (dates, publisher names), copyright licenses, or author addresses.
3. COMPLETENESS: The question must be a complete grammatical sentence ending with '?'.
4. TARGET CLAIM & SEED ANSWER:
   - Must be complete, authoritative sentences directly answering the question.
   - Plain, natural English prose only; NO formal logic notation, turnstiles, or math equations.
5. ZERO ANSWER LEAKAGE IN THE QUESTION:
   - Connect the problem setup seamlessly to the question (e.g. "Under physicalist theories of mind, how does the identification of consciousness with brain activity reconcile with subjective epistemic gaps?").
   - NEVER leak the final answer or resolution inside a subordinate clause (e.g. NEVER ask "How does X relate to Y, given that [Answer]?"). Let the reasoning trace discover the resolution.

Return ONLY a valid JSON object matching this structure:
{
  "question": "A natural, self-contained, premise-grounded multi-hop reasoning question",
  "target_claim": "The core factual conclusion connecting the concepts (full authoritative sentence)",
  "key_premises": ["Factual premise 1", "Factual premise 2"],
  "seed_answer": "Clear, complete, natural final answer sentence"
}
Do NOT include conversational filler. Return ONLY JSON.
"""

def _sanitize_question(question: str) -> str:
    """
    Purges meta-references, text pointers, and 'according to the text' phrasing from questions.
    Ensures question is a completely self-contained, natural academic/philosophical inquiry.
    """
    if not question:
        return ""
    q = question.strip()

    # 1. Remove meta clauses anywhere in the question
    meta_patterns = [
        r',?\s*\baccording\s+to\s+(?:the\s+)?(?:provided\s+)?(?:text|document|passage|excerpt|article|paper|research|source\s+text)\b[,;]?',
        r'\b(?:according\s+to|based\s+on|as\s+stated\s+in|as\s+described\s+in|as\s+mentioned\s+in)\s+(?:the\s+)?(?:provided\s+)?(?:text|document|passage|excerpt|article|paper|research|source\s+text)\b[,;]?',
        r'\b(?:within|in)\s+(?:the\s+)?(?:provided\s+)?(?:text|document|passage|excerpt|article|paper)\b[,;]?',
        r'\bfrom\s+(?:the\s+)?(?:provided\s+)?(?:text|document|passage|excerpt|article|source\s+text)\b[,;]?',
        r'\bas\s+(?:noted|explained|outlined)\s+in\s+(?:the\s+)?(?:provided\s+)?(?:text|document|passage|article)\b[,;]?',
    ]
    for pat in meta_patterns:
        q = re.sub(pat, '', q, flags=re.IGNORECASE)

    # 2. Fix spacing and punctuation around question marks and commas
    q = re.sub(r'\s*,\s*\?', '?', q)
    q = re.sub(r'\s{2,}', ' ', q).strip()

    # 3. Clean trailing prepositions before '?' if stripped
    q = re.sub(r'\b(?:to|in|of|for|with|about|under)\s*\?', '?', q)

    # 4. Ensure trailing '?' and proper capitalization
    q = q.rstrip(".!,:; ")
    if q and not q.endswith("?"):
        q += "?"
    if q and q[0].islower():
        q = q[0].upper() + q[1:]
    return q


def _clean_discourse_sentence(sent: str) -> str:
    """Strips leading discourse markers, line numbers, and meta-pointers from sentences."""
    s = sent.strip()
    s = re.sub(r'^\d+[\s\.\:\)\-]+\s*', '', s)
    s = re.sub(
        r'^(?:for\s+example,?\s*|in\s+particular,?\s*|specifically,?\s*|furthermore,?\s*|moreover,?\s*|however,?\s*|consequently,?\s*|therefore,?\s*|third,?\s*|second,?\s*|first,?\s*|finally,?\s*|yet,?\s*|again,?\s*|notably,?\s*|in\s+addition,?\s*|as\s+a\s+result,?\s*)',
        '',
        s,
        flags=re.IGNORECASE
    )
    s = re.sub(
        r'^(?:according\s+to\s+(?:the\s+)?(?:document|text|author|passage|article|excerpt|research|source\s+text),?\s*)',
        '',
        s,
        flags=re.IGNORECASE
    )
    s = re.sub(
        r'^(?:based\s+on\s+(?:the\s+)?(?:document|text|passage|excerpt|article),?\s*)',
        '',
        s,
        flags=re.IGNORECASE
    )
    s = re.sub(
        r'^(?:as\s+stated\s+in\s+(?:the\s+)?(?:document|text|passage),?\s*)',
        '',
        s,
        flags=re.IGNORECASE
    )
    s = re.sub(
        r'^(?:a\s+related\s+point\s+is\s+that|this\s+(?:might\s+)?suggest(?:s)?\s+that|it\s+(?:is\s+)?(?:important|worth\s+noting|clear)\s+that|meaning\s+that|which\s+means\s+that|suggesting\s+that|indicating\s+that)\s+',
        '',
        s,
        flags=re.IGNORECASE
    )
    s = s.strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def _lower_first_if_not_proper(text: str) -> str:
    if not text:
        return ""
    words = text.split(maxsplit=1)
    first = words[0]
    rest = (" " + words[1]) if len(words) > 1 else ""
    proper_names = {
        "turing", "gödel", "godel", "abelard", "llull", "descartes", "copeland",
        "marder", "penrose", "lucas", "schwartz", "caltech", "unesco", "ai", "ctm"
    }
    if first.isupper() or first.lower() in proper_names:
        return text
    return first[0].lower() + first[1:] + rest


class QuestionGenerator:
    def __init__(self, llm_client: Optional[LocalLLMClient] = None):
        self.llm_client = llm_client

    def generate_from_chunk(
        self,
        chunk: Chunk,
        model: str = "qwen2.5:7b"
    ) -> Dict[str, Any]:
        """Generate a question from a paragraph-level seed."""
        if not self.llm_client or not self.llm_client.health_check():
            return self._heuristic_paragraph_seed(chunk)

        prompt = f"Document: {chunk.doc_name} (Page {chunk.page_number})\nSection: {chunk.section_title}\n\nText Excerpt:\n{chunk.text}\n\nJSON Output:"
        try:
            raw = self.llm_client.generate(
                prompt=prompt,
                system_prompt=PARAGRAPH_Q_SYSTEM_PROMPT,
                model=model,
                temperature=0.2,
                max_tokens=600,
            )
            data = self._parse_json_safely(raw)
            if data:
                data["question"] = _sanitize_question(data.get("question", ""))
            if data and self.validate_sample_quality(data):
                data["source_type"] = "paragraph_seed"
                data["provenance"] = {
                    "doc_name": chunk.doc_name,
                    "pages": [chunk.page_number],
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section_title,
                }
                data["context_text"] = chunk.text
                if not data.get("key_premises"):
                    data["key_premises"] = [s.strip() for s in chunk.text.split(".") if len(s.strip()) > 20][:3]
                return data
        except Exception as e:
            logger.warning(f"LLM question generation failed: {e}")

        return self._heuristic_paragraph_seed(chunk)

    def generate_from_multihop_path(
        self,
        path_info: Dict[str, Any],
        supporting_chunks: List[Chunk],
        model: str = "qwen2.5:7b"
    ) -> Dict[str, Any]:
        """Generate a multi-hop reasoning question spanning multiple KG nodes/pages."""
        context_snippets = "\n\n---\n\n".join([
            f"[Page {c.page_number} - {c.section_title}]:\n{c.text.strip()}"
            for c in supporting_chunks[:3]
        ])

        prop_texts = [p.get("statement", "") for p in path_info.get("propositions", []) if p.get("statement")]
        props_block = "\n".join(f"- {pt}" for pt in prop_texts) if prop_texts else ""

        prompt = f"Relational Chain:\n{path_info['path_repr']}\n\n"
        if props_block:
            prompt += f"Grounded Propositions along Chain:\n{props_block}\n\n"
        prompt += f"Supporting Text:\n{context_snippets}\n\nJSON Output:"

        try:
            raw = self.llm_client.generate(
                prompt=prompt,
                system_prompt=MULTIHOP_Q_SYSTEM_PROMPT,
                model=model,
                temperature=0.2,
                max_tokens=650,
            )
            data = self._parse_json_safely(raw)
            if data:
                data["question"] = _sanitize_question(data.get("question", ""))
            if data and self.validate_sample_quality(data):
                data["source_type"] = "multihop_kg_path"
                data["path_nodes"] = path_info.get("path_nodes", [])
                data["chunk_ids"] = path_info.get("chunk_ids", [])
                data["propositions"] = path_info.get("propositions", [])
                prov_docs = path_info.get("provenance_docs", [])
                if not prov_docs and supporting_chunks:
                    prov_docs = sorted(list(set(c.doc_name for c in supporting_chunks if c.doc_name)))
                primary_doc = prov_docs[0] if prov_docs else (supporting_chunks[0].doc_name if supporting_chunks else "unknown")
                data["provenance"] = {
                    "doc_name": primary_doc,
                    "documents": prov_docs,
                    "is_cross_document": len(prov_docs) > 1,
                    "pages": path_info.get("provenance_pages", []),
                    "path_repr": path_info["path_repr"],
                    "path_nodes": path_info.get("path_nodes", []),
                    "chunk_ids": path_info.get("chunk_ids", []),
                }
                data["context_text"] = context_snippets
                return data
        except Exception as e:
            logger.warning(f"LLM multi-hop question generation failed: {e}")

        return self._heuristic_multihop_seed(path_info, supporting_chunks)

    def _parse_json_safely(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM output, handling think tags, markdown blocks, and escape quirks."""
        if not text:
            return None

        # 1. Strip think / thought / reasoning blocks first
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<thought>.*?</thought>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<reasoning>.*?</reasoning>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = clean.strip()

        # 2. Check for fenced code block: ```json ... ``` or ``` ... ```
        fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "question" in data:
                    return data
            except Exception:
                pass

        # 3. Search for outermost balanced JSON object
        start_idx = clean.find('{')
        end_idx = clean.rfind('}')
        if start_idx != -1 and end_idx > start_idx:
            candidate = clean[start_idx:end_idx + 1]
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "question" in data:
                    return data
            except Exception:
                cleaned_cand = re.sub(r',\s*([\]}])', r'\1', candidate)
                try:
                    data = json.loads(cleaned_cand)
                    if isinstance(data, dict) and "question" in data:
                        return data
                except Exception:
                    pass

        return None

    def _heuristic_paragraph_seed(self, chunk: Chunk) -> Dict[str, Any]:
        """High-quality heuristic paragraph seed extraction without broken sentence fragments."""
        raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk.text) if len(s.strip()) > 30]
        clean_sentences = [
            _clean_discourse_sentence(s) for s in raw_sentences
            if s[0].isupper()
            and not any(bad in s.lower() for bad in ("table ", "figure ", "fig. ", "section ", "available online", "creativecommons", "license", "doi.org", "et al.", "published by"))
            and not s.endswith(":")
            and not s.lower().startswith(("associated with", "in addition to"))
        ]

        candidates = clean_sentences if clean_sentences else raw_sentences
        if not candidates:
            candidates = [chunk.text.strip()]

        first_sentence = _clean_discourse_sentence(candidates[0])
        concl = _clean_discourse_sentence(candidates[-1]) if len(candidates) > 1 else first_sentence
        if len(concl) < 35 or concl.endswith("..."):
            concl = first_sentence

        premises = [_clean_discourse_sentence(c) for c in candidates[:min(3, len(candidates))]] if len(candidates) > 1 else [first_sentence]

        words = re.findall(r'[a-zA-Z0-9_-]+', first_sentence)
        meaningful_words = [
            w for w in words
            if len(w) > 3 and w.lower() not in ("according", "stated", "which", "their", "there", "these", "those", "about", "using", "first", "since", "while", "where", "example", "particular")
        ]
        topic = " ".join(meaningful_words[:3]) if meaningful_words else "the central concept"

        if first_sentence and len(first_sentence.strip()) > 35 and first_sentence != concl:
            s_lead = first_sentence.strip().rstrip(".")
            question = f"Given that {_lower_first_if_not_proper(s_lead)}, what does this analysis establish regarding {topic}?"
        else:
            question = f"Regarding {topic}, how does the theoretical framework establish its core principles and properties?"

        question = _sanitize_question(question)

        return {
            "question": question,
            "target_claim": concl,
            "key_premises": premises,
            "seed_answer": concl,
            "source_type": "paragraph_seed_heuristic",
            "provenance": {
                "doc_name": chunk.doc_name,
                "pages": [chunk.page_number],
                "chunk_id": chunk.chunk_id,
                "section": chunk.section_title,
            },
            "context_text": chunk.text,
        }

    def _heuristic_multihop_seed(self, path_info: Dict[str, Any], chunks: List[Chunk]) -> Dict[str, Any]:
        src = path_info.get("source", "Entity A")
        tgt = path_info.get("target", "Entity B")
        path_nodes = path_info.get("path_nodes", [src, tgt])

        full_context = "\n\n---\n\n".join([
            f"[Page {c.page_number} - {c.section_title}]:\n{c.text.strip()}"
            for c in chunks if c.text.strip()
        ]) if chunks else ""

        # Search for substantive sentences in chunks mentioning path concepts
        src_sentences = []
        tgt_sentences = []
        for c in chunks:
            for s in re.split(r'(?<=[.!?])\s+', c.text):
                s_strip = s.strip()
                if len(s_strip) > 25 and not any(bad in s_strip.lower() for bad in ("table ", "figure ", "creativecommons", "license", "doi.org")):
                    cleaned_s = _clean_discourse_sentence(s_strip)
                    if src.lower() in s_strip.lower() and cleaned_s not in src_sentences:
                        src_sentences.append(cleaned_s)
                    if tgt.lower() in s_strip.lower() and cleaned_s not in tgt_sentences:
                        tgt_sentences.append(cleaned_s)

        # Formulate a natural, coherent academic inquiry where problem setup is directly connected to question without answer leakage
        if src_sentences and len(src_sentences[0].strip()) > 30 and not any(kw in src_sentences[0].lower() for kw in ("unclear", "conclude", "demonstrate", "proven", "therefore")):
            lead = src_sentences[0].strip().rstrip(".")
            lead_sub = _lower_first_if_not_proper(lead)
            if len(path_nodes) > 2:
                mid = path_nodes[1]
                question = f"Considering that {lead_sub}, how do intermediate principles of {mid} govern the relationship between {src} and {tgt}?"
            else:
                question = f"Considering that {lead_sub}, what theoretical mechanisms determine how {src} relates to {tgt}?"
        elif len(path_nodes) > 2:
            mid = path_nodes[1]
            question = f"Under the principles governing {src} and {mid}, how do these concepts determine the structure and implications of {tgt}?"
        else:
            question = f"In the conceptual framework of {src}, what theoretical mechanisms define its relationship with {tgt}?"

        question = _sanitize_question(question)

        # Ground target claim and seed answer in real text statements and path propositions
        propositions = path_info.get("propositions", [])
        prop_statements = [p["statement"] for p in propositions if p.get("statement")]
        prop_evidence = [p["evidence"] for p in propositions if p.get("evidence")]

        if prop_statements:
            if len(prop_statements) >= 2:
                target_claim = path_info.get("composite_claim") or f"Because {prop_statements[0].rstrip('.')}, {prop_statements[1].rstrip('.').lower()}."
                premises = prop_statements
                seed_answer = f"{prop_statements[0]} Consequently, {prop_statements[1].lower()}"
            else:
                target_claim = prop_statements[0]
                premises = prop_statements
                seed_answer = prop_statements[0]
        elif src_sentences:
            s1 = src_sentences[0].rstrip(".")
            distinct_tgt = next((s.rstrip(".") for s in tgt_sentences if s.strip() != s1), None)

            if distinct_tgt:
                s2 = distinct_tgt
                s2_sub = _lower_first_if_not_proper(s2)
                target_claim = f"{s1}, which directly informs how {s2_sub}."
                seed_answer = f"{s1}. In relation to this, {s2_sub}."
                premises = [s1 + ".", s2 + "."]
            else:
                target_claim = f"{s1}."
                seed_answer = f"{s1}."
                premises = [s1 + "."]
        elif tgt_sentences:
            s2 = tgt_sentences[0].rstrip(".")
            s2_sub = _lower_first_if_not_proper(s2)
            target_claim = f"{s2}."
            seed_answer = f"The evidence confirms that {s2_sub}."
            premises = [s2 + "."]
        else:
            target_claim = f"{src} forms a foundational framework establishing the core principles and constraints that govern {tgt}."
            seed_answer = f"{src} establishes the theoretical basis required to formalize and evaluate {tgt}."
            premises = [f"Foundational theoretical relationship connects {src} to {tgt}."]

        return {
            "question": question,
            "target_claim": target_claim,
            "key_premises": premises,
            "seed_answer": seed_answer,
            "source_type": "multihop_kg_path_heuristic",
            "path_nodes": path_nodes,
            "chunk_ids": path_info.get("chunk_ids", []),
            "propositions": propositions,
            "provenance": {
                "doc_name": path_info.get("doc_name") or (chunks[0].doc_name if chunks else "unknown"),
                "documents": path_info.get("provenance_docs") or ([chunks[0].doc_name] if chunks else []),
                "is_cross_document": path_info.get("is_cross_document", False),
                "pages": path_info.get("provenance_pages", []),
                "path_repr": path_info.get("path_repr", ""),
                "path_nodes": path_nodes,
                "chunk_ids": path_info.get("chunk_ids", []),
            },
            "context_text": full_context,
        }

    @staticmethod
    def validate_sample_quality_with_reason(q_res: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate that generated question and answer meet minimum quality criteria.
        Returns (False, reason) if answer is a crazy fragment, table pointer, or metadata question.
        """
        question = q_res.get("question", "").strip()
        answer = q_res.get("target_claim", q_res.get("seed_answer", q_res.get("answer", ""))).strip()
        reasoning = q_res.get("reasoning", "")

        if len(question) < 25 or not question.endswith("?"):
            return False, f"Question too short ({len(question)} chars) or does not end with '?'"
        if len(answer) < 25:
            return False, f"Answer too short or fragment: '{answer}' ({len(answer)} chars)"

        # Reject truncated questions
        if "..." in question or "…" in question:
            return False, f"Truncated question with ellipsis: '{question}'"

        # Reject questions containing forbidden meta-references to the text or document
        q_lower = question.lower()
        banned_meta_phrases = (
            "according to the text", "in the text", "the text explicitly", "the text states",
            "according to the document", "in the document", "within this text", "from the text",
            "according to the passage", "in the passage", "as stated in the text"
        )
        for bmp in banned_meta_phrases:
            if bmp in q_lower:
                return False, f"Question contains forbidden meta-reference: '{bmp}'"

        # Reject answers starting with lowercase or dangling prepositions/pointers
        if answer[0].islower() or answer.lower().startswith((
            "associated with", "according to table", "in table", "illustrated in table",
            "see table", "shown in table", "outlined in section"
        )):
            return False, f"Answer is a sentence fragment or pointer: '{answer}'"

        # Reject questions or answers regarding publication metadata, licenses, editorial dates, keywords
        a_lower = answer.lower()
        metadata_terms = (
            "copyright status", "editorial process", "timeline for the article",
            "received 29 september", "creative commons", "cc by-nc-nd",
            "terms are listed under", "listed under the 'keywords'",
            "which two institutions are located at the same address",
            "doi.org", "published by elsevier", "open access article",
            "elsevier ltd", "available online 21 november"
        )
        for term in metadata_terms:
            if term in q_lower:
                return False, f"Question asks about publication metadata/license: '{term}'"
            if term in a_lower:
                return False, f"Answer contains publication metadata/license: '{term}'"

        # Reject if reasoning purely references Elsevier license text or publication dates
        contaminated_reasoning_terms = (
            "cc by-nc-nd", "creative commons", "published by elsevier",
            "available online 21 november", "received 29 september", "0003-682x"
        )
        if any(term in reasoning.lower() for term in contaminated_reasoning_terms):
            return False, "Reasoning context contains publication metadata or Elsevier license text"

        # Reject non-propositional targets (isolated persons or author names as logical succedents)
        path_nodes = q_res.get("path_nodes", [])
        if path_nodes:
            last_node = str(path_nodes[-1]).strip().lower()
            non_prop_targets = {
                "searle", "john searle", "dehaene", "stanislas dehaene", "stanisla dehaene",
                "hakwan lau", "lau", "christof koch", "koch", "chalmers", "david chalmers",
                "fodor", "dennett", "lamme", "victor lamme", "judge"
            }
            if last_node in non_prop_targets:
                return False, f"Target path node is an entity/author rather than a propositional claim: '{path_nodes[-1]}'"

        return True, "Valid"

    @staticmethod
    def validate_sample_quality(q_res: Dict[str, Any]) -> bool:
        """
        Validate that generated question and answer meet minimum quality criteria.
        Returns False if answer is a crazy fragment, table pointer, or metadata question.
        """
        is_valid, _ = QuestionGenerator.validate_sample_quality_with_reason(q_res)
        return is_valid



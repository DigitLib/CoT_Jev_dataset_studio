"""
Thematic Knowledge Graph Builder using NetworkX.
Adheres to the Knwler architecture (Orbifold):
- Canonical schema and typed extraction.
- Incremental cross-chunk entity linking with alias consolidation and plural normalization.
- Safe 280-token extraction limits to maximize throughput on vLLM continuous batching.
- Thread-safe batch execution with main-thread progress reporting (prevents NoSessionContext).
- Robust multi-hop reasoning path discovery (>= 2 hops) across directed and undirected relational chains.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import networkx as nx
import json
import ast
import re
import logging
import time
import asyncio
import threading
import concurrent.futures
from dataclasses import dataclass, field

try:
    from knwler.config import Config as KnwlerConfig
    from knwler.models import Schema as KnwlerSchema, Graph as KnwlerGraph, ChunkGraph as KnwlerChunkGraph
    from knwler.chunking import Chunk as KnwlerChunk
    import knwler.llm
    import knwler.extraction
    from knwler.consolidation import consolidate_chunk_graphs
    KNWLER_AVAILABLE = True
except ImportError:
    KNWLER_AVAILABLE = False
    KnwlerConfig = None
    KnwlerSchema = None
    KnwlerGraph = None
    KnwlerChunkGraph = None
    KnwlerChunk = None

import os
from pathlib import Path
from datetime import datetime, timezone
from .chunker import Chunk
from .evidence import ground_evidence, summarize as summarize_evidence, SUPPORTED as EVIDENCE_SUPPORTED
from llm_backend.client import LocalLLMClient
from .concepts import (
    canonical_concept, concept_key, concept_issues, concept_quality, read_relation, copula_relation,
    repair_from_evidence, relation_from_statement, specialization_of, is_named_entity, inference_cue,
    fix_legacy_singular, MODALS as CONCEPT_MODALS, CONTRAST as CONCEPT_CONTRAST, check_inference,
    PREPOSITIONS as CONCEPT_PREPOSITIONS,
)
from nanogentzen.cot.relation_logic import (
    NEGATION_RE, NEGATION_TAIL_RE, EXCLUSION_RE, relation_logic, is_negated_phrase,
)

logger = logging.getLogger(__name__)

# Kept close to the original on purpose: an A/B run with the 2B extraction model on 30 chunks showed that
# long rule lists or a forced predicate vocabulary make small models write clauses as concept names and put
# "requires" / "is defined as" on relations the text does not state. This variant kept concept names as clean
# as before and raised supported inference edges; core.concepts.check_inference gates every inference label anyway.
EXTRACTION_SYSTEM_PROMPT = """You are an expert Knowledge Graph extractor for scientific, philosophical, and technical literature.
Given a document paragraph, extract 3 to 5 key relational facts.
Return ONLY a valid JSON array of objects:
[
  {
    "subject": "Entity A",
    "predicate": "relation",
    "object": "Entity B",
    "statement": "Complete natural language proposition asserting the relationship",
    "evidence": "Exact supporting excerpt from text",
    "strength": 8
  }
]
CRITICAL RULES:
1. Both "subject" and "object" MUST be concise conceptual noun phrases (1 to 4 words), never pronouns or clauses.
2. "predicate": prefer requires, implies, causes, excludes, is a kind of, is defined as, depends on, enables, undermines.
3. Connect to existing known canonical concepts whenever relevant.
4. Return ONLY the valid JSON array. No markdown explanations."""


GENERIC_BRIDGE_CONCEPTS = {
    "model", "models", "framework", "frameworks", "system", "systems",
    "method", "methods", "approach", "approaches", "analysis", "analyses",
    "evaluation", "evaluations", "generalization", "performance",
    "reasoning", "data", "dataset", "datasets", "result", "results",
    "study", "studies", "process", "processes", "theory", "theories",
    "concept", "concepts", "property", "properties", "function", "functions"
}

NON_PROPOSITIONAL_TARGETS = {
    "searle", "john searle", "dehaene", "stanislas dehaene", "stanisla dehaene",
    "hakwan lau", "lau", "christof koch", "koch", "chalmers", "david chalmers",
    "fodor", "dennett", "lamme", "victor lamme", "judge"
}


@dataclass
class Proposition:
    subject: str
    predicate: str
    obj: str
    statement: str
    evidence: str
    strength: int = 8
    chunk_id: str = ""
    doc_name: str = ""
    page_number: int = 1
    section: str = ""
    evidence_status: str = ""   # exact | fuzzy | located | unsupported ("" = not checked)
    extractor: str = ""         # llm | heuristic | knwler | manual | taxonomy
    raw_predicate: str = ""     # the relation phrase as extracted, before canonicalization
    claimed_predicate: str = "" # an inference predicate the evidence did not support (see _gate_inference)
    inference_check: str = ""   # why the evidence gate kept / demoted the predicate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "statement": self.statement,
            "evidence": self.evidence,
            "strength": self.strength,
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "section": self.section,
            "evidence_status": self.evidence_status,
            "extractor": self.extractor,
        }


class KnowledgeGraphBuilder:
    def __init__(self, llm_client: Optional[LocalLLMClient] = None, architect: Optional[Any] = None):
        self.llm_client = llm_client
        self.architect = architect
        self.graph = nx.MultiDiGraph()
        self._node_lookup: Dict[str, str] = {}
        self._key_lookup: Dict[str, str] = {}     # core.concepts.concept_key -> node
        self._path_cache: Dict[Tuple, List[Dict[str, Any]]] = {}
        self._graph_version: int = 0
        # "drop": edges whose evidence is not found in their chunk are not added; "keep": added, flagged
        self.evidence_policy: str = "drop"
        self.extraction_stats: Dict[str, int] = {}

    def clear(self):
        self.graph.clear()
        self._node_lookup.clear()
        self._key_lookup.clear()
        self._path_cache.clear()
        self._graph_version += 1
        self.extraction_stats = {}

    _STATS_LOCK = threading.Lock()   # extraction workers update the counters concurrently

    def _count(self, key: str, n: int = 1):
        with self._STATS_LOCK:
            self.extraction_stats[key] = self.extraction_stats.get(key, 0) + n

    @staticmethod
    def _safe_strength(v: Any) -> int:
        try:
            return max(1, min(10, int(round(float(v)))))
        except (TypeError, ValueError, OverflowError):   # OverflowError: "strength": Infinity in LLM JSON
            return 8

    def _gate_inference(self, predicate: str, evidence: str, subject: str = "",
                        obj: str = "") -> Tuple[str, str, str, bool]:
        """
        (predicate, claimed_predicate, check, flip). An inference predicate (requires, causes, excludes ...) is kept
        only if the evidence itself states that relation unhedged, in a sentence about the subject or object
        (core.concepts.check_inference); otherwise the edge becomes "relates to" and the LLM's label is kept as
        claimed_predicate. flip=True when the evidence states the relation the other way round (the LLM swapped
        subject and object). A wrong inference edge is worse than none: it becomes a premise of "verified" paths.
        """
        if relation_logic(predicate) == "none":
            return predicate, "", "", False
        res = check_inference(predicate, evidence, subject, obj)
        if res["ok"]:
            return predicate, "", res["reason"] + ("; direction reversed" if res["direction"] == "reverse" else ""), \
                res["direction"] == "reverse"
        return "relates to", predicate, res["reason"], False

    def _grounded(self, props: List["Proposition"], chunk: Chunk, extractor: str) -> List["Proposition"]:
        """Check every proposition's evidence against its chunk; apply evidence_policy and the inference gate."""
        out = []
        for pr in props:
            status, ev = ground_evidence(pr.evidence, chunk.text, pr.subject, pr.obj)
            pr.evidence_status, pr.evidence, pr.extractor = status, ev, extractor
            self._count(f"evidence_{status}")
            if status == "unsupported" and self.evidence_policy == "drop":
                self._count("dropped_unsupported")
                continue
            pr.predicate, pr.claimed_predicate, pr.inference_check, flip = self._gate_inference(
                pr.predicate, pr.evidence, pr.subject, pr.obj)
            if flip:
                pr.subject, pr.obj = pr.obj, pr.subject
                self._count("inference_reversed")
            if pr.claimed_predicate:
                self._count("inference_demoted")
            out.append(pr)
        return out

    def build_from_chunks(
        self,
        chunks: List[Chunk],
        model: str = "Sagicc/Occam-2B-CCoT",
        max_chunks: Optional[int] = None,
        concurrency: int = 8,
        progress_callback=None,
        base_url: Optional[str] = None
    ):
        """
        Extracts relational propositions across document chunks in parallel batches.
        All progress_callback invocations run strictly on the calling (main) thread.
        """
        target_chunks = chunks if max_chunks is None else chunks[:max_chunks]
        total = len(target_chunks)
        if total == 0:
            return

        active_client = self.llm_client
        if base_url:
            from llm_backend.client import LocalLLMClient
            active_client = LocalLLMClient(base_url=base_url)

        # Pre-flight check once instead of on every chunk
        if not active_client or not active_client.health_check():
            logger.warning("Backend offline. Generating heuristic knowledge graph.")
            for idx, ch in enumerate(target_chunks):
                for prop in self._grounded(self._heuristic_proposition_extract(ch), ch, "heuristic"):
                    self.add_proposition(prop)
                if progress_callback:
                    progress_callback(idx + 1, total, f"Heuristic KG extraction [{idx+1}/{total}] (p.{ch.page_number})")
            return

        if concurrency <= 1:
            for idx, chunk in enumerate(target_chunks):
                known_entities = self._relevant_concepts(chunk.text)
                propositions = self._extract_propositions_for_chunk(
                    chunk, model, known_entities=known_entities, client=active_client
                )
                for prop in propositions:
                    self.add_proposition(prop)
                if progress_callback:
                    progress_callback(idx + 1, total, f"Extracting KG [{idx+1}/{total}] (p.{chunk.page_number})")
        else:
            # Worker ONLY extracts data without calling Streamlit or acquiring GUI locks. Chunks run in waves so
            # each wave sees the concepts the previous ones created (one consistent name per concept).
            def _worker(item: Tuple[int, Chunk, List[str]]):
                idx, ch, known_entities = item
                props = self._extract_propositions_for_chunk(
                    ch, model, known_entities=known_entities, client=active_client
                )
                return idx, ch, props

            completed = 0
            wave = max(1, concurrency * 2)
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                for start in range(0, total, wave):
                    batch = list(enumerate(target_chunks))[start:start + wave]
                    # computed on the main thread, between waves: workers never read a graph being mutated
                    items = [(i, ch, self._relevant_concepts(ch.text)) for i, ch in batch]
                    futures = [executor.submit(_worker, item) for item in items]
                    # Collect results and update GUI progress strictly on the main thread
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            idx, ch, props = future.result()
                        except Exception as e:
                            logger.error(f"Worker task failed during KG extraction: {e}")
                            completed += 1
                            continue
                        for prop in props:
                            self.add_proposition(prop)
                        completed += 1
                        if progress_callback:
                            progress_callback(
                                completed,
                                total,
                                f"KG extraction [{completed}/{total}] (p.{ch.page_number}, workers={concurrency})"
                            )

    def append_from_chunks(
        self,
        chunks: List[Chunk],
        model: str = "Sagicc/Occam-2B-CCoT",
        concurrency: int = 8,
        progress_callback=None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Appends new chunks to the existing knowledge graph without clearing prior entities or relations.
        Chunks whose chunk_id already exists on any graph edge are skipped.
        """
        existing_cids = {
            data.get("chunk_id") for _, _, data in self.graph.edges(data=True) if data.get("chunk_id")
        }
        novel_chunks = [ch for ch in chunks if ch.chunk_id not in existing_cids]
        if not novel_chunks:
            logger.info("All provided chunks already exist in knowledge graph.")
            return {
                "novel_chunks": 0,
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
            }

        logger.info(f"Appending {len(novel_chunks)} new chunks to existing graph with {self.graph.number_of_nodes()} nodes.")
        self.build_from_chunks(
            chunks=novel_chunks,
            model=model,
            concurrency=concurrency,
            progress_callback=progress_callback,
            base_url=base_url,
        )
        return {
            "novel_chunks": len(novel_chunks),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
        }

    def _relevant_concepts(self, text: str, k: int = 15) -> List[str]:
        """
        Existing concepts the chunk talks about (every content word occurs in it), best first: well-formed,
        well-connected concepts. Passed to the LLM so it reuses one name per concept instead of inventing variants.
        """
        if self.graph.number_of_nodes() == 0 or not text:
            return []
        from .evidence import _words, _stem, _STOP
        stems = {_stem(w) for w in _words(text)}
        out = []
        for n, a in self.graph.nodes(data=True):
            ws = [w for w in _words(n) if w not in _STOP and len(w) >= 3]
            if ws and all(_stem(w) in stems for w in ws):
                out.append((-(a.get("concept_quality", 1.0) >= 0.6), -self.graph.degree(n), -len(ws), str(n)))
        out.sort()
        return [t[-1] for t in out[:k]]

    def build_from_chunks_knwler(
        self,
        chunks: List[Chunk],
        base_url: str = "http://localhost:8000/v1",
        model: str = "Sagicc/Occam-2B-CCoT",
        max_chunks: Optional[int] = None,
        concurrency: int = 8,
        progress_callback=None
    ):
        """
        Constructs Knowledge Graph using Knwler schemas, maintaining concurrency across fallbacks.
        """
        if not KNWLER_AVAILABLE:
            logger.warning("knwler package not installed. Falling back to direct extraction.")
            return self.build_from_chunks(
                chunks, model=model, max_chunks=max_chunks,
                concurrency=concurrency, progress_callback=progress_callback, base_url=base_url
            )

        target_chunks = chunks if max_chunks is None else chunks[:max_chunks]
        total = len(target_chunks)
        if total == 0:
            return

        resolved_url = base_url.replace("localhost", "127.0.0.1").rstrip("/")
        config = KnwlerConfig(
            backend="openai",
            base_url=resolved_url,
            api_key="not-needed",
            extraction_model=model,
            discovery_model=model,
            temperature=0.1,
            max_tokens=600,
            overlap_tokens=30,
            num_predict=650,
            use_cache=False
        )
        schema = KnwlerSchema(
            entity_types=["Concept", "Theory", "Person", "Formal System", "Process", "Mechanism"],
            relation_types=["defines", "implements", "requires", "enables", "implies", "contrasts with", "consists of", "models", "proves"]
        )

        def _get_val(item: Any, key: str, default: Any = "") -> Any:
            if isinstance(item, dict):
                return item.get(key, default)
            return getattr(item, key, default)

        async def _async_extract():
            sem = asyncio.Semaphore(max(1, concurrency))
            completed = 0
            extracted_relations = []

            async def _proc(idx: int, ch: Chunk):
                async with sem:
                    try:
                        raw = await knwler.extraction.extract_graph(ch.text[:2200], schema, config)
                        rels = raw.get("relations", []) if isinstance(raw, dict) else getattr(raw, "relations", [])
                        return ch, rels
                    except Exception as e:
                        logger.warning(f"Knwler extraction error on chunk {ch.chunk_id}: {e}")
                        h_props = self._grounded(self._heuristic_proposition_extract(ch), ch, "heuristic")
                        return ch, h_props

            tasks = [asyncio.create_task(_proc(i, c)) for i, c in enumerate(target_chunks)]
            for f in asyncio.as_completed(tasks):
                ch, rel_or_props = await f
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, f"Knwler extraction [{completed}/{total}] (p.{ch.page_number})")
                for r in rel_or_props:
                    if isinstance(r, Proposition):
                        self.add_proposition(r)
                    else:
                        extracted_relations.append((ch, r))

            return extracted_relations

        # snapshot: if Knwler fails part-way, the fallback must not re-add relations on top of the ones
        # Knwler already wrote (that duplicated edges)
        pre_graph = self.graph.copy()
        pre_lookups = (dict(self._node_lookup), dict(self._key_lookup))
        try:
            relations = asyncio.run(_async_extract())
            for ch, r in relations:
                s = self._canonicalize(str(_get_val(r, "source") or _get_val(r, "subject") or ""))
                p = self._canonicalize_predicate(str(_get_val(r, "type") or _get_val(r, "relation") or "relates to"))
                o = self._canonicalize(str(_get_val(r, "target") or _get_val(r, "object") or ""))
                desc = str(_get_val(r, "description") or _get_val(r, "statement") or f"{s} {p} {o}.")
                strength = self._safe_strength(_get_val(r, "strength", 8))
                if s and o and s.lower() != o.lower():
                    prop = Proposition(
                        subject=s, predicate=p, obj=o, statement=desc,
                        evidence=str(_get_val(r, "evidence") or ""),     # never the chunk prefix as fake evidence
                        strength=strength, chunk_id=ch.chunk_id, doc_name=ch.doc_name,
                        page_number=ch.page_number, section=ch.section_title
                    )
                    for gp in self._grounded([prop], ch, "knwler"):
                        self.add_proposition(gp)
        except Exception as ex:
            logger.error(f"Knwler orchestration failed: {ex}. Falling back to standard pipeline.")
            self.graph.clear()
            self.graph.add_nodes_from(pre_graph.nodes(data=True))
            self.graph.add_edges_from(pre_graph.edges(keys=True, data=True))
            self._node_lookup, self._key_lookup = dict(pre_lookups[0]), dict(pre_lookups[1])
            self._path_cache.clear()
            self._graph_version += 1
            self.build_from_chunks(
                chunks, model=model, max_chunks=max_chunks,
                concurrency=concurrency, progress_callback=progress_callback, base_url=base_url
            )

    def _extract_propositions_for_chunk(
        self,
        chunk: Chunk,
        model: str,
        known_entities: Optional[List[str]] = None,
        client: Optional[Any] = None
    ) -> List[Proposition]:
        active_client = client or self.llm_client
        if not active_client:
            self._count("chunks_heuristic_fallback")
            return self._grounded(self._heuristic_proposition_extract(chunk), chunk, "heuristic")

        clamped_text = chunk.text[:2200].strip()
        prompt = f"Paragraph (Source: {chunk.doc_name}, Page {chunk.page_number}):\n{clamped_text}\n"
        if known_entities:
            prompt += f"\nExisting concept names (reuse them exactly when the paragraph is about them): {', '.join(known_entities[:15])}\n"
        prompt += "\nJSON Triples:"

        try:
            res = active_client.generate(
                prompt=prompt,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                model=model,
                temperature=0.1,
                max_tokens=600,
            )
            data = self._parse_triples_json(res)
            if data:
                propositions = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    s = self._canonicalize(str(item.get("subject", "") or ""))
                    raw_p = str(item.get("predicate", "") or "")
                    o = self._canonicalize(str(item.get("object", "") or ""))
                    statement = str(item.get("statement", "") or "").strip()
                    # "argues" + "against echo chambers": the preposition belongs to the relation, not the concept
                    ow = o.split()
                    if len(ow) > 1 and ow[0].lower() in CONCEPT_PREPOSITIONS:
                        raw_p = f"{raw_p} {ow[0].lower()}".strip()
                        o = self._canonicalize(" ".join(ow[1:]))
                    sw = s.split()
                    if len(sw) > 1 and sw[0].lower() in CONCEPT_PREPOSITIONS:
                        s = self._canonicalize(" ".join(sw[1:]))
                    p, swap = self._canonical_relation(raw_p, statement, o)
                    if swap:
                        s, o = o, s
                    if s and p and o and s.lower() != o.lower():
                        if self._is_valid_entity(s) and self._is_valid_entity(o):
                            statement = statement or f"{s} {p} {o}."
                            evidence = str(item.get("evidence", "") or "").strip()   # checked against the chunk below
                            strength = self._safe_strength(item.get("strength", 8))
                            propositions.append(Proposition(
                                subject=s, predicate=p, obj=o, statement=statement,
                                evidence=evidence, strength=strength, chunk_id=chunk.chunk_id,
                                doc_name=chunk.doc_name, page_number=chunk.page_number, section=chunk.section_title,
                                raw_predicate=raw_p,
                            ))
                if propositions:
                    self._count("chunks_llm")
                    return self._grounded(propositions, chunk, "llm")
        except Exception as e:
            logger.warning(f"Proposition extraction failed on chunk {chunk.chunk_id}: {e}")

        self._count("chunks_heuristic_fallback")
        return self._grounded(self._heuristic_proposition_extract(chunk), chunk, "heuristic")

    def _parse_triples_json(self, text: str) -> Optional[List[Dict[str, Any]]]:
        if not text:
            return None
        # Remove think and reasoning tags
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<reasoning>.*?</reasoning>', '', clean, flags=re.DOTALL | re.IGNORECASE).strip()
        # Remove markdown code fence markers
        clean = re.sub(r'```(?:json)?', '', clean, flags=re.IGNORECASE).strip('` \t\r\n')

        # 1. Standard array format [...]
        arr_start = clean.find('[')
        if arr_start != -1:
            arr_end = clean.rfind(']')
            if arr_end > arr_start:
                candidate = clean[arr_start:arr_end + 1]
                candidate = re.sub(r',\s*([\]}])', r'\1', candidate)
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    try:
                        parsed = ast.literal_eval(candidate)
                        if isinstance(parsed, list):
                            return parsed
                    except Exception:
                        pass

            # 2. Recovery for truncated array (cut off by token limit)
            last_brace = clean.rfind('}')
            if last_brace > arr_start:
                candidate = clean[arr_start:last_brace + 1].strip()
                candidate = candidate.rstrip(',') + ']'
                candidate = re.sub(r',\s*([\]}])', r'\1', candidate)
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        logger.info(f"Recovered {len(parsed)} triples from truncated JSON stream.")
                        return parsed
                except Exception:
                    try:
                        parsed = ast.literal_eval(candidate)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            return parsed
                    except Exception:
                        pass

        # 3. Handle single object format { "subject": ... }
        obj_start = clean.find('{')
        if obj_start != -1:
            obj_end = clean.rfind('}')
            if obj_end > obj_start:
                candidate = clean[obj_start:obj_end + 1]
                candidate = re.sub(r',\s*}', '}', candidate)
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "subject" in parsed:
                        return [parsed]
                except Exception:
                    try:
                        parsed = ast.literal_eval(candidate)
                        if isinstance(parsed, dict) and "subject" in parsed:
                            return [parsed]
                    except Exception:
                        pass

        # 4. Fallback: Check if valid JSON array exists inside reasoning block if content was empty
        if "<think>" in text or "<reasoning>" in text:
            raw_blocks = re.findall(r'<(?:think|reasoning)>(.*?)</(?:think|reasoning)>', text, flags=re.DOTALL | re.IGNORECASE)
            for blk in raw_blocks:
                b_start = blk.find('[')
                b_end = blk.rfind(']')
                if b_start != -1 and b_end > b_start:
                    candidate = blk[b_start:b_end + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            return parsed
                    except Exception:
                        pass

        return None

    def _canonicalize_predicate(self, pred: str) -> str:
        """
        Canonical predicate, keeping polarity: "does not require" -> "not requires",
        "lacks" -> "excludes". See nanogentzen/relation_logic.py for what each one means logically.
        """
        raw = pred.strip().lower()
        negated = is_negated_phrase(raw)
        p = NEGATION_RE.sub(" ", raw)
        p = re.sub(r"\b(?:does|do|did|has|have|had)\b", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        p = re.sub(r'^(?:can|could|might|may|would|will|must|should|is|are|was|were|be|been|being)\s+', '', p)
        neg = "not " if negated else ""
        if EXCLUSION_RE.search(p):
            return neg + "excludes"
        core_verbs = [
            ("causes", r'\b(?:causes?|caused|produces?|triggers?|induces?)\b'),
            ("contrasts with", r'\b(?:differs?\s+from|contrasts?\s+with|distinguishes?)\b'),
            ("replaces", r'\b(?:replaces?|supersedes?)\b'),
            ("enables", r'\b(?:enables?|facilitates?|allows?|permits?)\b'),
            ("requires", r'\b(?:requires?|demands?|necessitates?)\b'),
            ("proves", r'\b(?:proves?|demonstrates?|shows?)\b'),
            ("implies", r'\b(?:impl(?:y|ies|ied)|entails?|leads?\s+to|results?\s+in)\b'),
            ("constrains", r'\b(?:constrains?|limits?|restricts?)\b'),
            ("formulates", r'\b(?:formulates?|devises?|develops?|introduces?|proposes?)\b'),
            ("is defined as", r'\b(?:defined\s+as|refers?\s+to|defined\s+in\s+terms\s+of)\b'),
            # anchored: "involves a kind of misdirection" is not "is a kind of"
            ("is a kind of", r'^(?:an?\s+)?(?:kinds?|types?|forms?)\s+of\b'),
            ("consists of", r'\b(?:consists?\s+of|composed\s+of)\b'),
            ("depends on", r'\b(?:depends?\s+on|rel(?:y|ies)\s+on)\b'),
            ("implements", r'\b(?:implements?|executes?|realizes?)\b'),
            ("models", r'\b(?:models?|simulates?|represents?)\b'),
            ("characterizes", r'\b(?:characterizes?|characterised?)\b'),
        ]
        for canon, pattern in core_verbs:
            if re.search(pattern, p):
                return neg + canon
        direct, swap = read_relation(p)
        if direct and not swap:
            return neg + direct
        # free phrase: keep the verb and its particles ("argues against", "unaware of"), drop object words
        # ("advance practical" -> "advance", "suggest advertising" -> "suggest", "believed the" -> "believed")
        words = [w for w in p.split() if w not in ("the", "a", "an")]
        if not words:
            return neg + "relates to"
        particles = {"on", "of", "for", "to", "in", "with", "from", "against", "about", "by", "as", "into", "upon",
                     "over", "between", "through", "out", "up", "down", "off", "away", "beyond", "across", "around",
                     "among", "within", "without", "under", "behind", "toward", "towards", "no", "none"}
        base = [words[0]]
        for w in words[1:3]:
            if w in particles:
                base.append(w)
            else:
                break
        return neg + " ".join(base)

    def _canonical_relation(self, pred: str, statement: str = "", obj: str = "") -> Tuple[str, bool]:
        """
        (canonical predicate, swap subject/object). Handles what the verb table cannot:
          "is necessary for" / "is a prerequisite for" -> requires, swapped ("A is necessary for B" = B requires A)
          "is a Y" (statement "X is a Y")            -> is a kind of
        """
        raw = (pred or "").strip().lower()
        negated = is_negated_phrase(raw)
        canon, swap = read_relation(NEGATION_RE.sub(" ", raw))
        if canon and swap:
            return ("not " if negated else "") + canon, True
        p = self._canonicalize_predicate(pred)
        if p in ("is", "are", "be", "is a", "is an") and statement:
            p = copula_relation(statement, obj) or p
        return p, False

    MAX_CONCEPT_WORDS = 6

    def _canonicalize(self, name: str) -> str:
        """
        Display name of a concept. The phrase is never cut mid-phrase (older builds kept the first 4 words,
        producing nodes such as "Respect For Practical And"): see core/concepts.py canonical_concept().
        """
        clean = canonical_concept(name, max_words=self.MAX_CONCEPT_WORDS)
        words = clean.split()
        if not words:
            return ""
        # the head noun is singularized: the last word, or the word before "of" ("Rights Of Access")
        head = next((i - 1 for i, w in enumerate(words) if i > 0 and w.lower() in ("of", "for", "to")), len(words) - 1)
        words[head] = self._singular(words[head])
        return " ".join(self._case(w) for w in words)

    _NO_SINGULAR = {
        "ethics", "physics", "mathematics", "economics", "politics", "linguistics", "statistics", "semantics",
        "species", "series", "news", "means", "lens", "chaos", "ethos", "pathos", "logos", "cosmos", "bias",
        "gas", "corpus", "status", "consensus", "apparatus", "axis", "basis", "thesis", "crisis",
    }
    _IRREGULAR = {"analyses": "analysis", "hypotheses": "hypothesis", "theses": "thesis", "criteria": "criterion",
                  "phenomena": "phenomenon", "data": "data", "bases": "basis", "crises": "crisis", "axes": "axis",
                  "indices": "index", "matrices": "matrix", "vertices": "vertex"}

    @classmethod
    def _singular(cls, w: str) -> str:
        low = w.lower()
        if low in cls._IRREGULAR:
            rep = cls._IRREGULAR[low]
            return rep.upper() if w.isupper() else (rep.capitalize() if w[:1].isupper() else rep)
        if w.isupper() or low in cls._NO_SINGULAR or len(w) <= 3 or low.endswith(("ss", "us", "is", "ics", "sis")):
            return w
        if low.endswith("ies") and len(w) > 4:
            return w[:-3] + ("Y" if w.isupper() else "y")
        if low.endswith(("sses", "xes", "ches", "shes", "zes")):
            return w[:-2]                            # Processes -> Process, Approaches -> Approach
        if low.endswith("s"):
            return w[:-1]
        return w

    @staticmethod
    def _case(w: str) -> str:
        """Title case, but keep acronyms (LLM, DNA) and mixed-case names (GraphQL, iPhone)."""
        if (len(w) > 1 and w.isupper()) or any(c.isupper() for c in w[1:]):
            return w
        return w[:1].upper() + w[1:]

    def _resolve_to_existing_node(self, name: str) -> str:
        if not name:
            return ""
        norm_key = re.sub(r'[^a-z0-9]', '', name.lower())
        if norm_key in self._node_lookup:
            return self._node_lookup[norm_key]
        ck = concept_key(name)            # word order / plural / possessive variants of the same concept
        if ck and ck in self._key_lookup and self.graph.has_node(self._key_lookup[ck]):
            self._node_lookup[norm_key] = self._key_lookup[ck]
            return self._key_lookup[ck]

        for existing in self.graph.nodes():
            exist_key = re.sub(r'[^a-z0-9]', '', existing.lower())
            if norm_key == exist_key:
                self._node_lookup[norm_key] = existing
                return existing
            if len(norm_key) > 4 and len(exist_key) > 4:
                if norm_key.rstrip('s') == exist_key.rstrip('s'):
                    self._node_lookup[norm_key] = existing
                    return existing

        self._node_lookup[norm_key] = name
        return name

    # concept issues that make a name unusable as a node (clauses / generic words are kept, with low quality)
    REJECT_CONCEPT_ISSUES = {"empty", "truncated", "pronoun", "numeric", "citation"}

    def _is_valid_entity(self, name: str) -> bool:
        if not name or len(name) < 2 or len(name) > 70:
            return False
        words = name.split()
        if len(words) > self.MAX_CONCEPT_WORDS:
            return False
        if self.REJECT_CONCEPT_ISSUES & set(concept_issues(name, self.MAX_CONCEPT_WORDS)):
            return False
        invalid_first = {"in", "on", "at", "by", "for", "with", "from", "of", "to", "as", "the", "a", "an", "and", "or"}
        if words[0].lower() in invalid_first:
            return False
        return True

    def add_proposition(self, prop: Proposition):
        if not prop.subject or not prop.obj:
            return
        sub = self._resolve_to_existing_node(prop.subject)
        obj = self._resolve_to_existing_node(prop.obj)

        if not sub or not obj or sub.lower() == obj.lower():
            return
        if not self._is_valid_entity(sub) or not self._is_valid_entity(obj):
            return

        for n in (sub, obj):
            if not self.graph.has_node(n):
                iss = concept_issues(n, self.MAX_CONCEPT_WORDS)
                self.graph.add_node(n, label=n, doc_name=prop.doc_name, page_number=prop.page_number,
                                    concept_issues=iss, concept_quality=concept_quality(n, 1, iss))
                self._key_lookup.setdefault(concept_key(n), n)

        self.graph.add_edge(
            sub, obj,
            predicate=prop.predicate,
            statement=prop.statement,
            evidence=prop.evidence,
            strength=prop.strength,
            chunk_id=prop.chunk_id,
            doc_name=prop.doc_name,
            page_number=prop.page_number,
            section=prop.section,
            evidence_status=prop.evidence_status,
            extractor=prop.extractor or "manual",
            raw_predicate=prop.raw_predicate or "",
            **({"claimed_predicate": prop.claimed_predicate} if prop.claimed_predicate else {}),
            **({"inference_check": prop.inference_check} if prop.inference_check else {}),
        )
        self._graph_version += 1
        self._path_cache.clear()

    def _heuristic_proposition_extract(self, chunk: Chunk) -> List[Proposition]:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk.text) if len(s.strip()) > 25]
        propositions = []
        patterns = [
            ("excludes", EXCLUSION_RE.pattern),
            ("implies", r'\b(?:impl(?:y|ies)|entails?|leads? to|results? in)\b'),
            ("requires", r'\b(?:requires?|demands?|necessitates?)\b'),
            ("enables", r'\b(?:enables?|facilitates?|allows?)\b'),
            ("characterizes", r'\b(?:characteri[sz]es?|defines?)\b'),
            ("contrasts with", r'\b(?:contrasts? with|differs? from|distinguishes?)\b'),
            ("is a property of", r'\b(?:is a property of|is an attribute of)\b'),
            ("applies to", r'\b(?:appl(?:y|ies) to|pertains? to)\b'),
            ("depends on", r'\b(?:depends? on|rel(?:y|ies) on)\b'),
            ("implements", r'\b(?:implements?|executes?|reali[sz]es?)\b'),
        ]

        for sent in sentences:
            for pred, regex in patterns:
                m = re.search(regex, sent, re.IGNORECASE)
                if m:
                    raw_sub = sent[:m.start()].strip()
                    raw_obj = sent[m.end():].strip()
                    neg_tail = NEGATION_TAIL_RE.search(raw_sub)
                    if neg_tail:                      # "X does not require Y" -> not requires
                        raw_sub = raw_sub[:neg_tail.start()].strip()
                        pred = "not " + pred
                    sub_cand = re.split(r'[,;]|\b(?:which|that|because)\b', raw_sub)[-1].strip()
                    obj_cand = re.split(r'[,;]|\b(?:which|that|because)\b', raw_obj)[0].strip()

                    sub = self._canonicalize(sub_cand)
                    obj = self._canonicalize(obj_cand)

                    if sub and obj and sub.lower() != obj.lower() and self._is_valid_entity(sub) and self._is_valid_entity(obj):
                        propositions.append(Proposition(
                            subject=sub, predicate=pred, obj=obj,
                            statement=sent.strip(),
                            evidence=sent.strip(),
                            strength=8,
                            chunk_id=chunk.chunk_id,
                            doc_name=chunk.doc_name,
                            page_number=chunk.page_number,
                            section=chunk.section_title
                        ))
                        break
            if len(propositions) >= 5:
                break
        return propositions

    @staticmethod
    def _pick_edge(edge_dict: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
        """Among parallel edges u->v, prefer one that licenses an inference (excludes / entails)."""
        edges = list(edge_dict.values())
        rank = {"excludes": 0, "entails": 1, "none": 2}
        return min(edges, key=lambda e: (rank[relation_logic(e.get("predicate"))], -int(e.get("strength", 0) or 0)))

    # ------------------------------------------------------------------ document topics
    def doc_topic_overlap(self) -> Dict[Tuple[str, str], float]:
        """
        Topic overlap of every document pair: overlap coefficient |A ∩ B| / min(|A|, |B|) of the specific
        (non-generic) concepts each document contributes. Same-topic papers share many concepts; two unrelated
        papers that happen to mention one common term score close to 0. Cached per graph version.
        """
        cache = getattr(self, "_topic_cache", None)
        if cache and cache[0] == self._graph_version:
            return cache[1]
        ents: Dict[str, Set[str]] = {}
        for u, v, d in self.graph.edges(data=True):
            doc = d.get("doc_name") or ""
            for n in (u, v):
                if str(n).lower().strip() not in GENERIC_BRIDGE_CONCEPTS:
                    ents.setdefault(doc, set()).add(n)
        docs = sorted(ents)
        out: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(docs):
            for b in docs[i + 1:]:
                m = min(len(ents[a]), len(ents[b])) or 1
                out[(a, b)] = out[(b, a)] = round(len(ents[a] & ents[b]) / m, 4)
        self._topic_cache = (self._graph_version, out)
        return out

    # ------------------------------------------------------------------ path search
    def _neighbors(self, n: str, undirected: bool) -> List[Tuple[str, bool]]:
        """(neighbor, walked_forward) pairs, highest-degree neighbors first (deterministic)."""
        out = [(v, True) for v in self.graph.successors(n)]
        if undirected:
            out += [(u, False) for u in self.graph.predecessors(n) if not self.graph.has_edge(n, u)]
        seen, uniq = set(), []
        for v, fwd in out:
            if v not in seen:
                seen.add(v)
                uniq.append((v, fwd))
        uniq.sort(key=lambda x: (-self.graph.degree(x[0]), str(x[0])))
        return uniq

    def _enumerate_paths(self, sources: List[str], min_hops: int, max_hops: int, undirected: bool,
                         max_candidates: int) -> List[Tuple[List[str], List[bool]]]:
        """Simple paths of min..max hops from each source (one DFS per source, not one per source/target pair)."""
        found: List[Tuple[List[str], List[bool]]] = []

        def dfs(path: List[str], dirs: List[bool]):
            if len(found) >= max_candidates:
                return
            hops = len(path) - 1
            if hops >= min_hops and str(path[-1]).lower().strip() not in NON_PROPOSITIONAL_TARGETS:
                found.append((list(path), list(dirs)))
            if hops >= max_hops:
                return
            for v, fwd in self._neighbors(path[-1], undirected):
                if v in path:
                    continue
                path.append(v)
                dirs.append(fwd)
                dfs(path, dirs)
                path.pop()
                dirs.pop()

        for src in sources:
            dfs([src], [])
            if len(found) >= max_candidates:
                logger.info(f"Path search stopped at {max_candidates} candidates")
                break
        return found

    @staticmethod
    def _verify_path(item: Dict[str, Any], active_architect: Any) -> None:
        """Decide one path with nanoGentzen and store the verdict and logic tree on it (in place)."""
        try:
            tree_res = active_architect.generate_best_logic_tree(
                premises=[item["composite_claim"]], conclusion=item["target"],
                path_nodes=item["path_nodes"], path_repr=item["path_repr"])
            item.update(
                gentzen_verified=tree_res.get("is_verified", False), logic_tree_res=tree_res,
                logic_tree_text=tree_res.get("logic_tree_text", ""), gentzen_sequent=tree_res.get("sequent", ""),
                curry_howard_term=tree_res.get("curry_howard_term", ""), proof_tree=tree_res.get("proof_tree"),
                provability_score=tree_res.get("provability_score", 1.0),
                epistemic_status=tree_res.get("epistemic_status",
                                              "PROVEN_TRUE" if tree_res.get("is_verified") else "INTUITIONISTIC_UNDECIDED"),
                relevance=tree_res.get("relevance", {}), tree_depth=tree_res.get("depth", item["hops"]),
                node_count=tree_res.get("node_count", len(item["path_nodes"])))
        except Exception as e:
            logger.warning(f"nanoGentzen verification failed on path {item['path_repr']}: {e}")
            item["gentzen_verified"] = False

    def _inference_neighbors(self, n: str) -> List[Tuple[str, bool]]:
        """
        Steps that license an inference from n: forward along an entailment or exclusion edge, or backward
        against an exclusion edge (exclusion is symmetric). Deterministic order.
        """
        out = []
        for v in self.graph.successors(n):
            if any(relation_logic(e.get("predicate")) != "none" for e in self.graph.get_edge_data(n, v).values()):
                out.append((v, True))
        for u in self.graph.predecessors(n):
            if u in self.graph.successors(n):
                continue
            if any(relation_logic(e.get("predicate")) == "excludes" for e in self.graph.get_edge_data(u, n).values()):
                out.append((u, False))
        out.sort(key=lambda x: (not x[1], str(x[0])))
        return out

    def _step_excludes(self, u: str, v: str, fwd: bool) -> bool:
        edata = self.graph.get_edge_data(u, v) if fwd else self.graph.get_edge_data(v, u)
        return bool(edata) and relation_logic(self._pick_edge(edata).get("predicate")) == "excludes"

    def _quality(self, n: str) -> float:
        a = self.graph.nodes[n]
        q = a.get("concept_quality")
        return concept_quality(str(n), a.get("mentions", 1)) if q is None else float(q)

    TAXONOMY_POLICIES = ("any", "last", "none")

    def _is_taxonomy_step(self, u: str, v: str, fwd: bool) -> bool:
        edata = self.graph.get_edge_data(u, v) if fwd else self.graph.get_edge_data(v, u)
        return bool(edata) and str(self._pick_edge(edata).get("predicate", "")).strip().lower() == "is a kind of"

    def _enumerate_inference_paths(self, min_hops: int, max_hops: int, max_candidates: int,
                                   min_concept_quality: float = 0.0,
                                   taxonomy_policy: str = "any") -> List[Tuple[List[str], List[bool]]]:
        """
        Logic-first path search: simple paths over inference edges only, from EVERY node that has one (not just
        the highest-degree nodes). Nothing follows from an excluded concept, so a path stops after an exclusion.
        These are exactly the paths nanoGentzen can prove; the general search finds them only by chance.
        taxonomy_policy (see find_multihop_paths) prunes "is a kind of" steps during the search, so the
        candidate budget is not spent on paths that would be dropped.
        """
        found: List[Tuple[List[str], List[bool]]] = []
        ok = (lambda n: self._quality(n) >= min_concept_quality) if min_concept_quality > 0 else (lambda n: True)
        sources = sorted((n for n in self.graph.nodes() if ok(n) and self._inference_neighbors(n)), key=str)

        def dfs(path: List[str], dirs: List[bool]):
            if len(found) >= max_candidates:
                return
            hops = len(path) - 1
            if hops >= min_hops and str(path[-1]).lower().strip() not in NON_PROPOSITIONAL_TARGETS:
                found.append((list(path), list(dirs)))
            if hops >= max_hops or (hops and self._step_excludes(path[-2], path[-1], dirs[-1])):
                return
            if hops and taxonomy_policy == "last" and self._is_taxonomy_step(path[-2], path[-1], dirs[-1]):
                return                # a taxonomy step may only end a path
            for v, fwd in self._inference_neighbors(path[-1]):
                if v in path or not ok(v) or (not fwd and hops + 1 < min_hops):
                    continue          # a backward exclusion can only be the last hop
                if taxonomy_policy != "any" and self._is_taxonomy_step(path[-1], v, fwd) and \
                        (taxonomy_policy == "none" or hops + 1 < min_hops):
                    continue
                path.append(v)
                dirs.append(fwd)
                dfs(path, dirs)
                path.pop()
                dirs.pop()

        for src in sources:
            dfs([src], [])
            if len(found) >= max_candidates:
                logger.info(f"Inference path search stopped at {max_candidates} candidates")
                break
        return found

    def find_multihop_paths(
        self,
        max_hops: int = 3,
        min_hops: int = 2,
        limit: Optional[int] = 50,
        intra_document_only: bool = True,
        scope_doc_name: Optional[str] = None,
        cross_document_only: bool = False,
        architect: Optional[Any] = None,
        verify_with_gentzen: bool = True,
        require_entailment: bool = False,
        min_topic_overlap: float = 0.05,
        max_sources: int = 40,
        max_candidates: int = 5000,
        logic_first: bool = True,
        verified_only: bool = False,
        min_concept_quality: float = 0.5,
        taxonomy_policy: str = "any",
        max_paths_per_hub: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Multi-hop paths with provenance, ranked, verified only for the paths returned.

        Each item carries `edges` (direction + logical reading of every hop), `logic_basis`
        (kg_entailment_chain / kg_relational / kg_unlabelled), evidence status per hop, and for cross-document
        paths `doc_topic_overlap` (lowest topic overlap between consecutive documents on the path).

        Cross-document paths are kept only when every document hand-over joins documents that share a topic
        (doc_topic_overlap >= min_topic_overlap, see doc_topic_overlap()) and the bridging concept is not a
        generic term such as "Model" or "Framework". require_entailment=True returns only entailment chains.
        logic_first=True first enumerates every chain of inference edges from every node (the general search
        only starts at the max_sources highest-degree nodes and finds such chains by chance).
        min_concept_quality skips paths through nodes that are not meaningful concepts (core/concepts.py:
        truncated phrases, clauses, pronoun fragments, generic words); 0 disables the check.
        verified_only=True keeps only paths nanoGentzen proves (PROVEN_TRUE): every entailment chain is verified
        before `limit` is applied, so the limit counts verified paths.
        taxonomy_policy decides where an "is a kind of" step may stand:
            "any"   anywhere (default; relational datasets such as Jev)
            "last"  at most one, as the last hop: "A requires B, B is a kind of C" concludes about C. A taxonomy
                    step first ("Artificial Consciousness is a kind of Consciousness, Consciousness requires X")
                    copies one claim about a hub onto every subtype, producing near-identical paths that turn a
                    theory-specific claim into a claim about each kind
            "none"  no taxonomy steps
        max_paths_per_hub caps how many returned paths may pass THROUGH one concept (as an interior node), so one
        hub cannot fill the dataset; paths are taken in rank order. None disables the cap.
        Ranking: entailment chains, supported evidence, more distinct source chunks, higher edge strength.
        """
        if taxonomy_policy not in self.TAXONOMY_POLICIES:
            raise ValueError(f"taxonomy_policy must be one of {self.TAXONOMY_POLICIES}, not {taxonomy_policy!r}")
        from nanogentzen.cot.parser_cot import path_logic_report

        if cross_document_only:
            intra_document_only = False
        if verified_only:
            verify_with_gentzen = True
            require_entailment = True     # only entailment chains can be proven: cheap prefilter

        active_architect = architect or self.architect
        if verify_with_gentzen and active_architect is None:
            try:
                from nanogentzen.logical_architect import LogicalArchitect
                active_architect = LogicalArchitect()
                self.architect = active_architect
            except Exception as e:
                logger.warning(f"Could not initialize LogicalArchitect for path verification: {e}")
                active_architect = None

        cache_key = (self._graph_version, self.graph.number_of_nodes(), self.graph.number_of_edges(),
                     max_hops, min_hops, limit, intra_document_only, scope_doc_name, cross_document_only,
                     verify_with_gentzen and (active_architect is not None), require_entailment,
                     round(float(min_topic_overlap), 4), max_sources, max_candidates, logic_first, verified_only,
                     round(float(min_concept_quality), 3), taxonomy_policy, max_paths_per_hub)
        if cache_key in self._path_cache:
            return list(self._path_cache[cache_key])   # a copy: callers sorting/filtering must not edit the cache
        if self.graph.number_of_nodes() < 3:
            self._path_cache[cache_key] = []
            return []

        topic = self.doc_topic_overlap()
        nodes_by_degree = sorted(self.graph.nodes(), key=lambda n: (-self.graph.degree(n), str(n)))
        sources = nodes_by_degree[:max_sources]

        def build(path: List[str], dirs: List[bool]) -> Optional[Dict[str, Any]]:
            if min_concept_quality > 0 and any(self._quality(n) < min_concept_quality for n in path):
                return None
            edge_details, chunk_ids, pages, props, edge_docs, strengths = [], [], set(), [], [], []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                fwd = dirs[i]
                edata_dict = self.graph.get_edge_data(u, v) if fwd else self.graph.get_edge_data(v, u)
                if not edata_dict:
                    return None
                edata = self._pick_edge(edata_dict)
                pred = edata.get("predicate", "relates to")
                d_name = edata.get("doc_name", "")
                edge_details.append(f"({u}) --[{pred}]--> ({v})" if fwd else f"({u}) --[{pred}] <-- ({v})")
                if edata.get("chunk_id"):
                    chunk_ids.append(edata["chunk_id"])
                pages.add(edata.get("page_number", 1))
                edge_docs.append(d_name)
                strengths.append(self._safe_strength(edata.get("strength", 8)))
                props.append({"predicate": str(pred).strip().lower(),
                              "statement": edata.get("statement", f"{u} {pred} {v}."), "evidence": edata.get("evidence", ""),
                              "page_number": edata.get("page_number", 1), "doc_name": d_name,
                              "evidence_status": edata.get("evidence_status", ""), "extractor": edata.get("extractor", "")})
            tax = [str(pp.get("predicate", "")) == "is a kind of" for pp in props]
            if taxonomy_policy == "none" and any(tax):
                return None
            if taxonomy_policy == "last" and (sum(tax) > 1 or any(tax[:-1])):
                return None
            distinct_docs = sorted({d for d in edge_docs if d})
            if scope_doc_name and any(d != scope_doc_name for d in distinct_docs):
                return None
            is_cross = len(distinct_docs) > 1
            if (intra_document_only and is_cross) or (cross_document_only and not is_cross):
                return None
            overlap = None
            if is_cross:
                overlaps = []
                for i in range(1, len(edge_docs)):
                    a, b = edge_docs[i - 1], edge_docs[i]
                    if a and b and a != b:
                        if str(path[i]).lower().strip() in GENERIC_BRIDGE_CONCEPTS:
                            return None                      # generic bridge: not a shared topic
                        overlaps.append(topic.get((a, b), 0.0))
                overlap = min(overlaps) if overlaps else 0.0
                if overlap < min_topic_overlap:
                    return None
            path_repr = " -> ".join(edge_details)
            rep = path_logic_report(path, path_repr)
            if require_entailment and rep["logic_basis"] != "kg_entailment_chain":
                return None
            stmts = [pp["statement"].rstrip(".") for pp in props]
            comp = (f"Because {', and '.join(stmts[:-1])}, {stmts[-1][:1].lower() + stmts[-1][1:]}."
                    if len(stmts) > 1 else f"{stmts[0]}.")
            ev_ok = all(pp.get("evidence_status") in EVIDENCE_SUPPORTED for pp in props)
            score = (2.0 * (rep["logic_basis"] == "kg_entailment_chain") + 1.0 * ev_ok
                     + 0.5 * len(set(chunk_ids)) / max(1, len(props)) + 0.5 * (sum(strengths) / len(strengths)) / 10.0)
            return {
                "source": path[0], "target": path[-1], "hops": len(path) - 1, "path_nodes": list(path),
                "path_repr": path_repr, "provenance_pages": sorted(pages), "provenance_docs": distinct_docs,
                "doc_name": distinct_docs[0] if distinct_docs else "unknown", "is_cross_document": is_cross,
                "doc_topic_overlap": overlap, "chunk_ids": chunk_ids, "propositions": props, "composite_claim": comp,
                "gentzen_verified": False, "edges": rep["edges"], "logic_basis": rep["logic_basis"],
                "logic_broken_at": rep["broken_at"], "evidence_statuses": [pp.get("evidence_status", "") for pp in props],
                "evidence_supported": ev_ok, "rank_score": round(score, 4), "taxonomy_hops": sum(tax),
            }

        intra: List[Dict[str, Any]] = []
        cross: List[Dict[str, Any]] = []
        seen = set()
        def candidates():
            if logic_first:
                yield "logic", self._enumerate_inference_paths(min_hops, max_hops, max_candidates, min_concept_quality,
                                                           taxonomy_policy)
            if require_entailment and logic_first:
                return                    # the general search cannot add entailment chains the logic search missed
            yield "directed", self._enumerate_paths(sources, min_hops, max_hops, False, max_candidates)
            if len(intra) + len(cross) < 15:   # undirected only when the graph is sparse
                yield "undirected", self._enumerate_paths(sources, min_hops, max_hops, True, max_candidates)

        for _kind, cand in candidates():
            for path, dirs in cand:
                key = tuple(path)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    item = build(path, dirs)
                except Exception as e:                   # one malformed edge must not hide the others, but say so
                    logger.warning(f"Skipping path {' -> '.join(map(str, path))}: {e}")
                    continue
                if item is not None:
                    (cross if item["is_cross_document"] else intra).append(item)

        rank = lambda it: (-it["rank_score"], it["hops"], it["path_repr"])  # noqa: E731
        intra.sort(key=rank)
        cross.sort(key=rank)
        if verified_only:
            if active_architect is None:
                logger.warning("verified_only: no nanoGentzen architect available, returning no paths")
                self._path_cache[cache_key] = []
                return []
            for item in intra + cross:
                self._verify_path(item, active_architect)
            intra = [it for it in intra if it.get("gentzen_verified")]
            cross = [it for it in cross if it.get("gentzen_verified")]
        if max_paths_per_hub:
            intra, cross = self._cap_hubs(intra, max_paths_per_hub), self._cap_hubs(cross, max_paths_per_hub)
        if cross_document_only:
            selected = cross[:limit] if limit else cross
        elif intra_document_only:
            selected = intra[:limit] if limit else intra
        elif not limit:
            selected = cross + intra
        else:
            selected, ci, ii = [], 0, 0
            while len(selected) < limit and (ci < len(cross) or ii < len(intra)):
                if ci < len(cross):
                    selected.append(cross[ci]); ci += 1
                if len(selected) < limit and ii < len(intra):
                    selected.append(intra[ii]); ii += 1

        if verify_with_gentzen and active_architect is not None:
            for item in selected:
                if "logic_tree_res" not in item:
                    self._verify_path(item, active_architect)

        self._path_cache[cache_key] = selected
        return list(selected)

    @staticmethod
    def _cap_hubs(items: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
        """Keep paths in order while no interior concept has been passed through `cap` times already."""
        used: Dict[str, int] = {}
        out = []
        for it in items:
            inner = it["path_nodes"][1:-1]
            if any(used.get(n, 0) >= cap for n in inner):
                continue
            for n in inner:
                used[n] = used.get(n, 0) + 1
            out.append(it)
        return out

    # ------------------------------------------------------------------ concept quality & repair
    CANONICAL_PREDICATES = {
        "requires", "implies", "causes", "excludes", "is a kind of", "is defined as", "depends on", "enables",
        "constrains", "consists of", "contrasts with", "models", "formulates", "replaces", "proves", "implements",
        "characterizes", "undermines", "applies to", "relates to",
    }

    def _score_node(self, n: str, mentions: Optional[int] = None, evidence: Optional[List[str]] = None) -> None:
        a = self.graph.nodes[n]
        iss = concept_issues(n, self.MAX_CONCEPT_WORDS)
        if mentions is None:
            mentions = len({d.get("chunk_id") for _, _, d in self.graph.in_edges(n, data=True)}
                           | {d.get("chunk_id") for _, _, d in self.graph.out_edges(n, data=True)} - {None, ""})
        a["concept_issues"] = iss
        a["mentions"] = int(mentions)
        a["concept_quality"] = concept_quality(n, mentions, iss)
        if evidence is not None:
            a["named_entity"] = is_named_entity(n, evidence)

    def _merge_node(self, src: str, dst: str) -> None:
        """Move every edge of src onto dst (self-loops dropped), keep src as an alias of dst, remove src."""
        if src == dst or not self.graph.has_node(src):
            return
        for u, v, d in list(self.graph.in_edges(src, data=True)):
            if u != dst:
                self.graph.add_edge(u, dst, **d)
        for u, v, d in list(self.graph.out_edges(src, data=True)):
            if v != dst:
                self.graph.add_edge(dst, v, **d)
        aliases = set(self.graph.nodes[dst].get("aliases", [])) | set(self.graph.nodes[src].get("aliases", [])) | {src}
        self.graph.nodes[dst]["aliases"] = sorted(aliases - {dst})
        self.graph.remove_node(src)

    def _rename_node(self, old: str, new: str) -> str:
        if not new or new == old:
            return old
        if self.graph.has_node(new):
            self._merge_node(old, new)
        else:
            nx.relabel_nodes(self.graph, {old: new}, copy=False)
            self.graph.nodes[new]["label"] = new
            self.graph.nodes[new]["aliases"] = sorted(set(self.graph.nodes[new].get("aliases", [])) | {old})
        return new

    def _reindex(self) -> None:
        self._node_lookup = {re.sub(r"[^a-z0-9]", "", str(n).lower()): n for n in self.graph.nodes()}
        self._key_lookup = {}
        for n in self.graph.nodes():
            self._key_lookup.setdefault(concept_key(str(n)), n)
        self._graph_version += 1
        self._path_cache.clear()

    def repair_concepts(self, chunks: Optional[List[Chunk]] = None, add_taxonomy: bool = True,
                        reread_predicates: bool = True, min_taxonomy_quality: float = 0.6,
                        embed_fn: Optional[Any] = None, sense_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Make the nodes of an existing graph meaningful concepts, without re-extracting (see core/concepts.py):

          1. predicates re-read from each edge's statement: object words dropped from free predicates
             ("advance practical" -> "advance"), inference phrasings recognised ("is based on" -> depends on,
             "is necessary for" -> requires with subject and object swapped)
          2. truncated names completed from the edge evidence ("Principle Of Informed Cognitive" ->
             "Principle Of Informed Cognitive Agency"); a leading preposition moves into the predicate
             ("argues" + "Against Policy" -> "argues against" + "Policy")
          3. variants of one concept merged (same concept_key: word order, plural, possessive), kept as aliases
          4. taxonomy edges from the names themselves: "Personal Autonomy" --[is a kind of]--> "Autonomy"
             (extractor "taxonomy", evidence_status "lexical"). With embed_fn (texts -> vectors) each candidate is
             sense-checked: the two concepts, each embedded with the statements the text makes about it, must be
             similar (>= sense_threshold), which rejects homonyms such as "Marketing Agency" vs moral "Agency".
          5. every node scored: concept_issues, mentions (chunks that mention it), concept_quality, named_entity

        Returns a report with before / after counts. Path discovery skips nodes below min_concept_quality.
        """
        G = self.graph
        rep: Dict[str, Any] = {"nodes_before": G.number_of_nodes(), "edges_before": G.number_of_edges()}
        inf_before = sum(relation_logic(d.get("predicate")) != "none" for _, _, d in G.edges(data=True))

        # 1. predicates (before renaming: statements still contain the old names). Inference upgrades only
        #    through core.concepts.inference_cue, which refuses modal, contrastive or indirect phrasings: a wrong
        #    inference edge would create false "verified" paths.
        upgraded, reversed_n, cleaned = 0, 0, 0
        if reread_predicates:
            for u, v, k, d in list(G.edges(keys=True, data=True)):
                old = d.get("predicate", "") or ""
                if relation_logic(old) != "none" or old.startswith("not ") or d.get("claimed_predicate"):
                    continue     # already an inference, negated, or demoted by the evidence gate: decided
                cue, raw_used = None, ""
                mid = relation_from_statement(d.get("statement", ""), u, v) or ""
                mid_toks = set(re.findall(r"[a-z']+", mid.lower()))
                if mid_toks & (CONCEPT_MODALS | CONCEPT_CONTRAST | {"not", "never", "no", "cannot"}) or "n't" in mid:
                    continue     # the text hedges or negates ("may be used to violate"): no inference, whatever the LLM wrote
                for raw in (d.get("raw_predicate") or old, mid):
                    if raw and (cue := inference_cue(raw)):
                        raw_used = raw
                        break
                if cue is None and old in ("is", "are", "be", "a", "an", "is a", "is an") and \
                        not concept_issues(u, self.MAX_CONCEPT_WORDS) and not concept_issues(v, self.MAX_CONCEPT_WORDS):
                    kind = copula_relation(d.get("statement", ""), v)
                    if kind:
                        cue, raw_used = (kind, False), old
                if cue is not None:
                    new, swap = cue
                    d2 = dict(d, predicate=new, predicate_before=old, raw_predicate=raw_used)
                    G.remove_edge(u, v, k)
                    G.add_edge(*((v, u) if swap else (u, v)), **d2)
                    upgraded += 1
                    reversed_n += bool(swap)
                    continue
                # not an inference: only drop object words an older build left after a VERB ("advance practical",
                # "suggest advertising", "holds the"); noun predicates ("incorrect information") and idioms stay
                simple = self._canonicalize_predicate(old)
                ow = [w for w in old.split() if w not in ("the", "a", "an")]
                first = ow[0] if ow else ""
                verbish = bool(re.search(r"(?:s|ed)$", first)) and len(first) > 3 and "-" not in first and not re.search(
                    r"(?:ous|ness|ics|ss|al|ive)$", first)
                negator = len(ow) > 1 and ow[1] in ("no", "not", "never", "none", "nothing")
                trailing_article = len(old.split()) == 2 and old.split()[1] in ("the", "a", "an")
                idiom = old in ("gives rise", "takes place", "makes sense", "pays attention", "takes account",
                                "takes into", "gives way", "makes possible")
                if simple and simple != old and ow[:len(simple.split())] == simple.split() and \
                        (verbish or trailing_article) and not idiom and not negator:
                    d["predicate_before"], d["predicate"] = old, simple
                    cleaned += 1
        rep["predicates_cleaned"] = cleaned

        # 1b. evidence gate on every inference edge (extracted or just upgraded); taxonomy edges are lexical
        demoted, flipped = [], []
        for u, v, k, d in list(G.edges(keys=True, data=True)):
            if d.get("extractor") == "taxonomy":
                continue
            new, claimed, why, flip = self._gate_inference(d.get("predicate", ""),
                                                           d.get("evidence", "") or d.get("statement", ""), u, v)
            if claimed:
                d["claimed_predicate"], d["predicate"], d["inference_check"] = claimed, new, why
                demoted.append((u, claimed, v, why))
            elif why:
                d["inference_check"] = why
                if flip:
                    G.remove_edge(u, v, k)
                    G.add_edge(v, u, **dict(d, direction_flipped=True))
                    flipped.append((v, d.get("predicate"), u))
        rep["inference_demoted"] = demoted
        rep["inference_flipped"] = flipped
        rep["predicates_upgraded"] = upgraded
        rep["edges_reversed"] = reversed_n

        # 2. names
        renamed, unrepairable, prep_moved = {}, [], 0
        for n in list(G.nodes()):
            if not G.has_node(n):
                continue
            fixed_sg = fix_legacy_singular(n)
            if fixed_sg != n:
                renamed[n] = fixed_sg
                n = self._rename_node(n, fixed_sg)
            iss = concept_issues(n, self.MAX_CONCEPT_WORDS)
            ws = n.split()
            if ws and ws[0].lower() in CONCEPT_PREPOSITIONS and len(ws) > 1:
                prep = ws[0].lower()
                for u, _, d in G.in_edges(n, data=True):
                    if not d.get("predicate", "").endswith(" " + prep):
                        d["predicate_before"] = d.get("predicate_before", d.get("predicate", ""))
                        d["predicate"] = f"{d.get('predicate', 'relates to')} {prep}".strip()
                prep_moved += 1
                new = self._canonicalize(" ".join(ws[1:]))
                if new and self._is_valid_entity(new):
                    renamed[n] = new
                    self._rename_node(n, new)
                    continue
            if "truncated" in iss:
                texts = [t for _, _, d in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True))
                         for t in (d.get("evidence", ""), d.get("statement", ""))]
                fixed = repair_from_evidence(n, texts)
                new = self._canonicalize(fixed) if fixed else ""
                if new and self._is_valid_entity(new):
                    renamed[n] = new
                    self._rename_node(n, new)
                else:
                    unrepairable.append(n)
        rep["renamed"] = renamed
        rep["prepositions_moved"] = prep_moved
        rep["unrepairable"] = unrepairable

        # 3. variants
        groups: Dict[str, List[str]] = {}
        for n in G.nodes():
            ck = concept_key(n)
            if ck:
                groups.setdefault(ck, []).append(n)
        merged = {}
        for ck, members in groups.items():
            if len(members) < 2:
                continue
            # fragments ("Systems With Humans In") are not merged into a good node: they would bring their edges
            members = [m for m in members if not (self.REJECT_CONCEPT_ISSUES & set(concept_issues(m, self.MAX_CONCEPT_WORDS)))]
            if len(members) < 2:
                continue
            members.sort(key=lambda m: (len(concept_issues(m, self.MAX_CONCEPT_WORDS)), -G.degree(m), len(m), m))
            keep = members[0]
            for m in members[1:]:
                self._merge_node(m, keep)
                merged[m] = keep
        rep["merged"] = merged

        # 5 (first pass). scores, so taxonomy only links well-formed concepts
        chunk_stems = None
        if chunks:
            from .evidence import _words, _stem, _STOP
            chunk_stems = [{_stem(w) for w in _words(c.text)} for c in chunks]
        def mentions_of(n):
            if chunk_stems is None:
                return None
            from .evidence import _words, _stem, _STOP
            ws = [_stem(w) for w in _words(n) if w not in _STOP and len(w) >= 3] or [_stem(w) for w in _words(n)]
            return sum(all(w in cs for w in ws) for cs in chunk_stems) if ws else 0
        for n in G.nodes():
            texts = [d.get("evidence", "") for _, _, d in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True))]
            self._score_node(n, mentions_of(n), texts)

        # 4. taxonomy. Edges added by an earlier repair under looser rules are re-checked first
        # ("Assess Consciousness" -> "Consciousness", "Scientific Theory" -> "Theory" no longer qualify)
        added, rejected_sense, tax_removed = [], [], []
        if add_taxonomy:
            for u, v, k, d in list(G.edges(keys=True, data=True)):
                if d.get("extractor") != "taxonomy":
                    continue
                if not specialization_of(u, v) or min(G.nodes[u].get("concept_quality", 0),
                                                      G.nodes[v].get("concept_quality", 0)) < min_taxonomy_quality:
                    G.remove_edge(u, v, k)
                    tax_removed.append((u, v))
        rep["taxonomy_removed"] = tax_removed
        if add_taxonomy:
            by_name = {str(n).lower(): n for n in G.nodes()}
            candidates = []
            for n in list(G.nodes()):
                if G.nodes[n].get("concept_quality", 0) < min_taxonomy_quality:
                    continue
                ws = n.split()
                for i in range(1, len(ws)):
                    g = by_name.get(" ".join(ws[i:]).lower())
                    if not g or g == n or G.nodes[g].get("concept_quality", 0) < min_taxonomy_quality:
                        continue
                    if not specialization_of(n, g):
                        continue
                    if any(d.get("predicate") == "is a kind of" for d in (G.get_edge_data(n, g) or {}).values()):
                        break
                    candidates.append((n, g))
                    break                      # nearest general concept only: Moral Responsibility -> Responsibility
            # "Christopher Wylie" / "Wylie", "Russian Internet Research Agency" / "Internet Research Agency":
            # a descriptor plus the same named entity is an alias, not a subclass
            aliases_merged = {}
            for n, g in list(candidates):
                if not (G.has_node(n) and G.has_node(g) and G.nodes[g].get("named_entity")):
                    continue
                extras = n.split()[:-len(g.split())]
                ev = [d.get("evidence", "") for _, _, d in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True))]
                # the extra words must themselves be proper names in the text ("Eric", "Christopher", "Google",
                # "Russian"); an all-caps or numbered variant ("SIMPLE COMPAS", "Article 15 GDPR") is a different thing
                proper = extras and len(extras) <= 2 and all(
                    not re.search(r"\d", w) and not (len(w) > 1 and w.isupper()) and
                    is_named_entity(w.rstrip(",").replace("'s", "").replace("’s", ""), ev) for w in extras)
                if not proper:
                    continue
                keep, drop = (n, g) if G.degree(n) > G.degree(g) else (g, n)   # the better-connected name stays
                self._merge_node(drop, keep)
                aliases_merged[drop] = keep
                candidates.remove((n, g))
            rep["aliases_merged"] = aliases_merged
            # a named entity is not a class: "Christopher Wylie" is not a kind of "Wylie"; pairs a sense check
            # rejected before stay rejected (a later run without embeddings must not add them back)
            known_rejected = {tuple(x[:2]) for x in G.graph.get("taxonomy_rejected", [])}
            candidates = [(n, g) for n, g in candidates if G.has_node(n) and G.has_node(g)
                          and not G.nodes[g].get("named_entity") and (n, g) not in known_rejected]
            sims: Dict[Tuple[str, str], float] = {}
            if embed_fn is not None and candidates:
                try:
                    import numpy as _np
                    # each concept = centroid of "name: statement" embeddings for up to 12 of its statements:
                    # a hub such as "Autonomy" (121 statements) is represented by how the text uses it overall
                    texts_of = {}
                    for x in {y for pair in candidates for y in pair}:
                        st = [d.get("statement", "") for _, _, d in list(G.in_edges(x, data=True)) + list(G.out_edges(x, data=True))
                              if d.get("extractor") != "taxonomy" and d.get("statement")]
                        step = max(1, len(st) // 12)
                        texts_of[x] = [f"{x}: {t}" for t in st[::step][:12]] or [x]
                    flat = [(x, t) for x, ts in texts_of.items() for t in ts]
                    vecs = []
                    for i in range(0, len(flat), 64):
                        vecs += embed_fn([t for _, t in flat[i:i + 64]])
                    acc: Dict[str, Any] = {}
                    for (x, _), v in zip(flat, vecs):
                        v = _np.asarray(v, dtype=float)
                        acc[x] = acc.get(x, 0) + v / (_np.linalg.norm(v) or 1.0)
                    E = {x: v / (_np.linalg.norm(v) or 1.0) for x, v in acc.items()}
                    sims = {(a, b): float(E[a] @ E[b]) for a, b in candidates if a in E and b in E}
                except Exception as e:
                    logger.warning(f"taxonomy sense check unavailable ({e}); edges added unchecked")
                    sims = {}
            for n, g in candidates:
                sim = sims.get((n, g))
                if sim is not None and sim < sense_threshold:
                    rejected_sense.append((n, g, round(sim, 3)))
                    continue
                G.add_edge(n, g, predicate="is a kind of", statement=f"{n} is a kind of {g}.",
                           evidence=f"'{n}' names a more specific form of '{g}' (head-noun match).",
                           strength=6, chunk_id="", doc_name=G.nodes[n].get("doc_name", ""),
                           page_number=G.nodes[n].get("page_number", 1), section="", evidence_status="lexical",
                           extractor="taxonomy", raw_predicate="",
                           sense_similarity=None if sim is None else round(sim, 3), sense_checked=sim is not None)
                added.append((n, g))
        rep["taxonomy_added"] = added
        rep["taxonomy_rejected_sense"] = rejected_sense
        if rejected_sense:
            prev = [list(x) for x in G.graph.get("taxonomy_rejected", [])]
            G.graph["taxonomy_rejected"] = prev + [[a, b, sim] for a, b, sim in rejected_sense]
        self._reindex()
        rep["nodes_after"] = G.number_of_nodes()
        rep["edges_after"] = G.number_of_edges()
        rep["inference_edges_before"] = inf_before
        rep["inference_edges_after"] = sum(relation_logic(d.get("predicate")) != "none" for _, _, d in G.edges(data=True))
        rep["concepts"] = self.concept_summary()
        return rep

    def concept_summary(self, threshold: float = 0.5) -> Dict[str, Any]:
        """How meaningful the concepts are: share above `threshold`, issue counts, distinct predicates."""
        G = self.graph
        qs = []
        issues: Dict[str, int] = {}
        for n, a in G.nodes(data=True):
            iss = a.get("concept_issues")
            if iss is None:
                iss = concept_issues(n, self.MAX_CONCEPT_WORDS)
            q = a.get("concept_quality")
            q = concept_quality(n, a.get("mentions", 1), iss) if q is None else q
            qs.append(q)
            for i in iss:
                issues[i] = issues.get(i, 0) + 1
        n = len(qs) or 1
        deg1 = sum(1 for x in G.nodes() if G.degree(x) <= 1)
        return {
            "nodes": len(qs), "meaningful": sum(q >= threshold for q in qs),
            "meaningful_share": round(sum(q >= threshold for q in qs) / n, 3),
            "issues": dict(sorted(issues.items(), key=lambda kv: -kv[1])),
            "isolated_share": round(deg1 / n, 3),
            "components": nx.number_weakly_connected_components(G) if len(qs) else 0,
            "distinct_predicates": len({d.get("predicate") for _, _, d in G.edges(data=True)}),
            "inference_edges": sum(relation_logic(d.get("predicate")) != "none" for _, _, d in G.edges(data=True)),
            "taxonomy_edges": sum(d.get("extractor") == "taxonomy" for _, _, d in G.edges(data=True)),
        }

    # ------------------------------------------------------------------ persistence
    KG_FORMAT = "nanogentzen-studio-kg"
    KG_FORMAT_VERSION = 1

    def to_dict(self, chunks: Optional[List[Chunk]] = None, communities: Optional[List[Dict[str, Any]]] = None,
                kg_id: Optional[str] = None) -> Dict[str, Any]:
        """Graph + (optionally) the chunks it was built from and the detected communities, as plain JSON data."""
        from dataclasses import asdict
        return {
            "format": self.KG_FORMAT,
            "version": self.KG_FORMAT_VERSION,
            "kg_id": kg_id,
            "documents": sorted(self.graph_documents()),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "nodes": [{"id": n, **dict(a)} for n, a in self.graph.nodes(data=True)],
            "edges": [{"u": u, "v": v, **dict(a)} for u, v, a in self.graph.edges(data=True)],
            "chunks": [asdict(c) for c in (chunks or [])],
            "communities": communities or [],
            "extraction_stats": dict(self.extraction_stats),
            "graph_attrs": {k: v for k, v in self.graph.graph.items() if k in ("taxonomy_rejected",)},
        }

    def graph_documents(self) -> set:
        """Names of the source documents the graph's relations were extracted from."""
        return {d.get("doc_name") for _, _, d in self.graph.edges(data=True) if d.get("doc_name")}

    def save(self, path: str, chunks: Optional[List[Chunk]] = None,
             communities: Optional[List[Dict[str, Any]]] = None, kg_id: Optional[str] = None) -> Path:
        """Write the graph to `path` (JSON). Written to a temp file first, so a crash never leaves half a graph."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(chunks, communities, kg_id=kg_id), ensure_ascii=False, default=str),
                       encoding="utf-8")
        os.replace(tmp, out)
        return out

    def load(self, path: str) -> Dict[str, Any]:
        """Replace the current graph with the one saved at `path`. Returns {"chunks": [Chunk], "communities": [...], ...}."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != self.KG_FORMAT:
            raise ValueError(f"{path} is not a saved Studio knowledge graph")
        self.clear()
        for n in data.get("nodes", []):
            attrs = {k: v for k, v in n.items() if k != "id"}
            self.graph.add_node(n["id"], **attrs)
            self._node_lookup[re.sub(r'[^a-z0-9]', '', str(n["id"]).lower())] = n["id"]
            self._key_lookup.setdefault(concept_key(str(n["id"])), n["id"])
        for e in data.get("edges", []):
            attrs = {k: v for k, v in e.items() if k not in ("u", "v")}
            self.graph.add_edge(e["u"], e["v"], **attrs)
        self.extraction_stats = dict(data.get("extraction_stats", {}))
        self.graph.graph.update(data.get("graph_attrs", {}) or {})
        for n, a in self.graph.nodes(data=True):
            if "concept_quality" not in a:          # saved by an older build: score from the name alone
                iss = concept_issues(str(n), self.MAX_CONCEPT_WORDS)
                a["concept_issues"], a["concept_quality"] = iss, concept_quality(str(n), 1, iss)
        self._graph_version += 1
        self._path_cache.clear()
        fields = set(Chunk.__dataclass_fields__)
        chunks = [Chunk(**{k: v for k, v in c.items() if k in fields}) for c in data.get("chunks", [])]
        return {"chunks": chunks, "communities": data.get("communities", []), "saved_at": data.get("saved_at"),
                "num_nodes": len(data.get("nodes", [])), "num_edges": len(data.get("edges", [])),
                "kg_id": data.get("kg_id")}

    @staticmethod
    def saved_info(path: str) -> Optional[Dict[str, Any]]:
        """Header of a saved graph (without loading it), or None if there is none."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            docs = sorted(({c.get("doc_name", "") for c in data.get("chunks", [])}
                           | {e.get("doc_name", "") for e in data.get("edges", [])}
                           | set(data.get("documents") or [])) - {"", None})
            return {"saved_at": data.get("saved_at"), "num_nodes": len(data.get("nodes", [])),
                    "num_edges": len(data.get("edges", [])), "num_chunks": len(data.get("chunks", [])), "documents": docs}
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 4) if self.graph.number_of_nodes() > 0 else 0.0,
            "evidence": summarize_evidence(d.get("evidence_status") or "unchecked" for _, _, d in self.graph.edges(data=True)),
            "extraction": dict(self.extraction_stats),
            "concepts": self.concept_summary() if self.graph.number_of_nodes() else {},
        }
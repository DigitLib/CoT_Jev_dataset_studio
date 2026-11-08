"""
Batch Dataset Generator module for nanoGentzen CoT Studio.
Automates end-to-end dataset creation across all discovered Knowledge Graph paths
and document chunks once single-sample manual verification is approved.
"""

from typing import List, Dict, Any, Optional, Callable, Set, Tuple
import json
import logging
import re
import threading
import concurrent.futures
from datetime import datetime, timezone

from .chunker import Chunk, is_boilerplate
from .question_generator import QuestionGenerator, _sanitize_question
from .cot_generator import CoTGenerator
from .dataset_manager import DatasetManager, DatasetSample
from nanogentzen.logical_architect import LogicalArchitect
from nanogentzen.cot.architect import with_proof_record
from llm_backend.client import provenance_model_name

logger = logging.getLogger(__name__)


def decide_sample_status(requested: str, is_verified: bool, is_grounded: bool, is_fallback: bool) -> Tuple[str, List[str]]:
    """
    Status a generated sample is stored with. `requested` is what the batch was asked to use; "accepted"
    is only granted to samples with a verified proof, a grounded trace and a real (non-template) LLM output.
    Everything else is stored as "draft" with the reasons, for review in the Studio.
    """
    if requested != "accepted":
        return requested, []
    reasons = []
    if not is_verified:
        reasons.append("proof not verified")
    if not is_grounded:
        reasons.append("reasoning not grounded in the logic tree")
    if is_fallback:
        reasons.append("LLM output unavailable; template reasoning")
    return ("accepted" if not reasons else "draft"), reasons


class BatchDatasetGenerator:
    def __init__(
        self,
        q_generator: QuestionGenerator,
        architect: LogicalArchitect,
        cot_generator: CoTGenerator,
        db_manager: DatasetManager,
    ):
        self.q_generator = q_generator
        self.architect = architect
        self.cot_generator = cot_generator
        self.db_manager = db_manager
        # one extra LLM call per sample: a plain step-by-step answer used as the token-savings baseline
        self.measure_token_savings = True
        self.stop_event = threading.Event()

    def request_stop(self) -> None:
        """Ask a running generate_batch to stop: queued seeds are cancelled, in-flight ones still finish and
        are saved. Safe to call from any thread (e.g. a Streamlit rerun triggered by a Stop button)."""
        self.stop_event.set()

    def generate_batch(
        self,
        source_mode: str,
        kg_paths: List[Dict[str, Any]],
        chunks: List[Chunk],
        model: str,
        max_samples: Optional[int] = None,
        concurrency: int = 1,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any], str], None]] = None,
        stop_signal: Optional[Callable[[], bool]] = None,
        status: str = "accepted"
    ) -> Dict[str, Any]:
        """
        Orchestrates complete dataset generation over all provided seeds.
        Supports parallel worker concurrency (ideal for vLLM continuous batching).

        Stopping: request_stop() (or stop_signal() returning True, checked on the calling thread) cancels all
        queued seeds; each worker saves its own sample, so work that is already in flight is kept. If the
        calling thread is interrupted (Streamlit stops a script run when a button is clicked), the pool is
        shut down without waiting for the queue, instead of blocking until every seed has been generated.
        """
        seeds = self._assemble_seeds(source_mode, kg_paths, chunks)
        if max_samples and max_samples > 0:
            seeds = seeds[:max_samples]

        self.stop_event.clear()
        total_seeds = len(seeds)
        generated_samples: List[DatasetSample] = []
        counters = {"verified": 0, "clarity": 0.0, "done": 0}
        seen_questions: Set[str] = set()
        dedup_lock = threading.Lock()
        db_lock = threading.Lock()

        def _stopped() -> bool:
            if stop_signal is not None and not self.stop_event.is_set():
                try:
                    if stop_signal():
                        self.stop_event.set()
                except Exception:
                    pass
            return self.stop_event.is_set()

        def _handle_seed(idx: int, seed: Dict[str, Any]) -> Optional[Tuple[DatasetSample, Dict[str, Any], bool, float]]:
            # runs on a worker thread: only the thread-safe event is consulted here
            if self.stop_event.is_set():
                return None
            try:
                res = self._synthesize_seed(
                    seed=seed,
                    chunks=chunks,
                    model=model,
                    status=status,
                    seen_questions=seen_questions,
                    dedup_lock=dedup_lock
                )
            except Exception as e:
                logger.error(f"Failed to generate batch sample {idx}: {e}")
                return None
            if res:
                sample, _sd, is_ver, clar = res
                with db_lock:   # saved by the worker, so an interrupted caller never loses finished samples
                    self.db_manager.save_sample(sample)
                    generated_samples.append(sample)
                    counters["verified"] += int(bool(is_ver))
                    counters["clarity"] += clar
            return res

        def _report(result, idx: int):
            with db_lock:
                counters["done"] += 1     # skipped and failed seeds count as processed, so progress never stalls
                done = counters["done"]
            if progress_callback:
                if result:
                    sample, sample_dict, _v, _c = result
                    progress_callback(done, total_seeds, sample_dict,
                                      f"Sample [{done}/{total_seeds}] generated: {sample.sample_id} (workers={concurrency})")
                else:
                    progress_callback(done, total_seeds, {}, f"Seed {idx} skipped [{done}/{total_seeds}]")

        if concurrency <= 1:
            for idx, seed in enumerate(seeds, start=1):
                if _stopped():
                    logger.info(f"Batch generation interrupted by user at sample {idx-1}/{total_seeds}.")
                    break
                if progress_callback:
                    progress_callback(counters["done"], total_seeds, {}, f"Generating sample {idx}/{total_seeds} ({seed['type']})...")
                _report(_handle_seed(idx, seed), idx)
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
            interrupted = True    # set to False unless an exception (e.g. Streamlit stopping the run) escapes
            try:
                futures_map = {executor.submit(_handle_seed, idx, seed): idx for idx, seed in enumerate(seeds, start=1)}
                for future in concurrent.futures.as_completed(futures_map):
                    if _stopped():
                        logger.info("Batch generation stopped by user. Cancelling queued seeds.")
                        break
                    _report(future.result(), futures_map[future])
                interrupted = False
            finally:
                if interrupted or self.stop_event.is_set():
                    self.stop_event.set()                       # workers skip anything not yet started
                # a requested stop waits for the (at most `concurrency`) in-flight seeds, so the returned counts
                # match what was saved; an interrupted caller returns at once and the workers save in the background
                executor.shutdown(wait=not interrupted, cancel_futures=self.stop_event.is_set())

        with db_lock:
            n = len(generated_samples)
            avg_clarity = round(counters["clarity"] / max(1, n), 2)
            verification_rate = round(counters["verified"] / max(1, n), 2)
            samples = list(generated_samples)

        return {
            "total_requested": total_seeds,
            "generated_count": n,
            "verified_count": counters["verified"],
            "verification_rate": verification_rate,
            "avg_clarity": avg_clarity,
            "samples": samples,
            "stopped": self.stop_event.is_set(),
        }

    def _assemble_seeds(
        self,
        mode: str,
        kg_paths: List[Dict[str, Any]],
        chunks: List[Chunk]
    ) -> List[Dict[str, Any]]:
        seeds = []
        substantive_chunks = [c for c in chunks if not is_boilerplate(c.text)]
        if not substantive_chunks and chunks:
            substantive_chunks = chunks

        if mode in ("kg_paths", "combined"):
            for p in kg_paths:
                seeds.append({"type": "kg_path", "data": p})
        if mode in ("chunks", "combined"):
            for c in substantive_chunks:
                seeds.append({"type": "chunk", "data": c})

        # Default fallback if empty
        if not seeds and substantive_chunks:
            for c in substantive_chunks:
                seeds.append({"type": "chunk", "data": c})
        return seeds

    def _synthesize_seed(
        self,
        seed: Dict[str, Any],
        chunks: List[Chunk],
        model: str,
        status: str,
        seen_questions: Set[str],
        dedup_lock: threading.Lock
    ) -> Optional[Tuple[DatasetSample, Dict[str, Any], bool, float]]:
        """Worker function that processes a single seed into a complete verified ClearCoT sample."""
        seed_type = seed["type"]
        item = seed["data"]

        # 1. Generate Question & Premises
        if seed_type == "kg_path":
            chunk_map = {c.chunk_id: c for c in chunks}
            exact_cids = item.get("chunk_ids", [])
            exact_supporting = [chunk_map[cid] for cid in exact_cids if cid in chunk_map and not is_boilerplate(chunk_map[cid].text)]

            if exact_supporting:
                supporting = exact_supporting[:3]
            else:
                pages = set(item.get("provenance_pages", []))
                prov_chunks = [c for c in chunks if c.page_number in pages and not is_boilerplate(c.text)]
                path_entities = [e for e in item.get("path_nodes", [item.get("source", ""), item.get("target", "")]) if e]

                target_pool = prov_chunks if prov_chunks else [c for c in chunks if not is_boilerplate(c.text)]
                scored_chunks = []
                for c in target_pool:
                    score = sum(1 for e in path_entities if any(w.lower() in c.text.lower() for w in e.split() if len(w) > 3))
                    scored_chunks.append((score, c))
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                supporting = [c for _, c in scored_chunks[:3]] or target_pool[:3] or chunks[:3]

            q_res = self.q_generator.generate_from_multihop_path(item, supporting, model=model)
        else:
            if is_boilerplate(item.text):
                return None
            q_res = self.q_generator.generate_from_chunk(item, model=model)

        # Quality validation gate
        if not QuestionGenerator.validate_sample_quality(q_res):
            logger.warning(f"Discarding low quality or boilerplate sample: {q_res.get('question')}")
            return None

        # Deduplication check
        norm_q = re.sub(r'[^a-zA-Z0-9]', '', q_res.get("question", "").lower())
        with dedup_lock:
            if norm_q in seen_questions:
                logger.info(f"Skipping duplicate question: {q_res.get('question')}")
                return None
            seen_questions.add(norm_q)

        target_claim = q_res.get("target_claim", q_res.get("seed_answer", ""))
        premises = q_res.get("key_premises", [target_claim])
        path_nodes = q_res.get("path_nodes", None)
        path_repr = q_res.get("provenance", {}).get("path_repr") or q_res.get("path_repr")
        context_text = q_res.get("context_text", "")

        # 2. Construct nanoGentzen Best Logic Tree (or reuse pre-verified tree from KG Path detection)
        if isinstance(item, dict) and item.get("logic_tree_res"):
            tree_res = item["logic_tree_res"]
        else:
            tree_res = self.architect.generate_best_logic_tree(
                premises=premises,
                conclusion=target_claim,
                path_nodes=path_nodes,
                path_repr=path_repr,
            )

        # 3. Build CoT along Logic Tree via user-selected LLM
        cot_res = self.cot_generator.build_cot_from_logic_tree(
            question=q_res["question"],
            target_claim=target_claim,
            logic_tree_text=tree_res["logic_tree_text"],
            context_text=context_text,
            model=model,
            temperature=0.25,
            relevance_info=tree_res.get("relevance", None),
            curry_howard_term=tree_res.get("curry_howard_term", ""),
            path_nodes=path_nodes,
            key_premises=premises,
            is_verified=tree_res.get("is_verified", False),
            epistemic_status=tree_res.get("epistemic_status", "PROVEN_TRUE"),
        )

        # 4. Assemble sample
        r_text = cot_res.get("reasoning", "")
        r_tokens = int(len(r_text.split()) * 1.3)
        final_answer = cot_res.get("answer", target_claim).strip()
        final_question = _sanitize_question(cot_res.get("question", q_res["question"]).strip())

        if len(final_answer) < 20 or final_answer[0].islower():
            final_answer = target_claim

        naive_meta = self.cot_generator.generate_naive_cot(
            question=final_question,
            target_claim=target_claim,
            context_text=context_text,
            model=model,
            return_meta=True,
        ) if hasattr(self.cot_generator, "generate_naive_cot") else None
        if isinstance(naive_meta, dict) and isinstance(naive_meta.get("text"), str):
            naive_baseline = naive_meta["text"]
        elif isinstance(naive_meta, str):                 # older / custom generators return the text only
            naive_baseline, naive_meta = naive_meta, {"text": naive_meta, "synthetic": False}
        else:
            naive_baseline = q_res.get("seed_answer", "").strip() or final_answer
            naive_meta = {"text": naive_baseline, "synthetic": True}
        baseline = (self.cot_generator.generate_standard_cot(final_question, context_text, model=model)
                    if self.measure_token_savings and hasattr(self.cot_generator, "generate_standard_cot") else None)
        if not isinstance(baseline, str):
            baseline = None
        measured = CoTGenerator.measure_token_savings(r_text, baseline)
        tokens_saved = measured if measured is not None else 0     # 0 = not measured, never a made-up value
        clarity = cot_res.get("clarity_score", 0.90)
        gr_score = cot_res.get("grounding_score", 1.0)
        is_verified = tree_res.get("is_verified", False)
        epistemic_stat = tree_res.get("epistemic_status", "PROVEN_TRUE" if is_verified else "INTUITIONISTIC_UNDECIDED")
        ch_term = tree_res.get("curry_howard_term", "")
        is_grounded = bool(cot_res.get("is_grounded", False))
        is_fallback = bool(cot_res.get("is_fallback", False))
        final_status, status_reasons = decide_sample_status(status, is_verified, is_grounded, is_fallback)
        provenance = with_proof_record(q_res.get("provenance", {}), tree_res)
        provenance["generation"] = {
            "model": provenance_model_name(model),
            "is_fallback": is_fallback,
            "fallback_error": cot_res.get("error") if is_fallback else None,
            "is_grounded": is_grounded,
            "grounding_score": cot_res.get("grounding_score"),
            "logic_basis": tree_res.get("logic_basis"),
            "naive_cot_synthetic": bool(naive_meta.get("synthetic")),
            "token_savings_measured": measured is not None,
            "token_savings_baseline": "standard_cot" if measured is not None else None,
            "status_reasons": status_reasons,
        }

        sample = DatasetSample(
            sample_id="",
            doc_name=q_res.get("provenance", {}).get("doc_name", "unknown"),
            question=final_question,
            reasoning=r_text,
            answer=final_answer,
            naive_cot=naive_baseline,
            status=final_status,
            gentzen_verified=is_verified,
            gentzen_sequent=tree_res.get("sequent", ""),
            proof_tree_json=json.dumps(tree_res.get("proof_tree", {})),
            provability_score=tree_res.get("provability_score", 1.0),
            tokens_saved_pct=tokens_saved,
            clarity_score=clarity,
            provenance_json=json.dumps(provenance, default=str),
            created_at=datetime.now(timezone.utc).isoformat(),
            curry_howard_term=ch_term,
            epistemic_status=epistemic_stat,
            grounding_score=gr_score,
        )

        sample_dict = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "reasoning": sample.reasoning,
            "answer": sample.answer,
            "sequent": sample.gentzen_sequent,
            "verified": is_verified,
            "clarity_score": clarity,
            "grounding_score": gr_score,
            "is_grounded": is_grounded,
            "curry_howard_term": ch_term,
            "epistemic_status": epistemic_stat,
            "status": final_status,
            "status_reasons": status_reasons,
            "tokens_saved_pct": tokens_saved,
        }

        return (sample, sample_dict, is_verified, clarity)


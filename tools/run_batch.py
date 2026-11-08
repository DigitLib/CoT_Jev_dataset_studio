#!/usr/bin/env python3
"""
run_batch.py - Headless batch synthesis and dataset generation CLI for the Studio.

Enables end-to-end extraction, multi-hop path discovery, formal nanoGentzen verification,
and dataset exports without keeping a browser or Streamlit session running.

Example:
    # Generate ClearCoT dataset from uploads using local Ollama/vLLM
    python tools/run_batch.py --limit 30 --mode clearcot --model qwen2.5:7b

    # Generate both ClearCoT and Jev scenarios with native thinking traces
    python tools/run_batch.py --mode all --cot-mode native --model spark-x

    # Use existing latest.json knowledge graph instead of re-extracting
    python tools/run_batch.py --use-existing-kg --limit 50
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure Studio root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    UPLOADS_DIR, EXPORTS_DIR, DB_PATH,
    DEFAULT_MODEL_PATH,
)
from core import kg_store
from llm_backend.client import LocalLLMClient
from core.document_parser import DocumentParser
from core.chunker import DocumentChunker
from core.kg_builder import KnowledgeGraphBuilder
from core.community_detector import CommunityDetector
from core.cot_generator import CoTGenerator
from core.dataset_manager import DatasetManager
from core.batch_generator import BatchDatasetGenerator
from core.question_generator import QuestionGenerator
from core.chunker import is_boilerplate
from core.jev_generator import JevScenarioGenerator
from core.jev_dataset_manager import JevDatasetManager
from nanogentzen.engine import load_engine, detect_model_version

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_batch")


def parse_args():
    p = argparse.ArgumentParser(description="Headless batch dataset generator for nanoGentzen Studio")
    p.add_argument("--mode", choices=["clearcot", "jev", "all"], default="all",
                   help="Which dataset pipeline to run (default: all)")
    p.add_argument("--input-dir", type=str, default=str(UPLOADS_DIR),
                   help="Directory containing PDF/DOCX/TXT literature to ingest")
    p.add_argument("--model", type=str, default="qwen2.5:7b",
                   help="Teacher LLM model name")
    p.add_argument("--base-url", type=str, default="http://localhost:11434/v1",
                   help="OpenAI-compatible local server URL (Ollama, LM Studio, vLLM)")
    p.add_argument("--gentzen-path", type=str, default=str(DEFAULT_MODEL_PATH),
                   help="Path to nanoGentzen model folder or safetensors file")
    p.add_argument("--cot-mode", choices=["prompted", "native"], default="prompted",
                   help="Reasoning trace format: prompted or native (<think> channel)")
    p.add_argument("--limit", type=int, default=20,
                   help="Maximum candidate paths/scenarios to process")
    p.add_argument("--kg", type=str, default=None,
                   help="Reuse a saved knowledge graph (graph id from data/kg/index.json, or a path to a .json) "
                        "instead of re-chunking and extracting")
    p.add_argument("--use-existing-kg", action="store_true",
                   help="Reuse the last active saved graph (same as --kg <last active id>)")
    p.add_argument("--kg-name", type=str, default=None,
                   help="Graph id to save a newly built graph under (default: derived from the input documents)")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow a newly built graph to replace an existing saved graph with the same id")
    p.add_argument("--kg-concurrency", type=int, default=4, help="Parallel KG extraction workers")
    p.add_argument("--concurrency", type=int, default=4, help="Parallel ClearCoT generation workers")
    p.add_argument("--no-repair", dest="repair", action="store_false",
                   help="Skip concept repair after KG extraction (the GUI runs it by default)")
    p.add_argument("--taxonomy-policy", choices=["any", "last", "none"], default="any",
                   help="Where an 'is a kind of' hop may stand in ClearCoT paths (GUI default: any)")
    p.add_argument("--jev-source", choices=["paths", "chunks"], default="paths",
                   help="Seed Jev scenarios from KG paths (falls back to chunks if there are none) or from chunks")
    p.add_argument("--jev-workflow", type=str, default=None, help="Jev workflow slug (default: inferred per seed)")
    p.add_argument("--all-samples", dest="verified_only", action="store_false",
                   help="Also export samples that are not kernel-verified (default: verified only)")
    p.add_argument("--split-strategy", choices=["auto", "document", "path", "family"], default="auto",
                   help="Train/val/test grouping split strategy")
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing headless batch runner...")

    # 1. Initialize Clients & Engines
    llm_client = LocalLLMClient(base_url=args.base_url)
    engine_ver = detect_model_version(args.gentzen_path)
    logger.info(f"Loading nanoGentzen engine (detected version: v{engine_ver}) from {args.gentzen_path}...")
    engine = load_engine(model_path=args.gentzen_path)

    # Architect
    from nanogentzen.logical_architect import LogicalArchitect
    architect = LogicalArchitect(engine=engine)

    # 2. Ingest Literature or Load Saved KG
    kg_builder = KnowledgeGraphBuilder(llm_client=llm_client, architect=architect)
    comm_detector = CommunityDetector()
    chunks = []

    active_kg_id = None   # stamped into every generated sample's provenance
    kg_ref = args.kg
    if kg_ref is None and args.use_existing_kg:
        last = kg_store.last_active()
        if last is None:
            logger.error("--use-existing-kg: there is no saved knowledge graph.")
            sys.exit(2)
        kg_ref = last["kg_id"]

    if kg_ref:
        if kg_ref.endswith(".json") and Path(kg_ref).exists():
            logger.info(f"Loading Knowledge Graph file {kg_ref}...")
            restored = kg_builder.load(kg_ref)
            active_kg_id = restored.get("kg_id") or Path(kg_ref).stem
        else:
            if kg_store.get_graph(kg_ref) is None:
                known = ", ".join(g["kg_id"] for g in kg_store.list_graphs()) or "none"
                logger.error(f"No saved knowledge graph {kg_ref!r} (saved graphs: {known}).")
                sys.exit(2)
            logger.info(f"Loading saved Knowledge Graph '{kg_ref}'...")
            restored = kg_store.load_graph(kg_builder, kg_ref)
            active_kg_id = kg_ref
        chunks = restored.get("chunks", [])
        logger.info(f"Restored {kg_builder.graph.number_of_nodes()} entities, "
                    f"{kg_builder.graph.number_of_edges()} relations, and {len(chunks)} chunks.")
    else:
        logger.info(f"Ingesting documents from {args.input_dir}...")
        parser = DocumentParser()
        chunker = DocumentChunker(strategy="paragraph")
        in_path = Path(args.input_dir)
        doc_files = sorted([f for f in in_path.glob("*") if f.suffix.lower() in (".pdf", ".docx", ".txt", ".md")])

        if not doc_files:
            logger.warning(f"No document files found in {args.input_dir}.")
        else:
            for df in doc_files:
                logger.info(f"Parsing {df.name}...")
                pages = parser.parse_file(str(df))
                c = chunker.chunk_pages(df.name, pages)
                chunks.extend(c)
            logger.info(f"Extracted {len(chunks)} chunks across {len(doc_files)} document(s).")

            base_id = kg_store.sanitize_kg_id(args.kg_name) if args.kg_name else \
                kg_store.kg_id_for(df.name for df in doc_files)
            if kg_store.get_graph(base_id) is not None and not args.overwrite:
                logger.error(f"A saved knowledge graph '{base_id}' already exists. Reuse it with --kg {base_id}, "
                             f"pick another --kg-name, or pass --overwrite.")
                sys.exit(2)

            logger.info("Extracting Knowledge Graph propositions...")
            kg_builder.build_from_chunks(chunks, model=args.model, max_chunks=len(chunks),
                                         concurrency=args.kg_concurrency, base_url=args.base_url)
            if args.repair:
                # same as the Knowledge Graph tab's default "Repair & consolidate concepts after extraction":
                # completes truncated names, re-reads relations, adds taxonomy edges (without it a graph usually
                # has almost no inference chains for ClearCoT)
                rep = kg_builder.repair_concepts(chunks=chunks, embed_fn=None, sense_threshold=0.45)
                logger.info(f"Concept repair: inference edges {rep.get('inference_edges_before')} -> "
                            f"{rep.get('inference_edges_after')}, taxonomy edges added {len(rep.get('taxonomy_added', []))}.")
            logger.info("Detecting topic communities...")
            comm_res = comm_detector.detect_communities(kg_builder.graph)
            try:
                active_kg_id = base_id
                path = kg_store.save_graph(kg_builder, base_id, chunks=chunks,
                                           communities=comm_res.get("communities", []))
                logger.info(f"Saved knowledge graph '{base_id}' to {path}.")
            except Exception as e:
                logger.warning(f"Could not save the knowledge graph: {e}")

    # 3. ClearCoT Pipeline (same flow as the Logic Tree & ClearCoT tab's batch generator)
    if args.mode in ("clearcot", "all"):
        logger.info("=== Running ClearCoT Batch Pipeline ===")
        cot_gen = CoTGenerator(llm_client=llm_client, reasoning_mode=args.cot_mode)
        db_manager = DatasetManager(db_path=str(DB_PATH))
        db_manager.kg_id = active_kg_id
        batch_gen = BatchDatasetGenerator(
            q_generator=QuestionGenerator(llm_client=llm_client),
            architect=architect,
            cot_generator=cot_gen,
            db_manager=db_manager,
        )

        logger.info(f"Finding kernel-verified multi-hop entailment paths (limit: {args.limit})...")
        paths = kg_builder.find_multihop_paths(
            require_entailment=True,
            verified_only=True,
            limit=args.limit,
            taxonomy_policy=args.taxonomy_policy,
            max_paths_per_hub=8,
            min_hops=2,
            max_hops=3,
            architect=architect,
        )
        logger.info(f"Found {len(paths)} kernel-verified entailment paths.")

        if paths:
            results = batch_gen.generate_batch(
                source_mode="kg_paths",
                kg_paths=paths,
                chunks=chunks,
                model=args.model,
                concurrency=args.concurrency,
                status="accepted",
            )
            logger.info(f"ClearCoT: {results['generated_count']} samples generated, {results['verified_count']} "
                        f"verified, avg clarity {results['avg_clarity']:.2f}.")

        sft_path = EXPORTS_DIR / "clearcot_sft_dataset.jsonl"
        dpo_path = EXPORTS_DIR / "clearcot_dpo_pairs.jsonl"
        parquet_path = EXPORTS_DIR / "clearcot_dataset.parquet"
        c_sft = db_manager.export_jsonl(str(sft_path), only_accepted=True,
                                        split_strategy=args.split_strategy, only_verified=args.verified_only)
        c_dpo = db_manager.export_dpo_pairs(str(dpo_path), only_accepted=True, skip_synthetic_rejected=True,
                                            split_strategy=args.split_strategy, only_verified=args.verified_only)
        c_pq = db_manager.export_parquet(str(parquet_path), only_accepted=True,
                                         split_strategy=args.split_strategy, only_verified=args.verified_only)
        skipped = getattr(db_manager, "last_dpo_skipped", {})
        logger.info(f"ClearCoT exports: {c_sft} SFT, {c_dpo} DPO pairs ({skipped}), {c_pq} Parquet rows.")

    # 4. Jev Decision Pipeline (same flow as the Jev tab's batch generator)
    if args.mode in ("jev", "all"):
        logger.info("=== Running Jev Decision Batch Pipeline ===")
        jev_gen = JevScenarioGenerator(llm_client=llm_client, engine=engine)
        jev_manager = JevDatasetManager(db_path=str(DB_PATH))
        jev_manager.kg_id = active_kg_id
        chunk_map = {c.chunk_id: c for c in chunks}
        substantive = [c for c in chunks if not is_boilerplate(c.text)] or chunks

        jev_paths = kg_builder.find_multihop_paths(
            limit=args.limit, min_hops=2, max_hops=3, verify_with_gentzen=False) if args.jev_source != "chunks" else []
        targets = [("kg", p) for p in jev_paths]
        if args.jev_source == "chunks" or not targets:
            targets = [("chunk", c) for c in substantive[:args.limit]]
        logger.info(f"Generating Jev scenarios from {len(targets)} {'KG paths' if targets and targets[0][0] == 'kg' else 'chunks'}...")

        base = verified_n = cf_n = 0
        for i, (kind, item) in enumerate(targets, start=1):
            try:
                if kind == "kg":
                    supporting = [chunk_map[cid] for cid in item.get("chunk_ids", []) if cid in chunk_map] or substantive[:3]
                    wf = args.jev_workflow or jev_gen.infer_workflow(
                        doc_name=item.get("doc_name", ""), path_nodes=item.get("path_nodes", []),
                        context_text=item.get("composite_claim", ""))
                    sc = jev_gen.generate_from_multihop_path(path_info=item, supporting_chunks=supporting,
                                                             workflow=wf, model=args.model)
                else:
                    wf = args.jev_workflow or jev_gen.infer_workflow(
                        doc_name=item.doc_name, path_nodes=[item.section_title] if item.section_title else None,
                        context_text=item.text)
                    sc = jev_gen.generate_from_chunk(chunk=item, workflow=wf, model=args.model)
                verified = bool(sc.get("gentzen_verified"))
                parent_id = jev_manager.save_scenario(sc, status="accepted" if verified else "draft")
                base += 1
                if verified:
                    verified_n += 1
                    for v in jev_gen.make_counterfactuals(sc, max_variants=2):
                        v["provenance"]["counterfactual"]["of"] = parent_id
                        jev_manager.save_scenario(v, status="accepted" if v.get("gentzen_verified") else "draft")
                        cf_n += 1
            except Exception as e:
                logger.warning(f"Jev seed {i} failed: {e}")
            if i % 10 == 0:
                logger.info(f"Jev [{i}/{len(targets)}]: {base} scenarios ({verified_n} verified), {cf_n} counterfactuals")
        logger.info(f"Jev: {base} base scenarios ({verified_n} verified) + {cf_n} counterfactual variants.")

        td_path = EXPORTS_DIR / "typed_decisions_dataset.parquet"
        s1mb_path = EXPORTS_DIR / "open_jev_s1mb_dataset.parquet"
        j_jsonl = EXPORTS_DIR / "jev_decisions_dataset.jsonl"
        c_td = jev_manager.export_typed_decisions_parquet(str(td_path), only_accepted=True, column_mode="positional",
                                                         split_strategy=args.split_strategy, only_verified=args.verified_only)
        c_s1 = jev_manager.export_open_jev_parquet(str(s1mb_path), only_accepted=True,
                                                  split_strategy=args.split_strategy, only_verified=args.verified_only)
        c_jl = jev_manager.export_jev_jsonl(str(j_jsonl), only_accepted=True,
                                           split_strategy=args.split_strategy, only_verified=args.verified_only)
        logger.info(f"Jev exports: {c_td} typed-decisions rows, {c_s1} S1MB rows, {c_jl} JSONL records.")

    logger.info("Batch processing complete.")


if __name__ == "__main__":
    main()

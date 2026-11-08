"""
Dataset Manager and SQLite persistence module for ClearCoT datasets.
Supports sample curation, filtering, and multi-format exports (JSONL, Parquet, DPO preference pairs).
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import sqlite3
import json
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
import logging

from .dataset_splits import assign_splits, path_key, DEFAULT_RATIOS, document_groups, sample_documents
from .cot_generator import make_reasoning_direct, naive_trace_leak

logger = logging.getLogger(__name__)


import contextlib


@dataclass
class DatasetSample:
    sample_id: str
    doc_name: str
    question: str
    reasoning: str
    answer: str
    naive_cot: str = ""
    status: str = "draft"  # "draft", "verified", "accepted", "rejected"
    gentzen_verified: bool = False
    gentzen_sequent: str = ""
    proof_tree_json: str = "{}"
    provability_score: float = 0.0
    tokens_saved_pct: int = 0
    clarity_score: float = 0.0
    provenance_json: str = "{}"
    created_at: str = ""
    curry_howard_term: str = ""
    epistemic_status: str = "PROVEN_TRUE"
    grounding_score: float = 1.0

    def __post_init__(self):
        if not self.sample_id:
            self.sample_id = f"S-{uuid.uuid4().hex[:12].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


TAG_CLEAN_REGEX = re.compile(
    r'</?(?:reasoning|think(?:ing)?|thought(?:s|_process)?|ThinkingTraces?|answer)>',
    re.IGNORECASE
)


def _safe_json(text: Optional[str]) -> Dict[str, Any]:
    """Parsed JSON object, or {} for empty / malformed text (one bad row must not abort an export)."""
    if not text:
        return {}
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else {"value": v}
    except Exception:
        return {}


class DatasetManager:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            from config import DB_PATH
            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.kg_id: Optional[str] = None   # active knowledge graph; stamped into new samples' provenance
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    @contextlib.contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    sample_id TEXT PRIMARY KEY,
                    doc_name TEXT,
                    question TEXT,
                    reasoning TEXT,
                    answer TEXT,
                    naive_cot TEXT,
                    status TEXT,
                    gentzen_verified INTEGER,
                    gentzen_sequent TEXT,
                    proof_tree_json TEXT,
                    provability_score REAL,
                    tokens_saved_pct INTEGER,
                    clarity_score REAL,
                    provenance_json TEXT,
                    created_at TEXT,
                    curry_howard_term TEXT DEFAULT '',
                    epistemic_status TEXT DEFAULT 'PROVEN_TRUE',
                    grounding_score REAL DEFAULT 1.0
                )
            """)
            # Schema auto-migration for existing databases
            for col, col_def in [
                ("curry_howard_term", "TEXT DEFAULT ''"),
                ("epistemic_status", "TEXT DEFAULT 'PROVEN_TRUE'"),
                ("grounding_score", "REAL DEFAULT 1.0"),
            ]:
                try:
                    cursor.execute(f"ALTER TABLE samples ADD COLUMN {col} {col_def}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

    def save_sample(self, sample: DatasetSample) -> str:
        from core.kg_store import stamp_kg_id
        sample.provenance_json = stamp_kg_id(sample.provenance_json, self.kg_id)
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO samples (
                    sample_id, doc_name, question, reasoning, answer, naive_cot,
                    status, gentzen_verified, gentzen_sequent, proof_tree_json,
                    provability_score, tokens_saved_pct, clarity_score,
                    provenance_json, created_at, curry_howard_term, epistemic_status,
                    grounding_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample.sample_id,
                sample.doc_name,
                sample.question,
                sample.reasoning,
                sample.answer,
                sample.naive_cot,
                sample.status,
                1 if sample.gentzen_verified else 0,
                sample.gentzen_sequent,
                sample.proof_tree_json,
                sample.provability_score,
                sample.tokens_saved_pct,
                sample.clarity_score,
                sample.provenance_json,
                sample.created_at,
                sample.curry_howard_term,
                sample.epistemic_status,
                sample.grounding_score,
            ))
            conn.commit()
        return sample.sample_id

    def get_samples(self, status: Optional[str] = None) -> List[DatasetSample]:
        samples = []
        with self._connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("SELECT * FROM samples WHERE status = ? ORDER BY created_at DESC", (status,))
            else:
                cursor.execute("SELECT * FROM samples ORDER BY created_at DESC")
            rows = cursor.fetchall()
            for r in rows:
                ch_term = r[15] if len(r) > 15 and r[15] is not None else ""
                ep_stat = r[16] if len(r) > 16 and r[16] is not None else ("PROVEN_TRUE" if bool(r[7]) else "INTUITIONISTIC_UNDECIDED")
                gr_score = float(r[17]) if len(r) > 17 and r[17] is not None else 1.0
                samples.append(DatasetSample(
                    sample_id=r[0],
                    doc_name=r[1],
                    question=r[2],
                    reasoning=r[3],
                    answer=r[4],
                    naive_cot=r[5] or "",
                    status=r[6],
                    gentzen_verified=bool(r[7]),
                    gentzen_sequent=r[8] or "",
                    proof_tree_json=r[9] or "{}",
                    provability_score=r[10] or 0.0,
                    tokens_saved_pct=r[11] or 0,
                    clarity_score=r[12] or 0.0,
                    provenance_json=r[13] or "{}",
                    created_at=r[14],
                    curry_howard_term=ch_term,
                    epistemic_status=ep_stat,
                    grounding_score=gr_score,
                ))
        return samples

    def update_sample(self, sample: DatasetSample):
        self.save_sample(sample)

    def delete_sample(self, sample_id: str):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM samples WHERE sample_id = ?", (sample_id,))
            conn.commit()

    def clear_all(self, doc_name: Optional[str] = None):
        """Clear all samples, or only samples belonging to a specific document."""
        with self._connection() as conn:
            cursor = conn.cursor()
            if doc_name and doc_name != "All Documents":
                cursor.execute("DELETE FROM samples WHERE doc_name = ?", (doc_name,))
            else:
                cursor.execute("DELETE FROM samples")
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        samples = self.get_samples()
        total = len(samples)
        if total == 0:
            return {
                "total_samples": 0,
                "accepted_count": 0,
                "verified_count": 0,
                "acceptance_rate": 0.0,
                "verification_rate": 0.0,
                "avg_tokens_saved_pct": 0,
                "avg_clarity": 0.0,
            }
        accepted = sum(1 for s in samples if s.status == "accepted")
        verified = sum(1 for s in samples if s.gentzen_verified)
        tokens_saved = [s.tokens_saved_pct for s in samples if s.tokens_saved_pct > 0]
        avg_saved = int(sum(tokens_saved) / len(tokens_saved)) if tokens_saved else 0
        clarities = [s.clarity_score for s in samples if s.clarity_score > 0]
        avg_clarity = round(sum(clarities) / len(clarities), 2) if clarities else 0.0
        return {
            "total_samples": total,
            "accepted_count": accepted,
            "verified_count": verified,
            "acceptance_rate": round(accepted / total, 2),
            "verification_rate": round(verified / total, 2),
            "avg_tokens_saved_pct": avg_saved,
            "avg_clarity": avg_clarity,
        }

    def split_map(self, samples: List["DatasetSample"], strategy: str = "auto", ratios=DEFAULT_RATIOS) -> Dict[str, str]:
        """{sample_id: split}, grouped by document, then KG path / chunk (see core/dataset_splits.py)."""
        provs = {}
        for s in samples:
            try:
                provs[s.sample_id] = json.loads(s.provenance_json) if s.provenance_json else {}
            except Exception:
                provs[s.sample_id] = {}
        docs = {s.sample_id: sample_documents(s.doc_name, provs[s.sample_id]) for s in samples}
        groups = document_groups(docs.values())
        items = [(s.sample_id, {"document": groups.get(docs[s.sample_id][0]) if docs[s.sample_id] else None,
                                "path": path_key(provs[s.sample_id]), "family": s.sample_id}) for s in samples]
        mapping, info = assign_splits(items, strategy, ratios)
        self.last_split_info = info
        return mapping

    def global_split_map(self, strategy: str = "auto", ratios=DEFAULT_RATIOS) -> Dict[str, str]:
        """
        Splits computed once over every sample in the database, so a sample has the same split in the SFT
        JSONL, the DPO pairs and the Parquet file. (Splitting each export's own filtered subset assigned the
        same id to different splits in different files: train/test leakage between SFT and DPO.)
        """
        everything = self.get_samples()
        return self.split_map(everything, strategy, ratios) if everything else {}

    def _export_samples(self, only_accepted: bool = False, only_verified: bool = False,
                        drop_negative_savings: bool = True, doc_name: Optional[str] = None,
                        min_clarity: float = 0.0) -> List[DatasetSample]:
        """
        Samples for an export. only_verified keeps kernel-verified (PROVEN_TRUE) samples only.
        drop_negative_savings drops samples whose ClearCoT trace is longer than the naive one
        (tokens_saved_pct < 0): they contradict the point of the dataset.
        """
        samples = self.get_samples()
        if only_accepted:
            samples = [s for s in samples if s.status == "accepted"]
        if only_verified:
            samples = [s for s in samples if s.gentzen_verified]
        if drop_negative_savings:
            samples = [s for s in samples if (s.tokens_saved_pct or 0) >= 0]
        if doc_name:
            samples = [s for s in samples if s.doc_name == doc_name]
        if min_clarity and min_clarity > 0:
            samples = [s for s in samples if (s.clarity_score or 0) >= min_clarity]
        return samples

    def export_jsonl(self, output_path: str, only_accepted: bool = False, split_strategy: str = "auto",
                     only_verified: bool = False, doc_name: Optional[str] = None, min_clarity: float = 0.0) -> int:
        samples = self._export_samples(only_accepted, only_verified, doc_name=doc_name, min_clarity=min_clarity)
        splits = self.global_split_map(split_strategy) if samples else {}
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                clean_r = make_reasoning_direct(TAG_CLEAN_REGEX.sub('', s.reasoning or "").strip())
                clean_a = TAG_CLEAN_REGEX.sub('', s.answer or "").strip()
                line = {
                    "id": s.sample_id,
                    "split": splits.get(s.sample_id, "train"),
                    "document": s.doc_name,
                    "question": s.question,
                    "reasoning": clean_r,
                    "answer": clean_a,
                    "conversations": [
                        {"role": "user", "content": s.question},
                        {"role": "assistant", "content": f"<think>\n{clean_r}\n</think>\n\n{clean_a}"}
                    ],
                    "gentzen_verified": s.gentzen_verified,
                    "sequent": s.gentzen_sequent,
                    "curry_howard_term": s.curry_howard_term,
                    "epistemic_status": s.epistemic_status,
                    "tokens_saved_pct": s.tokens_saved_pct,
                    "clarity_score": s.clarity_score,
                    "grounding_score": s.grounding_score,
                    "provenance": _safe_json(s.provenance_json),
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
        return count

    def export_parquet(self, output_path: str, only_accepted: bool = False, split_strategy: str = "auto",
                       only_verified: bool = False, doc_name: Optional[str] = None, min_clarity: float = 0.0) -> int:
        """Export dataset as an Apache Parquet table for high-throughput training harnesses."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("Pandas not installed; cannot export Parquet.")
            return 0
        samples = self._export_samples(only_accepted, only_verified, doc_name=doc_name, min_clarity=min_clarity)
        if not samples:
            return 0
        splits = self.global_split_map(split_strategy)
        records = [
            {
                "id": s.sample_id,
                "split": splits.get(s.sample_id, "train"),
                "document": s.doc_name,
                "question": s.question,
                "reasoning": make_reasoning_direct(TAG_CLEAN_REGEX.sub('', s.reasoning or "").strip()),
                "answer": TAG_CLEAN_REGEX.sub('', s.answer or "").strip(),
                "gentzen_verified": s.gentzen_verified,
                "sequent": s.gentzen_sequent,
                "curry_howard_term": s.curry_howard_term,
                "epistemic_status": s.epistemic_status,
                "tokens_saved_pct": s.tokens_saved_pct,
                "clarity_score": s.clarity_score,
                "status": s.status,
                "created_at": s.created_at,
            }
            for s in samples
        ]
        df = pd.DataFrame(records)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return len(records)

    SYNTHETIC_NAIVE_PREFIX = "Okay, let me think through this question carefully. We need to analyze"

    def _naive_is_synthetic(self, s: "DatasetSample") -> bool:
        try:
            gen = (json.loads(s.provenance_json or "{}") or {}).get("generation") or {}
        except Exception:
            gen = {}
        return bool(gen.get("naive_cot_synthetic")) or (s.naive_cot or "").strip().startswith(self.SYNTHETIC_NAIVE_PREFIX)

    def export_dpo_pairs(self, output_path: str, only_accepted: bool = True, skip_synthetic_rejected: bool = True,
                         split_strategy: str = "auto", only_verified: bool = False, doc_name: Optional[str] = None,
                         min_clarity: float = 0.0) -> int:
        """
        Export contrastive DPO preference pairs (Chosen: ClearCoT <think>, Rejected: Naive bloated CoT <think>).
        only_accepted: the chosen side must be a sample you (or the batch gate) accepted; rejected and draft
        samples never become "chosen". skip_synthetic_rejected: drop pairs whose rejected side would be the
        fixed template (the model would only learn to avoid one phrasing).
        """
        samples = self._export_samples(only_accepted, only_verified, doc_name=doc_name, min_clarity=min_clarity)
        if skip_synthetic_rejected:
            samples = [s for s in samples if s.naive_cot and not self._naive_is_synthetic(s)]
        splits = self.global_split_map(split_strategy) if samples else {}
        self.last_dpo_skipped = {"meta_leak": 0}
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                if not s.reasoning or not s.answer:
                    continue
                # Skip contaminated samples where tags leaked
                if "<reasoning>" in s.answer or "<answer>" in s.reasoning:
                    continue
                if s.reasoning.strip() == s.answer.strip():
                    continue

                clean_r = make_reasoning_direct(TAG_CLEAN_REGEX.sub('', s.reasoning).strip())
                clean_a = TAG_CLEAN_REGEX.sub('', s.answer).strip()

                # Build a distinct, conversational, or unstructured rejected reasoning trace
                if s.naive_cot and s.naive_cot.strip() != clean_r and s.naive_cot.strip() != clean_a and len(s.naive_cot.strip().split()) > 25:
                    rejected_r = TAG_CLEAN_REGEX.sub('', s.naive_cot).strip()
                else:
                    # Construct an authentic verbose, rambling "naive" baseline reasoning trace for contrast
                    topic = s.question.rstrip("?").strip()
                    if topic.lower().startswith(("how ", "why ", "what ", "in ")):
                        topic = re.sub(r'^(?:how\s+does|how\s+do|why\s+does|what\s+is|what\s+are|in\s+what\s+way\s+does)\s+', '', topic, flags=re.IGNORECASE)

                    rejected_r = (
                        f"Okay, let me think through this question carefully. We need to analyze {topic.strip()}. "
                        f"First, looking at the general background, there are several distinct factors to consider. "
                        f"Wait, does the primary principle directly imply this conclusion? It's not immediately obvious, "
                        f"because there are competing interpretations and multiple definitions involved. "
                        f"So if we examine the broader context without strict formal logical steps, "
                        f"it seems that {(clean_a[:1].lower() + clean_a[1:].rstrip('.')) or 'this'} makes sense as the general outcome. "
                        f"Yes, that should be the main takeaway."
                    )

                # Purge any accidental prompt leakage from rejected reasoning
                rejected_r = re.sub(r'\b(?:the\s+question\s+(?:is\s+)?asking\s+for|the\s+answer\s+should\s+be)\s+(?:a\s+)?(?:verbose|rambling|internal\s+monologue|typical\s+conversational\s+markers)[^.]*\.\s*', '', rejected_r, flags=re.IGNORECASE)
                rejected_r = re.sub(r'^(?:##\s*Reasoning\s*)?Okay,\s*let\s*me\s*(?:carefully\s*)?(?:examine|think|look|unpack)[^.]*?(?:asking\s+for|should\s+be|typical)\s+(?:a\s+)?(?:verbose|rambling|internal\s+monologue|conversational\s+markers)[^.]*\.\s*', '', rejected_r, flags=re.IGNORECASE)
                rejected_r = TAG_CLEAN_REGEX.sub('', rejected_r).strip()

                # A negative that echoes its generation instructions is trivially separable: never export it
                if naive_trace_leak(rejected_r):
                    self.last_dpo_skipped["meta_leak"] += 1
                    continue

                # Ensure rejected reasoning never accidentally matches chosen reasoning or answer
                if rejected_r == clean_r or rejected_r == clean_a:
                    rejected_r = f"To answer the question, we can deduce that {clean_a.lower()} without formal constraints."

                pair = {
                    "id": s.sample_id,
                    "split": splits.get(s.sample_id, "train"),
                    "prompt": s.question,
                    "chosen": f"<think>\n{clean_r}\n</think>\n\n{clean_a}",
                    "rejected": f"<think>\n{rejected_r}\n</think>\n\n{clean_a}",
                    "metadata": {
                        "tokens_saved_pct": s.tokens_saved_pct,
                        "gentzen_verified": s.gentzen_verified,
                        "curry_howard_term": s.curry_howard_term,
                        "epistemic_status": s.epistemic_status,
                        "grounding_score": s.grounding_score,
                    }
                }
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                count += 1

        return count

    def export_layout_blocks_jsonl(self, blocks: List[Any], output_path: str) -> int:
        """Export raw layout-analyzed blocks with bounding boxes and labels to JSONL."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for b in blocks:
                rec = {
                    "block_id": getattr(b, "block_id", f"b_{count}"),
                    "page_number": getattr(b, "page_number", 1),
                    "label": getattr(b, "label", getattr(b, "block_type", "text")),
                    "bbox": getattr(b, "bbox", []),
                    "text": getattr(b, "text", ""),
                    "header_level": getattr(b, "header_level", 0),
                    "reading_order": getattr(b, "reading_order", count),
                    "confidence": getattr(b, "confidence", 1.0),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
        return count

    def export_chunks_parquet(self, chunks: List[Any], output_path: str) -> int:
        """Export extracted knowledge chunks with embeddings and provenance to Apache Parquet."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("Pandas not installed; cannot export Parquet.")
            return 0
        if not chunks:
            return 0
        records = [
            {
                "chunk_id": c.chunk_id,
                "doc_name": c.doc_name,
                "page_number": c.page_number,
                "section_title": c.section_title,
                "chunk_type": c.chunk_type,
                "token_estimate": c.token_estimate,
                "text": c.text,
                "bbox": str(getattr(c, "bbox", [])) if getattr(c, "bbox", None) else "",
                "layout_label": getattr(c, "layout_label", "") or c.chunk_type,
                "has_embedding": c.embedding is not None,
                "embedding": c.embedding if c.embedding is not None else [],
            }
            for c in chunks
        ]
        df = pd.DataFrame(records)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return len(records)

    def export_clean_markdown(self, pages: List[Any], output_path: str) -> int:
        """Reconstruct clean hierarchical markdown from structured pages without visual noise."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        for page in pages:
            blocks = getattr(page, "blocks", [])
            for b in blocks:
                b_type = getattr(b, "block_type", "paragraph")
                text = getattr(b, "text", "").strip()
                if not text:
                    continue
                if b_type == "heading":
                    h_level = getattr(b, "header_level", 2)
                    prefix = "#" * max(1, min(h_level or 2, 4))
                    lines.append(f"\n{prefix} {text}\n")
                elif b_type == "table":
                    lines.append(f"\n{text}\n")
                elif b_type == "list-item":
                    lines.append(f"- {text}")
                elif b_type in ("footnote", "page-header", "page-footer", "reference", "references"):
                    continue
                else:
                    lines.append(f"\n{text}\n")
        content = "\n".join(lines).strip()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return len(content)

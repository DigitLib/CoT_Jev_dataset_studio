"""
Jev Dataset Manager (core/jev_dataset_manager.py).
Handles SQLite persistence and multi-format exports for Jev Decision Datasets,
including:
  1. `typed-decisions` Parquet export (with dynamic flattened target columns)
  2. `S1MB` / Open-Jev Parquet export (with judgment distributions & input state_json)
  3. SFT / Multi-task Jev JSONL export.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import sqlite3
import json
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import logging
import contextlib

from .dataset_splits import assign_splits, path_key, DEFAULT_RATIOS, document_groups, sample_documents

logger = logging.getLogger(__name__)


@dataclass
class JevSample:
    sample_id: str
    doc_name: str
    workflow: str
    split: str = "train"
    state_json: str = "{}"
    questions_json: str = "{}"
    gold_json: str = "{}"
    factors_json: str = "{}"
    policy_rules_json: str = "[]"
    target_decision: str = ""
    gentzen_verified: bool = True
    gentzen_sequent: str = ""
    curry_howard_term: str = ""
    epistemic_status: str = "PROVEN_TRUE"
    status: str = "accepted"  # "draft", "accepted", "rejected"
    provenance_json: str = "{}"
    created_at: str = ""

    def __post_init__(self):
        if not self.sample_id:
            self.sample_id = f"JEV-{uuid.uuid4().hex[:12].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# Convenience alias for JevSample
JevDecisionSample = JevSample


class JevDatasetManager:
    """Manages SQLite storage and multi-format dataset exports for Jev Decision Scenarios."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            from config import DB_PATH
            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.kg_id: Optional[str] = None   # active knowledge graph; stamped into new samples' provenance
        self._init_db()

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
                CREATE TABLE IF NOT EXISTS jev_samples (
                    sample_id TEXT PRIMARY KEY,
                    doc_name TEXT,
                    workflow TEXT,
                    split TEXT,
                    state_json TEXT,
                    questions_json TEXT,
                    gold_json TEXT,
                    factors_json TEXT,
                    policy_rules_json TEXT,
                    target_decision TEXT,
                    gentzen_verified INTEGER,
                    gentzen_sequent TEXT,
                    curry_howard_term TEXT,
                    epistemic_status TEXT,
                    status TEXT,
                    provenance_json TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()

    def save_scenario(self, scenario: Dict[str, Any], status: str) -> str:
        """Store a scenario dict from JevScenarioGenerator (GUI and tools/run_batch.py share this)."""
        prov = scenario.get("provenance") or {}
        return self.save_sample(JevSample(
            sample_id="",
            doc_name=prov.get("doc_name", "unknown"),
            workflow=scenario.get("workflow", "domain_policy"),
            split="train",
            state_json=json.dumps(scenario.get("state", {})),
            questions_json=json.dumps(scenario.get("questions", {})),
            gold_json=json.dumps(scenario.get("gold", {}), default=str),
            factors_json=json.dumps(scenario.get("factors", {})),
            policy_rules_json=json.dumps(scenario.get("policy_rules", [])),
            target_decision=scenario.get("target_decision", ""),
            gentzen_verified=bool(scenario.get("gentzen_verified", False)),   # never assumed
            gentzen_sequent=scenario.get("gentzen_sequent", ""),
            curry_howard_term=scenario.get("curry_howard_term", ""),
            epistemic_status=scenario.get("epistemic_status") or "INTUITIONISTIC_UNDECIDED",
            status=status,
            provenance_json=json.dumps(prov, default=str),
        ))

    def save_sample(self, sample: JevSample) -> str:
        from core.kg_store import stamp_kg_id
        sample.provenance_json = stamp_kg_id(sample.provenance_json, self.kg_id)
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO jev_samples (
                    sample_id, doc_name, workflow, split, state_json, questions_json,
                    gold_json, factors_json, policy_rules_json, target_decision,
                    gentzen_verified, gentzen_sequent, curry_howard_term, epistemic_status,
                    status, provenance_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample.sample_id,
                sample.doc_name,
                sample.workflow,
                sample.split,
                sample.state_json,
                sample.questions_json,
                sample.gold_json,
                sample.factors_json,
                sample.policy_rules_json,
                sample.target_decision,
                1 if sample.gentzen_verified else 0,
                sample.gentzen_sequent,
                sample.curry_howard_term,
                sample.epistemic_status,
                sample.status,
                sample.provenance_json,
                sample.created_at,
            ))
            conn.commit()
        return sample.sample_id

    def get_samples(
        self,
        status: Optional[str] = None,
        workflow: Optional[str] = None,
    ) -> List[JevSample]:
        samples = []
        with self._connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM jev_samples WHERE 1=1"
            params = []
            if status:
                query += " AND status = ?"
                params.append(status)
            if workflow and workflow != "All Workflows":
                query += " AND workflow = ?"
                params.append(workflow)
            query += " ORDER BY created_at DESC"

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            for r in rows:
                samples.append(JevSample(
                    sample_id=r[0],
                    doc_name=r[1],
                    workflow=r[2],
                    split=r[3] or "train",
                    state_json=r[4] or "{}",
                    questions_json=r[5] or "{}",
                    gold_json=r[6] or "{}",
                    factors_json=r[7] or "{}",
                    policy_rules_json=r[8] or "[]",
                    target_decision=r[9] or "",
                    gentzen_verified=bool(r[10]),
                    gentzen_sequent=r[11] or "",
                    curry_howard_term=r[12] or "",
                    epistemic_status=r[13] or "PROVEN_TRUE",
                    status=r[14] or "accepted",
                    provenance_json=r[15] or "{}",
                    created_at=r[16] or "",
                ))
        return samples

    def delete_sample(self, sample_id: str):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM jev_samples WHERE sample_id = ?", (sample_id,))
            conn.commit()

    def clear_all(self, workflow: Optional[str] = None):
        with self._connection() as conn:
            cursor = conn.cursor()
            if workflow and workflow != "All Workflows":
                cursor.execute("DELETE FROM jev_samples WHERE workflow = ?", (workflow,))
            else:
                cursor.execute("DELETE FROM jev_samples")
            conn.commit()

    def purge_legacy_samples(self) -> int:
        """Removes legacy placeholder records (e.g. System_Architecture, empty observations, document_compliance defaults)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM jev_samples
                WHERE state_json LIKE '%System_Architecture%'
                   OR state_json LIKE '%"observations": {}%'
                   OR state_json LIKE '%"grounded_premise": ""%'
                   OR workflow = 'document_compliance'
                """
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    def get_stats(self) -> Dict[str, Any]:
        samples = self.get_samples()
        total = len(samples)
        if total == 0:
            return {
                "total_samples": 0,
                "verified_count": 0,
                "verification_rate": 0.0,
                "unique_workflows": 0,
                "workflow_breakdown": {},
            }
        verified = sum(1 for s in samples if s.gentzen_verified)
        workflows = {}
        for s in samples:
            workflows[s.workflow] = workflows.get(s.workflow, 0) + 1

        return {
            "total_samples": total,
            "verified_count": verified,
            "verification_rate": round(verified / total, 2),
            "unique_workflows": len(workflows),
            "workflow_breakdown": workflows,
        }

    # ------------------------------------------------------------------ splits
    @staticmethod
    def _prov(s: "JevSample") -> Dict[str, Any]:
        try:
            return json.loads(s.provenance_json) if s.provenance_json else {}
        except Exception:
            return {}

    def _group_keys(self, s: "JevSample", doc_groups: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        prov = self._prov(s)
        family = ((prov.get("counterfactual") or {}).get("of")) or s.sample_id
        docs = sample_documents(s.doc_name, prov)
        doc_key = (doc_groups or {}).get(docs[0], docs[0]) if docs else None
        return {"document": doc_key, "path": path_key(prov) or f"family:{family}", "family": family}

    def _doc_groups(self, samples: List["JevSample"]) -> Dict[str, str]:
        return document_groups(sample_documents(s.doc_name, self._prov(s)) for s in samples)

    def split_map(self, samples: List["JevSample"], strategy: str = "auto",
                  ratios=DEFAULT_RATIOS) -> Dict[str, str]:
        """{sample_id: split}; parent scenarios and their counterfactual variants always share a split,
        and documents joined by a cross-document scenario are one group."""
        groups = self._doc_groups(samples)
        self._last_doc_groups = groups
        mapping, info = assign_splits([(s.sample_id, self._group_keys(s, groups)) for s in samples], strategy, ratios)
        self.last_split_info = info
        return mapping

    def _group_id(self, s: "JevSample", level: str) -> str:
        key = (self._group_keys(s, getattr(self, "_last_doc_groups", None)).get(level)
               if level in ("document", "path", "family") else None)
        key = key or (s.doc_name + s.workflow)
        return f"{s.workflow}:{hashlib.sha256(str(key).encode('utf-8')).hexdigest()}"

    @staticmethod
    def export_gold(s: "JevSample") -> Dict[str, Any]:
        """
        Gold labels as they may be exported. For a sample the kernel did not verify, the label is the LLM's
        intended answer, not a fact: it is moved to `intended_label` and `label` / `probabilities` are cleared,
        so an export that includes unverified rows (only_verified=False) never presents a guess as gold.
        """
        gold = json.loads(s.gold_json) if s.gold_json else {}
        if s.gentzen_verified or not isinstance(gold, dict):
            return gold
        out = {}
        for q_id, g in gold.items():
            if isinstance(g, dict):
                g = dict(g)
                g["intended_label"] = g.get("label")
                g["label"] = None
                g["confidence"] = None
                g["probabilities"] = {}
            out[q_id] = g
        return out

    def _export_samples(self, only_accepted: bool, only_verified: bool = False) -> List["JevSample"]:
        samples = self.get_samples()
        if only_accepted:
            samples = [s for s in samples if s.status == "accepted"]
        if only_verified:
            samples = [s for s in samples if s.gentzen_verified]
        return samples

    def export_typed_decisions_parquet(
        self,
        output_path: str,
        only_accepted: bool = False,
        split_strategy: str = "auto",
        column_mode: str = "question_id",
        only_verified: bool = False,
    ) -> int:
        """
        Exports samples into the canonical Jev `typed-decisions` Parquet schema:
        ['id', 'workflow', 'split', 'state', 'questions', 'gold', 'factors', 'n_questions',
         '{q}__label', '{q}__confidence', '{q}__probabilities']
        column_mode "question_id": target columns named after each question id (canonical; fine when ids
        repeat across samples). "positional": q1__label, q1__id, q2__label ... so LLM-chosen ids that
        differ per sample do not create one sparse column set per sample.
        split_strategy: see core/dataset_splits.py ("stored" keeps the split saved with each sample).
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("Pandas not installed; cannot export Parquet.")
            return 0

        samples = self._export_samples(only_accepted, only_verified)
        if not samples:
            return 0
        splits = self.split_map(samples, split_strategy)

        rows = []
        for s in samples:
            state_obj = json.loads(s.state_json) if s.state_json else {}
            questions_obj = json.loads(s.questions_json) if s.questions_json else {}
            gold_obj = self.export_gold(s)
            factors_obj = json.loads(s.factors_json) if s.factors_json else {}

            row: Dict[str, Any] = {
                "id": s.sample_id,
                "workflow": s.workflow,
                "split": splits.get(s.sample_id, s.split),
                "state": json.dumps(state_obj),
                "questions": json.dumps(questions_obj),
                "gold": json.dumps(gold_obj),
                "factors": json.dumps(factors_obj),
                "n_questions": len(questions_obj),
            }

            # Flatten dynamic target columns for each decision question
            for pos, (q_id, q_gold) in enumerate(gold_obj.items(), 1):
                label = q_gold.get("label", "")
                conf = q_gold.get("confidence", 1.0)
                probs = q_gold.get("probabilities", {})
                col = f"q{pos}" if column_mode == "positional" else q_id
                if column_mode == "positional":
                    row[f"{col}__id"] = q_id
                row[f"{col}__label"] = "" if label is None else str(label)
                row[f"{col}__confidence"] = float(conf) if conf is not None else float("nan")
                row[f"{col}__probabilities"] = json.dumps(probs)
                row[f"{col}__verification"] = str(q_gold.get("verification", ""))

            rows.append(row)

        df = pd.DataFrame(rows)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return len(rows)

    def export_open_jev_parquet(
        self,
        output_path: str,
        only_accepted: bool = False,
        split_strategy: str = "auto",
        include_decisions: bool = True,
        include_rules: bool = True,
        only_verified: bool = False,
    ) -> int:
        """
        Exports samples into the canonical Open-Jev `S1MB` Parquet schema:
        ['input', 'targets', 'split', 'case_id', 'group_id', 'language', 'input_hash']
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("Pandas not installed; cannot export Parquet.")
            return 0

        samples = self._export_samples(only_accepted, only_verified)
        if not samples:
            return 0
        splits = self.split_map(samples, split_strategy)
        level = getattr(self, "last_split_info", {}).get("level", "document")

        records = []
        for s in samples:
            state_obj = json.loads(s.state_json) if s.state_json else {}
            questions_obj = json.loads(s.questions_json) if s.questions_json else {}
            gold_obj = self.export_gold(s)

            targets = []
            for q_id, q_data in questions_obj.items():
                g = gold_obj.get(q_id, {})
                criteria = q_data.get("criteria", {})
                ids = list(criteria.keys()) or [g.get("label", "true")]
                probs_dict = g.get("probabilities", {})
                probs_list = [probs_dict.get(k, 1.0 / max(1, len(ids))) for k in ids]

                targets.append({
                    "decision_id": q_id,
                    "kind": "judgment_distribution",
                    "annotation_kind": "hard_label",
                    "ids": ids,
                    "probabilities": probs_list,
                    "label": g.get("label", ""),
                })

            # The model input must contain everything the label depends on: the state, the decisions to make
            # (ids, types, options, criteria) and the policy rules that decide them.
            if include_decisions or include_rules:
                inp: Dict[str, Any] = {"state": state_obj}
                if include_decisions:
                    inp["decisions"] = [
                        {"decision_id": q_id, "type": q.get("type", "choice"),
                         "instructions": q.get("instructions", ""),
                         "options": list((q.get("criteria") or {}).keys()) or q.get("options") or [],
                         "criteria": q.get("criteria") or {}}
                        for q_id, q in questions_obj.items()]
                if include_rules:
                    inp["policy_rules"] = json.loads(s.policy_rules_json) if s.policy_rules_json else []
                state_str = json.dumps(inp, ensure_ascii=False, sort_keys=True)
            else:
                state_str = json.dumps(state_obj)
            inp_hash = hashlib.sha256(state_str.encode("utf-8")).hexdigest()

            records.append({
                "input": {"state_json": state_str},
                "targets": targets,
                "split": splits.get(s.sample_id, s.split),
                "case_id": f"{s.workflow}:{s.sample_id}",
                "group_id": self._group_id(s, level),
                "language": "en",
                "input_hash": inp_hash,
            })

        df = pd.DataFrame(records)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return len(records)

    def export_jev_jsonl(
        self,
        output_path: str,
        only_accepted: bool = False,
        split_strategy: str = "auto",
        only_verified: bool = False,
    ) -> int:
        """Exports rich JSONL records containing both Jev JSON structures and formal proof certificates."""
        samples = self._export_samples(only_accepted, only_verified)
        splits = self.split_map(samples, split_strategy) if samples else {}
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                line = {
                    "id": s.sample_id,
                    "doc_name": s.doc_name,
                    "workflow": s.workflow,
                    "split": splits.get(s.sample_id, s.split),
                    "state": json.loads(s.state_json) if s.state_json else {},
                    "questions": json.loads(s.questions_json) if s.questions_json else {},
                    "gold": self.export_gold(s),
                    "factors": json.loads(s.factors_json) if s.factors_json else {},
                    "policy_rules": json.loads(s.policy_rules_json) if s.policy_rules_json else [],
                    "gentzen_verified": s.gentzen_verified,
                    "sequent": s.gentzen_sequent,
                    "curry_howard_term": s.curry_howard_term,
                    "epistemic_status": s.epistemic_status,
                    "provenance": json.loads(s.provenance_json) if s.provenance_json else {},
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
        return count


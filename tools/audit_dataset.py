import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.question_generator import QuestionGenerator

def audit(clean=False):
    repo_root = Path(__file__).parent.parent
    db_path = repo_root / "data" / "clearcot.db"
    if not db_path.exists():
        print(f"Database does not exist at {db_path}.")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT sample_id, status, question, reasoning, answer FROM samples")
    rows = cursor.fetchall()
    print(f"Total samples in DB: {len(rows)}")

    valid_ids = []
    invalid_ids = []
    seen_qs = set()

    for sample_id, status, question, reasoning, answer in rows:
        sample_dict = {
            "question": question,
            "reasoning": reasoning,
            "answer": answer
        }
        is_valid, reason = QuestionGenerator.validate_sample_quality_with_reason(sample_dict)
        q_norm = (question or "").strip().lower()
        if is_valid:
            if q_norm in seen_qs:
                is_valid = False
                reason = f"Duplicate question: '{question[:50]}...'"
            else:
                seen_qs.add(q_norm)

        if is_valid:
            valid_ids.append(sample_id)
            print(f"  [VALID] {sample_id} ({status}): {question[:60]}... -> {answer[:40]}...")
        else:
            invalid_ids.append((sample_id, reason))
            print(f"  [INVALID] {sample_id} ({status}): {reason}")
            print(f"            Q: {question[:70]}")
            print(f"            A: {answer[:70]}")

    print(f"\nSummary: {len(valid_ids)} valid, {len(invalid_ids)} invalid.")

    if clean and invalid_ids:
        print("\nPurging invalid samples from DB...")
        for sample_id, _ in invalid_ids:
            cursor.execute("DELETE FROM samples WHERE sample_id = ?", (sample_id,))
        conn.commit()
        print(f"Successfully deleted {len(invalid_ids)} invalid samples from DB.")

    # Re-export clean datasets if clean or export requested
    if clean or "--export" in sys.argv:
        from core.dataset_manager import DatasetManager
        mgr = DatasetManager(str(db_path))
        sft_count = mgr.export_jsonl("data/exports/clearcot_sft_dataset.jsonl", only_accepted=False)
        dpo_count = mgr.export_dpo_pairs("data/exports/clearcot_dpo_pairs.jsonl")
        parquet_count = mgr.export_parquet("data/exports/clearcot_dataset.parquet", only_accepted=False)
        print(f"Re-exported clean SFT dataset ({sft_count} samples) to: data/exports/clearcot_sft_dataset.jsonl")
        print(f"Re-exported clean DPO pairs ({dpo_count} pairs) to: data/exports/clearcot_dpo_pairs.jsonl")
        print(f"Re-exported clean Parquet dataset ({parquet_count} samples) to: data/exports/clearcot_dataset.parquet")

    conn.close()

if __name__ == "__main__":
    clean_flag = "--clean" in sys.argv
    audit(clean=clean_flag)


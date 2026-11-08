"""
make_samples.py - cut the small, tracked example set (data/exports/samples/) from a full exports folder.

    python tools/make_samples.py SRC_EXPORTS_DIR [--out data/exports/samples] [--n 20]

Takes about N records per dataset family, stratified over split and epistemic status (Jev also over
workflow), and keeps the SAME ids across the three ClearCoT files and across the three Jev files, so a sample
can be followed from SFT to DPO to Parquet. Run tools/audit_exports.py on the result before committing it.
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _stratified(rows, key, n, seed=0):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    out, order = [], sorted(groups)
    while len(out) < n and any(groups.values()):
        for k in order:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def _read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _write_jsonl(p: Path, rows):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("--out", default=str(ROOT / "data" / "exports" / "samples"))
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    # ClearCoT: choose ids that have a DPO pair first, so all three files show the same samples
    sft = _read_jsonl(src / "clearcot_sft_dataset.jsonl")
    dpo = {r["id"]: r for r in _read_jsonl(src / "clearcot_dpo_pairs.jsonl")}
    if sft:
        with_pair = [r for r in sft if r["id"] in dpo] or sft
        pick = _stratified(with_pair, lambda r: (r.get("split"), r.get("epistemic_status")), a.n)
        ids = [r["id"] for r in pick]
        _write_jsonl(out / "clearcot_sft_dataset.jsonl", pick)
        _write_jsonl(out / "clearcot_dpo_pairs.jsonl", [dpo[i] for i in ids if i in dpo])
        pq = src / "clearcot_dataset.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
            df[df["id"].isin(ids)].to_parquet(out / "clearcot_dataset.parquet", index=False)
        print(f"ClearCoT: {len(pick)} samples ({sum(i in dpo for i in ids)} with DPO pairs)")

    jev = _read_jsonl(src / "jev_decisions_dataset.jsonl")
    if jev:
        pick = _stratified(jev, lambda r: (r.get("split"), r.get("epistemic_status"), r.get("workflow")), a.n)
        ids = {r["id"] for r in pick}
        _write_jsonl(out / "jev_decisions_dataset.jsonl", pick)
        td = src / "typed_decisions_dataset.parquet"
        if td.exists():
            df = pd.read_parquet(td)
            df[df["id"].isin(ids)].to_parquet(out / "typed_decisions_dataset.parquet", index=False)
        s1 = src / "open_jev_s1mb_dataset.parquet"
        if s1.exists():
            df = pd.read_parquet(s1)
            df[df["case_id"].astype(str).str.split(":").str[-1].isin(ids)].to_parquet(
                out / "open_jev_s1mb_dataset.parquet", index=False)
        print(f"Jev: {len(pick)} records")


if __name__ == "__main__":
    main()

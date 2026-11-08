"""
audit_exports.py - release checks for an exports folder (data/exports/ or data/exports/samples/).

    python tools/audit_exports.py                       # data/exports/
    python tools/audit_exports.py data/exports/samples  # the tracked samples
    python tools/audit_exports.py DIR --json report.json

Checks (exit code 1 if any FAIL):
  * a sample id has the same split in SFT JSONL, DPO JSONL and ClearCoT Parquet (and in the three Jev files)
  * no DPO `rejected` trace echoes its generation instructions
  * ClearCoT: gentzen_verified => PROVEN_TRUE; Jev: gentzen_verified => PROVEN_TRUE or NO_OPTION_PROVABLE,
    and every verified gold label has a checked certificate (or is a certified undecided)
  * no state-completed (PROVEN_UNDER_ASSUMPTIONS) record is marked verified
  * no local filesystem paths (/home/..., C:\\...) anywhere in the files
  * reports (WARN): label vocabulary per workflow, confidence spread, workflows, documents, splits, row counts
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.cot_generator import naive_trace_leak  # noqa: E402

LOCAL_PATH_RE = re.compile(r"(?:/home/[^/\s\"']+/|/Users/[^/\s\"']+/|[A-Za-z]:\\\\(?:Users|exper)\\\\)")
JEV_VERIFIED = {"PROVEN_TRUE", "NO_OPTION_PROVABLE"}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, level, check, detail):
        self.rows.append({"level": level, "check": check, "detail": detail})
        print(f"[{level:4}] {check}: {detail}")

    @property
    def failed(self):
        return any(r["level"] == "FAIL" for r in self.rows)


def _jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else None


def _parquet(p: Path):
    if not p.exists():
        return None
    import pandas as pd
    return pd.read_parquet(p)


def _split_consistency(rep, name, maps):
    maps = {k: v for k, v in maps.items() if v is not None}
    ids = set().union(*maps.values()) if maps else set()
    bad = [i for i in ids if len({m[i] for m in maps.values() if i in m}) > 1]
    if len(maps) < 2:
        rep.add("WARN", f"{name} split consistency", f"only {list(maps)} present")
    elif bad:
        rep.add("FAIL", f"{name} split consistency", f"{len(bad)} id(s) in different splits across {list(maps)}, e.g. {bad[:3]}")
    else:
        rep.add("PASS", f"{name} split consistency", f"{len(ids)} ids agree across {list(maps)}")


def audit(d: Path) -> Report:
    rep = Report()
    sft = _jsonl(d / "clearcot_sft_dataset.jsonl")
    dpo = _jsonl(d / "clearcot_dpo_pairs.jsonl")
    ccp = _parquet(d / "clearcot_dataset.parquet")
    jev = _jsonl(d / "jev_decisions_dataset.jsonl")
    tdp = _parquet(d / "typed_decisions_dataset.parquet")
    s1 = _parquet(d / "open_jev_s1mb_dataset.parquet")

    counts = {n: (len(x) if x is not None else None) for n, x in
              [("sft", sft), ("dpo", dpo), ("clearcot_parquet", ccp), ("jev", jev), ("typed_decisions", tdp), ("s1mb", s1)]}
    rep.add("INFO", "row counts", counts)

    # ---------------- ClearCoT
    _split_consistency(rep, "ClearCoT", {
        "sft": {r["id"]: r["split"] for r in sft} if sft else None,
        "dpo": {r["id"]: r["split"] for r in dpo} if dpo else None,
        "parquet": dict(zip(ccp["id"], ccp["split"])) if ccp is not None else None,
    })
    if dpo:
        leaks = [(r["id"], naive_trace_leak(r["rejected"])) for r in dpo if naive_trace_leak(r.get("rejected", ""))]
        rep.add("FAIL" if leaks else "PASS", "DPO negatives", f"{len(leaks)} echo their instructions, e.g. {leaks[:3]}"
                if leaks else f"{len(dpo)} rejected traces read as genuine reasoning")
    if sft:
        bad = [r["id"] for r in sft if r.get("gentzen_verified") and r.get("epistemic_status") != "PROVEN_TRUE"]
        rep.add("FAIL" if bad else "PASS", "ClearCoT verified => PROVEN_TRUE", f"{len(bad)} violations {bad[:3]}")
        saved = sorted(r.get("tokens_saved_pct") or 0 for r in sft)
        rep.add("INFO", "ClearCoT tokens_saved_pct", {"mean": round(sum(saved) / len(saved), 1),
                                                       "median": saved[len(saved) // 2], "min": saved[0], "max": saved[-1]})
        rep.add("INFO", "ClearCoT documents", dict(collections.Counter(r.get("document") for r in sft)))
        rep.add("INFO", "ClearCoT splits", dict(collections.Counter(r.get("split") for r in sft)))

    # ---------------- Jev
    _split_consistency(rep, "Jev", {
        "jsonl": {r["id"]: r["split"] for r in jev} if jev else None,
        "typed_decisions": dict(zip(tdp["id"], tdp["split"])) if tdp is not None else None,
        "s1mb": {str(c).split(":")[-1]: sp for c, sp in zip(s1["case_id"], s1["split"])} if s1 is not None else None,
    })
    if jev:
        bad_status, bad_cert, assumed = [], [], []
        labels = collections.defaultdict(collections.Counter)
        confs = collections.Counter()
        for r in jev:
            if r.get("gentzen_verified") and r.get("epistemic_status") not in JEV_VERIFIED:
                bad_status.append(r["id"])
            comp = ((r.get("provenance") or {}).get("nanogentzen") or {}).get("state_completion") or {}
            if r.get("gentzen_verified") and comp.get("applied"):
                assumed.append(r["id"])   # also catches records written before PROVEN_UNDER_ASSUMPTIONS existed
            for g in (r.get("gold") or {}).values():
                labels[r.get("workflow")][g.get("label")] += 1
                confs[round(float(g.get("confidence") or 0), 3)] += 1
                if g.get("verification") == "PROVEN_UNDER_ASSUMPTIONS" and r.get("gentzen_verified"):
                    assumed.append(r["id"])
                if r.get("gentzen_verified") and g.get("verification") == "PROVEN_TRUE" and not g.get("certificate_checked"):
                    bad_cert.append(r["id"])
        rep.add("FAIL" if bad_status else "PASS", "Jev verified => PROVEN_TRUE | NO_OPTION_PROVABLE",
                f"{len(bad_status)} violations {bad_status[:3]}")
        rep.add("FAIL" if bad_cert else "PASS", "Jev verified labels carry a checked certificate",
                f"{len(bad_cert)} violations {bad_cert[:3]}")
        rep.add("FAIL" if assumed else "PASS", "Jev state completion never verified", f"{len(assumed)} violations {assumed[:3]}")
        for wf, c in labels.items():
            long = [l for l in c if l and len(re.split(r"[_\s-]+", str(l))) >= 4]
            rep.add("WARN" if long or len(c) > 4 else "INFO", f"Jev labels ({wf})", dict(c))
        rep.add("INFO", "Jev confidence values", dict(confs.most_common(6)))
        rep.add("INFO", "Jev workflows", dict(collections.Counter(r.get("workflow") for r in jev)))
        rep.add("INFO", "Jev statuses", dict(collections.Counter(r.get("epistemic_status") for r in jev)))

    # ---------------- local paths
    hits = []
    for f in sorted(d.glob("*")):
        if f.suffix in (".jsonl", ".json", ".md"):
            text = f.read_text(encoding="utf-8", errors="replace")
        elif f.suffix == ".parquet":
            text = _parquet(f).astype(str).to_csv()
        else:
            continue
        m = LOCAL_PATH_RE.search(text)
        if m:
            hits.append(f"{f.name}: {text[max(0, m.start() - 20):m.end() + 40]!r}")
    rep.add("FAIL" if hits else "PASS", "no local filesystem paths", hits[:3] if hits else "none found")
    return rep


def main():
    ap = argparse.ArgumentParser(description="Release checks for exported datasets")
    ap.add_argument("dir", nargs="?", default=str(ROOT / "data" / "exports"))
    ap.add_argument("--json", help="write the report to this file")
    a = ap.parse_args()
    rep = audit(Path(a.dir))
    if a.json:
        Path(a.json).write_text(json.dumps(rep.rows, indent=1, default=str), encoding="utf-8")
    print("\nRESULT:", "FAIL" if rep.failed else "PASS")
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()

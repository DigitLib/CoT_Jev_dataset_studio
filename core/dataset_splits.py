"""
dataset_splits.py - deterministic, leakage-free train / validation / test splits.

Samples generated from the same document, the same KG path, or as counterfactual variants of one
scenario share text and facts. Splitting them independently leaks test content into training.
assign_splits() keeps every group in one split:

    strategy "document"  group = source document; documents joined by a cross-document sample
                         (a KG path through two same-topic PDFs) form one group     (strongest isolation)
             "path"      group = KG path / source chunk         (when there are too few documents)
             "family"    group = sample + its counterfactual variants
             "stored"    keep the split stored with each sample (no reassignment)
             "auto"      the first of document / path / family that yields at least `min_groups` groups
                         and gives every split at least half its target share

Groups are taken in a fixed (hash) order and each goes to the split furthest below its target share,
so the result is reproducible and close to the requested ratios.
"""
import hashlib
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SPLITS = ("train", "validation", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)
LEVELS = ("document", "path", "family")


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _place(groups: Dict[str, List[str]], target: List[float]) -> Tuple[Dict[str, str], List[int]]:
    """Greedy: biggest groups first (ties by hash), each to the split furthest below its target share."""
    counts = [0, 0, 0]
    out: Dict[str, str] = {}
    for key in sorted(groups, key=lambda k: (-len(groups[k]), _h(k))):
        need = [(target[i] - counts[i]) / target[i] if target[i] > 0 else float("-inf") for i in range(3)]
        i = max(range(3), key=lambda j: (need[j], -j))
        counts[i] += len(groups[key])
        for sid in groups[key]:
            out[sid] = SPLITS[i]
    return out, counts


def _balance(counts: List[int], target: List[float]) -> float:
    """Worst split's share of its target (1.0 = on target); splits with a zero target are ignored."""
    shares = [min(1.0, c / t) for c, t in zip(counts, target) if t > 0]
    return min(shares) if shares else 1.0


def assign_splits(
    items: Sequence[Tuple[str, Dict[str, str]]],
    strategy: str = "auto",
    ratios: Sequence[float] = DEFAULT_RATIOS,
    min_groups: int = 3,
    min_balance: float = 0.5,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """
    items: (sample_id, {"document": key, "path": key, "family": key}) per sample.
    Returns ({sample_id: split}, info) where info has the level used, the per-split counts and `balance`
    (the worst split's share of its target count).

    "auto" takes the first level (document, path, family) with at least `min_groups` groups whose split
    reaches `min_balance` -- three documents of 108, 22 and 1 samples have enough groups, but would give a
    1-sample test set, so the path level is used instead. If no level is balanced, the best one is used and
    info["warning"] says so. An explicit strategy is never changed, only warned about.
    """
    if strategy == "stored":
        return {}, {"level": "stored"}
    if strategy != "auto" and strategy not in LEVELS:
        raise ValueError(f"unknown split strategy {strategy!r}")
    levels = LEVELS if strategy == "auto" else (strategy,)

    total = len(items)
    r = [max(0.0, float(x)) for x in ratios]
    s = sum(r) or 1.0
    target = [total * x / s for x in r]

    tried: List[Dict[str, object]] = []
    best = None   # (balance, -order, level, groups, out, counts)
    for order, lv in enumerate(levels):
        groups: Dict[str, List[str]] = {}
        for sid, keys in items:
            groups.setdefault(keys.get(lv) or sid, []).append(sid)
        if strategy == "auto" and len(groups) < min_groups and lv != levels[-1]:
            tried.append({"level": lv, "groups": len(groups), "reason": f"fewer than {min_groups} groups"})
            continue
        out, counts = _place(groups, target)
        bal = _balance(counts, target)
        cand = (bal, -order, lv, groups, out, counts)
        if best is None or cand[:2] > best[:2]:
            best = cand
        if bal >= min_balance:
            best = cand
            break
        tried.append({"level": lv, "groups": len(groups), "counts": dict(zip(SPLITS, counts)),
                      "reason": f"unbalanced ({bal:.2f} of target in the worst split)"})

    bal, _, level, groups, out, counts = best
    info: Dict[str, object] = {"level": level, "groups": len(groups), "counts": dict(zip(SPLITS, counts)),
                               "balance": round(bal, 3)}
    tried = [t for t in tried if t["level"] != level]
    if tried:
        info["levels_skipped"] = tried
    if bal < min_balance:
        c = dict(zip(SPLITS, counts))
        info["warning"] = (f"the '{level}' split is unbalanced ({c}): a split has under {min_balance:.0%} of its "
                           "target; choose another split level" if strategy != "auto" else
                           f"no split level reaches {min_balance:.0%} of every target; '{level}' is the most "
                           f"balanced ({c})")
    return out, info


def path_key(provenance: Dict) -> Optional[str]:
    """KG path / source-chunk identity of a sample's provenance."""
    if not isinstance(provenance, dict):
        return None
    if provenance.get("path_repr"):
        return "path:" + str(provenance["path_repr"])
    if provenance.get("path_nodes"):
        return "path:" + " -> ".join(map(str, provenance["path_nodes"]))
    cids = provenance.get("chunk_ids") or ([provenance["chunk_id"]] if provenance.get("chunk_id") else [])
    if cids:
        return "chunks:" + ",".join(sorted(map(str, cids)))
    return None


def document_groups(doc_lists: Iterable[Sequence[str]]) -> Dict[str, str]:
    """
    {document: group key}, where documents that appear together in any sample (a cross-document KG path
    joins them) are in the same group (union-find). Without that, a sample built from documents A and B
    could sit in train while other samples from B sit in test.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for docs in doc_lists:
        docs = [d for d in docs if d]
        for d in docs:
            find(d)
        for d in docs[1:]:
            ra, rb = find(docs[0]), find(d)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {d: "docs:" + find(d) for d in parent}


def sample_documents(doc_name: str, provenance: Dict) -> List[str]:
    docs = []
    if isinstance(provenance, dict):
        docs = list(provenance.get("documents") or provenance.get("provenance_docs") or [])
    if doc_name and doc_name not in docs:
        docs.insert(0, doc_name)
    return docs


def describe_split(info: Optional[Dict[str, object]]) -> Tuple[str, Optional[str]]:
    """(one-line summary, warning or None) of an assign_splits info dict, for the UI."""
    if not info:
        return "", None
    if info.get("level") == "stored":
        return "Split: stored per sample", None
    counts = info.get("counts") or {}
    line = (f"Split by {info.get('level')} ({info.get('groups')} groups): "
            + " / ".join(f"{k} {counts.get(k, 0)}" for k in SPLITS))
    skipped = [f"{t['level']}: {t['reason']}" for t in info.get("levels_skipped") or []]
    if skipped:
        line += " · skipped " + "; ".join(skipped)
    return line, info.get("warning")


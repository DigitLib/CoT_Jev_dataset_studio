"""
Named knowledge-graph store: one graph file per topic instead of a single `latest.json` slot.

* A graph's id (`kg_id`) is derived from its source documents (`kg_id_for`): one document gives the slugged
  file stem, several give `<first-stem>+<N-1>-<hash8>`. Users may also type their own (sanitised) name.
* Graphs live in `<KG_DIR>/<kg_id>.json`. `<KG_DIR>/index.json` keeps a small header per graph (documents,
  size, save time), so listing graphs never parses the multi-MB files, and remembers the last active graph.
* Graph files found on disk but missing from the index (e.g. a `latest.json` written by an older version) are
  registered under their file stem without being renamed or rewritten.
"""

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

INDEX_NAME = "index.json"
_BACKUP_SUFFIX = ".before_repair.json"
_LEGACY_BACKUPS = {"before_concept_repair.json"}
_MAX_SLUG = 60


def _kg_dir(kg_dir: Optional[Path]) -> Path:
    if kg_dir is not None:
        return Path(kg_dir)
    from config import KG_DIR
    return Path(KG_DIR)


def slugify(text: str, max_len: int = _MAX_SLUG) -> str:
    """Filename-safe lower-case slug (ASCII letters, digits and single dashes)."""
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len].strip("-") or "graph"


def kg_id_for(doc_names: Iterable[str]) -> str:
    """Deterministic graph id for a set of source documents."""
    docs = sorted({str(d) for d in doc_names if d})
    if not docs:
        return "untitled-graph"
    first = slugify(Path(docs[0]).stem)
    if len(docs) == 1:
        return first
    digest = hashlib.sha1("\n".join(docs).encode("utf-8")).hexdigest()[:8]
    return f"{first[:40]}+{len(docs) - 1}-{digest}"


def sanitize_kg_id(name: str) -> str:
    """A user-typed graph name reduced to a safe id; never a path ('/', '..', drive letters are removed)."""
    name = Path(str(name).strip().replace("\\", "/")).name
    if name.lower().endswith(".json"):
        name = name[:-5]
    s = re.sub(r"[^A-Za-z0-9+_.-]+", "-", name).strip("-.")
    s = re.sub(r"\.{2,}", ".", s)
    if s.lower() == INDEX_NAME[:-5]:
        s = "index-graph"
    return s[:80] or "untitled-graph"


def graph_path(kg_id: str, kg_dir: Optional[Path] = None) -> Path:
    return _kg_dir(kg_dir) / f"{sanitize_kg_id(kg_id)}.json"


def backup_path(kg_id: str, kg_dir: Optional[Path] = None) -> Path:
    return _kg_dir(kg_dir) / f"{sanitize_kg_id(kg_id)}{_BACKUP_SUFFIX}"


# ------------------------------------------------------------------ index
def _read_index(kg_dir: Path) -> Dict[str, Any]:
    p = kg_dir / INDEX_NAME
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("graphs"), dict):
            return data
    except Exception:
        pass
    return {"version": 1, "last_active": None, "graphs": {}}


def _write_index(kg_dir: Path, index: Dict[str, Any]) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    tmp = kg_dir / (INDEX_NAME + ".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, kg_dir / INDEX_NAME)


def _is_graph_file(p: Path) -> bool:
    return (p.suffix == ".json" and p.name != INDEX_NAME and not p.name.endswith(_BACKUP_SUFFIX)
            and p.name not in _LEGACY_BACKUPS)


def sync_index(kg_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Register graph files that are on disk but not in the index; drop entries whose file is gone."""
    kg_dir = _kg_dir(kg_dir)
    index = _read_index(kg_dir)
    graphs = index["graphs"]
    changed = False
    for kid in [k for k, e in graphs.items() if not (kg_dir / e.get("file", "")).exists()]:
        graphs.pop(kid)
        changed = True
    known_files = {e.get("file") for e in graphs.values()}
    if kg_dir.exists():
        from core.kg_builder import KnowledgeGraphBuilder
        for p in sorted(kg_dir.glob("*.json")):
            if not _is_graph_file(p) or p.name in known_files:
                continue
            info = KnowledgeGraphBuilder.saved_info(str(p))
            if not info:
                continue
            kid = p.stem if p.stem not in graphs else f"{p.stem}-{hashlib.sha1(p.name.encode()).hexdigest()[:6]}"
            graphs[kid] = {"file": p.name, "documents": info.get("documents", []),
                           "num_nodes": info.get("num_nodes", 0), "num_edges": info.get("num_edges", 0),
                           "num_chunks": info.get("num_chunks", 0), "saved_at": info.get("saved_at")}
            changed = True
    if index.get("last_active") not in graphs:
        index["last_active"] = None
        changed = True
    if changed:
        _write_index(kg_dir, index)
    return index


def list_graphs(kg_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Saved graphs, newest first: [{kg_id, file, documents, num_nodes, num_edges, num_chunks, saved_at}]."""
    index = sync_index(kg_dir)
    rows = [{"kg_id": k, **e} for k, e in index["graphs"].items()]
    return sorted(rows, key=lambda r: str(r.get("saved_at") or ""), reverse=True)


def get_graph(kg_id: str, kg_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    e = sync_index(kg_dir)["graphs"].get(kg_id)
    return {"kg_id": kg_id, **e} if e else None


def last_active(kg_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    index = sync_index(kg_dir)
    kid = index.get("last_active")
    if kid and kid in index["graphs"]:
        return {"kg_id": kid, **index["graphs"][kid]}
    rows = list_graphs(kg_dir)
    return rows[0] if rows else None


def unique_kg_id(base: str, kg_dir: Optional[Path] = None) -> str:
    """`base`, or `base-2`, `base-3` ... so that no existing graph file is overwritten."""
    kg_dir = _kg_dir(kg_dir)
    base = sanitize_kg_id(base)
    taken = set(sync_index(kg_dir)["graphs"])
    kid, n = base, 2
    while kid in taken or graph_path(kid, kg_dir).exists():
        kid, n = f"{base}-{n}", n + 1
    return kid


# ------------------------------------------------------------------ save / load / delete
def save_graph(kg_builder: Any, kg_id: str, chunks: Optional[list] = None,
               communities: Optional[list] = None, kg_dir: Optional[Path] = None) -> Path:
    """Write the graph to `<kg_id>.json`, update the index and mark it as the last active graph."""
    kg_dir = _kg_dir(kg_dir)
    kg_id = sanitize_kg_id(kg_id)
    path = graph_path(kg_id, kg_dir)
    kg_builder.save(str(path), chunks=chunks, communities=communities, kg_id=kg_id)
    index = sync_index(kg_dir)
    index["graphs"][kg_id] = {
        "file": path.name, "documents": sorted(kg_builder.graph_documents()),
        "num_nodes": kg_builder.graph.number_of_nodes(), "num_edges": kg_builder.graph.number_of_edges(),
        "num_chunks": len(chunks or []), "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    index["last_active"] = kg_id
    _write_index(kg_dir, index)
    return path


def load_graph(kg_builder: Any, kg_id: str, kg_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load a saved graph into `kg_builder`, mark it active and return the restore info (with `kg_id`)."""
    kg_dir = _kg_dir(kg_dir)
    entry = get_graph(kg_id, kg_dir)
    if not entry:
        raise FileNotFoundError(f"No saved knowledge graph named {kg_id!r}")
    restored = kg_builder.load(str(kg_dir / entry["file"]))
    restored["kg_id"] = kg_id
    restored["documents"] = entry.get("documents", [])
    set_last_active(kg_id, kg_dir)
    return restored


def set_last_active(kg_id: Optional[str], kg_dir: Optional[Path] = None) -> None:
    kg_dir = _kg_dir(kg_dir)
    index = sync_index(kg_dir)
    index["last_active"] = kg_id if kg_id in index["graphs"] else None
    _write_index(kg_dir, index)


def delete_graph(kg_id: str, kg_dir: Optional[Path] = None) -> bool:
    """Delete a saved graph file (and its repair backup) and remove it from the index."""
    kg_dir = _kg_dir(kg_dir)
    index = sync_index(kg_dir)
    entry = index["graphs"].pop(kg_id, None)
    if not entry:
        return False
    for p in (kg_dir / entry["file"], backup_path(kg_id, kg_dir)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    if index.get("last_active") == kg_id:
        index["last_active"] = None
    _write_index(kg_dir, index)
    return True


def backup_before_repair(kg_id: str, kg_dir: Optional[Path] = None) -> Optional[Path]:
    """Copy the saved file of `kg_id` to `<kg_id>.before_repair.json` (the previous save, before a repair)."""
    kg_dir = _kg_dir(kg_dir)
    entry = get_graph(kg_id, kg_dir)
    if not entry:
        return None
    src, dst = kg_dir / entry["file"], backup_path(kg_id, kg_dir)
    try:
        dst.write_bytes(src.read_bytes())
        return dst
    except Exception:
        return None


def stamp_kg_id(provenance_json: Optional[str], kg_id: Optional[str]) -> Optional[str]:
    """Add {"kg_id": ...} to a sample's provenance JSON (kept if already present), so every dataset row can be
    traced to the knowledge graph it was generated from."""
    if not kg_id:
        return provenance_json
    try:
        prov = json.loads(provenance_json) if provenance_json else {}
    except Exception:
        return provenance_json
    if not isinstance(prov, dict) or prov.get("kg_id"):
        return provenance_json
    prov["kg_id"] = kg_id
    return json.dumps(prov, ensure_ascii=False, default=str)

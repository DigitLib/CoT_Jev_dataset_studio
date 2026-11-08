"""Session-state side of the named knowledge-graph store (core/kg_store.py): restore, reset, save."""

from typing import Any, Dict, List, Optional

from core import kg_store
from gui._paths import clear_discovered_paths

KG_ID_KEY = "kg_id"


def restore_graph(state: Dict[str, Any], kg_builder: Any, kg_id: str, restore_chunks: bool = False) -> Dict[str, Any]:
    """Load saved graph `kg_id` into the builder. Its chunks replace the Library only when restore_chunks=True."""
    restored = kg_store.load_graph(kg_builder, kg_id)
    if restore_chunks:
        state["chunks"] = restored["chunks"]
    state["communities"] = restored["communities"]
    state["kg_built"] = True
    state[KG_ID_KEY] = kg_id
    state.pop("selected_path", None)
    clear_discovered_paths(state)
    return restored


def reset_graph(state: Dict[str, Any], kg_builder: Any, clear_chunks: bool = False) -> None:
    """Empty the in-memory graph and everything derived from it. Saved files are never touched."""
    kg_builder.clear()
    for k in ("communities", "kg_repair_report", "selected_path", KG_ID_KEY):
        state.pop(k, None)
    state["kg_built"] = False
    clear_discovered_paths(state)
    if clear_chunks:
        state["chunks"] = []


def save_current(state: Dict[str, Any], kg_builder: Any, chunks: Optional[List[Any]] = None,
                 kg_id: Optional[str] = None) -> str:
    """Save the current graph under `kg_id` (default: the session's graph id, else one derived from its documents)."""
    kid = kg_id or state.get(KG_ID_KEY) or kg_store.unique_kg_id(kg_store.kg_id_for(kg_builder.graph_documents()))
    kg_store.save_graph(kg_builder, kid, chunks=chunks, communities=state.get("communities", []))
    state[KG_ID_KEY] = kid
    return kid

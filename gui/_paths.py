"""
Shared KG-path bookkeeping for the Studio tabs.

The Knowledge Graph tab is where paths are discovered (with the user's filters: limit, document scope,
entailment-only, kernel-verified-only). It publishes that list here, and the Jev tab reuses it instead of
running its own, differently-filtered search. Results are keyed by a graph signature, so a list
discovered for an older graph is never reused after the graph changes.
"""

from typing import Any, Dict, List, Optional, Tuple

DISCOVERED_KEY = "kg_discovered_paths"
DISCOVERED_SIG_KEY = "kg_discovered_signature"
DISCOVERED_PARAMS_KEY = "kg_discovered_params"
JEV_CACHE_KEY = "jev_rediscovered_paths_cache"


def graph_signature(kg_builder: Any) -> Optional[Tuple[int, int, int, int]]:
    """Identity, version (bumped by clear/load/repair) and size of the current graph; None without a usable graph."""
    graph = getattr(kg_builder, "graph", None) if kg_builder is not None else None
    if graph is None or graph.number_of_nodes() <= 1:
        return None
    return (id(graph), int(getattr(kg_builder, "_graph_version", 0)),
            graph.number_of_nodes(), graph.number_of_edges())


def publish_discovered_paths(state: Dict[str, Any], kg_builder: Any, paths: List[Dict[str, Any]],
                             params: Dict[str, Any]) -> None:
    """Called by the Knowledge Graph tab after its path discovery."""
    state[DISCOVERED_KEY] = list(paths)
    state[DISCOVERED_SIG_KEY] = graph_signature(kg_builder)
    state[DISCOVERED_PARAMS_KEY] = dict(params)


def clear_discovered_paths(state: Dict[str, Any]) -> None:
    for k in (DISCOVERED_KEY, DISCOVERED_SIG_KEY, DISCOVERED_PARAMS_KEY, JEV_CACHE_KEY):
        state.pop(k, None)


def get_jev_paths(state: Dict[str, Any], kg_builder: Any, rediscover: bool = False) -> Tuple[List[Dict[str, Any]], str]:
    """
    Paths offered by the Jev tab, and a short description of where they came from.

    Default: exactly the paths the Knowledge Graph tab discovered for the current graph.
    rediscover=True: a Jev-specific relational search (no kernel verification - Jev logic comes from the
    scenario rules, not the path), honouring the KG tab's document scope, limit and topic overlap plus the
    Jev tab's own `jev_require_entailment` switch. Cached per graph signature and parameters.
    """
    sig = graph_signature(kg_builder)
    if sig is None:
        return [], "no knowledge graph"

    if not rediscover:
        if state.get(DISCOVERED_SIG_KEY) == sig and state.get(DISCOVERED_KEY) is not None:
            return list(state[DISCOVERED_KEY]), "from Knowledge Graph tab"
        return [], "open the Knowledge Graph tab's Paths view to discover paths"

    kg_params = state.get(DISCOVERED_PARAMS_KEY) or {}
    params = {
        "require_entailment": bool(state.get("jev_require_entailment", False)),
        "min_topic_overlap": float(state.get("kg_min_topic_overlap", 0.05)),
        "limit": int(state.get("kg_path_limit", 100)),
        "intra_document_only": bool(state.get("intra_doc_only", True)),
        "scope_doc_name": kg_params.get("scope_doc_name"),
    }
    cache_key = (sig, tuple(sorted(params.items())))
    cache = state.get(JEV_CACHE_KEY)
    if cache and cache.get("key") == cache_key:
        return list(cache["paths"]), cache["source"]

    paths = kg_builder.find_multihop_paths(max_hops=3, min_hops=2, verify_with_gentzen=False, **params)
    source = f"re-discovered for Jev, limit {params['limit']}"
    if not paths:
        paths = kg_builder.find_multihop_paths(max_hops=3, min_hops=1, verify_with_gentzen=False, **params)
        source = f"re-discovered for Jev incl. 1-hop, limit {params['limit']}"
    state[JEV_CACHE_KEY] = {"key": cache_key, "paths": list(paths), "source": source}
    return list(paths), source

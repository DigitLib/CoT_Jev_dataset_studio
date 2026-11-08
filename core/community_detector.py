"""
Community detection module for thematic segmentation of the Knowledge Graph.
Clusters concepts into thematic topics using Louvain / modularity algorithms,
identifies central hub concepts with network centrality, and synthesizes
Knwler-style topic labels and descriptions.
"""

from typing import Dict, List, Any, Optional
import networkx as nx
import re
import json
import logging

from llm_backend.client import LocalLLMClient

logger = logging.getLogger(__name__)

COMMUNITY_COLORS = [
    "#4A90E2", "#50E3C2", "#F5A623", "#E74C3C", "#9B59B6",
    "#1ABC9C", "#34495E", "#F39C12", "#D35400", "#27AE60",
    "#2980B9", "#8E44AD", "#16A085", "#2C3E50", "#BDC3C7"
]

COMMUNITY_LABEL_PROMPT = """You are an expert thematic analyst for knowledge graphs in science, philosophy, and technology.
Given a thematic cluster of concepts and their relations extracted from research literature, generate a concise academic topic title (2 to 4 words) and a 1-sentence summary description.

Concepts:
{concepts}

Relations:
{relations}

Output valid JSON only:
{{
  "topic_title": "Concise 2-4 Word Topic Title",
  "description": "One authoritative sentence summarizing the core theme interconnecting these concepts."
}}
"""

class CommunityDetector:
    def __init__(self, llm_client: Optional[LocalLLMClient] = None):
        self.llm_client = llm_client

    def detect_communities(
        self,
        graph: nx.MultiDiGraph,
        model: str = "qwen2.5:7b"
    ) -> Dict[str, Any]:
        """
        Detect thematic communities, assign colors, identify true central hubs,
        and generate Knwler-style topic labels and descriptions.
        """
        if graph.number_of_nodes() == 0:
            return {"node_to_community": {}, "communities": []}

        # Convert to weighted undirected graph for modularity clustering
        u_graph = nx.Graph()
        for u, v, data in graph.edges(data=True):
            if u != v:
                weight = float(data.get("strength", 8))
                if u_graph.has_edge(u, v):
                    u_graph[u][v]["weight"] += weight
                else:
                    u_graph.add_edge(u, v, weight=weight)

        for n in graph.nodes():
            if not u_graph.has_node(n):
                u_graph.add_node(n)

        # Community detection algorithm
        try:
            from networkx.algorithms.community import louvain_communities
            raw_communities = list(louvain_communities(u_graph, weight="weight", seed=42))
        except Exception:
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                raw_communities = list(greedy_modularity_communities(u_graph, weight="weight"))
            except Exception:
                raw_communities = [set(c) for c in nx.connected_components(u_graph)]

        # Sort communities by size descending
        raw_communities.sort(key=lambda c: len(c), reverse=True)

        node_to_comm = {}
        communities_summary = []

        for c_idx, members in enumerate(raw_communities):
            members_list = list(members)
            color = COMMUNITY_COLORS[c_idx % len(COMMUNITY_COLORS)]

            # Determine central hub node using internal degree and global centrality
            subg = u_graph.subgraph(members_list)
            internal_degrees = dict(subg.degree(weight="weight"))
            global_degrees = dict(u_graph.degree(members_list))

            # Pick node with highest internal weight, tie-break with global degree
            sorted_by_centrality = sorted(
                members_list,
                key=lambda n: (internal_degrees.get(n, 0), global_degrees.get(n, 0)),
                reverse=True
            )
            hub = sorted_by_centrality[0] if sorted_by_centrality else members_list[0]

            # Clean hub node display string (strip trailing hyphens/prefixes)
            clean_hub = re.sub(r'[\s\-–—]+(?:re|pre|co|non|anti|un)[\s\-–—]*$', '', hub, flags=re.IGNORECASE)
            clean_hub = clean_hub.strip(" -—–") or hub

            for node in members_list:
                node_to_comm[node] = c_idx
                if graph.has_node(node):
                    graph.nodes[node]["community"] = c_idx
                    graph.nodes[node]["color"] = color

            # Synthesize Knwler-style topic label & description
            topic_title, desc = self._generate_community_label(
                subg=subg,
                hub=clean_hub,
                members=sorted_by_centrality,
                graph=graph,
                model=model
            )

            is_primary = len(members_list) >= 3

            communities_summary.append({
                "community_id": c_idx,
                "topic_title": topic_title,
                "label": f"Theme {c_idx + 1}: {topic_title}",
                "description": desc,
                "hub_node": clean_hub,
                "color": color,
                "size": len(members_list),
                "is_primary": is_primary,
                "members": sorted(members_list),
            })

        return {
            "node_to_community": node_to_comm,
            "communities": communities_summary
        }

    def _generate_community_label(
        self,
        subg: nx.Graph,
        hub: str,
        members: List[str],
        graph: nx.MultiDiGraph,
        model: str
    ) -> tuple[str, str]:
        """Synthesize descriptive topic title and summary for community."""
        # Check if LLM is available
        if self.llm_client and self.llm_client.health_check() and len(members) >= 3:
            try:
                top_concepts = members[:8]
                relations = []
                for u, v, data in graph.edges(top_concepts, data=True):
                    if v in top_concepts:
                        relations.append(f"{u} -> {data.get('predicate', 'relates to')} -> {v}")

                prompt = COMMUNITY_LABEL_PROMPT.format(
                    concepts=", ".join(top_concepts),
                    relations="\n".join(relations[:10]) or "Interconnected conceptual relationships"
                )
                raw = self.llm_client.generate(
                    prompt=prompt,
                    model=model,
                    temperature=0.2,
                    max_tokens=200
                )
                data = self._parse_json(raw)
                if data and "topic_title" in data:
                    t = data["topic_title"].strip().strip('"\'')
                    d = data.get("description", "").strip() or f"Thematic cluster centered on {hub}."
                    return t, d
            except Exception as e:
                logger.warning(f"Community labeling via LLM failed: {e}")

        # Robust heuristic fallback
        if len(members) == 1:
            return hub, f"Isolated concept: {hub}."
        elif len(members) == 2:
            return f"{members[0]} & {members[1]}", f"Dyadic conceptual relationship linking {members[0]} and {members[1]}."
        else:
            top_two = members[:2]
            topic_title = f"{top_two[0]} & {top_two[1]}"
            desc = f"Thematic domain centered on {hub}, interconnecting {len(members)} core concepts across the knowledge graph."
            return topic_title, desc

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        match = re.search(r'\{.*?\}', clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return None

"""Knowledge graph service for sampling concept paths and prompt payloads."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from states.agent_state import Paper


@dataclass
class KnowledgeGraphContext:
    """Container for sampled knowledge graph context."""

    nodes: List[Dict[str, str]]
    edges: List[Dict[str, str]]
    path: List[str]
    summary: str
    prompt: str


class KnowledgeGraphService:
    """Builds lightweight knowledge graphs from seed data and documents."""

    def __init__(self, seed_path: Optional[Path] = None):
        self.graph = nx.Graph()
        self.seed_path = seed_path or Path(__file__).with_name("seed_graph.json")
        self._load_seed_graph()

    def _load_seed_graph(self):
        if not self.seed_path.exists():
            return
        data = json.loads(self.seed_path.read_text(encoding="utf-8"))
        for node in data.get("nodes", []):
            self.graph.add_node(
                node["id"],
                label=node.get("label", node["id"]),
                description=node.get("description", ""),
                domain=node.get("domain", "general"),
            )
        for edge in data.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                relation=edge.get("relation", "related_to"),
            )

    def ingest_papers(self, papers: List[Paper]):
        """Augment the graph with nodes derived from papers, extracting key entities."""
        import re
        
        for paper in papers:
            paper_node = f"paper::{paper.id}"
            self.graph.add_node(
                paper_node,
                label=paper.title[:80],
                description=paper.abstract[:200] if paper.abstract else "",
                domain=paper.field or "paper",
            )
            
            # Extract key entities from title and abstract
            text = f"{paper.title} {paper.abstract or ''}".lower()
            
            # Extract materials (common patterns)
            material_patterns = [
                r'\b(silk|fibroin|amyloid|fibril|hydrogel|polymer|composite|nanoparticle|graphene|protein|peptide)\w*\b',
                r'\b(collagen|elastin|keratin|chitin|cellulose)\w*\b',
            ]
            materials = set()
            for pattern in material_patterns:
                matches = re.findall(pattern, text)
                materials.update(matches)
            
            # Extract diseases/conditions
            disease_patterns = [
                r'\b(syndrome|disease|disorder|pathology|condition)\w*\b',
                r'\b(cogan|autoimmune|vasculitis|keratitis|hearing|impairment)\w*\b',
            ]
            conditions = set()
            for pattern in disease_patterns:
                matches = re.findall(pattern, text)
                conditions.update(matches)
            
            # Extract methods/techniques
            method_patterns = [
                r'\b(molecular dynamics|md|simulation|modeling|dft|fem|experiment|synthesis|characterization)\w*\b',
                r'\b(therapy|treatment|drug|biologic|tocilizumab|antibody)\w*\b',
            ]
            methods = set()
            for pattern in method_patterns:
                matches = re.findall(pattern, text)
                methods.update(matches)
            
            # Create nodes and edges for extracted entities
            for material in materials:
                if len(material) > 3:  # Filter short matches
                    mat_node = f"material::{material}"
                    self.graph.add_node(
                        mat_node,
                        label=material.title(),
                        description=f"Material/component mentioned in research",
                        domain=paper.field or "material",
                    )
                    self.graph.add_edge(paper_node, mat_node, relation="discusses")
            
            for condition in conditions:
                if len(condition) > 3:
                    cond_node = f"condition::{condition}"
                    self.graph.add_node(
                        cond_node,
                        label=condition.title(),
                        description=f"Medical condition or pathology",
                        domain="medicine",
                    )
                    self.graph.add_edge(paper_node, cond_node, relation="addresses")
            
            for method in methods:
                if len(method) > 3:
                    meth_node = f"method::{method}"
                    self.graph.add_node(
                        meth_node,
                        label=method.title(),
                        description=f"Research method or technique",
                        domain=paper.field or "method",
                    )
                    self.graph.add_edge(paper_node, meth_node, relation="uses")
            
            # Author connections
            for author in paper.authors[:5]:
                author_node = f"author::{author}"
                if not self.graph.has_node(author_node):
                    self.graph.add_node(
                        author_node,
                        label=author,
                        description="Researcher/Author",
                        domain="author",
                    )
                self.graph.add_edge(author_node, paper_node, relation="authored")
            
            # Field connections
            if paper.field:
                field_node = f"field::{paper.field}"
                if not self.graph.has_node(field_node):
                    self.graph.add_node(
                        field_node,
                        label=paper.field.replace("_", " ").title(),
                        description="Research field",
                        domain="field",
                    )
                self.graph.add_edge(field_node, paper_node, relation="belongs_to")
            
            # Connect materials to conditions if both present
            for mat in materials:
                for cond in conditions:
                    if len(mat) > 3 and len(cond) > 3:
                        mat_node = f"material::{mat}"
                        cond_node = f"condition::{cond}"
                        if self.graph.has_node(mat_node) and self.graph.has_node(cond_node):
                            self.graph.add_edge(mat_node, cond_node, relation="potentially_treats")

    def map_keywords(self, text: str) -> List[str]:
        """Map keywords from the query to nodes in the graph."""
        keywords = {token.strip().lower() for token in text.split() if len(token) > 3}
        matched = []
        for node, data in self.graph.nodes(data=True):
            label = data.get("label", "").lower()
            for keyword in keywords:
                if keyword in label:
                    matched.append(node)
                    break
        return matched

    def _random_node(self) -> Optional[str]:
        if not self.graph.nodes:
            return None
        return random.choice(list(self.graph.nodes))

    def sample_path(
        self,
        query: str,
        strategy: str = "random",
        max_steps: int = 12,
    ) -> KnowledgeGraphContext:
        """Sample a subgraph based on the query."""
        matched_nodes = self.map_keywords(query)
        
        # Try multiple node pairs for richer paths
        if len(matched_nodes) >= 2:
            source, target = matched_nodes[0], matched_nodes[1]
        elif len(matched_nodes) == 1:
            source = matched_nodes[0]
            # Find a diverse target (different domain if possible)
            source_domain = self.graph.nodes[source].get("domain", "")
            candidates = [n for n in self.graph.nodes() 
                         if self.graph.nodes[n].get("domain", "") != source_domain]
            target = random.choice(candidates) if candidates else self._random_node()
        else:
            # Random exploration - pick nodes from different domains
            nodes_by_domain = {}
            for node, data in self.graph.nodes(data=True):
                domain = data.get("domain", "general")
                if domain not in nodes_by_domain:
                    nodes_by_domain[domain] = []
                nodes_by_domain[domain].append(node)
            
            if len(nodes_by_domain) >= 2:
                domains = list(nodes_by_domain.keys())
                source = random.choice(nodes_by_domain[domains[0]])
                target = random.choice(nodes_by_domain[domains[1]])
            else:
                source = self._random_node()
                target = self._random_node()

        if not source or not target or source == target:
            return self._empty_context(query)

        try:
            if strategy == "shortest":
                path = nx.shortest_path(self.graph, source=source, target=target)
            else:
                # Random walk biased toward target neighborhood with longer exploration
                path = self._random_walk(source, target, max_steps)
                
                # If path is too short, extend it with additional random steps
                if len(path) < 5 and max_steps > 5:
                    # Add more exploration
                    current = path[-1]
                    for _ in range(max_steps - len(path)):
                        neighbors = list(self.graph.neighbors(current))
                        if neighbors:
                            current = random.choice(neighbors)
                            if current not in path:  # Avoid cycles
                                path.append(current)
                        else:
                            break
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return self._empty_context(query)

        nodes, edges = self._subgraph_from_path(path)

        summary = self._summarize_path(path, nodes, edges)

        prompt = self._build_prompt(path, nodes, edges, summary, query)

        return KnowledgeGraphContext(
            nodes=nodes,
            edges=edges,
            path=path,
            summary=summary,
            prompt=prompt,
        )

    def _random_walk(self, source: str, target: str, max_steps: int) -> List[str]:
        path = [source]
        current = source
        for _ in range(max_steps):
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
            # bias toward nodes closer to target
            def score(node):
                try:
                    return 1 / (1 + nx.shortest_path_length(self.graph, node, target))
                except nx.NetworkXNoPath:
                    return 0.1

            weights = [score(n) for n in neighbors]
            total = sum(weights)
            if total == 0:
                next_node = random.choice(neighbors)
            else:
                probs = [w / total for w in weights]
                next_node = random.choices(neighbors, probs)[0]
            path.append(next_node)
            current = next_node
            if current == target:
                break
        if path[-1] != target and nx.has_path(self.graph, current, target):
            tail = nx.shortest_path(self.graph, current, target)
            path.extend(tail[1:])
        return path

    def _subgraph_from_path(
        self, path: List[str]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        nodes = []
        edges = []
        for node in path:
            data = self.graph.nodes[node]
            nodes.append(
                {
                    "id": node,
                    "label": data.get("label", node),
                    "description": data.get("description", ""),
                    "domain": data.get("domain", "general"),
                }
            )
        for i in range(len(path) - 1):
            rel = self.graph.get_edge_data(path[i], path[i + 1], default={}).get(
                "relation", "related_to"
            )
            edges.append(
                {
                    "source": path[i],
                    "target": path[i + 1],
                    "relation": rel,
                }
            )
        return nodes, edges

    def _summarize_path(
        self,
        path: List[str],
        nodes: List[Dict[str, str]],
        edges: List[Dict[str, str]],
    ) -> str:
        if not path:
            return ""
        pieces = []
        for i, node_id in enumerate(path):
            node = next((n for n in nodes if n["id"] == node_id), None)
            if not node:
                continue
            label = node["label"]
            desc = node["description"]
            pieces.append(f"{label}: {desc}" if desc else label)
            if i < len(edges):
                rel = edges[i]["relation"]
                pieces.append(f" --[{rel}]--> ")
        return "".join(pieces)

    def _build_prompt(
        self,
        path: List[str],
        nodes: List[Dict[str, str]],
        edges: List[Dict[str, str]],
        summary: str,
        query: str,
    ) -> str:
        prompt = [
            "Knowledge Graph Context",
            f"User Query: {query}",
            f"Path: {' -> '.join(nodes[i]['label'] for i in range(len(path)) if i < len(nodes))}",
            "Nodes:",
        ]
        for node in nodes:
            prompt.append(
                f"- {node['label']} ({node['domain']}): {node.get('description', '')}"
            )
        prompt.append("Edges:")
        for edge in edges:
            source_label = next(
                (n["label"] for n in nodes if n["id"] == edge["source"]), edge["source"]
            )
            target_label = next(
                (n["label"] for n in nodes if n["id"] == edge["target"]), edge["target"]
            )
            prompt.append(
                f"- {source_label} --[{edge['relation']}]--> {target_label}"
            )
        prompt.append(f"\nNarrative Summary:\n{summary}")
        return "\n".join(prompt)

    def _empty_context(self, query: str) -> KnowledgeGraphContext:
        return KnowledgeGraphContext(
            nodes=[],
            edges=[],
            path=[],
            summary="No relevant knowledge graph path found.",
            prompt=f"No structured graph context available for query: {query}",
        )


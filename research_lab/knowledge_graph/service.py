"""Knowledge graph service for building graphs from RAG papers and path sampling."""

from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import random
from dataclasses import dataclass

from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingManager
from config.settings import settings


@dataclass
class GraphPath:
    """Represents a path through the knowledge graph."""
    nodes: List[str]
    edges: List[Tuple[str, str, str]]  # (source, relationship, target)
    subgraph: Dict[str, Any]  # JSON representation of subgraph


@dataclass
class PathSamplingResult:
    """Result of path sampling."""
    path: GraphPath
    path_type: str  # "random" or "shortest"
    total_nodes: int
    total_edges: int


class KnowledgeGraphService:
    """
    Service for building and querying knowledge graphs from RAG papers.
    
    Builds a graph where:
    - Nodes = concepts/entities (materials, methods, properties, etc.)
    - Edges = relationships between concepts
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        field: Optional[str] = None
    ):
        """
        Initialize the knowledge graph service.
        
        Args:
            vector_store: Vector store containing papers
            field: Optional field filter
        """
        self.vector_store = vector_store
        self.field = field
        self.graph = nx.MultiDiGraph()  # MultiDiGraph to support multiple edge types
        self.embedding_manager = EmbeddingManager()
        self._node_embeddings: Dict[str, List[float]] = {}
        self._built = False
    
    def build_graph(
        self,
        max_papers: Optional[int] = None,
        min_entities_per_paper: int = 3
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from papers in the vector store.
        
        Args:
            max_papers: Maximum number of papers to process (None = all)
            min_entities_per_paper: Minimum entities to extract per paper
            
        Returns:
            Statistics about the built graph
        """
        if self._built:
            return self._get_graph_stats()
        
        # Get all papers from vector store
        try:
            # Query for all documents (we'll filter by field if needed)
            all_docs = self.vector_store._collection.get(
                where={"doc_type": "paper"} if self.field is None else {
                    "doc_type": "paper",
                    "field": self.field
                },
                limit=max_papers or 1000
            )
        except Exception as e:
            # If collection is empty or error, return empty stats
            return {
                "nodes": 0,
                "edges": 0,
                "papers_processed": 0,
                "status": "empty"
            }
        
        papers_processed = 0
        
        # Process each paper to extract entities and relationships
        for i, doc_id in enumerate(all_docs.get("ids", [])):
            if max_papers and i >= max_papers:
                break
            
            try:
                doc_data = self.vector_store._collection.get(
                    ids=[doc_id],
                    include=["documents", "metadatas"]
                )
                
                if not doc_data.get("documents"):
                    continue
                
                content = doc_data["documents"][0]
                metadata = doc_data.get("metadatas", [{}])[0]
                
                # Extract entities and relationships from paper
                entities, relationships = self._extract_entities_and_relationships(
                    content,
                    metadata.get("title", ""),
                    min_entities_per_paper
                )
                
                # Add to graph
                for entity in entities:
                    if entity not in self.graph:
                        self.graph.add_node(entity, type="concept", papers=set())
                    self.graph.nodes[entity]["papers"].add(doc_id)
                
                for source, rel, target in relationships:
                    if source not in self.graph:
                        self.graph.add_node(source, type="concept", papers=set())
                    if target not in self.graph:
                        self.graph.add_node(target, type="concept", papers=set())
                    
                    self.graph.add_edge(source, target, relationship=rel, paper_id=doc_id)
                    self.graph.nodes[source]["papers"].add(doc_id)
                    self.graph.nodes[target]["papers"].add(doc_id)
                
                papers_processed += 1
                
            except Exception as e:
                # Skip papers that fail to process
                continue
        
        # Convert paper sets to counts for serialization
        for node in self.graph.nodes():
            if "papers" in self.graph.nodes[node]:
                self.graph.nodes[node]["paper_count"] = len(self.graph.nodes[node]["papers"])
                del self.graph.nodes[node]["papers"]
        
        self._built = True
        return self._get_graph_stats()
    
    def _extract_entities_and_relationships(
        self,
        content: str,
        title: str,
        min_entities: int
    ) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Extract entities (concepts) and relationships from paper content.
        
        Uses LLM to extract:
        - Entities: materials, methods, properties, mechanisms, etc.
        - Relationships: "possesses", "enables", "improves", "requires", etc.
        
        Args:
            content: Paper content (title + abstract)
            title: Paper title
            min_entities: Minimum entities to extract
            
        Returns:
            Tuple of (entities_list, relationships_list)
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import json
        import re
        
        # Use LLM to extract entities and relationships
        llm_kwargs = {
            "model": settings.openai_model,
            "temperature": 0.1,
            "openai_api_key": settings.openai_api_key
        }
        if settings.openai_base_url:
            llm_kwargs["openai_api_base"] = settings.openai_base_url
        
        llm = ChatOpenAI(**llm_kwargs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting scientific concepts and relationships from research papers.

Extract:
1. **Entities**: Materials, methods, properties, mechanisms, processes, applications (e.g., "silk fibroin", "molecular dynamics", "tensile strength", "self-assembly")
2. **Relationships**: How entities relate (e.g., "possesses", "enables", "improves", "requires", "is_composed_of", "exhibits")

Return a JSON object with:
{{
    "entities": ["entity1", "entity2", ...],
    "relationships": [
        ["source_entity", "relationship_type", "target_entity"],
        ...
    ]
}}

Be specific and extract at least {min_entities} entities. Focus on scientific concepts, not generic terms."""),
            ("human", """Extract entities and relationships from this research paper:

Title: {title}

Content:
{content}

Return only valid JSON, no markdown formatting.""")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "title": title,
                "content": content[:3000],  # Limit content length
                "min_entities": min_entities
            })
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get("entities", [])
                relationships = [tuple(r) for r in data.get("relationships", []) if len(r) == 3]
                return entities, relationships
            else:
                # Fallback: try to parse as JSON directly
                data = json.loads(response)
                entities = data.get("entities", [])
                relationships = [tuple(r) for r in data.get("relationships", []) if len(r) == 3]
                return entities, relationships
                
        except Exception as e:
            # Fallback: extract simple entities from title and content
            # Simple keyword extraction as fallback
            words = (title + " " + content[:500]).lower()
            # Extract capitalized phrases (likely entities)
            entities = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', title + " " + content[:500])))
            entities = [e for e in entities if len(e) > 3][:min_entities]
            relationships = []
            return entities, relationships
    
    def sample_path(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        path_type: str = "random",
        max_length: int = 10,
        random_waypoints: int = 2
    ) -> PathSamplingResult:
        """
        Sample a path through the knowledge graph.
        
        Args:
            source: Source node (if None, randomly selected)
            target: Target node (if None, randomly selected)
            path_type: "random" or "shortest"
            max_length: Maximum path length
            random_waypoints: Number of random waypoints for random paths
            
        Returns:
            PathSamplingResult with path and subgraph
        """
        if not self._built:
            self.build_graph()
        
        if self.graph.number_of_nodes() == 0:
            # Return empty path if graph is empty
            return PathSamplingResult(
                path=GraphPath(nodes=[], edges=[], subgraph={}),
                path_type=path_type,
                total_nodes=0,
                total_edges=0
            )
        
        # Select source and target nodes
        all_nodes = list(self.graph.nodes())
        
        if source is None:
            source = random.choice(all_nodes)
        elif source not in self.graph:
            # Find closest node by name similarity
            source = self._find_closest_node(source, all_nodes)
        
        if target is None:
            target = random.choice(all_nodes)
            # Ensure target is different from source
            while target == source and len(all_nodes) > 1:
                target = random.choice(all_nodes)
        elif target not in self.graph:
            target = self._find_closest_node(target, all_nodes)
        
        # Sample path
        if path_type == "shortest":
            path_nodes = self._shortest_path(source, target, max_length)
        else:  # random
            path_nodes = self._random_path(source, target, max_length, random_waypoints)
        
        if not path_nodes:
            # If no path found, return single node
            path_nodes = [source]
        
        # Extract edges along path
        edges = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if self.graph.has_edge(u, v):
                # Get all relationships between u and v
                edge_data = self.graph.get_edge_data(u, v)
                if edge_data:
                    # Get first relationship type
                    rel_type = list(edge_data.values())[0].get("relationship", "related_to")
                    edges.append((u, rel_type, v))
        
        # Build subgraph
        subgraph = self._build_subgraph_json(path_nodes, edges)
        
        return PathSamplingResult(
            path=GraphPath(nodes=path_nodes, edges=edges, subgraph=subgraph),
            path_type=path_type,
            total_nodes=len(path_nodes),
            total_edges=len(edges)
        )
    
    def _shortest_path(
        self,
        source: str,
        target: str,
        max_length: int
    ) -> List[str]:
        """Find shortest path between source and target."""
        try:
            path = nx.shortest_path(self.graph, source, target)
            if len(path) > max_length:
                # Truncate to max_length
                path = path[:max_length]
            return path
        except nx.NetworkXNoPath:
            return []
    
    def _random_path(
        self,
        source: str,
        target: str,
        max_length: int,
        random_waypoints: int
    ) -> List[str]:
        """
        Generate a random path with waypoints.
        
        Uses a randomized Dijkstra-like approach with waypoints.
        """
        all_nodes = list(self.graph.nodes())
        path = [source]
        
        # Add random waypoints
        waypoints = []
        for _ in range(random_waypoints):
            waypoint = random.choice(all_nodes)
            if waypoint not in path and waypoint != target:
                waypoints.append(waypoint)
        
        # Build path through waypoints
        current = source
        for waypoint in waypoints:
            try:
                segment = nx.shortest_path(self.graph, current, waypoint)
                # Add segment to path (avoid duplicates)
                for node in segment[1:]:
                    if node not in path:
                        path.append(node)
                    if len(path) >= max_length:
                        break
                current = waypoint
            except nx.NetworkXNoPath:
                continue
            
            if len(path) >= max_length:
                break
        
        # Final segment to target
        try:
            final_segment = nx.shortest_path(self.graph, current, target)
            for node in final_segment[1:]:
                if node not in path:
                    path.append(node)
                if len(path) >= max_length:
                    break
        except nx.NetworkXNoPath:
            pass
        
        return path[:max_length]
    
    def _find_closest_node(self, query: str, nodes: List[str]) -> str:
        """Find closest node by name similarity."""
        query_lower = query.lower()
        best_match = nodes[0]
        best_score = 0
        
        for node in nodes:
            node_lower = node.lower()
            # Simple similarity: check if query is substring or vice versa
            if query_lower in node_lower or node_lower in query_lower:
                score = len(query_lower) / max(len(node_lower), 1)
                if score > best_score:
                    best_score = score
                    best_match = node
        
        return best_match
    
    def _build_subgraph_json(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        """Build JSON representation of subgraph."""
        # Get node information
        node_data = {}
        for node in nodes:
            node_info = self.graph.nodes[node]
            node_data[node] = {
                "type": node_info.get("type", "concept"),
                "paper_count": node_info.get("paper_count", 0)
            }
        
        # Build edge list
        edge_list = []
        for source, rel, target in edges:
            edge_list.append({
                "source": source,
                "relationship": rel,
                "target": target
            })
        
        return {
            "nodes": node_data,
            "edges": edge_list,
            "path": " → ".join([f"{nodes[i]} --{edges[i][1] if i < len(edges) else 'related_to'}--> {nodes[i+1]}" 
                                for i in range(len(nodes)-1)])
        }
    
    def _get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "papers_processed": sum(
                self.graph.nodes[n].get("paper_count", 0) 
                for n in self.graph.nodes()
            ),
            "status": "built" if self._built else "empty"
        }
    
    def get_node_definitions(self, nodes: List[str]) -> Dict[str, str]:
        """
        Get definitions/descriptions for nodes.
        
        Uses LLM to generate definitions based on graph context.
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import json
        
        llm_kwargs = {
            "model": settings.openai_model,
            "temperature": 0.1,
            "openai_api_key": settings.openai_api_key
        }
        if settings.openai_base_url:
            llm_kwargs["openai_api_base"] = settings.openai_base_url
        
        llm = ChatOpenAI(**llm_kwargs)
        
        # Get context for each node (neighbors, relationships)
        node_contexts = {}
        for node in nodes:
            if node in self.graph:
                neighbors = list(self.graph.neighbors(node))
                edges_out = [(node, self.graph[node][n].get("relationship", "related_to"), n) 
                            for n in neighbors[:5]]  # Limit to 5 neighbors
                node_contexts[node] = {
                    "neighbors": neighbors[:5],
                    "relationships": edges_out
                }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a scientific ontologist. Provide clear, concise definitions for scientific concepts based on their graph context.

For each concept, provide:
1. A brief definition (1-2 sentences)
2. Key properties or characteristics
3. How it relates to other concepts in the graph

Return JSON: {{"concept_name": "definition text", ...}}"""),
            ("human", """Provide definitions for these concepts:

{node_contexts}

Return only valid JSON.""")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "node_contexts": json.dumps(node_contexts, indent=2)
            })
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response)
        except Exception:
            # Fallback: return simple definitions
            return {node: f"{node} is a scientific concept in the knowledge graph." for node in nodes}


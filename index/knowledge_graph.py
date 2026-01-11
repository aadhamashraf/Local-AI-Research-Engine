"""
Knowledge Graph - Entity and relationship extraction and storage
"""

import networkx as nx
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from pathlib import Path
import json

from llm.ollama_client import OllamaClient
from llm.prompts import ENTITY_EXTRACTION_PROMPT, RELATIONSHIP_EXTRACTION_PROMPT


class KnowledgeGraph:
    """Knowledge graph for storing entities and relationships."""
    
    def __init__(
        self,
        db_path: str = "./data/graph.db",
        ollama_client: Optional[OllamaClient] = None
    ):
        """
        Initialize knowledge graph.
        
        Args:
            db_path: Path to SQLite database
            ollama_client: Optional Ollama client for entity extraction
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.client = ollama_client
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        
        # Initialize database
        self._init_database()
        
        # Load existing graph
        self.load()
        
        logger.info(f"Initialized knowledge graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                label TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                type TEXT NOT NULL,
                evidence TEXT,
                metadata TEXT,
                FOREIGN KEY (source) REFERENCES nodes(id),
                FOREIGN KEY (target) REFERENCES nodes(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique node identifier
            node_type: Type of node (Concept, Method, Author, Paper, Tool)
            label: Display label
            metadata: Additional metadata
        """
        metadata = metadata or {}
        
        self.graph.add_node(
            node_id,
            type=node_type,
            label=label,
            **metadata
        )
    
    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        evidence: str = "",
        metadata: Dict[str, Any] = None
    ):
        """
        Add an edge to the graph.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of relationship
            evidence: Supporting evidence
            metadata: Additional metadata
        """
        metadata = metadata or {}
        
        self.graph.add_edge(
            source,
            target,
            type=edge_type,
            evidence=evidence,
            **metadata
        )
    
    def extract_and_add_from_chunk(
        self,
        chunk: Dict[str, Any],
        max_relationships: int = 10
    ):
        """
        Extract entities and relationships from a chunk and add to graph.
        
        Args:
            chunk: Chunk dictionary with content and metadata
            max_relationships: Maximum relationships to extract
        """
        if not self.client:
            logger.warning("No Ollama client provided, skipping extraction")
            return
        
        content = chunk.get("content", "")
        source_file = chunk.get("metadata", {}).get("source", "Unknown")
        
        try:
            # Extract entities
            entities = self._extract_entities(content)
            
            # Add nodes
            all_entity_ids = []
            for entity_type, entity_list in entities.items():
                for entity_name in entity_list:
                    entity_id = self._create_entity_id(entity_name, entity_type)
                    all_entity_ids.append((entity_id, entity_name, entity_type))
                    
                    if not self.graph.has_node(entity_id):
                        self.add_node(
                            entity_id,
                            entity_type,
                            entity_name,
                            {"source": source_file}
                        )
            
            # Extract relationships
            if len(all_entity_ids) > 1:
                relationships = self._extract_relationships(
                    content,
                    [name for _, name, _ in all_entity_ids]
                )
                
                # Add edges
                for rel in relationships[:max_relationships]:
                    source_name = rel.get("source", "")
                    target_name = rel.get("target", "")
                    rel_type = rel.get("type", "related_to")
                    evidence = rel.get("evidence", "")
                    
                    # Find matching entity IDs
                    source_id = self._find_entity_id(source_name, all_entity_ids)
                    target_id = self._find_entity_id(target_name, all_entity_ids)
                    
                    if source_id and target_id:
                        self.add_edge(
                            source_id,
                            target_id,
                            rel_type,
                            evidence,
                            {"source": source_file}
                        )
            
            logger.debug(f"Extracted {len(all_entity_ids)} entities and added to graph")
            
        except Exception as e:
            logger.warning(f"Failed to extract entities from chunk: {e}")
    
    def get_related_entities(
        self,
        entity_id: str,
        max_depth: int = 2,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get entities related to a given entity.
        
        Args:
            entity_id: Starting entity ID
            max_depth: Maximum traversal depth
            max_results: Maximum number of results
            
        Returns:
            List of related entities with relationship info
        """
        if not self.graph.has_node(entity_id):
            return []
        
        related = []
        visited = set()
        
        # BFS traversal
        queue = [(entity_id, 0)]
        
        while queue and len(related) < max_results:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Get neighbors
            for neighbor in self.graph.neighbors(current_id):
                if neighbor not in visited:
                    # Get edge data
                    edges = self.graph.get_edge_data(current_id, neighbor)
                    
                    for edge_key, edge_data in edges.items():
                        related.append({
                            "entity_id": neighbor,
                            "label": self.graph.nodes[neighbor].get("label", neighbor),
                            "type": self.graph.nodes[neighbor].get("type", "Unknown"),
                            "relationship": edge_data.get("type", "related_to"),
                            "evidence": edge_data.get("evidence", ""),
                            "depth": depth + 1
                        })
                    
                    if depth + 1 < max_depth:
                        queue.append((neighbor, depth + 1))
        
        return related[:max_results]
    
    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by label.
        
        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        query_lower = query.lower()
        results = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            label = node_data.get("label", "").lower()
            node_type = node_data.get("type", "")
            
            # Filter by type if specified
            if entity_type and node_type != entity_type:
                continue
            
            # Check if query matches
            if query_lower in label:
                results.append({
                    "entity_id": node_id,
                    "label": node_data.get("label", ""),
                    "type": node_type,
                    "metadata": {k: v for k, v in node_data.items() if k not in ["label", "type"]}
                })
        
        return results[:limit]
    
    def save(self):
        """Save graph to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM edges")
        cursor.execute("DELETE FROM nodes")
        
        # Save nodes
        for node_id, node_data in self.graph.nodes(data=True):
            metadata = {k: v for k, v in node_data.items() if k not in ["type", "label"]}
            
            cursor.execute(
                "INSERT INTO nodes (id, type, label, metadata) VALUES (?, ?, ?, ?)",
                (
                    node_id,
                    node_data.get("type", "Unknown"),
                    node_data.get("label", node_id),
                    json.dumps(metadata)
                )
            )
        
        # Save edges
        for source, target, edge_data in self.graph.edges(data=True):
            metadata = {k: v for k, v in edge_data.items() if k not in ["type", "evidence"]}
            
            cursor.execute(
                "INSERT INTO edges (source, target, type, evidence, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    source,
                    target,
                    edge_data.get("type", "related_to"),
                    edge_data.get("evidence", ""),
                    json.dumps(metadata)
                )
            )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved knowledge graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def load(self):
        """Load graph from SQLite database."""
        if not self.db_path.exists():
            logger.info("No existing graph database found")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load nodes
        cursor.execute("SELECT id, type, label, metadata FROM nodes")
        for row in cursor.fetchall():
            node_id, node_type, label, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            self.add_node(node_id, node_type, label, metadata)
        
        # Load edges
        cursor.execute("SELECT source, target, type, evidence, metadata FROM edges")
        for row in cursor.fetchall():
            source, target, edge_type, evidence, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            self.add_edge(source, target, edge_type, evidence, metadata)
        
        conn.close()
        
        logger.info(f"Loaded knowledge graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using LLM."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:2000])
        
        response = self.client.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=500
        )
        
        try:
            entities = json.loads(response)
            return entities
        except json.JSONDecodeError:
            logger.warning("Failed to parse entity extraction response")
            return {}
    
    def _extract_relationships(
        self,
        text: str,
        entities: List[str]
    ) -> List[Dict[str, str]]:
        """Extract relationships using LLM."""
        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
            text=text[:2000],
            entities=", ".join(entities)
        )
        
        response = self.client.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=800
        )
        
        try:
            relationships = json.loads(response)
            return relationships
        except json.JSONDecodeError:
            logger.warning("Failed to parse relationship extraction response")
            return []
    
    def _create_entity_id(self, name: str, entity_type: str) -> str:
        """Create a unique entity ID."""
        # Normalize name
        normalized = name.lower().replace(" ", "_")
        return f"{entity_type.lower()}:{normalized}"
    
    def _find_entity_id(
        self,
        name: str,
        entity_list: List[Tuple[str, str, str]]
    ) -> Optional[str]:
        """Find entity ID by name from a list of (id, name, type)."""
        name_lower = name.lower()
        
        for entity_id, entity_name, entity_type in entity_list:
            if entity_name.lower() == name_lower:
                return entity_id
        
        return None

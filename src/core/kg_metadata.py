"""
Knowledge Graph Metadata Layer

This module provides a metadata layer that sits between the Query Router
and retrieval sources (Vector Store, DuckDB). It expands queries by finding
related sources based on entity-to-data relationships.

Key Features:
- Query expansion via entity extraction and source linking
- Auto-population during document/table ingestion
- MCP tools for direct graph interaction
- Persistent JSON storage (data/kg/)
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime


@dataclass
class KGNode:
    """A node in the knowledge graph."""
    id: str
    type: str  # document, table, column, entity, topic
    name: str
    source: str  # 'vector', 'sql', 'manual'
    properties: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'KGNode':
        return cls(**data)


@dataclass
class KGEdge:
    """An edge (relationship) in the knowledge graph."""
    source_id: str
    target_id: str
    relationship: str  # references, describes, contains, manages, related_to, tagged_with
    weight: float = 1.0
    properties: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def id(self) -> str:
        """Unique edge ID based on source, target, and relationship."""
        return f"{self.source_id}--{self.relationship}-->{self.target_id}"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'KGEdge':
        return cls(**data)


@dataclass
class RetrievalPlan:
    """Plan for expanded retrieval based on KG analysis."""
    original_query: str
    vector_queries: list[str] = field(default_factory=list)
    sql_hints: list[str] = field(default_factory=list)  # table names to consider
    entities_found: list[str] = field(default_factory=list)
    expanded: bool = False  # True if KG added additional sources


class KnowledgeGraphMetadata:
    """
    Metadata layer for query expansion and guided retrieval.
    
    Sits between Query Router and retrieval sources to:
    - Extract entities from questions
    - Find linked documents/tables for each entity
    - Expand retrieval scope to include related sources
    
    Usage:
        kg = KnowledgeGraphMetadata(storage_path="data/kg")
        plan = kg.expand_query("What's Alice's budget?", "unstructured")
        # plan.sql_hints might include 'budgets' table if linked to 'Alice'
    """
    
    def __init__(self, storage_path: str = "data/kg"):
        """
        Initialize the knowledge graph.
        
        Args:
            storage_path: Directory for persistent JSON storage
        """
        self.storage_path = storage_path
        self.nodes: dict[str, KGNode] = {}
        self.edges: dict[str, KGEdge] = {}
        
        # Create storage directory if needed
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing graph
        self._load()
    
    # =========================================================================
    # Query Expansion (Core Feature)
    # =========================================================================
    
    def expand_query(self, question: str, query_type: str) -> RetrievalPlan:
        """
        Analyze question and expand retrieval scope based on KG links.
        
        Args:
            question: User's question
            query_type: Current classification (structured/unstructured/hybrid)
            
        Returns:
            RetrievalPlan with expanded queries and SQL hints
        """
        plan = RetrievalPlan(original_query=question)
        plan.vector_queries = [question]
        
        # 1. Extract potential entities from question
        entities = self._extract_entities(question)
        plan.entities_found = entities
        
        if not entities:
            return plan
        
        # 2. Find linked sources for each entity
        for entity_name in entities:
            # Look for entity node
            entity_node = self._find_node_by_name(entity_name, node_type='entity')
            if not entity_node:
                continue
            
            # Get all linked sources
            linked = self.get_related_sources(entity_node.id)
            
            for node in linked:
                if node.type == 'table':
                    if node.name not in plan.sql_hints:
                        plan.sql_hints.append(node.name)
                        plan.expanded = True
                elif node.type == 'document':
                    # Add document name as additional vector query
                    if node.name not in plan.vector_queries:
                        plan.vector_queries.append(node.name)
                        plan.expanded = True
        
        # 3. Also check if question mentions known table names directly
        for node in self.nodes.values():
            if node.type == 'table':
                if node.name.lower() in question.lower():
                    if node.name not in plan.sql_hints:
                        plan.sql_hints.append(node.name)
                        plan.expanded = True
        
        return plan
    
    def _extract_entities(self, text: str) -> list[str]:
        """
        Extract potential entity names from text.
        
        Uses simple heuristics:
        - Capitalized words (proper nouns)
        - Known entity names from the graph
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entity name strings
        """
        entities = []
        
        # 1. Find capitalized words (potential proper nouns)
        words = text.split()
        for word in words:
            # Clean punctuation
            clean = word.strip('.,?!:;"\'()[]{}')
            if len(clean) > 1 and clean[0].isupper() and not clean.isupper():
                # Skip common sentence starters
                if clean.lower() not in {'what', 'who', 'where', 'when', 'why', 'how', 
                                          'is', 'are', 'the', 'a', 'an', 'this', 'that',
                                          'can', 'could', 'would', 'should', 'do', 'does'}:
                    if clean not in entities:
                        entities.append(clean)
        
        # 2. Check for known entity names (case-insensitive)
        text_lower = text.lower()
        for node in self.nodes.values():
            if node.type == 'entity':
                if node.name.lower() in text_lower:
                    if node.name not in entities:
                        entities.append(node.name)
        
        return entities
    
    def _find_node_by_name(self, name: str, node_type: Optional[str] = None) -> Optional[KGNode]:
        """Find a node by name (case-insensitive)."""
        name_lower = name.lower()
        for node in self.nodes.values():
            if node.name.lower() == name_lower:
                if node_type is None or node.type == node_type:
                    return node
        return None
    
    def get_related_sources(self, node_id: str) -> list[KGNode]:
        """
        Get all nodes linked to the given node.
        
        Args:
            node_id: ID of the node to find relations for
            
        Returns:
            List of related KGNode objects
        """
        related = []
        
        for edge in self.edges.values():
            if edge.source_id == node_id:
                if edge.target_id in self.nodes:
                    related.append(self.nodes[edge.target_id])
            elif edge.target_id == node_id:
                if edge.source_id in self.nodes:
                    related.append(self.nodes[edge.source_id])
        
        return related
    
    # =========================================================================
    # Node Management
    # =========================================================================
    
    def add_node(self, node: KGNode) -> KGNode:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self._save()
        return node
    
    def add_document_node(self, doc_id: str, name: str, metadata: Optional[dict] = None) -> KGNode:
        """Add a document node (from vector store)."""
        node = KGNode(
            id=f"doc:{doc_id}",
            type='document',
            name=name,
            source='vector',
            properties=metadata or {}
        )
        return self.add_node(node)
    
    def add_table_node(self, table_name: str, schema: Optional[dict] = None) -> KGNode:
        """Add a table node (from DuckDB)."""
        node = KGNode(
            id=f"table:{table_name}",
            type='table',
            name=table_name,
            source='sql',
            properties=schema or {}
        )
        return self.add_node(node)
    
    def add_entity_node(self, name: str, entity_type: str = 'unknown') -> KGNode:
        """Add an entity node (person, organization, etc.)."""
        # Normalize ID
        entity_id = f"entity:{name.lower().replace(' ', '_')}"
        
        # Check if exists
        if entity_id in self.nodes:
            return self.nodes[entity_id]
        
        node = KGNode(
            id=entity_id,
            type='entity',
            name=name,
            source='manual',
            properties={'entity_type': entity_type}
        )
        return self.add_node(node)
    
    def get_node(self, node_id: str) -> Optional[KGNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its edges."""
        if node_id not in self.nodes:
            return False
        
        # Remove node
        del self.nodes[node_id]
        
        # Remove related edges
        edges_to_remove = [
            edge_id for edge_id, edge in self.edges.items()
            if edge.source_id == node_id or edge.target_id == node_id
        ]
        for edge_id in edges_to_remove:
            del self.edges[edge_id]
        
        self._save()
        return True
    
    # =========================================================================
    # Edge Management
    # =========================================================================
    
    def link(
        self, 
        source_id: str, 
        target_id: str, 
        relationship: str,
        weight: float = 1.0,
        properties: Optional[dict] = None
    ) -> Optional[KGEdge]:
        """
        Create a relationship between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship: Type of relationship
            weight: Edge weight for ranking (default 1.0)
            properties: Additional edge properties
            
        Returns:
            Created edge, or None if nodes don't exist
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        edge = KGEdge(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            weight=weight,
            properties=properties or {}
        )
        
        self.edges[edge.id] = edge
        self._save()
        return edge
    
    def unlink(self, source_id: str, target_id: str, relationship: str) -> bool:
        """Remove a specific relationship."""
        edge_id = f"{source_id}--{relationship}-->{target_id}"
        if edge_id in self.edges:
            del self.edges[edge_id]
            self._save()
            return True
        return False
    
    def get_edges(self, node_id: str, relationship: Optional[str] = None) -> list[KGEdge]:
        """Get all edges for a node, optionally filtered by relationship."""
        result = []
        for edge in self.edges.values():
            if edge.source_id == node_id or edge.target_id == node_id:
                if relationship is None or edge.relationship == relationship:
                    result.append(edge)
        return result
    
    # =========================================================================
    # Search and Query
    # =========================================================================
    
    def search_nodes(
        self, 
        query: str, 
        node_type: Optional[str] = None,
        limit: int = 10
    ) -> list[KGNode]:
        """
        Search nodes by name.
        
        Args:
            query: Search string (case-insensitive)
            node_type: Optional filter by type
            limit: Maximum results
            
        Returns:
            List of matching nodes
        """
        query_lower = query.lower()
        results = []
        
        for node in self.nodes.values():
            if node_type and node.type != node_type:
                continue
            if query_lower in node.name.lower():
                results.append(node)
                if len(results) >= limit:
                    break
        
        return results
    
    def find_path(
        self, 
        source_id: str, 
        target_id: str, 
        max_depth: int = 3
    ) -> Optional[list[str]]:
        """
        Find a path between two nodes (BFS).
        
        Returns:
            List of node IDs representing the path, or None if no path exists
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        if source_id == target_id:
            return [source_id]
        
        # BFS
        from collections import deque
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            for neighbor in self.get_related_sources(current):
                if neighbor.id == target_id:
                    return path + [target_id]
                
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [neighbor.id]))
        
        return None
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        node_types = {}
        for node in self.nodes.values():
            node_types[node.type] = node_types.get(node.type, 0) + 1
        
        rel_types = {}
        for edge in self.edges.values():
            rel_types[edge.relationship] = rel_types.get(edge.relationship, 0) + 1
        
        return {
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'node_types': node_types,
            'relationship_types': rel_types
        }
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def _save(self):
        """Save graph to JSON files."""
        # Save nodes
        nodes_path = os.path.join(self.storage_path, 'nodes.json')
        with open(nodes_path, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self.nodes.items()}, f, indent=2)
        
        # Save edges
        edges_path = os.path.join(self.storage_path, 'edges.json')
        with open(edges_path, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self.edges.items()}, f, indent=2)
    
    def _load(self):
        """Load graph from JSON files."""
        # Load nodes
        nodes_path = os.path.join(self.storage_path, 'nodes.json')
        if os.path.exists(nodes_path):
            with open(nodes_path, 'r') as f:
                data = json.load(f)
                self.nodes = {k: KGNode.from_dict(v) for k, v in data.items()}
        
        # Load edges
        edges_path = os.path.join(self.storage_path, 'edges.json')
        if os.path.exists(edges_path):
            with open(edges_path, 'r') as f:
                data = json.load(f)
                self.edges = {k: KGEdge.from_dict(v) for k, v in data.items()}
    
    def clear(self):
        """Clear all nodes and edges."""
        self.nodes = {}
        self.edges = {}
        self._save()

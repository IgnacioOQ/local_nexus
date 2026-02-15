"""
MCP Server: Local Nexus Client Agent

Exposes the Unified Nexus Architecture (RAG + Data Warehouse + KG)
to LLM agents via MCP for local question answering.

Components:
  - DuckDB: Structured data queries (Text2SQL)
  - ChromaDB: Unstructured document search (RAG)
  - Knowledge Graph: Query expansion and entity linking
  - Unified Engine: Query routing and answer synthesis

Usage:
    python -m src.mcp.server
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

# MCP imports (graceful fallback if not installed)
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, Resource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    Tool = None
    TextContent = None
    Resource = None


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "db_path": "data/warehouse.db",
    "vector_store_path": "data/vectordb",
    "kg_store_path": "data/kg",
}


# =============================================================================
# Engine Singleton
# =============================================================================

_engine: Optional[Any] = None


def get_engine():
    """Get or create the Unified Engine instance."""
    global _engine
    if _engine is None:
        from src.core.unified_engine import UnifiedEngine
        from src.core.database import DatabaseManager

        # Initialize database
        db = DatabaseManager(CONFIG["db_path"])

        # Initialize vector store (may fail if chromadb not installed)
        vector_store = None
        try:
            from src.core.vector_store import VectorStore
            vector_store = VectorStore(db_path=CONFIG["vector_store_path"])
        except ImportError:
            pass

        # Initialize Knowledge Graph Metadata
        kg_metadata = None
        try:
            from src.core.kg_metadata import KnowledgeGraphMetadata
            kg_metadata = KnowledgeGraphMetadata(storage_path=CONFIG["kg_store_path"])
        except Exception:
            pass

        # Initialize LLM
        llm_func = None
        try:
            from src.core.llm import init_gemini, DEFAULT_MODEL
            from google import genai

            if init_gemini():
                def gemini_call(prompt: str) -> str:
                    api_key = os.getenv("GEMINI_API_KEY")
                    client = genai.Client(api_key=api_key)
                    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    return response.text

                llm_func = gemini_call
        except ImportError:
            pass

        _engine = UnifiedEngine(
            vector_store=vector_store,
            db_connection=db.get_connection(),
            kg_metadata=kg_metadata,
            llm_func=llm_func
        )

    return _engine


# =============================================================================
# Tool Registry
# =============================================================================

TOOLS: Dict[str, Dict[str, Any]] = {}


def register_tool(name: str, description: str, input_schema: Dict[str, Any]):
    """Decorator to register MCP tools."""
    def decorator(func):
        TOOLS[name] = {
            "description": description,
            "schema": input_schema,
            "handler": func
        }
        return func
    return decorator


# =============================================================================
# Query Interface Tools
# =============================================================================

@register_tool(
    name="unified_query",
    description="""Answer a question using the Unified Nexus engine.

Automatically routes to appropriate data source(s):
- STRUCTURED: SQL over DuckDB (counts, sums, aggregations)
- UNSTRUCTURED: Semantic search over documents (policies, concepts)
- HYBRID: Combines sources when needed

The Knowledge Graph expands queries by finding related tables/documents.

This is your primary tool for answering user questions about data.""",
    input_schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Natural language question"
            },
            "force_type": {
                "type": "string",
                "enum": ["auto", "structured", "unstructured", "hybrid"],
                "description": "Force a specific retrieval path (default: auto)",
                "default": "auto"
            }
        },
        "required": ["question"]
    }
)
async def unified_query(question: str, force_type: str = "auto") -> Dict[str, Any]:
    """Execute unified query across all data sources."""
    engine = get_engine()

    force = None if force_type == "auto" else force_type
    result = engine.query(question, force_type=force)

    return {
        "status": "success" if not result.error else "error",
        "query_type": result.query_type,
        "answer": result.answer,
        "sql_query": result.sql_query,
        "sources_count": len(result.sources),
        "decomposed_queries": result.decomposed_queries,
        "error": result.error
    }


@register_tool(
    name="classify_query",
    description="""Classify a question without executing it.

Returns the query type (structured/unstructured/hybrid) and confidence.
Useful for understanding how the system would handle a question.""",
    input_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Question to classify"}
        },
        "required": ["question"]
    }
)
async def classify_query(question: str) -> Dict[str, Any]:
    """Classify query type without execution."""
    engine = get_engine()
    query_type, confidence, scores = engine.router.classify_with_confidence(question)

    return {
        "status": "success",
        "query_type": query_type.value,
        "confidence": confidence,
        "scores": scores,
        "explanation": engine.router.get_routing_explanation(question)
    }


# =============================================================================
# Structured Data Tools
# =============================================================================

@register_tool(
    name="execute_sql",
    description="""Execute SQL directly against DuckDB.

Use for complex queries, custom joins, or when you need precise control.
For simple questions, prefer unified_query.

Results limited to prevent context overflow.""",
    input_schema={
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "SQL query"},
            "limit": {"type": "integer", "description": "Max rows (default: 100)", "default": 100}
        },
        "required": ["sql"]
    }
)
async def execute_sql(sql: str, limit: int = 100) -> Dict[str, Any]:
    """Execute SQL query."""
    engine = get_engine()

    if not engine.text2sql:
        return {"status": "error", "error": "Text2SQL not configured"}

    # Add limit if not present
    sql_upper = sql.upper()
    if "LIMIT" not in sql_upper:
        sql = f"{sql.rstrip(';')} LIMIT {limit}"

    result = engine.text2sql.execute_sql(sql)

    if result.success:
        return {
            "status": "success",
            "sql": result.sql,
            "columns": result.columns,
            "row_count": result.row_count,
            "data": result.data[:limit] if result.data else []
        }
    else:
        return {
            "status": "error",
            "sql": sql,
            "error": result.error,
            "suggestion": "Use describe_table to check schema"
        }


@register_tool(
    name="list_tables",
    description="List all tables in the data warehouse with basic info.",
    input_schema={"type": "object", "properties": {}}
)
async def list_tables() -> Dict[str, Any]:
    """List available tables."""
    engine = get_engine()

    if not engine.text2sql:
        return {"status": "error", "error": "Text2SQL not configured"}

    tables = engine.text2sql.get_available_tables()

    table_info = []
    for table_name in tables:
        schema = engine.text2sql.get_table_info(table_name)
        if schema:
            table_info.append({
                "name": table_name,
                "columns": len(schema.columns),
                "rows": schema.row_count
            })

    return {
        "status": "success",
        "tables": table_info
    }


@register_tool(
    name="describe_table",
    description="""Get detailed table information: columns, types, sample data.

Use before writing SQL to understand table structure.""",
    input_schema={
        "type": "object",
        "properties": {
            "table_name": {"type": "string", "description": "Table name"}
        },
        "required": ["table_name"]
    }
)
async def describe_table(table_name: str) -> Dict[str, Any]:
    """Describe table schema."""
    engine = get_engine()

    if not engine.text2sql:
        return {"status": "error", "error": "Text2SQL not configured"}

    schema = engine.text2sql.get_table_info(table_name)

    if not schema:
        tables = engine.text2sql.get_available_tables()
        return {
            "status": "error",
            "error": f"Table '{table_name}' not found",
            "available_tables": tables
        }

    return {
        "status": "success",
        "table": table_name,
        "columns": schema.columns,
        "row_count": schema.row_count,
        "sample_rows": schema.sample_data
    }


# =============================================================================
# Unstructured Data Tools
# =============================================================================

@register_tool(
    name="semantic_search",
    description="""Search documents by meaning (not keywords).

Good for finding policies, concepts, context on topics.
Returns most relevant document chunks with metadata.""",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Semantic search query"},
            "top_k": {"type": "integer", "description": "Number of results", "default": 5},
            "source_filter": {"type": "string", "description": "Filter by source type"}
        },
        "required": ["query"]
    }
)
async def semantic_search(query: str, top_k: int = 5, source_filter: str = None) -> Dict[str, Any]:
    """Semantic document search."""
    engine = get_engine()

    if not engine.vector_store:
        return {"status": "error", "error": "Vector store not configured"}

    where = {"source": source_filter} if source_filter else None
    results = engine.vector_store.query(query, n_results=top_k, where=where)

    formatted_results = []
    if results.get('documents') and results['documents'][0]:
        docs = results['documents'][0]
        ids = results.get('ids', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        for i, doc in enumerate(docs):
            formatted_results.append({
                "id": ids[i] if i < len(ids) else None,
                "text": doc[:500] + "..." if len(doc) > 500 else doc,
                "source": metadatas[i].get("source", "unknown") if i < len(metadatas) else "unknown",
                "relevance": round(1 - distances[i], 3) if i < len(distances) else 0
            })

    return {
        "status": "success",
        "query": query,
        "results": formatted_results
    }


@register_tool(
    name="list_document_sources",
    description="List all document sources that have been ingested.",
    input_schema={"type": "object", "properties": {}}
)
async def list_document_sources() -> Dict[str, Any]:
    """List ingested document sources."""
    engine = get_engine()

    if not engine.vector_store:
        return {"status": "error", "error": "Vector store not configured"}

    stats = engine.vector_store.get_stats()

    return {
        "status": "success",
        "document_count": stats.get("count", 0),
        "collection_name": stats.get("name", "documents")
    }


# =============================================================================
# Knowledge Graph Tools
# =============================================================================

# Helper to get KG instance
def _get_kg():
    """Get KnowledgeGraphMetadata instance."""
    engine = get_engine()
    return engine.kg_metadata


@register_tool(
    name="kg_get_related",
    description="""Get all sources (tables/documents) linked to an entity.

Use for questions like:
- "What data sources are related to Alice Chen?"
- "What tables mention this project?"

Returns linked nodes with relationship types.""",
    input_schema={
        "type": "object",
        "properties": {
            "entity_name": {"type": "string", "description": "Entity name to find relations for"}
        },
        "required": ["entity_name"]
    }
)
async def kg_get_related(entity_name: str) -> Dict[str, Any]:
    """Get sources linked to an entity."""
    kg = _get_kg()
    
    if not kg:
        return {"status": "error", "error": "Knowledge Graph not configured"}
    
    # Find entity node
    entity_node = kg._find_node_by_name(entity_name, node_type='entity')
    if not entity_node:
        # Try adding as new entity
        return {
            "status": "not_found",
            "message": f"Entity '{entity_name}' not found in Knowledge Graph",
            "suggestion": "Use kg_list_entities to see available entities"
        }
    
    related = kg.get_related_sources(entity_node.id)
    
    return {
        "status": "success",
        "entity": entity_name,
        "related_sources": [
            {
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "source": node.source
            }
            for node in related
        ]
    }


@register_tool(
    name="kg_find_path",
    description="""Find connection path between two nodes in the Knowledge Graph.

Use for questions like:
- "How is customer X related to product Y?"
- "What connects this document to that table?"

Returns the shortest path if one exists.""",
    input_schema={
        "type": "object",
        "properties": {
            "from_node": {"type": "string", "description": "Starting node name"},
            "to_node": {"type": "string", "description": "Target node name"},
            "max_depth": {"type": "integer", "description": "Maximum path length", "default": 3}
        },
        "required": ["from_node", "to_node"]
    }
)
async def kg_find_path(from_node: str, to_node: str, max_depth: int = 3) -> Dict[str, Any]:
    """Find path between two nodes."""
    kg = _get_kg()
    
    if not kg:
        return {"status": "error", "error": "Knowledge Graph not configured"}
    
    # Find source node
    source = kg._find_node_by_name(from_node)
    target = kg._find_node_by_name(to_node)
    
    if not source:
        return {"status": "not_found", "message": f"Node '{from_node}' not found"}
    if not target:
        return {"status": "not_found", "message": f"Node '{to_node}' not found"}
    
    path = kg.find_path(source.id, target.id, max_depth)
    
    if path:
        # Resolve IDs to names
        path_names = [kg.nodes[node_id].name for node_id in path if node_id in kg.nodes]
        return {
            "status": "success",
            "path": path_names,
            "path_length": len(path) - 1
        }
    else:
        return {
            "status": "no_path",
            "message": f"No path found between '{from_node}' and '{to_node}' within {max_depth} hops"
        }


@register_tool(
    name="kg_list_entities",
    description="List all known entities in the Knowledge Graph.",
    input_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Maximum entities to return", "default": 20}
        }
    }
)
async def kg_list_entities(limit: int = 20) -> Dict[str, Any]:
    """List all known entities."""
    kg = _get_kg()
    
    if not kg:
        return {"status": "error", "error": "Knowledge Graph not configured"}
    
    entities = kg.search_nodes("", node_type="entity", limit=limit)
    stats = kg.get_stats()
    
    return {
        "status": "success",
        "total_nodes": stats.get("node_count", 0),
        "total_edges": stats.get("edge_count", 0),
        "entities": [
            {"name": e.name, "type": e.properties.get("entity_type", "unknown")}
            for e in entities
        ]
    }


@register_tool(
    name="kg_add_link",
    description="""Create a relationship between two nodes.

Use to manually link entities to tables/documents.
Example: Link "Alice Chen" to "budgets" table with "manages" relationship.""",
    input_schema={
        "type": "object",
        "properties": {
            "source_name": {"type": "string", "description": "Source node name"},
            "target_name": {"type": "string", "description": "Target node name"},
            "relationship": {"type": "string", "description": "Relationship type (e.g., manages, references, describes)"}
        },
        "required": ["source_name", "target_name", "relationship"]
    }
)
async def kg_add_link(source_name: str, target_name: str, relationship: str) -> Dict[str, Any]:
    """Create a relationship between nodes."""
    kg = _get_kg()
    
    if not kg:
        return {"status": "error", "error": "Knowledge Graph not configured"}
    
    # Find or create source node
    source = kg._find_node_by_name(source_name)
    if not source:
        # Create as entity
        source = kg.add_entity_node(source_name)
    
    # Find target node
    target = kg._find_node_by_name(target_name)
    if not target:
        return {"status": "not_found", "message": f"Target node '{target_name}' not found"}
    
    edge = kg.link(source.id, target.id, relationship)
    
    if edge:
        return {
            "status": "success",
            "message": f"Created link: {source_name} --[{relationship}]--> {target_name}",
            "edge_id": edge.id
        }
    else:
        return {
            "status": "error",
            "error": "Failed to create link"
        }


@register_tool(
    name="kg_search",
    description="Search the Knowledge Graph by name or type.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (name substring)"},
            "node_type": {"type": "string", "description": "Filter by type (entity, table, document, topic)"},
            "limit": {"type": "integer", "description": "Max results", "default": 10}
        },
        "required": ["query"]
    }
)
async def kg_search(query: str, node_type: str = None, limit: int = 10) -> Dict[str, Any]:
    """Search nodes in the Knowledge Graph."""
    kg = _get_kg()
    
    if not kg:
        return {"status": "error", "error": "Knowledge Graph not configured"}
    
    results = kg.search_nodes(query, node_type=node_type, limit=limit)
    
    return {
        "status": "success",
        "query": query,
        "results": [
            {
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "source": node.source
            }
            for node in results
        ]
    }


# =============================================================================
# Ingestion Tools
# =============================================================================

@register_tool(
    name="ingest_document",
    description="""Ingest a document into the vector store for semantic search.

Supports: PDF, TXT, DOCX, MD files.
After ingestion, the document becomes searchable via semantic_search.""",
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the document file"},
            "source_name": {"type": "string", "description": "Human-readable source identifier"},
        },
        "required": ["file_path"]
    }
)
async def ingest_document(file_path: str, source_name: str = None) -> Dict[str, Any]:
    """Ingest document into vector store."""
    engine = get_engine()

    if not engine.vector_store:
        return {"status": "error", "error": "Vector store not configured"}

    if not os.path.exists(file_path):
        return {"status": "error", "error": f"File not found: {file_path}"}

    try:
        from src.core.document_ingestion import DocumentIngester

        ingester = DocumentIngester(engine.vector_store)
        result = ingester.ingest_file(file_path, source_name=source_name)

        return {
            "status": "success",
            "file": file_path,
            "chunks_created": result.get("chunks", 0),
            "source": source_name or os.path.basename(file_path)
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# System Tools
# =============================================================================

@register_tool(
    name="get_system_status",
    description="Check health and statistics of all Unified Nexus components.",
    input_schema={"type": "object", "properties": {}}
)
async def get_system_status() -> Dict[str, Any]:
    """Get system status."""
    engine = get_engine()
    stats = engine.get_stats()

    return {
        "status": "healthy",
        "components": {
            "vector_store": stats.get("has_vector_store", False),
            "database": stats.get("has_db_connection", False),
            "knowledge_graph": stats.get("kg_metadata") is not None,
            "llm": stats.get("has_llm", False)
        },
        "details": stats
    }


@register_tool(
    name="clear_cache",
    description="Clear query decomposition cache. Use after data changes.",
    input_schema={"type": "object", "properties": {}}
)
async def clear_cache() -> Dict[str, Any]:
    """Clear engine caches."""
    engine = get_engine()
    engine.clear_cache()

    return {
        "status": "success",
        "message": "Query decomposition cache cleared"
    }


# =============================================================================
# MCP Server Creation
# =============================================================================

def create_server() -> Optional[Any]:
    """Create and configure the MCP server."""
    if not MCP_AVAILABLE:
        return None

    server = Server("local-nexus-client")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """Return all registered tools."""
        return [
            Tool(name=name, description=tool["description"], inputSchema=tool["schema"])
            for name, tool in TOOLS.items()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Execute a tool by name."""
        if name not in TOOLS:
            return [TextContent(type="text", text=json.dumps({
                "error": "UnknownTool",
                "message": f"Tool '{name}' not found",
                "available": list(TOOLS.keys())
            }))]

        try:
            handler = TOOLS[name]["handler"]
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = handler(**arguments)
            return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({
                "error": type(e).__name__,
                "message": str(e)
            }))]

    @server.list_resources()
    async def list_resources() -> List[Resource]:
        """Expose read-only resources."""
        return [
            Resource(
                uri="config://nexus/schema",
                name="Database Schema",
                description="Current data warehouse schema"
            ),
            Resource(
                uri="config://nexus/sources",
                name="Document Sources",
                description="List of ingested document sources"
            )
        ]

    return server


async def run_server():
    """Run the MCP server via stdio."""
    if not MCP_AVAILABLE:
        print("MCP package not installed. Install with: pip install mcp")
        return

    server = create_server()
    if server:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())


# =============================================================================
# Synchronous Tool Access (for non-MCP usage)
# =============================================================================

def call_tool_sync(name: str, **kwargs) -> Dict[str, Any]:
    """
    Call a tool synchronously (for use outside MCP context).

    Example:
        result = call_tool_sync("unified_query", question="How many customers?")
    """
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}

    handler = TOOLS[name]["handler"]

    if asyncio.iscoroutinefunction(handler):
        # Run async handler in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(handler(**kwargs))
    else:
        return handler(**kwargs)


def get_available_tools() -> List[str]:
    """Get list of available tool names."""
    return list(TOOLS.keys())


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    if MCP_AVAILABLE:
        asyncio.run(run_server())
    else:
        print("MCP package not installed.")
        print("Available tools (sync mode):", get_available_tools())

        # Demo: run a query
        print("\nDemo: Running unified_query...")
        result = call_tool_sync("unified_query", question="What tables are available?")
        print(json.dumps(result, indent=2, default=str))

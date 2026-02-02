# Client MCP Agent Skill
- id: skills.client_mcp_agent
- status: active
- type: agent_skill
- owner: local-assistant
- last_checked: 2025-01-29
<!-- content -->
This skill enables a coding assistant running **locally on the client's computer** to implement and extend MCP protocols that expose the **Unified Nexus Architecture** (RAG + Data Warehouse + Knowledge Graph). The agent becomes the intelligent interface through which users query their local data ecosystem.

## Role Definition
- id: skills.client_mcp_agent.role
- status: active
- type: context
<!-- content -->
You are a **Local Nexus Client Agent**—an AI assistant running on the user's machine with MCP access to:

1. **DuckDB Data Warehouse** — Structured tables, SQL queries, aggregations
2. **ChromaDB Vector Store** — Unstructured documents, semantic search
3. **Knowledge Graph Metadata Layer** — Entity-data links, query expansion, guided retrieval
4. **Unified Query Engine** — Orchestration layer that routes and combines retrieval paths

Your purpose is to **answer questions** by intelligently leveraging these data sources, and to **help developers extend** the MCP protocol when new capabilities are needed.

## Architecture Context
- id: skills.client_mcp_agent.architecture
- status: active
- type: context
<!-- content -->
The Unified Nexus follows the **TAG (Table-Augmented Generation)** paradigm:

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Question                              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Router (LLM)                           │
│         Classifies: structured / unstructured / hybrid          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              ★ Knowledge Graph Metadata Layer ★                 │
│                                                                 │
│  • Extracts entities from question                              │
│  • Finds linked tables/documents for each entity                │
│  • Expands retrieval scope with SQL hints                       │
│  • Upgrades query type to HYBRID when needed                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       │                                               │
       ▼                                               ▼
┌─────────────┐                                 ┌─────────────┐
│ Vector Store│                                 │   DuckDB    │
│ (ChromaDB)  │                                 │ (Text2SQL)  │
│             │                                 │             │
│ Semantic    │                                 │ Structured  │
│ Search      │                                 │ Queries     │
└──────┬──────┘                                 └──────┬──────┘
       │                                               │
       └───────────────────────┬───────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Context Assembly & Answer Generation               │
│         (Quality LLM synthesizes final response)                │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Many real questions require BOTH semantic understanding AND precise computation:
- "Which product has the most complaints about shipping?" → semantic + aggregation
- "Compare revenue trends with what the market report says" → structured + unstructured
- "Find customers who mentioned pricing concerns and have > $10K LTV" → semantic filter + computation

## MCP Tool Categories
- id: skills.client_mcp_agent.tool_categories
- status: active
- type: guideline
<!-- content -->
The Client MCP Agent should expose tools organized into these functional categories:

### Category 1: Query Interface
- id: skills.client_mcp_agent.tool_categories.query
- status: active
- type: protocol
<!-- content -->
High-level tools that abstract the complexity of multi-source retrieval.

| Tool | Purpose | When to Use |
|:-----|:--------|:------------|
| `unified_query` | Route and execute queries across all sources | Default for user questions |
| `classify_query` | Determine query type without executing | Debugging, planning |
| `explain_routing` | Show why a query was routed a certain way | Transparency, debugging |

**Tool: `unified_query`**
```python
@register_tool(
    name="unified_query",
    description="""
    Answer a natural language question using the Unified Nexus engine.
    Automatically routes to the appropriate data source(s):
    - STRUCTURED: SQL over DuckDB for counts, sums, comparisons
    - UNSTRUCTURED: Semantic search over documents for policies, concepts
    - GRAPH: Relationship traversal for connections, paths, hierarchies
    - HYBRID: Combines multiple sources when question requires both
    
    Use this as your primary tool for answering user questions about data.
    """,
    input_schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Natural language question to answer"
            },
            "force_type": {
                "type": "string",
                "enum": ["auto", "structured", "unstructured", "graph", "hybrid"],
                "description": "Force a specific retrieval path (default: auto)",
                "default": "auto"
            }
        },
        "required": ["question"]
    }
)
async def unified_query(question: str, force_type: str = "auto") -> dict:
    """Execute a unified query across all data sources."""
    # Implementation delegates to UnifiedEngine.query()
    pass
```

### Category 2: Structured Data Tools
- id: skills.client_mcp_agent.tool_categories.structured
- status: active
- type: protocol
<!-- content -->
Direct access to the DuckDB data warehouse for precise operations.

| Tool | Purpose | When to Use |
|:-----|:--------|:------------|
| `execute_sql` | Run raw SQL queries | Complex queries, joins |
| `text_to_sql` | Convert NL to SQL without executing | Validation, learning |
| `list_tables` | Show available tables | Discovery, schema exploration |
| `describe_table` | Get table schema and samples | Understanding data structure |
| `get_table_stats` | Row counts, value distributions | Data profiling |

**Tool: `execute_sql`**
```python
@register_tool(
    name="execute_sql",
    description="""
    Execute a SQL query against the DuckDB data warehouse.
    
    ⚠️ Use this for complex queries that text_to_sql might not handle well,
    or when you need precise control over the SQL.
    
    For simple questions, prefer unified_query which handles routing automatically.
    
    Returns results as a list of row dictionaries.
    Limit results to avoid overwhelming context (default: 100 rows).
    """,
    input_schema={
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "SQL query to execute"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum rows to return (default: 100)",
                "default": 100
            }
        },
        "required": ["sql"]
    }
)
async def execute_sql(sql: str, limit: int = 100) -> dict:
    """Execute SQL and return results."""
    pass
```

**Tool: `describe_table`**
```python
@register_tool(
    name="describe_table",
    description="""
    Get detailed information about a table in the data warehouse.
    
    Returns:
    - Column names and types
    - Row count
    - Sample rows (first 5)
    - Null counts per column
    - Unique value counts for low-cardinality columns
    
    Use this to understand table structure before writing queries.
    """,
    input_schema={
        "type": "object",
        "properties": {
            "table_name": {
                "type": "string",
                "description": "Name of the table to describe"
            }
        },
        "required": ["table_name"]
    }
)
async def describe_table(table_name: str) -> dict:
    """Get table schema and statistics."""
    pass
```

### Category 3: Unstructured Data Tools
- id: skills.client_mcp_agent.tool_categories.unstructured
- status: active
- type: protocol
<!-- content -->
Access to the ChromaDB vector store for semantic search.

| Tool | Purpose | When to Use |
|:-----|:--------|:------------|
| `semantic_search` | Find relevant documents | Policies, concepts, context |
| `list_document_sources` | Show ingested document sources | Discovery |
| `get_document_by_id` | Retrieve specific document | Follow-up, citation |
| `search_with_filter` | Semantic search with metadata filters | Scoped searches |

**Tool: `semantic_search`**
```python
@register_tool(
    name="semantic_search",
    description="""
    Search for documents semantically related to a query.
    
    Uses vector embeddings to find documents by meaning, not keywords.
    Good for:
    - Finding policies, procedures, guidelines
    - Understanding concepts and definitions
    - Locating relevant context for a topic
    
    Returns top-k most relevant document chunks with metadata.
    """,
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (semantic, not keyword-based)"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5
            },
            "source_filter": {
                "type": "string",
                "description": "Optional: filter by document source/type"
            }
        },
        "required": ["query"]
    }
)
async def semantic_search(query: str, top_k: int = 5, source_filter: str = None) -> dict:
    """Perform semantic search over documents."""
    pass
```

### Category 4: Knowledge Graph Tools
- id: skills.client_mcp_agent.tool_categories.knowledge_graph
- status: active
- type: protocol
<!-- content -->
Tools for query expansion and entity-data linking.

| Tool | Purpose | When to Use |
|:-----|:--------|:------------|
| `kg_get_related` | Get sources linked to an entity | Find what tables/docs relate to a person/concept |
| `kg_find_path` | Find paths between nodes | Relationship discovery |
| `kg_list_entities` | List known entities | Entity exploration |
| `kg_add_link` | Create entity-data links | Manual linking |
| `kg_search` | Search graph by name/type | Discovery |

**Tool: `kg_get_related`**
```python
@register_tool(
    name="kg_get_related",
    description="""
    Get all sources (tables/documents) linked to an entity.
    
    Use for questions like:
    - "What data sources are related to Alice Chen?"
    - "What tables mention this project?"
    
    Returns linked nodes with relationship types.
    """,
    input_schema={
        "type": "object",
        "properties": {
            "entity_name": {
                "type": "string",
                "description": "Entity name to find relations for"
            }
        },
        "required": ["entity_name"]
    }
)
async def kg_get_related(entity_name: str) -> dict:
    """Get sources linked to an entity."""
    kg = _get_kg()
    entity_node = kg._find_node_by_name(entity_name, node_type='entity')
    if not entity_node:
        return {"status": "not_found", "message": f"Entity '{entity_name}' not found"}
    
    related = kg.get_related_sources(entity_node.id)
    return {
        "status": "success",
        "entity": entity_name,
        "related_sources": [
            {"id": n.id, "name": n.name, "type": n.type}
            for n in related
        ]
    }
```

**Tool: `kg_add_link`**
```python
@register_tool(
    name="kg_add_link",
    description="""
    Create a relationship between two nodes.
    
    Use to manually link entities to tables/documents.
    Example: Link "Alice Chen" to "budgets" table with "manages" relationship.
    """,
    input_schema={
        "type": "object",
        "properties": {
            "source_name": {"type": "string", "description": "Source node name"},
            "target_name": {"type": "string", "description": "Target node name"},
            "relationship": {"type": "string", "description": "Relationship type"}
        },
        "required": ["source_name", "target_name", "relationship"]
    }
)
async def kg_add_link(source_name: str, target_name: str, relationship: str) -> dict:
    """Create a relationship between nodes."""
    pass
```

### Category 5: Ingestion Tools
- id: skills.client_mcp_agent.tool_categories.ingestion
- status: active
- type: protocol
<!-- content -->
Tools for adding new data to the system.

| Tool | Purpose | When to Use |
|:-----|:--------|:------------|
| `ingest_document` | Add document to vector store | New unstructured data |
| `ingest_table` | Add CSV/Excel to data warehouse | New structured data |
| `add_graph_edge` | Add relationship to graph | New connections |
| `refresh_schema` | Update Text2SQL schema cache | After table changes |

**Tool: `ingest_document`**
```python
@register_tool(
    name="ingest_document",
    description="""
    Ingest a document into the vector store for semantic search.
    
    Supports: PDF, TXT, DOCX, MD files.
    
    The document will be:
    1. Chunked into smaller pieces
    2. Embedded using sentence-transformers
    3. Stored in ChromaDB with metadata
    
    After ingestion, the document becomes searchable via semantic_search.
    """,
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the document file"
            },
            "source_name": {
                "type": "string",
                "description": "Human-readable source identifier"
            },
            "metadata": {
                "type": "object",
                "description": "Additional metadata (date, author, category, etc.)"
            }
        },
        "required": ["file_path"]
    }
)
async def ingest_document(file_path: str, source_name: str = None, metadata: dict = None) -> dict:
    """Ingest document into vector store."""
    pass
```

### Category 6: System Tools
- id: skills.client_mcp_agent.tool_categories.system
- status: active
- type: protocol
<!-- content -->
Tools for system introspection and maintenance.

| Tool | Purpose | When to Use |
|:-----|:--------|:------------|
| `get_system_status` | Check health of all components | Diagnostics |
| `get_data_stats` | Summary of available data | Overview |
| `clear_cache` | Reset query caches | After data changes |
| `export_query_log` | Get recent query history | Debugging, auditing |

## Operational Patterns
- id: skills.client_mcp_agent.operational_patterns
- status: active
- type: guideline
<!-- content -->

### Pattern 1: Question Answering Flow
- id: skills.client_mcp_agent.operational_patterns.qa_flow
- status: active
- type: protocol
<!-- content -->
The standard flow for answering user questions:

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Assess the question                                     │
│ - Is it about structured data? (counts, totals, comparisons)    │
│ - Is it about unstructured data? (policies, concepts, docs)     │
│ - Is it about relationships? (connections, paths, hierarchies)  │
│ - Does it require multiple sources? (hybrid)                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Choose retrieval strategy                               │
│ - Simple questions → unified_query (let engine decide)          │
│ - Complex SQL needed → execute_sql directly                     │
│ - Specific document needed → semantic_search + get_document     │
│ - Relationship questions → find_connections + get_neighbors     │
│ - Multi-faceted → multiple tools in sequence                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Execute and synthesize                                  │
│ - Call appropriate tool(s)                                      │
│ - Verify results make sense                                     │
│ - If insufficient, try alternative approach                     │
│ - Synthesize final answer from all retrieved context            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                         Final Answer
```

### Pattern 2: Exploratory Analysis
- id: skills.client_mcp_agent.operational_patterns.exploration
- status: active
- type: protocol
<!-- content -->
When users want to understand their data:

```python
# 1. Start with system overview
stats = await get_data_stats()

# 2. Explore relevant tables
tables = await list_tables()
for t in relevant_tables:
    schema = await describe_table(t)
    
# 3. Sample the data
sample = await execute_sql(f"SELECT * FROM {table} LIMIT 10")

# 4. Check available documents
sources = await list_document_sources()

# 5. Formulate specific queries based on understanding
```

### Pattern 3: Hybrid Question Resolution
- id: skills.client_mcp_agent.operational_patterns.hybrid
- status: active
- type: protocol
<!-- content -->
For questions requiring both structured and unstructured data:

**Example**: "Which products have the most complaints about shipping?"

```python
# Step 1: Get products with complaint counts (structured)
sql_result = await execute_sql("""
    SELECT product_id, COUNT(*) as complaint_count
    FROM complaints
    GROUP BY product_id
    ORDER BY complaint_count DESC
    LIMIT 10
""")

# Step 2: For top products, search for shipping-related complaints (semantic)
for product in sql_result['data'][:5]:
    shipping_complaints = await semantic_search(
        query=f"shipping delivery delay problem {product['product_id']}",
        source_filter="complaints"
    )
    
# Step 3: Synthesize findings
# Combine numerical rankings with semantic themes
```

### Pattern 4: Iterative Refinement
- id: skills.client_mcp_agent.operational_patterns.refinement
- status: active
- type: protocol
<!-- content -->
When initial results are insufficient:

```python
# Initial attempt
result = await unified_query("What's our refund policy for damaged items?")

# If result lacks specificity, try targeted search
if not_specific_enough(result):
    docs = await semantic_search(
        query="refund policy damaged items return",
        top_k=10  # Get more candidates
    )
    
# If still insufficient, check for related policies
related = await semantic_search(
    query="return policy warranty damage defect",
    top_k=5
)

# Combine all findings for comprehensive answer
```

## Error Handling Guidelines
- id: skills.client_mcp_agent.error_handling
- status: active
- type: guideline
<!-- content -->

### SQL Errors
- id: skills.client_mcp_agent.error_handling.sql
- status: active
- type: guideline
<!-- content -->
```python
# When SQL fails, provide actionable recovery
{
    "error": "SQLError",
    "message": "Column 'ship_date' not found in table 'orders'",
    "suggestion": "Available date columns: order_date, delivery_date, created_at",
    "schema_hint": await describe_table("orders")
}
```

### Empty Results
- id: skills.client_mcp_agent.error_handling.empty
- status: active
- type: guideline
<!-- content -->
```python
# When no results found, suggest alternatives
{
    "error": "NoResults",
    "message": "No documents found matching 'remote work policy 2024'",
    "suggestions": [
        "Try broader search: 'remote work policy'",
        "Available policy documents: HR_policies.pdf, Employee_handbook.docx",
        "Check if document has been ingested: list_document_sources()"
    ]
}
```

### Ambiguous Queries
- id: skills.client_mcp_agent.error_handling.ambiguous
- status: active
- type: guideline
<!-- content -->
When a query could be interpreted multiple ways:

1. **Ask for clarification** if the ambiguity significantly affects the answer
2. **Make a reasonable assumption and state it** if you can proceed
3. **Provide multiple interpretations** if quick to compute both

```
User: "How many customers last month?"

Response: "I found 1,247 new customers in December 2024. 
If you meant active customers (those who made a purchase), 
that number is 8,432. Which metric were you looking for?"
```

## Extension Guidelines
- id: skills.client_mcp_agent.extension
- status: active
- type: guideline
<!-- content -->
When users need capabilities not yet exposed via MCP:

### Adding New Tools
- id: skills.client_mcp_agent.extension.new_tools
- status: active
- type: task
<!-- content -->
1. **Identify the capability gap** — What can't the current tools do?
2. **Check if it's a composition** — Can existing tools be combined?
3. **Design the interface** — What inputs/outputs make sense for an LLM?
4. **Implement with informative errors** — Help the LLM recover from mistakes
5. **Document with examples** — Show when and how to use the tool

### Tool Design Checklist
- id: skills.client_mcp_agent.extension.checklist
- status: active
- type: task
<!-- content -->
- [ ] Name is descriptive and action-oriented (`analyze_trends` not `trends`)
- [ ] Description explains WHEN to use, not just WHAT it does
- [ ] Input schema has descriptions for all parameters
- [ ] Required vs optional parameters are correctly marked
- [ ] Return format is documented
- [ ] Errors include recovery suggestions
- [ ] Tool is idempotent where possible

### Common Extension Requests
- id: skills.client_mcp_agent.extension.common
- status: active
- type: context
<!-- content -->

| Request | Implementation Approach |
|:--------|:------------------------|
| "Search with date range" | Add date filter parameter to `semantic_search` |
| "Compare two time periods" | New tool `compare_periods` wrapping SQL |
| "Export results to CSV" | New tool `export_results` with format options |
| "Schedule recurring query" | New tool `schedule_query` (requires job system) |
| "Alert on threshold" | New tool `create_alert` (requires monitoring) |

## MCP Server Template
- id: skills.client_mcp_agent.server_template
- status: active
- type: context
<!-- content -->
Complete template for the Client MCP Agent server:

```python
"""
MCP Server: Local Nexus Client Agent

Exposes the Unified Nexus Architecture (RAG + Data Warehouse + Graph)
to LLM agents via MCP for local question answering.

Components:
  - DuckDB: Structured data queries (Text2SQL)
  - ChromaDB: Unstructured document search (RAG)
  - Graph Store: Relationship traversal (optional)
  - Unified Engine: Query routing and answer synthesis
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource

# Import Unified Nexus components
from src.core.unified_engine import UnifiedEngine
from src.core.vector_store import VectorStore
from src.core.text2sql import Text2SQLEngine
from src.core.query_router import QueryRouter, QueryType

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "db_path": "data/warehouse.duckdb",
    "vector_store_path": "data/vectordb",
    "graph_store_path": "data/graphdb",  # Optional
}

# =============================================================================
# Server Initialization
# =============================================================================

server = Server("local-nexus-client")

# Initialize the Unified Engine (lazy loading)
_engine: Optional[UnifiedEngine] = None


def get_engine() -> UnifiedEngine:
    """Get or create the Unified Engine instance."""
    global _engine
    if _engine is None:
        # Initialize with your LLM client
        from src.llm import get_llm_client
        llm = get_llm_client()
        
        _engine = UnifiedEngine(
            db_path=CONFIG["db_path"],
            vector_store_path=CONFIG["vector_store_path"],
            llm_client=llm
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
- GRAPH: Relationship traversal (connections, paths)
- HYBRID: Combines sources when needed

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
                "enum": ["auto", "structured", "unstructured", "graph", "hybrid"],
                "default": "auto"
            }
        },
        "required": ["question"]
    }
)
async def unified_query(question: str, force_type: str = "auto") -> Dict[str, Any]:
    """Execute unified query across all data sources."""
    engine = get_engine()
    
    if force_type != "auto":
        # Override router decision
        type_map = {
            "structured": QueryType.STRUCTURED,
            "unstructured": QueryType.UNSTRUCTURED,
            "graph": QueryType.GRAPH,
            "hybrid": QueryType.HYBRID
        }
        # Implementation would pass forced type to engine
    
    result = engine.query(question)
    
    return {
        "status": "success",
        "query_type": result["query_type"],
        "answer": result["answer"],
        "sources_used": _summarize_sources(result["retrieval"]),
        "confidence": _estimate_confidence(result)
    }


@register_tool(
    name="classify_query",
    description="""Classify a question without executing it.

Returns the query type (structured/unstructured/graph/hybrid) and confidence.
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
    details = engine.router.get_classification_details(question)
    return details


# =============================================================================
# Structured Data Tools
# =============================================================================

@register_tool(
    name="execute_sql",
    description="""Execute SQL directly against DuckDB.

Use for complex queries, custom joins, or when you need precise control.
For simple questions, prefer unified_query.

⚠️ Results limited to prevent context overflow.""",
    input_schema={
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "SQL query"},
            "limit": {"type": "integer", "default": 100}
        },
        "required": ["sql"]
    }
)
async def execute_sql(sql: str, limit: int = 100) -> Dict[str, Any]:
    """Execute SQL query."""
    engine = get_engine()
    
    # Add limit if not present
    if "LIMIT" not in sql.upper():
        sql = f"{sql.rstrip(';')} LIMIT {limit}"
    
    try:
        result = engine.text2sql.execute_sql(sql)
        return {
            "status": "success",
            "sql": sql,
            "row_count": len(result),
            "data": result[:limit]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "sql": sql,
            "suggestion": "Use describe_table to check schema"
        }


@register_tool(
    name="list_tables",
    description="List all tables in the data warehouse with row counts.",
    input_schema={"type": "object", "properties": {}}
)
async def list_tables() -> Dict[str, Any]:
    """List available tables."""
    engine = get_engine()
    schema = engine.text2sql.get_schema()
    
    return {
        "status": "success",
        "tables": [
            {"name": name, "columns": len(info["columns"]), "rows": info.get("row_count", "unknown")}
            for name, info in schema.items()
        ]
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
    schema = engine.text2sql.get_schema()
    
    if table_name not in schema:
        return {
            "status": "error",
            "error": f"Table '{table_name}' not found",
            "available_tables": list(schema.keys())
        }
    
    info = schema[table_name]
    
    # Get sample rows
    sample = engine.text2sql.execute_sql(f"SELECT * FROM {table_name} LIMIT 5")
    
    return {
        "status": "success",
        "table": table_name,
        "columns": info["columns"],
        "row_count": info.get("row_count"),
        "sample_rows": sample
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
            "top_k": {"type": "integer", "default": 5},
            "source_filter": {"type": "string", "description": "Filter by source"}
        },
        "required": ["query"]
    }
)
async def semantic_search(query: str, top_k: int = 5, source_filter: str = None) -> Dict[str, Any]:
    """Semantic document search."""
    engine = get_engine()
    
    where = {"source": source_filter} if source_filter else None
    results = engine.vector_store.search(query, top_k=top_k, where=where)
    
    return {
        "status": "success",
        "query": query,
        "results": [
            {
                "text": r["text"],
                "source": r["metadata"].get("source", "unknown"),
                "relevance": 1 - r["distance"]  # Convert distance to similarity
            }
            for r in results
        ]
    }


@register_tool(
    name="list_document_sources",
    description="List all document sources that have been ingested.",
    input_schema={"type": "object", "properties": {}}
)
async def list_document_sources() -> Dict[str, Any]:
    """List ingested document sources."""
    engine = get_engine()
    
    # Query ChromaDB for unique sources
    # This is a simplified implementation
    stats = engine.vector_store.get_stats()
    
    return {
        "status": "success",
        "total_documents": stats["total_documents"],
        "persist_directory": stats["persist_directory"]
    }


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
    stats = engine.get_system_stats()
    
    return {
        "status": "healthy",
        "components": {
            "vector_store": stats["vector_store"],
            "data_warehouse": {
                "tables": stats["database_schema"]
            }
        }
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _summarize_sources(retrieval: dict) -> List[str]:
    """Summarize which sources were used in retrieval."""
    sources = []
    if retrieval.get("type") == "structured":
        sources.append(f"SQL: {retrieval.get('sql', 'N/A')}")
    elif retrieval.get("type") == "unstructured":
        doc_count = len(retrieval.get("documents", []))
        sources.append(f"Documents: {doc_count} chunks")
    elif retrieval.get("type") == "hybrid":
        sources.extend(_summarize_sources(retrieval.get("structured", {})))
        sources.extend(_summarize_sources(retrieval.get("unstructured", {})))
    return sources


def _estimate_confidence(result: dict) -> str:
    """Estimate confidence in the result."""
    retrieval = result.get("retrieval", {})
    if retrieval.get("success"):
        return "high" if retrieval.get("type") != "hybrid" else "medium"
    return "low"


# =============================================================================
# MCP Handlers
# =============================================================================

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
        result = await handler(**arguments) if asyncio.iscoroutinefunction(handler) else handler(**arguments)
        return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": type(e).__name__,
            "message": str(e)
        }))]


# =============================================================================
# Entry Point
# =============================================================================

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
```

## Resources (Read-Only Context)
- id: skills.client_mcp_agent.resources
- status: active
- type: context
<!-- content -->
In addition to tools, expose MCP Resources for read-only context:

```python
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
        ),
        Resource(
            uri="docs://nexus/query-examples",
            name="Query Examples",
            description="Example queries for each query type"
        )
    ]
```

## Integration with Agent Protocols
- id: skills.client_mcp_agent.integration
- status: active
- type: guideline
<!-- content -->
The Client MCP Agent should work harmoniously with your existing agent documentation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Context Stack                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Layer 1: Agent Persona (.md files)                            │
│   ├── CLIENT_MCP_AGENT.md     → This skill (how to use Nexus)   │
│   ├── DOMAIN_KNOWLEDGE.md     → Business concepts               │
│   └── COMMUNICATION_STYLE.md  → Tone and formatting             │
│                                                                 │
│   Layer 2: MCP Tools (this server)                              │
│   ├── unified_query           → Primary question answering      │
│   ├── execute_sql             → Direct SQL access               │
│   ├── semantic_search         → Document retrieval              │
│   └── [other tools]           → Additional capabilities         │
│                                                                 │
│   Layer 3: Data Layer (Unified Nexus)                           │
│   ├── DuckDB                  → Structured data                 │
│   ├── ChromaDB                → Unstructured documents          │
│   └── Knowledge Graph         → Entity links & query expansion  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The `.md` files tell the agent **what questions to answer and how to communicate**.
The MCP tools tell the agent **how to retrieve the information**.
The data layer **holds the actual knowledge**.

## Version History
- id: skills.client_mcp_agent.version_history
- status: active
- type: log
<!-- content -->
| Date | Version | Changes |
|------|---------|---------|
| 2026-02-02 | 1.1.0 | Replaced Graph Store with Knowledge Graph Metadata Layer; updated architecture diagram; added KG tools |
| 2025-01-29 | 1.0.0 | Initial skill definition adapted from MCP_AGENT.md |

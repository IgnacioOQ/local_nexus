# Unified Nexus Architecture Agent
- id: una-agent
- status: active
- type: agent_skill
- owner: local-nexus
- last_checked: 2026-02-02
<!-- content -->
This document describes the **current implementation** of the Unified Nexus Architecture in the local_nexus project. It serves as both a reference for developers and a skill definition for AI agents interacting with the system.

## Architecture Overview
- id: una-agent.overview
- status: active
- type: context
<!-- content -->
The Unified Nexus Architecture (UNA) combines three data retrieval paradigms into a single intelligent query system:

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

### Core Principle
- id: una-agent.overview.principle
- status: active
- type: context
<!-- content -->
The **TAG (Table-Augmented Generation)** paradigm recognizes that RAG and Data Warehouse approaches are **complementary, not competing**:

| Capability | RAG (Unstructured) | Data Warehouse (Structured) | Knowledge Graph |
|:-----------|:-------------------|:----------------------------|:----------------|
| **Data Type** | Documents, PDFs | Tables, CSVs | Relationships |
| **Query Style** | Semantic similarity | Exact SQL | Entity traversal |
| **Strengths** | Fuzzy matching | Aggregations | Entity-data linking |
| **Example** | "What's our PTO policy?" | "How many sales in Q3?" | "What data relates to Alice?" |

## Implemented Components
- id: una-agent.components
- status: active
- type: context
<!-- content -->

### Core Modules (`src/core/`)
- id: una-agent.components.core
- status: active
- type: context
<!-- content -->

| Module | Purpose | Key Classes |
|:-------|:--------|:------------|
| [unified_engine.py](file:///Users/ignacio/Documents/VS%20Code/GitHub%20Repositories/local_nexus/src/core/unified_engine.py) | Main orchestrator combining all retrieval paths | `UnifiedEngine`, `RetrievalResult` |
| [query_router.py](file:///Users/ignacio/Documents/VS%20Code/GitHub%20Repositories/local_nexus/src/core/query_router.py) | LLM-based query classification | `QueryRouter`, `classify()` |
| [kg_metadata.py](file:///Users/ignacio/Documents/VS%20Code/GitHub%20Repositories/local_nexus/src/core/kg_metadata.py) | Knowledge Graph for query expansion | `KnowledgeGraphMetadata`, `KGNode`, `KGEdge` |
| [vector_store.py](file:///Users/ignacio/Documents/VS%20Code/GitHub%20Repositories/local_nexus/src/core/vector_store.py) | ChromaDB wrapper for semantic search | `VectorStore` |
| [text2sql.py](file:///Users/ignacio/Documents/VS%20Code/GitHub%20Repositories/local_nexus/src/core/text2sql.py) | Natural language to SQL generation | `Text2SQL` |
| [document_ingestion.py](file:///Users/ignacio/Documents/VS%20Code/GitHub%20Repositories/local_nexus/src/core/document_ingestion.py) | Document chunking and embedding | `DocumentChunker` |
| [database.py](file:///Users/ignacio/Documents/VS%20Code/GitHub%20Repositories/local_nexus/src/core/database.py) | DuckDB connection management | `DuckDBConnection` |

### MCP Server (`src/mcp/`)
- id: una-agent.components.mcp
- status: active
- type: context
<!-- content -->

| Module | Purpose |
|:-------|:--------|
| [server.py](file:///Users/ignacio/Documents/VS%20Code/GitHub%20Repositories/local_nexus/src/mcp/server.py) | FastMCP server exposing 15 tools for unified data access |

**Available MCP Tools (15 total):**

| Category | Tool | Purpose |
|:---------|:-----|:--------|
| **Query** | `unified_query` | Route query to appropriate retrieval path(s) |
| **Query** | `classify_query` | Classify query type without executing |
| **Structured** | `execute_sql` | Run raw SQL against DuckDB |
| **Structured** | `list_tables` | List available tables |
| **Structured** | `describe_table` | Get table schema and samples |
| **Unstructured** | `semantic_search` | Search documents by meaning |
| **Unstructured** | `list_document_sources` | List ingested document sources |
| **Knowledge Graph** | `kg_get_related` | Get sources linked to an entity |
| **Knowledge Graph** | `kg_find_path` | Find path between nodes |
| **Knowledge Graph** | `kg_list_entities` | List all known entities |
| **Knowledge Graph** | `kg_add_link` | Create entity-data relationship |
| **Knowledge Graph** | `kg_search` | Search graph by name/type |
| **Ingestion** | `ingest_document` | Add document to vector store |
| **System** | `get_system_status` | Check component health |
| **System** | `clear_cache` | Reset query caches |

## Knowledge Graph Metadata Layer
- id: una-agent.kg-layer
- status: active
- type: context
<!-- content -->
The KG layer is positioned **between the Query Router and databases**, enabling intelligent query expansion.

### How Query Expansion Works
- id: una-agent.kg-layer.expansion
- status: active
- type: context
<!-- content -->

```python
# Example: "What is Alice Chen's budget?"

# Step 1: Query Router classifies as "unstructured"
query_type = "unstructured"

# Step 2: KG extracts entities
entities_found = ["Alice Chen"]

# Step 3: KG finds linked sources
# Alice Chen → manages → budgets (table)
sql_hints = ["budgets"]

# Step 4: KG upgrades query type
query_type = "hybrid"  # Now queries BOTH vector store AND SQL

# Result: Question answered with data from budgets table
```

### Entity Extraction
- id: una-agent.kg-layer.entities
- status: active
- type: context
<!-- content -->
Current entity extraction uses **simple heuristics**:
- Capitalized words (proper nouns)
- Known entity names already in the graph

> [!TIP]
> For production, consider LLM-based Named Entity Recognition (NER) for more accurate extraction.

### Node Types
- id: una-agent.kg-layer.node-types
- status: active
- type: context
<!-- content -->

| Type | Prefix | Purpose | Example |
|:-----|:-------|:--------|:--------|
| `entity` | `entity:` | People, projects, concepts | `entity:alice_chen` |
| `table` | `table:` | Database tables | `table:budgets` |
| `document` | `doc:` | Ingested documents | `doc:policies.pdf` |

### Persistence
- id: una-agent.kg-layer.persistence
- status: active
- type: context
<!-- content -->
KG data is persisted to JSON files in `data/kg/`:
- `nodes.json` - All graph nodes
- `edges.json` - All graph edges

## Data Flow Examples
- id: una-agent.data-flow
- status: active
- type: context
<!-- content -->

### Structured Query
- id: una-agent.data-flow.structured
- status: active
- type: context
<!-- content -->
```
User: "How many orders were placed last month?"
  │
  ├─ Router: STRUCTURED
  ├─ KG: No expansion needed
  ├─ Text2SQL: Generates SQL
  │    SELECT COUNT(*) FROM orders 
  │    WHERE order_date >= '2026-01-01'
  ├─ DuckDB: Executes, returns 1,247
  └─ LLM: "There were 1,247 orders placed last month."
```

### Unstructured Query
- id: una-agent.data-flow.unstructured
- status: active
- type: context
<!-- content -->
```
User: "What is our refund policy?"
  │
  ├─ Router: UNSTRUCTURED
  ├─ KG: No entity links found
  ├─ Vector Store: Semantic search
  │    → policies.pdf (chunk 3): "Refunds are processed within..."
  └─ LLM: "Our refund policy states that refunds are processed..."
```

### Hybrid Query (KG-Expanded)
- id: una-agent.data-flow.hybrid
- status: active
- type: context
<!-- content -->
```
User: "What is Alice Chen's budget allocation?"
  │
  ├─ Router: UNSTRUCTURED (initial)
  ├─ KG: Extracts "Alice Chen"
  │    └─ entity:alice_chen → manages → table:budgets
  ├─ KG: Upgrades to HYBRID
  ├─ Text2SQL + Vector Store (parallel)
  │    ├─ SQL: SELECT * FROM budgets WHERE manager='Alice Chen'
  │    └─ Semantic: Search for "budget" context
  └─ LLM: Synthesizes answer from both sources
```

## Suggested Improvements
- id: una-agent.suggestions
- status: active
- type: guideline
<!-- content -->
Based on the [UNIFIED_NEXUS_ARCHITECTURE.md](file:///Users/ignacio/Documents/VS%20Code/GitHub%20Repositories/local_nexus/mds/UNIFIED_NEXUS_ARCHITECTURE.md) reference design, the following enhancements are recommended:

### High Priority
- id: una-agent.suggestions.high
- status: active
- type: task
<!-- content -->

| Suggestion | Status | Effort | Benefit |
|:-----------|:-------|:-------|:--------|
| **Auto-populate KG on ingestion** | Not implemented | 2-3h | Automatic entity detection when documents/tables are ingested |
| **LLM-based entity extraction** | Partial (heuristics only) | 1h | More accurate entity detection vs. simple capitalization rules |
| **Two-tier LLM strategy** | Not implemented | 1h | Use cheap LLM for routing/SQL, quality LLM for final answer |
| **Query decomposition** | Implemented | - | Already supports breaking complex queries into sub-queries |

### Medium Priority
- id: una-agent.suggestions.medium
- status: active
- type: task
<!-- content -->

| Suggestion | Status | Effort | Benefit |
|:-----------|:-------|:-------|:--------|
| **Semantic SQL (column matching)** | Not implemented | 2h | Use embeddings to help with ambiguous column names |
| **Feedback loop / query logging** | Not implemented | 2h | Log queries and outcomes for fine-tuning |
| **Retry with alternative paths** | Partial | 1h | If one path fails, automatically try another |
| **Graph visualization UI** | Not implemented | 3h | Visual interface for exploring entity relationships |

### Low Priority (Future)
- id: una-agent.suggestions.low
- status: active
- type: context
<!-- content -->

| Suggestion | Effort | Notes |
|:-----------|:-------|:------|
| Multi-modal support (images) | 4h+ | Extend vector store for image embeddings |
| Temporal graphs | 3h | Track how relationships change over time |
| Graph embeddings (GNN) | Research | Use graph neural networks for semantic search over structure |
| Full GraphRAG | 8h+ | Microsoft's approach for document synthesis (many LLM calls) |

## Error Handling
- id: una-agent.error-handling
- status: active
- type: guideline
<!-- content -->

### Current Behavior
- id: una-agent.error-handling.current
- status: active
- type: context
<!-- content -->
- **Component isolation**: Failures in one component (e.g., VectorStore) don't crash the entire system
- **Lazy loading**: Heavy components are loaded on first use
- **Graceful degradation**: If KG is unavailable, queries proceed without expansion

### Recommended Additions
- id: una-agent.error-handling.recommended
- status: active
- type: guideline
<!-- content -->

> [!IMPORTANT]
> Consider adding these error handling patterns from the reference architecture:

1. **SQL Error Recovery**: When generated SQL fails, provide column suggestions from schema
2. **Empty Result Guidance**: When no results found, suggest broader search terms or alternative document sources
3. **Ambiguous Query Clarification**: When query type is uncertain, ask user or provide multiple interpretations

## Usage Guidelines
- id: una-agent.usage
- status: active
- type: guideline
<!-- content -->

### For Users
- id: una-agent.usage.users
- status: active
- type: context
<!-- content -->
1. **Ask naturally**: The system routes your question to the appropriate data source
2. **Use the KG toggle**: Enable/disable Knowledge Graph expansion in the sidebar
3. **Review query type**: The UI shows whether your query was routed to SQL, documents, or both

### For Developers
- id: una-agent.usage.developers
- status: active
- type: context
<!-- content -->
1. **Use `unified_query`** as the primary MCP tool for question answering
2. **Populate the KG** using `kg_add_link` to create entity→data relationships
3. **Force query type** with `force_type` parameter when you know the data source

### For AI Agents
- id: una-agent.usage.agents
- status: active
- type: protocol
<!-- content -->
When interacting with the Local Nexus via MCP:

1. **Start with `unified_query`** for most questions—it handles routing automatically
2. **Use `kg_list_entities`** to discover known entities before asking entity-specific questions  
3. **Use `describe_table`** before writing custom SQL to understand schema
4. **Call `get_system_status`** to verify component availability

## Version History
- id: una-agent.version-history
- status: active
- type: log
<!-- content -->

| Date | Version | Changes |
|:-----|:--------|:--------|
| 2026-02-02 | 1.0.0 | Initial UNA_AGENT.md documenting current architecture with KG Metadata Layer |

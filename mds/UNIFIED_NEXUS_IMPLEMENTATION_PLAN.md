# Unified Nexus Implementation Plan: RAG + Data Warehouse
- status: active
- type: implementation_plan
- id: unified-nexus-implementation
- last_checked: 2026-01-29
<!-- content -->

## Goal
- id: unified_nexus_implementation_plan_rag_data_warehouse.goal
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
Unify the existing **Data Warehouse** (DuckDB + CSV/Excel ingestion) with a **RAG architecture** to enable:
1. **Structured queries** → SQL over DuckDB tables  
2. **Unstructured queries** → Semantic search over documents  
3. **Hybrid queries** → Both combined  
4. **MCP protocols** → Agent tools for programmatic access

This plan incorporates proven patterns from [mcmp_chatbot](https://github.com/IgnacioOQ/mcmp_chatbot).

---

## Key Features
- id: unified_nexus_implementation_plan_rag_data_warehouse.key_features
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->

### 1. Smart Retrieval (Query Decomposition) ✅
- id: unified_nexus_implementation_plan_rag_data_warehouse.key_features.1_smart_retrieval_query_decomposition
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
The unified engine automatically breaks down complex multi-part questions into simpler sub-queries for more complete answers.

**How it works:**
- User asks: *"What are our top products and what do customers say about shipping?"*
- Engine decomposes into:
  1. *"What are our top products?"* (structured → SQL)
  2. *"What do customers say about shipping?"* (unstructured → vector search)
- Results from both sub-queries are combined and deduplicated
- LLM generates a unified response

**Implementation:**
```python
@functools.lru_cache(maxsize=128)
def decompose_query(self, user_question) -> tuple[str, ...]
```
- LRU cache prevents repeated LLM calls for identical questions
- Decomposition limited to 1-3 sub-queries to avoid over-fragmentation

### 2. Institutional Graph Layer 🔜
- id: unified_nexus_implementation_plan_rag_data_warehouse.key_features.2_institutional_graph_layer
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
A graph-based layer (`data/graph/`) to understand organizational structure and relationships that are better represented as graphs.

**Use cases:**
- Organizational hierarchy (reporting chains, teams)
- Customer relationships (accounts → contacts → interactions)
- Product dependencies (components → assemblies)
- Document references (policies → procedures → forms)

**Implementation:** `src/core/graph_store.py` (stub ready, full implementation in future phase)

### 3. Performance Optimization: Batch Vector Search ✅
- id: unified_nexus_implementation_plan_rag_data_warehouse.key_features.3_performance_optimization_batch_vector_search
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
The retrieval engine uses **batch querying** to minimize latency. By sending all decomposed sub-queries to ChromaDB in a single parallel batch request, we achieved an **~82% reduction in retrieval time**.

| Approach | Latency | Notes |
|:---------|:--------|:------|
| Sequential queries | ~0.43s | One query at a time |
| Batch queries | ~0.07s | All queries in parallel |
| **Improvement** | **~82%** | |

**Implementation:**
```python

# Single batch request for all sub-queries
- id: single_batch_request_for_all_sub_queries
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
results = self.vector_store.query(
    query_texts=decomposed_queries,  # List of queries
    n_results=top_k
)
```

Combined with deduplication (`seen_ids = set()`), this ensures no duplicate context even when multiple sub-queries return overlapping documents.

---

## Vector Store Selection
- id: single_batch_request_for_all_sub_queries.vector_store_selection
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
**Selected: ChromaDB** (local, lightweight, Python-native)

> [!NOTE]
> **Alternatives considered** (for future reference):
> | Option | Pros | Cons | When to Use |
> |:-------|:-----|:-----|:------------|
> | **LanceDB** | Columnar, fast, embedded | Newer, less community | High-volume analytics |
> | **pgvector** | SQL integration, familiar | Requires PostgreSQL | When consolidating to Postgres |
> | **Pinecone** | Managed, scalable | Cloud-only, costs | Enterprise scale |
> | **Weaviate** | GraphQL, rich features | Heavier setup | Complex schema needs |

---

## Data Types & Handling Strategy
- id: single_batch_request_for_all_sub_queries.data_types_handling_strategy
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
| Data Type | Current | New Handler | Storage | Notes |
|:----------|:--------|:------------|:--------|:------|
| **CSV/Excel** | ✅ `ingestion.py` | Keep | DuckDB | Tabular queries |
| **JSON (tabular)** | ✅ `ingestion.py` | Keep | DuckDB | Tabular queries |
| **Plain Text (.txt)** | ❌ | `DocumentIngester` | ChromaDB | Semantic search |
| **Markdown (.md)** | ❌ | `DocumentIngester` | ChromaDB | Header-aware chunking |
| **PDF** | ❌ | `DocumentIngester` | ChromaDB | Requires `pypdf` |
| **DOCX** | ❌ | `DocumentIngester` | ChromaDB | Requires `python-docx` |
| **JSON (nested)** | ❌ | MCP Tools | Both | Hybrid access |

---

## Proposed Changes
- id: single_batch_request_for_all_sub_queries.proposed_changes
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->

### New Core Components
- id: single_batch_request_for_all_sub_queries.proposed_changes.new_core_components
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
---

#### [NEW] `src/core/vector_store.py`
- id: single_batch_request_for_all_sub_queries.proposed_changes.new_core_components.new_srccorevector_storepy
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
ChromaDB wrapper implementing mcmp_chatbot patterns:

- **Batch queries**: Accept list of query texts for parallel retrieval
- **Upsert support**: Update existing docs, insert new ones
- **Metadata filtering**: `where` parameter for structured filters
- **Deduplication**: Content-hash based IDs
- **Embedding**: `all-MiniLM-L6-v2` (local, free)

```python

# Key method signature
- id: key_method_signature
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
def query(self, query_texts, n_results=3, where=None) -> dict
```

---

#### [NEW] `src/core/query_router.py`
- id: key_method_signature.new_srccorequery_routerpy
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
Classifies queries into `STRUCTURED` / `UNSTRUCTURED` / `HYBRID`:

- Keyword heuristics (fast, no LLM cost)
- LLM fallback for ambiguous cases
- `QueryType` enum for type safety

---

#### [NEW] `src/core/text2sql.py`
- id: key_method_signature.new_srccoretext2sqlpy
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
Natural language → SQL for DuckDB:

- Schema introspection with sample data
- SQL generation via LLM
- Query validation before execution

---

#### [NEW] `src/core/unified_engine.py`
- id: key_method_signature.new_srccoreunified_enginepy
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
Main orchestrator (inspired by mcmp_chatbot `RAGEngine`):

- **Query decomposition** with LRU caching (Smart Retrieval)
- **Batch vector search** for 82% latency reduction
- **Multi-source retrieval**: VectorStore + DuckDB + Graph
- **Context assembly**: Format results for LLM
- **MCP integration** (optional, toggleable)

```python

# Key method signatures
- type: plan
<!-- content -->
@functools.lru_cache(maxsize=128)
def decompose_query(self, user_question) -> list[str]

def retrieve_with_decomposition(self, question, top_k=3) -> list[dict]

def generate_response(self, query, use_mcp_tools=False) -> str
```

---

#### [NEW] `src/core/document_ingestion.py`
- type: task
<!-- content -->
Document processing pipeline:

- **Chunking strategies**: size-based, header-based (for MD)
- **File readers**: TXT, MD, PDF, DOCX
- **Metadata extraction**: source, type, timestamps

---

#### [NEW] `src/core/graph_store.py` ✅
- type: task
<!-- content -->
Institutional graph for organizational relationships:

- **Node types**: person, team, department, document, product, customer
- **Relationship types**: reports_to, manages, belongs_to, owns, references
- **Traversal queries**: path finding, subgraph extraction
- **Context extraction**: augments retrieval with organizational knowledge
- **Persistence**: JSON files in `data/graph/`

```python

# Key method signatures
- id: key_method_signatures
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
def add_node(self, node: GraphNode) -> bool
def add_edge(self, edge: GraphEdge) -> bool
def traverse(self, start_id, relationship, direction, max_depth) -> GraphQueryResult
def get_context_for_query(self, entity_names: list[str]) -> str
```

---

<!-- MERGED FROM NEWER VERSION -->

@functools.lru_cache(maxsize=128)
def decompose_query(self, user_question) -> list[str]

def retrieve_with_decomposition(self, question, top_k=3) -> list[dict]

def generate_response(self, query, use_mcp_tools=False) -> str
```

---

#### [NEW] `src/mcp/` (Phase 5)
- id: key_method_signatures.new_srcmcp_phase_5
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
MCP Server for structured data tools:

- `search_tables`: Query DuckDB metadata
- `get_schema`: Retrieve table schemas
- `execute_query`: Run validated SQL

---

### Modified Files
- id: key_method_signatures.modified_files
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
---

#### [MODIFY] `requirements.txt`
- id: key_method_signatures.modified_files.modify_requirementstxt
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
```diff
+ chromadb>=0.4.0
+ sentence-transformers>=2.0.0
+ rank-bm25>=0.2.0
+ pypdf>=3.0.0
+ python-docx>=0.8.0
```

---

#### [MODIFY] `src/app.py`
- id: key_method_signatures.modified_files.modify_srcapppy
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
- Add document uploader in sidebar
- Display query type badges in chat
- Show system stats (tables, documents)
- MCP toggle (optional)

---

## Implementation Phases
- id: key_method_signatures.implementation_phases
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->

### Phase 1: Vector Store Foundation (2h) ✅
- id: key_method_signatures.implementation_phases.phase_1_vector_store_foundation_2h
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
- [x] Create `src/core/vector_store.py`
- [x] Create `src/core/document_ingestion.py`  
- [x] Update `requirements.txt`
- [x] Test: ingest sample docs, run queries

### Phase 2: Query Routing (1h) ✅
- id: key_method_signatures.implementation_phases.phase_2_query_routing_1h
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
- [x] Create `src/core/query_router.py`
- [x] Test: classify diverse query types

### Phase 3: Text2SQL (2h) ✅
- id: key_method_signatures.implementation_phases.phase_3_text2sql_2h
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
- [x] Create `src/core/text2sql.py`
- [x] Test: generate SQL from natural language

### Phase 4: Unified Engine (3h) ✅
- id: key_method_signatures.implementation_phases.phase_4_unified_engine_3h
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
- [x] Create `src/core/unified_engine.py`
- [x] Implement query decomposition with caching
- [x] Integrate all retrieval paths
- [x] Test: end-to-end queries

### Phase 5: UI + MCP (2h) ✅
- id: key_method_signatures.implementation_phases.phase_5_ui_mcp_2h
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
- [x] Modify `src/app.py` for document upload
- [x] Create `src/mcp/` directory and server (12 tools)
- [x] Update `src/components/sidebar.py` with Tables/Documents/Graph tabs
- [x] Update `src/components/chat.py` with Unified Engine integration
- [x] Test: full workflow in Streamlit

---

## Verification Plan
- id: key_method_signatures.verification_plan
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->

### Automated Tests
- id: key_method_signatures.verification_plan.automated_tests
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
```bash

#### [NEW] `src/core/document_ingestion.py`
- id: key_method_signatures.new_srccoredocument_ingestionpy
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
Document processing pipeline:

- **Chunking strategies**: size-based, header-based (for MD)
- **File readers**: TXT, MD, PDF, DOCX
- **Metadata extraction**: source, type, timestamps

---

#### [NEW] `src/core/graph_store.py` ✅
- id: key_method_signatures.new_srccoregraph_storepy
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
Institutional graph for organizational relationships:

- **Node types**: person, team, department, document, product, customer
- **Relationship types**: reports_to, manages, belongs_to, owns, references
- **Traversal queries**: path finding, subgraph extraction
- **Context extraction**: augments retrieval with organizational knowledge
- **Persistence**: JSON files in `data/graph/`

```python

#### [NEW] `src/core/document_ingestion.py`
- id: key_method_signatures.new_srccoredocument_ingestionpy
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
Document processing pipeline:

- **Chunking strategies**: size-based, header-based (for MD)
- **File readers**: TXT, MD, PDF, DOCX
- **Metadata extraction**: source, type, timestamps

---

#### [NEW] `src/core/graph_store.py` ✅
- id: key_method_signatures.new_srccoregraph_storepy
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
Institutional graph for organizational relationships:

- **Node types**: person, team, department, document, product, customer
- **Relationship types**: reports_to, manages, belongs_to, owns, references
- **Traversal queries**: path finding, subgraph extraction
- **Context extraction**: augments retrieval with organizational knowledge
- **Persistence**: JSON files in `data/graph/`

```python

# After each phase
- id: after_each_phase
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
pytest tests/ -v
```

| Test File | Coverage |
|:----------|:---------|
| `test_vector_store.py` | Add/query/dedupe operations |
| `test_query_router.py` | Classification accuracy |
| `test_document_ingestion.py` | Chunking, file reading |
| `test_unified_engine.py` | End-to-end retrieval |

### Manual Verification
- id: after_each_phase.manual_verification
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
1. **Streamlit smoke test**: `streamlit run src/app.py`
2. **Structured query**: "How many rows in sales_data?"
3. **Unstructured query**: "What does the policy say about X?"
4. **Hybrid query**: "Which customers mentioned pricing and spent >$1K?"

---

## Key Patterns from mcmp_chatbot
- id: after_each_phase.key_patterns_from_mcmp_chatbot
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
| Pattern | Benefit | Implementation |
|:--------|:--------|:---------------|
| **Batch queries** | 82% latency reduction | `vs.query(query_texts=[...])` |
| **Query decomposition** | Better recall for complex Q | `@lru_cache` decorated method |
| **Deduplication** | No duplicate context | `seen_ids = set()` |
| **Metadata filtering** | Precise structured access | `where={"type": "event"}` |
| **MCP toggle** | Control latency/cost | Sidebar checkbox |

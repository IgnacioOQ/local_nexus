- status: active
- type: context
- context_dependencies: { "conventions": "MD_CONVENTIONS.md", "setup": "mds/PROJECT_SETUP.md", "master_plan": "mds/MASTER_PLAN.md", "agents": "AGENTS.md" }
<!-- content -->

**Local Nexus** is a privacy-first **Unified Intelligence Platform** that provides a **Streamlit Chatbot Interface** to combine a **Data Warehouse** (DuckDB) with **RAG** (Retrieval-Augmented Generation) and a **Knowledge Graph Metadata Layer**, answering complex questions that require both computation and semantic understanding.

> **Architecture**: Built on the **TAG (Table-Augmented Generation)** paradigm from Berkeley/Databricks research, unifying structured SQL queries with semantic document search, enhanced by KG-guided query expansion.

## Unified Nexus Architecture
- status: active
- type: context
<!-- content -->

```
                          ┌─────────────────────────────────────┐
                          │           User Question             │
                          └──────────────┬──────────────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────────────┐
                          │         Query Router (LLM)          │
                          │  Classifies: structured/unstructured│
                          │              /hybrid                │
                          └──────────────┬──────────────────────┘
                                         │
                                         ▼
                   ┌──────────────────────────────────────────┐
                   │   ★ Knowledge Graph Metadata Layer ★     │
                   │                                          │
                   │  • Expands queries with related sources   │
                   │  • Links entities → tables/documents      │
                   │  • Upgrades query type when needed        │
                   └────────────────────┬─────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
              ▼                                               ▼
    ┌─────────────────┐                         ┌─────────────────┐
    │   Vector Store  │                         │    DuckDB       │
    │   (ChromaDB)    │                         │  (Text2SQL)     │
    └────────┬────────┘                         └────────┬────────┘
             │                                           │
             └────────────────────┬─────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────────────────────────┐
                          │     Context Assembly & Generation   │
                          └─────────────────────────────────────┘
```

## Key Features
- status: active
- type: context
<!-- content -->

| Feature | Description |
|:--------|:------------|
| **Chatbot Interface** | Streamlit UI for natural language interaction and file management |
| **Query Routing** | Automatically classifies questions as structured (SQL), unstructured (RAG), or hybrid |
| **Knowledge Graph** | Entity-data links for query expansion and guided retrieval |
| **Text2SQL** | Converts natural language to DuckDB SQL with schema introspection |
| **Smart Retrieval** | Query decomposition for complex multi-part questions with LRU caching |
| **Batch Vector Search** | 82% latency reduction via parallel ChromaDB queries |
| **MCP Server** | 10+ tools for programmatic agent access |

## Core Capabilities
- status: active
- type: context
<!-- content -->
*   **Interactive Chatbot**: A **Streamlit**-based interface (`src/app.py`) for chatting with your data, visualizing SQL results, and managing file uploads.
*   **Data Warehouse (DuckDB)**: Ingests CSV/Excel/JSON into tables for SQL aggregations, joins, and filters.
*   **Document Search (ChromaDB)**: Semantic search over TXT, MD, PDF, DOCX files for policies and concepts.
*   **Knowledge Graph Metadata**: Links entities to tables/documents for query expansion and guided retrieval.
*   **Unified Engine**: Routes queries to appropriate source(s), expands via KG, and synthesizes answers.
*   **MCP Integration**: Expose all capabilities as agent tools via Model Context Protocol.

## MCP Protocol (Agent Tools)
- status: active
- type: context
<!-- content -->

The **Model Context Protocol (MCP)** server exposes all Unified Nexus capabilities to LLM agents, enabling programmatic access to structured data, documents, and relationships.

### Available Tools

| Tool | Category | Purpose |
|:-----|:---------|:--------|
| `unified_query` | Query | Answer questions using automatic routing |
| `classify_query` | Query | Classify query type without execution |
| `execute_sql` | Structured | Execute SQL directly against DuckDB |
| `list_tables` | Structured | List available tables with row counts |
| `describe_table` | Structured | Get schema, types, and sample data |
| `semantic_search` | Unstructured | Search documents by meaning |
| `list_document_sources` | Unstructured | List ingested document sources |
| `kg_get_related` | Knowledge Graph | Get sources linked to an entity |
| `kg_find_path` | Knowledge Graph | Find connection path between nodes |
| `kg_list_entities` | Knowledge Graph | List all known entities |
| `kg_add_link` | Knowledge Graph | Create a relationship between nodes |
| `kg_search` | Knowledge Graph | Search graph by name/type |
| `ingest_document` | Ingestion | Add documents to vector store |
| `get_system_status` | System | Check component health |
| `clear_cache` | System | Clear query decomposition cache |

### Usage

```bash
# Run as MCP server (for LLM integration)
python -m src.mcp.server

# Or use tools directly in Python
from src.mcp.server import call_tool_sync
result = call_tool_sync("unified_query", question="What are total sales?")
```

## Data Ingestion Pipeline
- status: active
- type: context
<!-- content -->
This section describes how the Local Warehouse processes raw information.
*   **Supported Formats**: CSV (`.csv`), Excel (`.xls`, `.xlsx`), JSON (`.json`).
*   **Process Flow**:
    1.  **Deduplication**: Calculates a SHA-256 hash of the file content. If the hash exists, upload is skipped.
    2.  **Normalization**: Uses **Pandas** to infer data types (Integers, Floats, Timestamps) and create a structured DataFrame.
    3.  **Sanitization**: Column names are converted to SQL-friendly `snake_case` (e.g., "Order Date" -> `order_date`).
    4.  **Persistence**: The DataFrame is serialized directly into a native **DuckDB** table. This is an automatic relational transformation.

## Deployment Modes
- status: active
- type: context
<!-- content -->

### 1. Local Mode (Privacy-First)
This is the intended production environment for Phase 1.
*   **Host**: Your local machine (localhost).
*   **Storage**: Your local hard drive (`data/warehouse.db`).
*   **Privacy**: Data **never** leaves your computer. The "Cloud" components (Phase 3) will only receive anonymized or specific queries you explicitly approve.

### 2. Cloud Demo (Public)
This is for demonstration and UI testing only.
*   **Host**: Streamlit Community Cloud.
*   **Storage**: Ephemeral. Any file you upload is deleted when the app reboots (which happens frequently).
*   **Privacy**: Not suitable for sensitive client data in this configuration.

## Directory Structure
- status: active
- type: context
<!-- content -->
```
local_nexus/
├── src/
│   ├── app.py                     # Main Streamlit Entry Point
│   ├── core/
│   │   ├── unified_engine.py      # Query routing + context assembly
│   │   ├── query_router.py        # STRUCTURED/UNSTRUCTURED/HYBRID classification
│   │   ├── text2sql.py            # Natural language → DuckDB SQL
│   │   ├── vector_store.py        # ChromaDB semantic search
│   │   ├── kg_metadata.py         # Knowledge Graph metadata layer
│   │   ├── document_ingestion.py  # TXT/MD/PDF/DOCX processing
│   │   ├── database.py            # DuckDB connection management
│   │   └── ingestion.py           # CSV/Excel/JSON → DuckDB
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── server.py              # MCP Server (15 agent tools)
│   ├── components/                # UI Components (Sidebar, Chat)
│   └── utils/                     # Utilities (Logging)
├── data/                          # Local Storage (Gitignored)
│   ├── warehouse.db               # DuckDB database
│   ├── vectordb/                  # ChromaDB persistence
│   └── kg/                        # Knowledge Graph JSON files
├── mds/                           # Project Documentation & Plans
├── tests/                         # Unit Tests (114 tests)
├── MD_CONVENTIONS.md              # Schema Specifications
└── AGENTS.md                      # Agent Instructions
```

## Getting Started
- status: active
- type: guideline
<!-- content -->
For detailed setup instructions, please refer to **[mds/PROJECT_SETUP.md](mds/PROJECT_SETUP.md)**.

### Quick Start
```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Application
streamlit run src/app.py
```

## Documentation Protocol
- status: active
- type: guideline
<!-- content -->
This project uses the **Markdown-JSON Hybrid Schema**.
*   **[MD_CONVENTIONS.md](MD_CONVENTIONS.md)**: The schema specification.
*   **[AGENTS.md](AGENTS.md)**: Operational guidelines for AI Agents.
*   **[mds/PROJECT_SETUP.md](mds/PROJECT_SETUP.md)**: Setup and installation guide.

## Known Issues
- status: active
- type: log
<!-- content -->
> [!IMPORTANT]
> **Initialization Blocking**: Internal component initialization (e.g., VectorStore, Engine) can block the entire chat flow if exceptions occur. The system has been patched to retry initialization, but check the console logs if the chat remains unresponsive.

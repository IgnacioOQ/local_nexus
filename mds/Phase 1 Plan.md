# Phase 1 Implementation: Local Nexus
- status: active
- type: plan
- id: implementation.phase1
- owner: user
- priority: critical
- context_dependencies: { "master_plan": "MASTER_PLAN.md", "conventions": "../MD_CONVENTIONS.md" }
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
This document outlines the tactical execution plan for building the **Local Nexus**, the client-side application of the Intelligent Control SaaS.

**Objective**: Create a self-contained local application that functions as:
1. **Data Warehouse**: Ingests and structures raw CSV/Excel files locally (DuckDB).
2. **Interface**: A chat-based UI for querying that data (Streamlit).

**Tech Stack Selection**:
* **Language**: Python 3.10+
* **Frontend**: Streamlit (Chosen for rapid iteration and native data support).
* **Database**: DuckDB (Embedded OLAP, zero-dependency, SQL compatible).
* **Data Processing**: Pandas / Polars.
* **Agent Framework**: **Google ADK (Local Mode)**. Use the ADK pattern (Python functions + type hints) for "Mock tools" to ensure seamless transition to Phase 3.

## Project Initialization & Structure
- status: done
- type: task
- id: implementation.phase1.init
- estimate: 1d
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Establish the repository structure to support modular growth into Phases 2 and 3.

**Directory Structure**:
```
/src
  /app.py            # Main Streamlit entry point
  /core
    /database.py     # DuckDB singleton wrapper
    /ingestion.py    # File processing logic
  /components
    /chat.py         # UI component for chat history
    /sidebar.py      # UI component for file management
  /utils
    /logger.py       # Telemetry logging (for future RL)
/tests               # Unit and integration tests
/data
  /raw               # Staging area for user uploads
  /warehouse.db      # Persistent DuckDB file
```

## Module 1: The Local Data Warehouse
- status: todo
- type: task
- id: implementation.phase1.warehouse
- blocked_by: [implementation.phase1.init]
- estimate: 1w
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Implement the persistence layer using DuckDB. This is the "Long-Term Memory" of the system.

### Database Manager Class
- status: todo
- type: task
- id: implementation.phase1.warehouse.manager
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Create a DatabaseManager class in src/core/database.py.

* **Connection**: Maintain a persistent connection to data/warehouse.db.
* **Schema Definition**:
    * **`metadata_registry`**:
        * `file_id` (UUID, PK)
        * `filename` (VARCHAR)
        * `upload_timestamp` (TIMESTAMP)
        * `file_hash` (VARCHAR)
        * `row_count` (INTEGER)
    * **`telemetry_log`**:
        * `log_id` (UUID, PK)
        * `user_id` (VARCHAR, nullable) -- Prepare for Phase 2 Auth
        * `timestamp` (TIMESTAMP)
        * `query_text` (VARCHAR)
        * `response_type` (VARCHAR)
        * `user_feedback` (INTEGER, nullable)
        * `synced_at` (TIMESTAMP, nullable) -- For Phase 2 Sync Logic
* **Methods**: `get_connection()`, `execute_query()`, `get_table_schema()`.

### Ingestion Service
- status: todo
- type: task
- id: implementation.phase1.warehouse.ingest
- blocked_by: [implementation.phase1.warehouse.manager]
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Create logic to handle user file uploads.

1. **Normalization**: Convert incoming Excel/CSV to a strict Pandas DataFrame.
2. **Type Enforcement**: Cast generic `object` columns to specific types (String, Int, Float, Bool, Timestamp) to ensure future BigQuery compatibility.
3. **Sanitization**: Clean column names (lowercase, snake_case) to make them SQL-friendly for the LLM later.
4. **Storage**: Use `duckdb.sql("CREATE TABLE ... AS SELECT ...")` to persist data.
5. **Versioning**: Calculate a content hash (SHA-256) of the file. If the hash exists, skip; if the filename exists but hash differs, create a new version (e.g., `sales_data_v2`).

## Module 2: The Chat Interface
- status: todo
- type: task
- id: implementation.phase1.ui
- blocked_by: [implementation.phase1.warehouse]
- estimate: 1w
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Implement the Streamlit frontend.

### Layout & Session State
- status: todo
- type: task
- id: implementation.phase1.ui.layout
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
* **Sidebar**: "Data Management". A file uploader widget and a list of currently available tables in DuckDB.
* **Main Area**: Chat container.
* **State Management**: Initialize `st.session_state` to hold:
  * `messages`: List of `{'role': 'user'|'assistant', 'content': str, 'data_ref': ...}`.
  * `active_tables`: List of tables currently in context.
  * `user_identity`: Dict `{'id': 'local-dev', 'role': 'admin'}` (Mock for Phase 2 Auth).

### Chat Logic & Mock Orchestrator
- status: todo
- type: task
- id: implementation.phase1.ui.chat
- blocked_by: [implementation.phase1.ui.layout]
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Since the Cloud Agents (Phase 3) are not ready, build a **Local Loopback** for testing.

1. **Input**: User types "Show me the last 5 rows of sales".
2. **Mock Processor (ADK Pattern)**:
   * Instead of regex, define a class `LocalAnalyst` with methods like `get_sales_data(limit: int)`.
   * Use **Pydantic** inputs to mirror ADK tool arguments.
   * *Goal*: Swap this Mock for a real `adk.Agent` in Phase 3 without frontend rewrites.
3. **Rendering**:
   * If response is Text: `st.markdown()`.
   * If response is Data: `st.dataframe()` or `st.bar_chart()`.

## module 3: Testing & Quality Assurance
- status: todo
- type: task
- id: implementation.phase1.testing
- blocked_by: [implementation.phase1.ui]
- estimate: 3d
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Establish comprehensive testing to ensure reliability before packaging.

### Unit Tests
- status: todo
- type: task
- id: implementation.phase1.testing.unit
- <!-- content -->
- **Framework**: `pytest`
- **Coverage**:
    - `ingestion.py`: Verify file hash assertions and schema normalization.
    - `database.py`: Test connection persistence and query execution with mock data.

### Integration Tests
- status: todo
- type: task
- id: implementation.phase1.testing.integration
- blocked_by: [implementation.phase1.testing.unit]
<!-- content -->
- **Flow**: Simulate a full user flow: User uploads CSV -> Ingestion -> stored in DB -> Query retrieves it.

## Module 4: Packaging & Distribution
- status: todo
- type: task
- id: implementation.phase1.packaging
- blocked_by: [implementation.phase1.testing]
- estimate: 2d
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Prepare the application for easy local deployment.

### Dependency Management
- status: todo
- type: task
- id: implementation.phase1.packaging.deps
<!-- content -->
- Create `requirements.txt` with locked versions.
- Create `environment.yml` for Conda users.

### Execution Scripts
- status: todo
- type: task
- id: implementation.phase1.packaging.scripts
<!-- content -->
- Create `run_app.sh` (Mac/Linux) and `run_app.bat` (Windows) to set up the environment and launch `streamlit run src/app.py`.

## Research Instrumentation (Pre-RL)
- status: todo
- type: task
- id: implementation.phase1.research
- priority: high
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
To prepare for the RL Agents in Phase 3, we must treat this phase as the "Data Collection" period.

* **Interaction Logging**: Every user query and subsequent system output must be logged to a JSONL file or the `telemetry_log` table.
* **Format**:
  ```json
  {
    "timestamp": "ISO8601",
    "state_snapshot": ["list_of_active_tables", "row_counts"],
    "action_user_query": "raw_text_input",
    "system_response_type": "table_render",
    "user_feedback": null
  }
  ```
* **Why**: This dataset will be used to offline-train the Orchestrator to classify intent (Analysis vs. Control) before we deploy the live model.
# Master Plan: Intelligent Control SaaS
- status: active
- type: plan
- id: master_plan.saas
- owner: product-manager
- priority: critical
- context_dependencies: { "manager": "../manager/MANAGER_AGENT.md", "conventions": "../MD_CONVENTIONS.md" }
- last_checked: 2026-01-23T15:14:25+01:00
<!-- content -->
This document serves as the central strategic plan for the **Intelligent Control & Analysis Platform**. It is a dual-engine AI system functioning as both a **Business Analyst** and an **Autonomous Operator** for SMBs and industrial clients. It combines LLM reasoning (Analysis) with RL control (Optimization).

## Executive Summary
- status: done
- type: context
- id: product.saas.summary
- last_checked: 2026-01-23T13:47:07+01:00
<!-- content -->
**The Vision**: To empower every SMB with a Fortune 500-grade Data Science & Operations team—instantly. We bridge the gap between raw data and profitable action by combining natural language intuition with rigorous mathematical implementation.

**The Problem**: Small and mid-sized businesses (SMBs) across Retail and Logistics are drowning in spreadsheets. They face complex inventory dilemmas and operational inefficiencies but cannot afford a dedicated Data Science team. They operate on "gut feeling" rather than optimal control.

**The Solution**: An **Intelligent Control SaaS** that functions as a dual-engine AI partner:
1.  **The "Virtual Chief Analyst" (Understanding)**: A Code Interpreter agent that turns "Why are sales down?" into instant, visualized, statistical truth—not just text summaries.
2.  **The "Autonomous Operator" (Action)**: An RL-driven Controller that turns "Optimize my stock" into precise, executed actions, solving complex math (like the Newsvendor problem) to maximize cash flow and minimize waste.

**Market Value**:
*   **Democratization**: Access to advanced analytics (Time-series forecasting, Hypothesis testing) via a simple chat interface.
*   **Operational Excellence**: Transitioning clients from *reactive* fire-fighting to *proactive*, mathematically optimal inventory management.
*   **Hybrid Architecture**: A privacy-conscious Local Nexus for daily work, connected to a scalable Cloud Brain for heavy computation.

## Technical Architecture
- status: active
- type: plan
- id: product.saas.arch
- last_checked: 2026-01-23T13:47:07+01:00
<!-- content -->
The system separates Analytical Queries (Code Execution) from Control Tasks (Model Inference).

### Core Components
- status: active
- type: context
- id: product.saas.arch.components
- last_checked: 2026-01-23T13:47:07+01:00
<!-- content -->
#### Chatbot Assistant App
- status: active
- type: context
- id: product.saas.arch.components.chatbot
- last_checked: 2026-01-23T19:47:31+01:00
<!-- content -->
Serves as the primary interface for users, functioning simultaneously as a mechanism for interaction and a local data warehouse. It facilitates data collection and user intent capture.

#### Internal Ecosystem of AI-Assistants
- status: active
- type: context
- id: product.saas.arch.components.ecosystem
- last_checked: 2026-01-23T19:47:31+01:00
<!-- content -->
A background orchestration layer where multiple specialized AI agents collaborate. These agents are internal-only and handle specific sub-tasks to ensure seamless system operation.

#### Cloud Infrastructure (BigQuery & Compute)
- status: active
- type: context
- id: product.saas.arch.components.cloud
- last_checked: 2026-01-23T19:47:31+01:00
<!-- content -->
The scalable backbone of the platform. It includes **Google BigQuery** for massive data warehousing and **Google Cloud Compute** for performant processing, ensuring reliability and speed.

#### Internal Algorithms Repository
- status: active
- type: context
- id: product.saas.arch.components.algorithms
- last_checked: 2026-01-23T19:47:31+01:00
<!-- content -->
The central library of data processing and control algorithms. This leverages **Vertex AI** for advanced data science modeling and optimization tasks, representing the core intellectual property of the analysis engine.

### Information Flow
- status: active
- type: context
- id: product.saas.arch.flow
- last_checked: 2026-01-23T13:47:07+01:00
<!-- content -->

#### AI Assistant Orchestration
- status: active
- type: context
- id: product.saas.arch.flow.orchestration
- last_checked: 2026-01-23T15:28:59+01:00
<!-- content -->
User interaction begins with the Chatbot App, which forwards requests to the Orchestrator (Vertex AI).
1.  **Intent Recognition**: The Orchestrator determines if the request is **Analysis** (informational) or **Control** (actionable).
2.  **Routing**:
    *   **Analysis**: Routed to Code Interpreter / Analyst Agent for data querying and visualization.
    *   **Control**: Routed to Planner / RL Agent for optimization and decision making.
3.  **Response**: Results are aggregated and returned to the Chatbot as natural language or UI components.

#### Control Loop
- status: active
- type: context
- id: product.saas.arch.flow.control
- last_checked: 2026-01-23T15:28:59+01:00
<!-- content -->
This high-frequency loop handles the autonomous optimization system:
1.  **Telemetry Ingest**: Raw data streams from the Client App/Warehouse are ingested into BigQuery.
2.  **State Estimation**: Processing algorithms convert raw telemetry into state vectors ($s_t$) suitable for model input.
3.  **Decision**: The Policy network ($\pi$) or Planer selects the optimal action ($a_t$) based on the current state.
4.  **Execution & Feedback**: The action is sent to the Controller for execution, and the outcome is recorded for offline re-training and refinement.

#### Human-AI Interaction
- status: active
- type: context
- id: product.saas.arch.flow.human_ai
- last_checked: 2026-01-23T19:51:07+01:00
<!-- content -->
Defines the protocols for how humans interact with the AI agents.

##### Developer-AI Interaction
- status: active
- type: protocol
- id: product.saas.arch.flow.human_ai.developer
- last_checked: 2026-01-23T19:51:07+01:00
<!-- content -->
Protocol for developers to configure, train, and debug agents. Involves direct access to internal logs, model weights, and the 'Analysis Sandbox' for safe code testing.

##### Client-AI Interaction
- status: active
- type: protocol
- id: product.saas.arch.flow.human_ai.client
- last_checked: 2026-01-23T19:51:07+01:00
<!-- content -->
Protocol for end-users. Restricted to natural language via the Chatbot App. No direct code execution allowed. Intent is parsed by the Orchestrator.

#### AI-Tools Protocols
- status: active
- type: protocol
- id: product.saas.arch.flow.tools
- last_checked: 2026-01-23T19:51:07+01:00
<!-- content -->
Protocols for how AI agents utilize external software and APIs. Adheres to the **Model Context Protocol (MCP)** to standardize tool definition, discovery, and execution.


### Knowledge Bases
- status: active
- type: context
- id: product.saas.arch.knowledge
- last_checked: 2026-01-23T20:00:00+01:00
<!-- content -->
Repository resources categorized by their function.

#### Agentic
- status: active
- type: context
- id: product.saas.arch.knowledge.agentic
- last_checked: 2026-01-23T20:00:00+01:00
<!-- content -->
- [MANAGER_AGENT](MANAGER_AGENT.md)
- [CLEANER_AGENT](../cleaner/CLEANER_AGENT.md)
- [REACT_ASSISTANT](../../AI_AGENTS/specialists/REACT_ASSISTANT.md)
- [RECSYS_AGENT](../../AI_AGENTS/specialists/RECSYS_AGENT.md)
- [CONTROL_AGENT](../../AI_AGENTS/specialists/CONTROL_AGENT.md)
- [UI_DESIG_ASSISTANT](../../AI_AGENTS/specialists/UI_DESIG_ASSISTANT.md)
- [LINEARIZE_AGENT](../../AI_AGENTS/specialists/LINEARIZE_AGENT.md)
- [MC_AGENT](../../AI_AGENTS/specialists/MC_AGENT.md)

#### Knowledge
- status: active
- type: context
- id: product.saas.arch.knowledge.general
- last_checked: 2026-01-23T20:00:00+01:00
<!-- content -->
- [README](../../README.md)
- [MD_CONVENTIONS](../../MD_CONVENTIONS.md)
- [AGENTS](../../AGENTS.md)
- [AGENTS_LOG](../../AGENTS_LOG.md)
- [DAG_Example](../../language/example/DAG_Example.md)

## Implementation Roadmap
- status: active
- type: plan
- id: product.saas.roadmap
- last_checked: 2026-01-23T21:44:23+01:00
<!-- content -->
This roadmap strips away enterprise complexity to focus on the core value proposition: a local app that acts as a data hub and a chat interface, connected to powerful cloud agents for execution.

### Phase 1: Local Nexus
- status: active
- type: plan
- id: implementation.phase1
- owner: user
- priority: critical
- estimate: 4w
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
**Tech Stack Selection**:
* **Language**: Python 3.10+
* **Frontend**: Streamlit (Chosen for rapid iteration and native data support).
* **Database**: DuckDB (Embedded OLAP, zero-dependency, SQL compatible).
* **Data Processing**: Pandas / Polars.
* **Agent Framework**: **Google ADK (Local Mode)**. Even in Phase 1, we will structure "Mock tools" using the ADK pattern (Python functions + type hints) to make the transition to Phase 3 seamless.

#### Project Initialization & Structure
- status: todo
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

#### Module 1: The Local Data Warehouse
- status: todo
- type: task
- id: implementation.phase1.warehouse
- blocked_by: [implementation.phase1.init]
- estimate: 1w
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Implement the persistence layer using DuckDB. This is the "Long-Term Memory" of the system.

##### Database Manager Class
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

##### Ingestion Service
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

#### Module 2: The Chat Interface
- status: todo
- type: task
- id: implementation.phase1.ui
- blocked_by: [implementation.phase1.warehouse]
- estimate: 1w
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Implement the Streamlit frontend.

##### Layout & Session State
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

##### Chat Logic & Mock Orchestrator
- status: todo
- type: task
- id: implementation.phase1.ui.chat
- blocked_by: [implementation.phase1.ui.layout]
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Since the Cloud Agents (Phase 3) are not ready, build a **Local Loopback** for testing.

1. **Input**: User types "Show me the last 5 rows of sales".
2. **Mock Processor (ADK Pattern)**:
   * Instead of random regex, define a class `LocalAnalyst` with methods like `get_sales_data(limit: int)`.
   * Use **Pydantic** to define the input schema for these methods, mirroring how ADK handles tool arguments.
   * This allows us to "swap" this Mock Processor for a real `adk.Agent` in Phase 3 without rewriting the frontend.
3. **Rendering**:
   * If response is Text: `st.markdown()`.
   * If response is Data: `st.dataframe()` or `st.bar_chart()`.

#### Module 3: Testing & Quality Assurance
- status: todo
- type: task
- id: implementation.phase1.testing
- blocked_by: [implementation.phase1.ui]
- estimate: 3d
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Establish comprehensive testing to ensure reliability before packaging.

##### Unit Tests
- status: todo
- type: task
- id: implementation.phase1.testing.unit
- <!-- content -->
- **Framework**: `pytest`
- **Coverage**:
    - `ingestion.py`: Verify file hash assertions and schema normalization.
    - `database.py`: Test connection persistence and query execution with mock data.

##### Integration Tests
- status: todo
- type: task
- id: implementation.phase1.testing.integration
- blocked_by: [implementation.phase1.testing.unit]
- <!-- content -->
- **Flow**: Simulate a full user flow: User uploads CSV -> Ingestion -> stored in DB -> Query retrieves it.

#### Module 4: Packaging & Distribution
- status: todo
- type: task
- id: implementation.phase1.packaging
- blocked_by: [implementation.phase1.testing]
- estimate: 2d
- last_checked: 2026-01-24T08:35:00+01:00
<!-- content -->
Prepare the application for easy local deployment.

##### Dependency Management
- status: todo
- type: task
- id: implementation.phase1.packaging.deps
- <!-- content -->
- Create `requirements.txt` with locked versions.
- Create `environment.yml` for Conda users.

##### Execution Scripts
- status: todo
- type: task
- id: implementation.phase1.packaging.scripts
- <!-- content -->
- Create `run_app.sh` (Mac/Linux) and `run_app.bat` (Windows) to set up the environment and launch `streamlit run src/app.py`.

#### Research Instrumentation (Pre-RL)
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

### Phase 2: The Cloud Bridge
- status: active
- type: plan
- id: implementation.phase2
- owner: user
- priority: critical
- estimate: 2w
- last_checked: 2026-01-24T08:50:00+01:00
- blocked_by: [implementation.phase1]
<!-- content -->
This document details the "Cloud Bridge" implementation. The goal is to establish a secure, scalable communication channel between the Local Nexus (Phase 1) and the Cloud Agents (Phase 3) using the Google Cloud Ecosystem.

**Objective**:
1.  **Infrastructure**: Provision serverless compute and storage on GCP.
2.  **Connectivity**: Build a secure API Gateway for the local app to "phone home".
3.  **Synchronization**: Create pipelines to mirror local data to the cloud for heavy processing.

**Tech Stack**:
*   **Compute**: Google Cloud Run (Serverless Container).
*   **Database**: Google BigQuery (Warehousing) & Firestore (NoSQL Metadata).
*   **API**: Python FastAPI.
*   **Auth**: Firebase Authentication.
*   **Deployment**: Terraform / gcloud CLI (via Antigravity MCP).

#### Module 1: Infrastructure Initialization (GCP)
- status: todo
- type: task
- id: implementation.phase2.infra
- estimate: 3d
- last_checked: 2026-01-24T08:50:00+01:00
<!-- content -->
Provision the necessary Google Cloud resources. We will favor "Infrastructure as Code" practices.

##### Project Setup & API Enablement
- status: todo
- type: task
- id: implementation.phase2.infra.setup
- priority: high
<!-- content -->
*   **Action**: Create a new GCP Project (e.g., `intelligent-control-prod`).
*   **Enable APIs**:
    *   `run.googleapis.com` (Cloud Run)
    *   `artifactregistry.googleapis.com` (Docker Images)
    *   `bigquery.googleapis.com` (Data Warehouse)
    *   `firestore.googleapis.com` (App State)

##### IaC & Deployment Workflow
- status: todo
- type: task
- id: implementation.phase2.infra.iac
- blocked_by: [implementation.phase2.infra.setup]
<!-- content -->
Define the infrastructure using Terraform or scriptable `gcloud` commands.
*   **Workflow**:
    1.  User prompts Antigravity to "Deploy Infrastructure".
    2.  Antigravity uses the terminal tool (or `gcloud` MCP) to execute the provisioning scripts.
    3.  Outputs (Service URLs, Bucket Names) are saved to `deployment_config.json`.

#### Module 2: Authentication & Security
- status: todo
- type: task
- id: implementation.phase2.auth
- blocked_by: [implementation.phase2.infra]
- estimate: 1w
- last_checked: 2026-01-24T08:50:00+01:00
<!-- content -->
Secure the bridge. The Local App must authenticate before sending data.

##### Identity Management (Firebase)
- status: todo
- type: task
- id: implementation.phase2.auth.firebase
<!-- content -->
*   **Setup**: Initialize a Firebase project linked to the GCP project.
*   **Client**: Integrate `firebase-admin` in the Cloud API and the JS/Python SDK in the Local App.
*   **Flow**:
    1.  Local User logs in.
    2.  Local App gets JWT Token.
    3.  API Gateway verifies JWT Token on every request.

##### Service Security
- status: todo
- type: task
- id: implementation.phase2.auth.iam
<!-- content -->
*   **Service Accounts**: Create a specific Service Account for the Cloud Run instance.
*   **Permissions**: Grant strictly necessary roles (e.g., `roles/bigquery.dataEditor`, `roles/storage.objectCreator`). **Do not use Owner role.**

#### Module 3: The API Gateway (Connector)
- status: todo
- type: task
- id: implementation.phase2.api
- blocked_by: [implementation.phase2.auth]
- estimate: 1w
- last_checked: 2026-01-24T08:50:00+01:00
<!-- content -->
Develop and deploy the central REST API.

##### Service Skeleton (FastAPI)
- status: todo
- type: task
- id: implementation.phase2.api.dev
<!-- content -->
Create `src/cloud/main.py`.
*   **Endpoints**:
    *   `POST /v1/telemetry`: Accepts JSON payloads of user interactions.
    *   `POST /v1/agent/task`: Submits a complex task for the Cloud Agents.
    *   `GET /v1/agent/status/{task_id}`: Polling endpoint for long-running jobs.

##### Containerization & Deploy
- status: todo
- type: task
- id: implementation.phase2.api.deploy
- blocked_by: [implementation.phase2.api.dev]
<!-- content -->
*   **Docker**: Create `Dockerfile` optimized for Python (multi-stage build).
*   **CI/CD**: Define a simple deployment script: `gcloud run deploy --source .`.

#### Module 4: Data Synchronization Pipeline
- status: todo
- type: task
- id: implementation.phase2.pipeline
- blocked_by: [implementation.phase2.api]
- estimate: 1w
- last_checked: 2026-01-24T08:50:00+01:00
<!-- content -->
Mechanisms to move large datasets from Local DuckDB to Cloud BigQuery.

##### Blob Storage Ingress
- status: todo
- type: task
- id: implementation.phase2.pipeline.gcs
<!-- content -->
For raw files (CSV/Excel) that are too large for JSON payloads.
*   **Mechanism**: Local App requests a Signed Upload URL from the API.
*   **Action**: Local App PUTs the file directly to a GCS Bucket (`raw-data-ingress`).

##### Warehouse Sync (BigQuery)
- status: todo
- type: task
- id: implementation.phase2.pipeline.bigquery
- blocked_by: [implementation.phase2.pipeline.gcs]
<!-- content -->
*   **Schema Mapping**: Map DuckDB types to BigQuery types.
*   **Validation**: Check incoming schema against existing BigQuery schema to reject breaking changes (Schema Drift defense).
*   **Validation**: Check incoming schema against existing BigQuery schema to reject breaking changes (Schema Drift defense).
*   **Trigger**: When a file lands in GCS, a Cloud Event triggers a "Loader" function (or the API itself) to load the CSV into BigQuery.
*   **ADK Compatibility**: Ensure the BigQuery dataset labels and descriptions are verbose. ADK's `BigQueryTool` uses these schema descriptions to understand how to query the data.

### Phase 3: The Cloud Agents
- status: todo
- type: plan
- id: implementation.phase3
- owner: user
- priority: critical
- estimate: 6w
- blocked_by: [implementation.phase2]
<!-- content -->
This document details the implementation of the "Brain" of the Intelligent Control SaaS: a multi-agent system built using the **Google Agent Development Kit (ADK)**.

**Objective**: Deploy a robust, observable, and scalable agent ecosystem handling:
1.  **Analysis**: Python-based data science and visualization.
2.  **Control**: RL/Control-theory optimization using a custom algorithm repository.
3.  **Orchestration**: Intelligent routing and state management.

**Tech Stack**:
*   **Framework**: Google ADK (Python SDK).
*   **Model**: Gemini 1.5 Pro (via Vertex AI).
*   **Runtime**: Cloud Run (Containerized Agents).
*   **Evaluation**: Vertex AI Gen AI Evaluation Service.

#### Architecture: The ADK Ecosystem
- status: todo
- type: plan
- id: implementation.phase3.arch
- estimate: 1w
<!-- content -->
We will leverage ADK's pattern for composable agents. The system will consist of a top-level **Coordinator Agent** and two specialized worker agents.

##### The Coordinator Pattern
- status: todo
- type: protocol
- id: implementation.phase3.arch.coordinator
<!-- content -->
Instead of a monolithic chain, we use a central `LlmAgent` acting as a router.
*   **Input**: Natural language user queries + State Context (from Phase 2).
*   **Decision**: Uses a `classify_intent` tool or few-shot prompting to decide:
    *   `ANALYSIS_REQUIRED` -> Delegate to Analyst Agent.
    *   `CONTROL_REQUIRED` -> Delegate to Controller Agent.
    *   `AMBIGUOUS` -> Ask clarifying questions.
*   **Output**: Aggregates responses from workers and formats the final answer for the user.

#### Module 1: The Analyst Agent (Data Scientist)
- status: todo
- type: task
- id: implementation.phase3.analyst
- blocked_by: [implementation.phase3.arch]
- estimate: 2w
<!-- content -->
**Role**: "Why is this happening?"
**Tools**: Code Execution, Data Visualization.

##### Data Science Tool Repository
- status: todo
- type: task
- id: implementation.phase3.analyst.repo
<!-- content -->
We will build a dedicated Python library (`src/lib_analysis`) that the agent learns to use.
*   **Structure**:
    ```python
    /src/lib_analysis
       /visualize.py   # High-level plot wrappers (plot_time_series, plot_distribution)
       /stats.py       # Hypothesis testing (anova, t_test)
       /clean.py       # Auto-cleaning utilities
    ```
*   **Integration**:
    *   Expose these functions as **ADK Tools**.
    *   Use type hints and docstrings heavily, as ADK uses these for tool definition verification.

##### Code Execution Sandbox
- status: todo
- type: task
- id: implementation.phase3.analyst.sandbox
<!-- content -->
*   **Mechanism**: The agent writes code that imports `lib_analysis`.
*   **Security**: Use ADK's `CodeExecutionTool` configured with a restricted environment (or E2B integration if ADK native support is insufficient).
*   **Output Handling**: Capture `stdout` (text) and generated artifacts (PNG/JSON) to pass back to the Coordinator.

#### Module 2: The Controller Agent (Optimizer)
- status: todo
- type: task
- id: implementation.phase3.controller
- blocked_by: [implementation.phase3.analyst]
- estimate: 2w
<!-- content -->
**Role**: "Optimize for X."
**Tools**: Optimization Algorithms, Simulation.

##### Control Algorithms Integration
- status: todo
- type: task
- id: implementation.phase3.controller.integration
<!-- content -->
Integrate the external repository [control_algorithms](https://github.com/IgnacioOQ/control_algorithms).
*   **Step 1**: Submodule or Package integration of the user's repository.
*   **Step 2**: Create an **ADK Tool Wrapper** (`src/tools/control_tools.py`) that exposes key algorithms as callable functions:
    *   `run_mpc_optimization(state_vector, constraints)`
    *   `solve_newsvendor(demand_dist, costs)`
    *   `simulate_scenario(initial_state, horizon)`
*   **Step 3**: Define the "State Schema". The Agent must know how to map the raw telemetry (from BigQuery/Phase 2) into the inputs required by these algorithms.

#### Module 3: Agent Development & Ops (ADK)
- status: todo
- type: task
- id: implementation.phase3.ops
- blocked_by: [implementation.phase3.controller]
- estimate: 1w
<!-- content -->
Establish the lifecycle for developing and improving these agents.

##### Evaluation Pipeline (GenAI Eval)
- status: todo
- type: task
- id: implementation.phase3.ops.eval
<!-- content -->
Use Google's Gen AI Evaluation Service to move beyond "vibes-based" testing.
*   **Trajectory Evaluation**: check if the Analyst Agent *actually* used the `visualize.py` tool or if it tried to hallucinate a plot.
*   **Golden Datasets**: Create a set of (Query, Expected_Tool_Call, Expected_Outcome) tuples.
*   **CI/CD**: Run `adk eval` as part of the deployment pipeline.

##### Deployment (Vertex AI)
- status: todo
- type: task
- id: implementation.phase3.ops.deploy
<!-- content -->
*   **Containerize**: Wrap the ADK agent server in a Docker container.
*   **Deploy**: Push to Cloud Run.
*   **Expose**: Connect the Cloud Run endpoint to the API Gateway created in Phase 2.

## Commercial Strategy
- status: active
- type: plan
- id: product.saas.commercial
- last_checked: 2026-01-24T09:40:55+01:00
<!-- content -->
This section outlines the strategy for monetization, user acquisition, and market validation.

### Frontline Trials
- status: todo
- type: plan
- id: product.saas.commercial.frontline
- estimate: 4w
<!-- content -->
**Objective**: Validate the product value proposition with real users in a low-stakes environment.
*   **Approach**: "Do things that don't scale." Direct outreach to friendly SMBs (Retail/Logistics).
*   **Goal**: 5-10 active users providing weekly feedback.
*   **Monetization**: Free or heavily discounted in exchange for feedback/testimonials.
*   **Metrics**: Engagement (Daily Active Users), "Magic Moments" (e.g., "This saved me 2 hours").

### Payment Schema
- status: todo
- type: plan
- id: product.saas.commercial.payment
- blocked_by: [product.saas.commercial.frontline]
- estimate: 2w
<!-- content -->
**Objective**: Build the infrastructure to capture value.
*   **Tech**: Stripe / Lemon Squeezy integration.
*   **Models**:
    *   **Freemium**: Local-only features are free.
    *   **Pro ($29/mo)**: Cloud sync + basic Analyst Agent usage (token capped).
    *   **Enterprise (Custom)**: Full Controller Agent access + dedicated support.
*   **Deliverable**: A seamless "Upgrade" flow within the Streamlit app.

### Marketing & Growth
- status: todo
- type: plan
- id: product.saas.commercial.marketing
- blocked_by: [product.saas.commercial.payment]
<!-- content -->
**Objective**: Scale awareness and acquisition.
*   **Content Marketing**: Blog posts/Videos demonstrating "Data Science for Non-Data Scientists" using our app.
*   **Outreach**: Targeted LinkedIn outreach to Operations Managers in Logistics/Retail.
*   **Channels**:
    *   **Organic**: SEO, GitHub (Open Source core?).
    *   **Paid**: Targeted ads on niche industry forums (later stage).

## Legals & Admin
- status: todo
- type: plan
- id: legal
- last_checked: 2026-01-24T09:57:25+01:00
<!-- content -->
This section details the administrative and legal infrastructure, divided by jurisdiction.

### US Branch (Headquarters)
- status: active
- type: plan
- id: legal.us
- owner: user
<!-- content -->
**Role**: Global Revenue Collection, Cloud Services Contracting, Intellectual Property Holder.

#### Banking & Cloud Accounting
- status: todo
- type: task
- id: legal.us.banking
- estimate: 1w
<!-- content -->
**Objective**: Establish the financial hub.
*   **Banking**:
    *   **Mercury**: Recommended (Zero fees, high yield).
    *   **Backup**: Novo / Grasshopper.
    *   **Action**: Apply with EIN and Articles of Organization.
*   **Accounting**:
    *   **QuickBooks Online**: Connect to Mercury.
    *   **Revenue**: Stripe / App Store payouts land here.
    *   **Expenses**: Pay Google Cloud (GCP/Workspace), GitHub, and EOR/Contractor fees from this account.

### Argentina Branch (Talent Hub)
- status: todo
- type: plan
- id: legal.ar
- owner: user
<!-- content -->
**Role**: Talent Acquisition, Software Development Center.

#### Entity Setup: S.R.L. (Sociedad de Responsabilidad Limitada)
- status: todo
- type: task
- id: legal.ar.setup
- estimate: 4w
<!-- content -->
**Objective**: Establish a local entity to hire full-time employees without EOR markup.
*   **Structure**:
    *   **Partners**: Requires 2 partners. Options:
        *   **Option A (Corporate Link)**: You 95% + US LLC 5% (*Requires US LLC IGJ registration*).
        *   **Option B (Fast Route)**: You 95% + Trusted Individual 5% (*Avoids US LLC paperwork*).
    *   **Capital**: ~ARS 100,000 (Symbolic). 25% paid at signing.
    *   **Manager**: Must have domicile in Argentina. (You can serve as Manager using your Argentine DNI/Passport if you maintain a local address).
*   **Process**:
    1.  **Name Reservation**: Check availability with IGJ.
    2.  **Bylaws (Contrato Social)**: Drafted by a local Notary Public (*Escribano*).
    3.  **Registration**: File with IGJ (Inspección General de Justicia).
    4.  **Tax ID**: Obtain CUIT from AFIP.
*   **US LLC Requirement**: To be a partner, the US LLC must register with IGJ under "Article 123" (Simplified in 2024, no longer need to prove assets abroad).

#### Hiring & Payroll
- status: todo
- type: task
- id: legal.ar.hiring
- blocked_by: [legal.ar.setup]
<!-- content -->
**Objective**: Hire local developers legally.
*   **Payroll**:
    *   **Registration**: Register as Employer (*Alta de Empleador*) with AFIP.
    *   **Service**: Use a local accounting firm (Estudio Contable) to process monthly payslips (*Recibos de Sueldo*) and F931 (Social Security).
*   **Benefits**:
    *   **Mandatory**: 13th Salary (*Aguinaldo*), Vacation (14 days), Health Insurance (*Obra Social*).
    *   **Perks**: USD Split-payment (part of salary paid abroad) is common for retention, but requires careful tax structuring (*consult local CPA*).

## Security & Safety Checks
- status: active
- type: guideline
- id: product.saas.security
- last_checked: 2026-01-23T13:47:07+01:00
<!-- content -->
-   **Indirect Execution**: Clients only submit natural language, never code.
-   **Repository Scoping**: Generated code can only import whitelisted libraries (`pandas`, `numpy`, `lib_analysis`). No `os` or `sys`.
-   **Simulation Isolation**: User-provided logic runs in `gVisor` sandboxes.
-   **Action Bounding**: Deterministic logic layer validates actions against safety constraints (e.g., `MAX_ORDER_LIMIT`) before execution.

## Research Directions
- status: active
- type: plan
- id: product.saas.research
- last_checked: 2026-01-23T13:47:07+01:00
<!-- content -->
-   **MBRL (DreamerV3)**: Learning World Models from telemetry to simulate environments.
-   **Safe RL**: Constrained MDPs (Lagrangian Relaxation) to ensure safety during exploration.
-   **Reflexion**: Agents that analyze their own tracebacks to iteratively fix code.

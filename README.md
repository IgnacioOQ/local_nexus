- status: active
- type: context
- context_dependencies: { "conventions": "MD_CONVENTIONS.md", "setup": "mds/PROJECT_SETUP.md", "master_plan": "mds/MASTER_PLAN.md" }
<!-- content -->

**Local Nexus** is the client-side application for the Intelligent Control SaaS. It serves as a privacy-first Data Warehouse and an AI-powered interface for Small and Medium Businesses (SMBs) to analyze their operations.

> **Phase 1 Status**: This project is currently in **Phase 1 (Local Data Warehouse)**. It focuses on ingesting CSV/Excel files into a local DuckDB instance and providing a chat interface via Streamlit.

## Features
- status: active
- type: context
<!-- content -->
*   **Local Data Warehouse**: Ingests raw Excel/CSV files and structure them into a high-performance **DuckDB** database (`data/warehouse.db`).
*   **Chat Interface**: A **Streamlit**-based chat UI that allows users to query their data in natural language (simulation/mock in Phase 1).
*   **Zero-Copying**: Uses DuckDB's direct querying capabilities to minimize data duplication.
*   **Telemetery**: Logs interactions to train future Reinforcement Learning models (Phase 3).

## Directory Structure
- status: active
- type: context
<!-- content -->
```
local_nexus/
├── src/
│   ├── app.py                # Main Entry Point
│   ├── core/                 # Backend Logic (Database, Ingestion)
│   ├── components/           # UI Components (Sidebar, Chat)
│   └── utils/                # Utilities (Logging)
├── data/                     # Local Storage (Gitignored)
├── mds/                      # Project Documentation & Plans
├── tests/                    # Unit Tests
├── MD_CONVENTIONS.md         # Schema Specifications
└── AGENTS.md                 # Agent Instructions
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

# Project Setup Guide: Local Nexus (Phase 1)
- status: active
- type: guideline
- context_dependencies: { "plan": "Phase 1 Plan.md", "master_plan": "MASTER_PLAN.md", "conventions": "../MD_CONVENTIONS.md" }
<!-- content -->
> **Purpose:** Instructions for initializing and running the Local Nexus application (Phase 1), which serves as a local data warehouse and chat interface.

---

## 1. Prerequisites
- status: active
- type: task
<!-- content -->
- **Python**: Version 3.10 or higher.
- **OS**: macOS, Linux, or Windows.
- **Git**: Installed and configured.

---

## 2. Directory Structure
- status: active
- type: task
<!-- content -->
The project follows a modular structure to support future expansion into Phases 2 (Cloud) and 3 (Agents).

```
local_nexus/
├── src/
│   ├── app.py                # Main Streamlit entry point
│   ├── core/
│   │   ├── database.py       # DuckDB singleton wrapper & Schema definitions
│   │   └── ingestion.py      # File processing logic (Pandas -> DuckDB)
│   ├── components/
│   │   ├── chat.py           # UI component for chat interface
│   │   └── sidebar.py        # UI component for file management
│   └── utils/
│       └── logger.py         # Telemetry logging
├── data/
│   ├── raw/                  # Staging area for user uploads (gitignored)
│   └── warehouse.db          # Persistent DuckDB database (gitignored)
├── tests/                    # Unit and Integration tests
├── requirements.txt          # Python dependencies
├── AGENTS.md                 # Agent context
├── HOUSEKEEPING.md           # Maintenance protocols
└── README.md                 # Project Overview
```

---

## 3. Installation
- status: active
- type: task
<!-- content -->

### 1. Clone & Navigate
```bash
git clone <repository_url>
cd local_nexus
```

### 2. Virtual Environment
It is recommended to use a virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Running the Application
- status: active
- type: task
<!-- content -->

To start the Local Nexus interface:
```bash
streamlit run src/app.py
```

The application will open in your default browser (usually at `http://localhost:8501`).

---

## 5. Development & Testing
- status: active
- type: task
<!-- content -->

### Run Tests
```bash
pytest tests/ -v
```

### Linting
```bash
pylint src/
```

---

## 6. Key Components
- status: active
- type: context
<!-- content -->

*   **DuckDB**: Used as the embedded OLAP database. Data is persisted in `data/warehouse.db`.
*   **Streamlit**: Powers the frontend UI for rapid iteration.
*   **Pandas**: Handles data transformation and normalization before storage.

---

## 7. Troubleshooting
- status: active
- type: guideline
<!-- content -->

- **Database Locks**: If DuckDB throws a lock error, ensure only one process (Streamlit app or script) is accessing `warehouse.db` at a time.
- **Port Conflicts**: If port 8501 is in use, Streamlit will automatically try the next available port.

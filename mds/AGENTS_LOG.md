# Agents Log
- status: active
- type: log
- context_dependencies: { "conventions": "../MD_CONVENTIONS.md", "agents": "../AGENTS.md", "project_root": "../README.md"}
<!-- content -->
Most recent event comes first

## Intervention History
- status: active
<!-- content -->
### Project Initialization: Local Nexus
- status: done
<!-- content -->
**Date:** 2026-01-24
**AI Assistant:** Antigravity (Phase 1 Setup)
**Summary:** Initialized the Local Nexus project (Phase 1).
- **Goal:** Transform the generic repository into the Local Nexus application structure.
- **Implementation:**
    - Updated `PROJECT_SETUP.md` with Local Nexus specifics.
    - Rewrote `README.md` to define the project.
    - Created source code skeleton (`src/app.py`, `src/core`, `src/components`).
    - Updated `HOUSEKEEPING.md` and executed initial status check.
- **Files Modified:** `README.md`, `PROJECT_SETUP.md`, `Phase 1 Plan.md`, `HOUSEKEEPING.md`, `src/*`, `requirements.txt`.

### Feature: Remove Metadata Tool
- status: active
<!-- content -->
**Date:** 2026-01-22
**AI Assistant:** Antigravity
**Summary:** Created `remove_meta.py` to reverse `migrate.py` effects and clean incomplete content.
- **Goal:** Allow removing metadata from markdowns and strip incomplete sections/content.
- **Implementation:**
    - Created `language/remove_meta.py` with strict metadata detection logic.
    - Added flags `--remove-incomplete-content` and `--remove-incomplete-sections`.
    - Created symlink `bin/language/remove_meta` -> `../../util/sh2py3.sh`.
- **Files Modified:** `language/remove_meta.py` [NEW], `bin/language/remove_meta` [NEW].

### Feature: CLI Improvements
- status: active
<!-- content -->
**Date:** 2026-01-22
**AI Assistant:** Antigravity
**Summary:** Improved Python CLIs in `manager` and `language` to be POSIX-friendly and support flexible I/O modes.
- **Goal:** Standardize CLI usage and support single/multi-file processing with checks.
- **Implementation:**
    - Created `language/cli_utils.py` for shared arg parsing.
    - Updated `migrate.py`, `importer.py` to support `-I` (in-line) and repeated `-i/-o`.
    - Updated `md_parser.py`, `visualization.py` to support file output.
    - Added `-h` to all tools.
- **Files Modified:** `language/*.py`, `manager/*.py`.

### Feature: Shell Wrapper for Python Scripts
- status: active
<!-- content -->
**Date:** 2026-01-22
**AI Assistant:** Antigravity
**Summary:** Created a generic shell wrapper `sh2py3.sh` and symlinks for python scripts.
- **Goal:** Allow execution of python scripts in `manager/` and `language/` from a central `bin/` directory.
- **Implementation:**
    - Created `util/sh2py3.sh` to determine script path from symlink invocation and execute with python/python3.
    - Created `bin/manager` and `bin/language` directories.
    - Created symlinks in `bin/` mapping to `util/sh2py3.sh` for all `.py` files in `manager/` and `language/`.
- **Files Modified:** `util/sh2py3.sh` [NEW], `bin/` directories [NEW].

### Fix: Chat Initialization & Internal Blocking
- status: active
<!-- content -->
**Date:** 2026-01-30
**AI Assistant:** Antigravity
**Summary:** Resolved a critical issue where the Chatbot became unresponsive due to cached initialization failures.
- **Goal:** Restore chatbot functionality and prevent silent failures.
- **Issue:** `UnifiedEngine` initialization failure (e.g., due to TypeErrors) led to `st.session_state.unified_engine` being set to `None`. Subsequent runs saw the key existed and assumed initialization was complete, blocking further attempts.
- **Fix:** Updated `src/components/chat.py` to retry initialization if the engine is `None`.
- **Architectural Insight:** Internal operations (like engine init) can block the entire chat flow if exceptions are swallowed or caching is too aggressive.
- **Files Modified:** `src/components/chat.py`, `src/core/document_ingestion.py` (fixed signature mismatch).

### Fix: Unified Engine Crash & Hang Diagnosis
- status: active
<!-- content -->
**Date:** 2026-01-30
**AI Assistant:** Antigravity (Crash Diagnosis)
**Summary:** Diagnosed and resolved multiple cascade failures causing the "Unified Engine" to crash or hang.
- **Issue 1: Streamlit Hard Crash on Toggle**
    - **Symptoms:** App completely crashed/exited when toggling "Unified Engine".
    - **Diagnosis:** Serialization Failure. `UnifiedEngine` contained a `DuckDB` connection object. When toggling state, Streamlit attempts to pickle `st.session_state` for the rerun. DuckDB connections are **not picklable**, causing a hard crash.
    - **Fix:** Refactored `src/components/chat.py` to move heavy, non-picklable resources (DB, VectorStore, Graph) to `st.cache_resource`. This keeps them out of the serializable session state.

- **Issue 2: Chatbot "Hang" / Unresponsiveness**
    - **Symptoms:** Chatbot showed "Thinking..." indefinitely or returned empty responses when Unified Engine was active.
    - **Diagnosis:** API Rate Limiting. The complex RAG pipeline (`QueryRouter` -> `Decomposer` -> `Text2SQL` -> `Generator`) passes through multiple LLM calls per user query. The "Free Tier" key was hitting the **5 RPM (Requests Per Minute)** limit immediately, causing 429 errors that were silently retried or swallowed.
    - **Fix:** User upgraded to a paid key. Confirmed via reproduction script `tests/reproduce_hang.py`.

- **Issue 3: Configuration Precedence**
    - **Symptoms:** Even with the new paid key, the app hit rate limits.
    - **Diagnosis:** Environment Variable Precedence. The local `.env` file (containing the old key) was being loaded by `python-dotenv` and overriding the `st.secrets` provided in Streamlit.
    - **Fix:** Updated `src/core/llm.py` to check `st.secrets` **before** checking `os.getenv`.

- **Issue 4: Model Deprecation**
    - **Symptoms:** `404 Error: models/gemini-flash-latest not found`.
    - **Diagnosis:** The hardcoded model string was invalid or deprecated.
    - **Fix:** Centralized model configuration in `src/core/llm.py` using constant `DEFAULT_MODEL`. Set to `gemini-2.0-flash-lite` (most cost-effective MCP model).

- **Feature: Granular Toggles**
    - **Implementation:** Split the single "Unified Engine" toggle into four component toggles (Vector, SQL, Graph, LLM) to allow isolating specific failure points in the future.
- **Files Modified:** `src/components/chat.py`, `src/core/llm.py`, `src/core/unified_engine.py`, `src/core/text2sql.py`, `src/mcp/server.py`.

# Housekeeping Protocol
- status: active
- type: recurring
- context_dependencies: { "conventions": "../MD_CONVENTIONS.md", "setup": "PROJECT_SETUP.md" }
<!-- content -->
1.  **Read AGENTS.md**: Ensure you align with the latest agent behaviors.
2.  **Verify Dependency Network**: Check import consistency in `src/`.
3.  **Run Tests**: Execute `pytest tests/` and report results.
4.  **Lint**: Run `pylint src/` (if available) or check for obvious syntax errors.
5.  **Report**: Overwrite the "Latest Report" section below with findings.
6.  **Log**: Add a summary entry to `AGENTS_LOG.md`.

# Current Project Housekeeping
- status: active
- type: recurring
<!-- content -->

## Dependency Network
- status: active
- type: context
<!-- content -->
*   **Frontend**: `src/app.py` -> (`src/components/chat.py`, `src/components/sidebar.py`)
*   **Backend**: 
    *   `src/components/sidebar.py` -> `src/core/ingestion.py`
    *   `src/core/ingestion.py` -> `src/core/database.py` (Persistence)
*   **Utils**: `src/utils/logger.py` (Imported largely everywhere)

## Latest Report
- status: active
- type: log
- last_checked: 2026-01-24T10:30:00+01:00
<!-- content -->
**Execution Date:** 2026-01-24

**Test Results:**
*   `tests/`: No tests found (Phase 1 Init).

**Summary:**
Project initialized. Directory structure verification passed. Application skeleton created.

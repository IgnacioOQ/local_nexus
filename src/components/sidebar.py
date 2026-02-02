import streamlit as st
import os
import tempfile
from src.core.ingestion import IngestionService


def render_sidebar():
    with st.sidebar:
        st.title("Local Nexus")

        # Tabs for different data types
        tab1, tab2 = st.tabs(["Tables", "Documents"])

        with tab1:
            render_structured_data_section()

        with tab2:
            render_document_section()

        st.divider()
        render_system_stats()


def render_structured_data_section():
    """Render structured data (CSV/Excel/JSON) upload and table list."""
    st.subheader("Structured Data")

    # File Uploader for tabular data
    uploaded_file = st.file_uploader(
        "Upload CSV, Excel, or JSON",
        type=['csv', 'xlsx', 'json'],
        key="structured_uploader"
    )

    if uploaded_file is not None:
        if st.button("Ingest Table", key="ingest_table_btn"):
            with st.spinner("Processing..."):
                service = IngestionService()
                success, message = service.process_file(uploaded_file)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    # Available Tables
    st.caption("Active Tables")

    db = IngestionService().db
    tables = db.get_active_tables()

    if not tables:
        st.info("No tables loaded.")
    else:
        for filename, rows, _ in tables:
            with st.expander(f"{filename}"):
                st.caption(f"Rows: {rows}")


def render_document_section():
    """Render document upload for RAG vector store."""
    st.subheader("Documents (RAG)")

    # Check if vector store is available
    try:
        from src.core.vector_store import VectorStore
        vector_store_available = True
    except ImportError:
        st.warning("ChromaDB not installed. Install with: `pip install chromadb`")
        return

    # File Uploader for documents
    uploaded_doc = st.file_uploader(
        "Upload TXT, MD, PDF, or DOCX",
        type=['txt', 'md', 'pdf', 'docx'],
        key="document_uploader"
    )

    if uploaded_doc is not None:
        source_name = st.text_input(
            "Source name (optional)",
            value=uploaded_doc.name,
            key="doc_source_name"
        )

        if st.button("Ingest Document", key="ingest_doc_btn"):
            with st.spinner("Ingesting document..."):
                try:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(uploaded_doc.name)[1]
                    ) as tmp:
                        tmp.write(uploaded_doc.getvalue())
                        tmp_path = tmp.name

                    # Ingest
                    from src.core.vector_store import VectorStore
                    from src.core.document_ingestion import DocumentIngester

                    vs = VectorStore(db_path="data/vectordb")
                    ingester = DocumentIngester(vs)
                    result = ingester.ingest_file(tmp_path, source_name=source_name)

                    # Clean up
                    os.unlink(tmp_path)

                    st.success(f"Ingested {result.get('chunks', 0)} chunks from {source_name}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Document stats
    st.caption("Document Store")
    try:
        from src.core.vector_store import VectorStore
        vs = VectorStore(db_path="data/vectordb")
        stats = vs.get_stats()
        st.metric("Documents", stats.get("count", 0))
    except Exception:
        st.info("No documents ingested yet.")



def render_system_stats():
    """Render system statistics."""
    st.caption("System Status")

    try:
        # Quick status indicators
        components = []

        # Database
        try:
            from src.core.database import DatabaseManager
            db = DatabaseManager()
            components.append("DuckDB: OK")
        except Exception:
            components.append("DuckDB: --")

        # Vector Store
        try:
            from src.core.vector_store import VectorStore
            components.append("ChromaDB: OK")
        except ImportError:
            components.append("ChromaDB: --")

        # LLM
        try:
            from src.core.llm import init_gemini
            if init_gemini():
                components.append("Gemini: OK")
            else:
                components.append("Gemini: --")
        except Exception:
            components.append("Gemini: --")

        st.caption(" | ".join(components))

    except Exception:
        pass

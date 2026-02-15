import streamlit as st
import os
from src.core.database import DatabaseManager


@st.cache_resource
def get_db_connection():
    """Cached database connection."""
    return DatabaseManager().get_connection()

@st.cache_resource
def get_vector_store():
    """Cached vector store."""
    from src.core.vector_store import VectorStore
    return VectorStore(db_path="data/vectordb")

@st.cache_resource
def get_llm_func():
    """Cached LLM function."""
    import google.generativeai as genai # Remove this
    from google import genai
    from src.core.llm import init_gemini, DEFAULT_MODEL
    import os
    
    if init_gemini():
        def gemini_call(prompt: str) -> str:
            api_key = os.getenv("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        return gemini_call
    return None

@st.cache_resource
def get_kg_metadata():
    """Cached Knowledge Graph Metadata."""
    from src.core.kg_metadata import KnowledgeGraphMetadata
    return KnowledgeGraphMetadata(storage_path="data/kg")

def get_engine_instance(use_vector: bool, use_db: bool, use_kg: bool, use_llm: bool):
    """
    Get Unified Engine instance with dynamic component selection.
    
    Note: We do NOT cache the engine object itself because its components change based on toggles.
    The heavy components (db_conn, vector_store, etc.) are still cached individually.
    """
    try:
        from src.core.unified_engine import UnifiedEngine
        
        # Get components (cached automatically by their own decorators)
        db_conn = get_db_connection() if use_db else None
        
        vector_store = None
        if use_vector:
            try:
                vector_store = get_vector_store()
            except Exception:
                pass
        
        kg_metadata = None
        if use_kg:
            try:
                kg_metadata = get_kg_metadata()
            except Exception as e:
                print(f"KG load error: {e}")
            
        llm_func = get_llm_func() if use_llm else None

        engine = UnifiedEngine(
            vector_store=vector_store,
            db_connection=db_conn,
            kg_metadata=kg_metadata,
            llm_func=llm_func
        )
        return engine
    except Exception as e:
        print(f"ERROR: UnifiedEngine initialization failed: {e}")
        return None

def get_unified_engine(use_vector=True, use_db=True, use_kg=True, use_llm=True):
    """Wrapper to get the engine with specific flags."""
    return get_engine_instance(use_vector, use_db, use_kg, use_llm)


def render_chat():
    st.header("Local Nexus Intelligence")

    # Check for Cloud/Memory mode
    is_cloud_path = "/mount/src" in os.path.abspath(__file__)

    if DatabaseManager().is_in_memory or is_cloud_path:
        st.warning("Cloud Demo Mode: Data will be lost on reboot. For production use, run locally.")

    # Engine components configuration
    st.sidebar.markdown("### Unified Engine Components")
    use_vector = st.sidebar.toggle("Vector Store (RAG)", value=True, help="Search documents (ChromaDB)")
    use_db = st.sidebar.toggle("Structured Data (SQL)", value=True, help="Query database (DuckDB)")
    use_kg = st.sidebar.toggle("Knowledge Graph", value=True, help="Query expansion via entity links")
    use_llm = st.sidebar.toggle("LLM Synthesis", value=True, help="Generate answers with Gemini")
    
    # Use unified engine if at least one data source is enabled
    use_unified = use_vector or use_db

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show query metadata if available
            if "query_type" in message:
                render_query_badge(message["query_type"])

            if "sql_query" in message and message["sql_query"]:
                with st.expander("SQL Query"):
                    st.code(message["sql_query"], language="sql")

            if "sources" in message and message["sources"]:
                with st.expander(f"Sources ({len(message['sources'])})"):
                    for i, src in enumerate(message["sources"][:3]):
                        st.caption(f"[{src.source}] {src.content[:200]}...")

    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            if use_unified:
                # Pass the feature flags to the generation function
                response = generate_unified_response(
                    prompt, 
                    use_vector=use_vector, 
                    use_db=use_db,
                    use_kg=use_kg,
                    use_llm=use_llm
                )
            else:
                response = generate_simple_response(prompt)

            st.markdown(response["content"])
            
            # ... (rest of function)

            # Show metadata
            if response.get("query_type"):
                render_query_badge(response["query_type"])

            if response.get("sql_query"):
                with st.expander("SQL Query"):
                    st.code(response["sql_query"], language="sql")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["content"],
                "query_type": response.get("query_type"),
                "sql_query": response.get("sql_query"),
                "sources": response.get("sources", [])
            })


def render_query_badge(query_type: str):
    """Render a badge showing the query type."""
    badges = {
        "structured": ("SQL", "blue"),
        "unstructured": ("RAG", "green"),
        "hybrid": ("Hybrid", "orange"),
        "error": ("Error", "red")
    }

    label, color = badges.get(query_type, ("Unknown", "gray"))
    st.caption(f"Query type: **{label}**")


def generate_unified_response(prompt: str, use_vector=True, use_db=True, use_kg=True, use_llm=True) -> dict:
    """Generate response using the Unified Engine."""
    engine = get_unified_engine(use_vector, use_db, use_kg, use_llm)

    if not engine:
        return {
            "content": "Unified Engine not available. Please check your configuration.",
            "query_type": "error"
        }

    with st.spinner("Analyzing query..."):
        try:
            print(f"DEBUG: calling engine.query with prompt: {prompt}")
            result = engine.query(prompt)
            print(f"DEBUG: engine.query returned: {result}")

            if result.error:
                return {
                    "content": f"Error: {result.error}",
                    "query_type": "error"
                }

            return {
                "content": result.answer,
                "query_type": result.query_type,
                "sql_query": result.sql_query,
                "sources": result.sources
            }

        except Exception as e:
            return {
                "content": f"An error occurred: {str(e)}",
                "query_type": "error"
            }


def generate_simple_response(prompt: str) -> dict:
    """Generate response using simple Gemini chat (fallback)."""
    with st.spinner("Thinking..."):
        try:
            from src.core.llm import get_gemini_response

            # Build message history for Gemini
            messages = st.session_state.messages + [{"role": "user", "content": prompt}]
            response_text = get_gemini_response(messages)

            return {"content": response_text}

        except Exception as e:
            return {
                "content": f"Error communicating with LLM: {str(e)}",
                "query_type": "error"
            }

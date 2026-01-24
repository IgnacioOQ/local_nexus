import streamlit as st
from src.core.ingestion import IngestionService

def render_sidebar():
    with st.sidebar:
        st.title("Data Management")
        
        # File Uploader
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            if st.button("Ingest Data"):
                service = IngestionService()
                success, message = service.process_file(uploaded_file)
                if success:
                    st.success(message)
                else:
                    st.error(message)

        st.divider()
        
        # Available Tables (Mock for now, easy to wire up to DB)
        st.subheader("Active Tables")
        # In a real scenario, query `metadata_registry` here
        st.info("No tables loaded.")

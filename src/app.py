import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
from src.components.sidebar import render_sidebar
from src.components.chat import render_chat

def main():
    st.set_page_config(
        page_title="Local Nexus",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "active_tables" not in st.session_state:
        st.session_state.active_tables = []

    # Layout
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()

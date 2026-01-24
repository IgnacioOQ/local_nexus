import streamlit as st
import pandas as pd

def render_chat():
    st.header("Local Nexus Intelligence")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "data" in message:
                st.dataframe(message["data"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Mock Response (Phase 1 Loopback)
        with st.chat_message("assistant"):
            response_text = f"Analyzed: {prompt} (Mock Response)"
            st.markdown(response_text)
            
            # Simple mock data response if "show" is in prompt
            if "show" in prompt.lower():
                mock_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
                st.dataframe(mock_data)
                st.session_state.messages.append({"role": "assistant", "content": response_text, "data": mock_data})
            else:
                st.session_state.messages.append({"role": "assistant", "content": response_text})

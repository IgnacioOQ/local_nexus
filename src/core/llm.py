import os
from google import genai
from dotenv import load_dotenv

DEFAULT_MODEL = "gemini-2.0-flash-lite"

def init_gemini():
    """Initializes the Gemini API with priority: Streamlit secrets > Environment variables."""
    load_dotenv()
    
    # 1. Try Streamlit secrets first (Priority for deployment/config)
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
    except (ImportError, FileNotFoundError):
        pass

    # 2. Fallback to Environment variables
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Warning: GEMINI_API_KEY not found in secrets or environment.")
        return False
    
    # genai.configure(api_key=api_key) # Deprecated
    # For google-genai, we instantiate Client with api_key specificially where needed, 
    # or we could set it potentially, but standard pattern is Client(api_key=...)
    # We will just ensure the env var is set if we found it in secrets, for convenience.
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    return True

def get_gemini_response(messages) -> str:
    """
    Sends a chat history to Gemini and returns the response text.
    
    Args:
        messages (list): List of dicts with "role" ('user' or 'model') and "parts" (content).
                         Streamlit uses "role" ('user' or 'assistant') and "content".
                         We need to convert formats.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
             # Try to find it again if helpful, or rely on init_gemini having run
             pass

        client = genai.Client(api_key=api_key)
        
        # Convert Streamlit history to GenAI history (if needed) or just use generate_content
        # The new SDK supports chat history nicely.
        
        # Streamlit: [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        # New SDK needs contents appropriately.
        # "role": "user" -> "user"
        # "role": "assistant" -> "model"
        
        # However, for simple generate_content with history, we might want to use chats.
        
        # Let's use clean chat history creation
        # client.chats.create(model=..., history=...)
        
        formatted_history = []
        for msg in messages[:-1]: # All except last
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append(
                genai.types.Content(
                    role=role,
                    parts=[genai.types.Part.from_text(text=msg["content"])]
                )
            )

        # Last message
        last_content = messages[-1]["content"]

        chat = client.chats.create(
            model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            history=formatted_history
        )
        
        response = chat.send_message(last_content)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {str(e)}"

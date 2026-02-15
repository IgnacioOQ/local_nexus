from google import genai
from src.core.llm import init_gemini, get_gemini_response
import os

def test_gemini_connection():
    print("Initializing Gemini...")
    if not init_gemini():
        print("Failed to initialize. Check API Key.")
        return

    print("Listing available models...")
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        for m in client.models.list():
            print(m.name)
    except Exception as e:
        print(f"Error listing models: {e}")

    print("Sending test message...")
    messages = [{"role": "user", "content": "Hello, this is a test from Local Nexus. Reply with 'Connection Successful'."}]
    response = get_gemini_response(messages)
    print(f"Response: {response}")

if __name__ == "__main__":
    test_gemini_connection()

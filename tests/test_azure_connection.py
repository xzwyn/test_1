# tests/test_azure_connection.py

import os
import sys
from dotenv import load_dotenv

# Adjust import path to access the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.azure_client import chat

def run_connection_test():
    """
    Checks the connection to the Azure OpenAI chat service.
    """
    print("--- Running Azure OpenAI Connection Test ---")
    
    # Load environment variables from the root .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    # Check if necessary environment variables are set
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        print("❌ FAIL: AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not found in .env file.")
        return

    print("Credentials found. Sending a test message to the chat API...")
    try:
        # Send a simple message and expect a simple response
        response = chat([{"role": "user", "content": "Reply with only the word: success"}])
        
        print(f"Received response: '{response.strip()}'")

        if "success" in response.lower():
            print("✅ PASS: Successfully connected to Azure OpenAI and received a valid response.")
        else:
            print("⚠️ WARN: Connected, but response was not as expected. Please check deployment status.")
    
    except Exception as e:
        print(f"❌ FAIL: An error occurred while trying to connect to Azure OpenAI.")
        print(f"   Error details: {e}")

if __name__ == "__main__":
    run_connection_test()
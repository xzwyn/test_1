# document_aligner/src/clients/azure_client.py

import os
from typing import List, Dict, Any, Optional
import numpy as np
from openai import AzureOpenAI

# NOTE: The main.py script already loads dotenv, but it's good practice
# for a module that relies on env vars to be able to load them itself.
from dotenv import load_dotenv
load_dotenv()


_client: Optional[AzureOpenAI] = None
_cfg = {
    "endpoint": None,
    "api_key": None,
    "api_version": None,
    "chat_deployment": None,
}

def _load_env():
    """Loads Azure configuration from environment variables."""
    _cfg["endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    _cfg["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
    _cfg["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    _cfg["chat_deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def _get_client() -> AzureOpenAI:
    """Initializes and returns a singleton AzureOpenAI client."""
    global _client
    if _client is not None:
        return _client
    
    _load_env()
    if not _cfg["endpoint"] or not _cfg["api_key"] or not _cfg["chat_deployment"]:
        raise RuntimeError(
            "Azure OpenAI client is not configured. "
            "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT in your .env file."
        )
    
    _client = AzureOpenAI(
        azure_endpoint=_cfg["endpoint"],
        api_key=_cfg["api_key"],
        api_version=_cfg["api_version"],
    )
    return _client

def chat(messages: List[Dict[str, Any]], temperature: float = 0.1, model: Optional[str] = None) -> str:
    """
    Sends a chat completion request to the configured Azure deployment.
    """
    client = _get_client()
    # The 'model' parameter must be your deployment name.
    deployment = model or _cfg["chat_deployment"]
    
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
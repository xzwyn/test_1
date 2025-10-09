import os
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

_chat_client: Optional[AzureOpenAI] = None
_embedding_client: Optional[AzureOpenAI] = None

_cfg = {
    "chat_endpoint": None,
    "chat_api_key": None,
    "chat_api_version": None,
    "chat_deployment": None,
    "embedding_endpoint": None,
    "embedding_api_key": None,
    "embedding_api_version": None,
    "embedding_deployment": None,
}

def _load_env():
    # Chat configuration
    _cfg["chat_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    _cfg["chat_api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
    _cfg["chat_api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    _cfg["chat_deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    # Embedding configuration
    _cfg["embedding_endpoint"] = os.getenv("AZURE_EMBEDDING_ENDPOINT")
    _cfg["embedding_api_key"] = os.getenv("AZURE_EMBEDDING_API_KEY")
    _cfg["embedding_api_version"] = os.getenv("AZURE_API_VERSION", "2024-02-01")
    _cfg["embedding_deployment"] = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

def _get_chat_client() -> AzureOpenAI:
    global _chat_client
    if _chat_client is not None:
        return _chat_client

    _load_env()
    if not _cfg["chat_endpoint"] or not _cfg["chat_api_key"] or not _cfg["chat_deployment"]:
        raise RuntimeError(
            "Azure OpenAI chat client is not configured. "
            "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT in your .env file."
        )

    _chat_client = AzureOpenAI(
        azure_endpoint=_cfg["chat_endpoint"],
        api_key=_cfg["chat_api_key"],
        api_version=_cfg["chat_api_version"],
    )
    return _chat_client

def _get_embedding_client() -> AzureOpenAI:
    global _embedding_client
    if _embedding_client is not None:
        return _embedding_client

    _load_env()
    if not _cfg["embedding_endpoint"] or not _cfg["embedding_api_key"] or not _cfg["embedding_deployment"]:
        raise RuntimeError(
            "Azure OpenAI embedding client is not configured. "
            "Set AZURE_EMBEDDING_ENDPOINT, AZURE_EMBEDDING_API_KEY, and AZURE_EMBEDDING_DEPLOYMENT_NAME in your .env file."
        )

    _embedding_client = AzureOpenAI(
        azure_endpoint=_cfg["embedding_endpoint"],
        api_key=_cfg["embedding_api_key"],
        api_version=_cfg["embedding_api_version"] or _cfg["chat_api_version"],
    )
    return _embedding_client

def chat(messages: List[Dict[str, Any]], temperature: float = 0.1, model: Optional[str] = None) -> str:
    client = _get_chat_client()
    deployment = model or _cfg["chat_deployment"]

    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""

def get_embeddings(texts: List[str], model: Optional[str]=None) -> List[List[float]]:
    client = _get_embedding_client()
    deployment = model or _cfg['embedding_deployment']

    if not deployment:
        raise ValueError("No embedding deployment specified. Please set AZURE_EMBEDDING_DEPLOYMENT_NAME in your .env file.")

    response = client.embeddings.create(
        input=texts,
        model=deployment
    )
    return [item.embedding for item in response.data]

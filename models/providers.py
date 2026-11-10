"""Model providers. Model choice is independent of framework choice.

Set model IDs in .env (GEMINI_MODEL, GROQ_MODEL, OLLAMA_MODEL). The exact ID is
recorded in every run's metadata.
"""
from __future__ import annotations

import os


def get_chat_model(name: str):
    """Return (langchain chat model, info dict)."""
    if name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        mid = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        m = ChatGoogleGenerativeAI(
            model=mid, google_api_key=os.getenv("GEMINI_API_KEY"), api_version="v1beta", temperature=0
        )
        return m, {"provider": "google_ai_studio", "model_id": mid}
    if name == "groq":
        from langchain_groq import ChatGroq

        mid = os.getenv("GROQ_MODEL")
        if not mid:
            raise RuntimeError("GROQ_MODEL is not set in .env (run python3 pick_groq.py)")
        return ChatGroq(model=mid, temperature=0), {"provider": "groq", "model_id": mid}
    if name == "ollama":
        from langchain_ollama import ChatOllama

        mid = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        return ChatOllama(model=mid, temperature=0), {"provider": "ollama_local", "model_id": mid}
    raise ValueError(f"Unknown model '{name}'. Use gemini, groq or ollama")

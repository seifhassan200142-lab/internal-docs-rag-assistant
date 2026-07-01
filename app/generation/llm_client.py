from __future__ import annotations

import httpx

from app.core.config import Settings
from app.generation.prompts import build_rag_messages
from app.schemas.models import RetrievedChunk


class LLMConfigurationError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        messages = build_rag_messages(question, chunks)
        provider = self.settings.llm_provider

        if provider == "openai":
            return self._chat_completions(
                base_url=self.settings.openai_base_url,
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_model,
                messages=messages,
            )
        if provider == "groq":
            return self._chat_completions(
                base_url=self.settings.groq_base_url,
                api_key=self.settings.groq_api_key,
                model=self.settings.groq_model,
                messages=messages,
            )
        if provider == "openai_compatible":
            if not self.settings.openai_compatible_base_url or not self.settings.openai_compatible_model:
                raise LLMConfigurationError(
                    "OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL must be set when LLM_PROVIDER=openai_compatible."
                )
            return self._chat_completions(
                base_url=self.settings.openai_compatible_base_url,
                api_key=self.settings.openai_compatible_api_key,
                model=self.settings.openai_compatible_model,
                messages=messages,
            )
        if provider == "ollama":
            return self._ollama_chat(messages=messages)

        raise LLMConfigurationError(f"Unsupported LLM provider: {provider}")

    def _chat_completions(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        if not api_key:
            raise LLMConfigurationError(
                "No LLM API key is configured. Set the required key in .env or use LLM_PROVIDER=ollama for a local model."
            )

        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "temperature": 0.1}

        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _ollama_chat(self, messages: list[dict[str, str]]) -> str:
        url = self.settings.ollama_base_url.rstrip("/") + "/api/chat"
        payload = {"model": self.settings.ollama_model, "messages": messages, "stream": False}

        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("message", {}).get("content", "").strip()

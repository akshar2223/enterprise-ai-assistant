"""
Abstracted LLM Client.

Design principle: every agent calls `LLMClient.chat()` or `LLMClient.chat_json()`.
To swap OpenAI for a local vLLM endpoint, set OPENAI_API_BASE in .env and
point to your vLLM server — the OpenAI-compatible REST API means zero
code changes in agents.

Why JSON mode?
  Agents that output structured data (Supervisor routing plan, SQL query,
  faithfulness score) use `chat_json()` which forces `response_format=json_object`.
  This eliminates fragile regex parsing of LLM output.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, APIError, RateLimitError
import asyncio

from config import get_settings

logger = logging.getLogger(__name__)

# Default system role used when no system prompt is supplied
_DEFAULT_SYSTEM = (
    "You are a precise, enterprise-grade AI assistant. "
    "Always respond factually and concisely."
)


class LLMClient:
    """
    Async LLM client backed by OpenAI Chat Completions API.

    Set OPENAI_BASE_URL environment variable to route requests to:
    - vLLM:    http://localhost:8080/v1
    - Ollama:  http://localhost:11434/v1
    Any OpenAI-compatible server works without changing this class.
    """

    def __init__(self, temperature: float = 0.0) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_chat_model
        self._default_temp = temperature

    # ── Primary API ───────────────────────────────────────────────────────────

    async def chat(
        self,
        user_message: str,
        system_prompt: str = _DEFAULT_SYSTEM,
        temperature: Optional[float] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Send a single chat turn and return the assistant's text reply.

        Args:
            user_message:  The user's latest message.
            system_prompt: Injected as the system role instruction.
            temperature:   Override default temperature (0.0 = deterministic).
            history:       Prior conversation turns [{role, content}, ...].
        """
        messages = self._build_messages(system_prompt, user_message, history)
        response = await self._call_with_retry(
            messages=messages,
            temperature=temperature if temperature is not None else self._default_temp,
        )
        return response.choices[0].message.content  # type: ignore[union-attr]

    async def chat_json(
        self,
        user_message: str,
        system_prompt: str = _DEFAULT_SYSTEM,
        temperature: Optional[float] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Same as `chat()` but forces JSON output and parses the result.

        Use this for all agents that return structured data (routing plan,
        SQL query object, faithfulness evaluation, etc.).

        Raises:
            ValueError: If the model returns invalid JSON (should not happen
                        with json_object response_format, but handled defensively).
        """
        messages = self._build_messages(system_prompt, user_message, history)
        response = await self._call_with_retry(
            messages=messages,
            temperature=temperature if temperature is not None else self._default_temp,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content  # type: ignore[union-attr]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned non-JSON: %s", raw[:200])
            raise ValueError(f"LLM output is not valid JSON: {exc}") from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 4,
    ):
        """Exponential backoff on rate-limit and transient server errors."""
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        for attempt in range(max_retries):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except RateLimitError:
                wait = 2 ** attempt
                logger.warning(
                    "OpenAI RateLimitError — retry %d/%d in %ds",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
            except APIError as exc:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning("OpenAI APIError %s — retry in %ds", exc.status_code, wait)
                await asyncio.sleep(wait)

        raise RuntimeError("LLM call failed after max retries")

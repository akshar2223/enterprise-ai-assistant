"""
OpenAI Embedding Client with:
- Semantic chunking (tiktoken, 512-token windows, 64-token stride)
- Batch embedding (100 texts per API call — safe within OpenAI rate limits)
- Exponential backoff retry (handles RateLimitError / APIError)
- Async-first design for FastAPI; sync wrapper provided for PySpark ETL

Chunking rationale:
  512 tokens ≈ ~380 words — enough context for semantic retrieval without
  diluting the embedding signal. 64-token stride preserves cross-chunk
  sentence continuity, matching the spec (FR-1.2).
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

import tiktoken
from openai import AsyncOpenAI, APIError, RateLimitError

from config import get_settings

logger = logging.getLogger(__name__)

# text-embedding-3-large uses cl100k_base tokenizer
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Practical batch size: 100 keeps us well within OpenAI's TPM limits
_BATCH_SIZE = 100

# RRF / HNSW recall is best with these values (per FR-1.2)
_CHUNK_TOKENS = 512
_STRIDE_TOKENS = 64

# Max tokens the model accepts (text-embedding-3-large hard limit)
_MODEL_MAX_TOKENS = 8191


class EmbeddingClient:
    """
    Singleton-friendly async embedding client.
    Instantiate once and reuse across the app lifecycle.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed(self, text: str) -> List[float]:
        """Embed a single string. Truncates to model limit automatically."""
        text = _truncate_to_limit(text)
        results = await self._embed_batch_with_retry([text])
        return results[0]

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings in batches.
        Returns embeddings in the same order as the input list.
        """
        truncated = [_truncate_to_limit(t) for t in texts]
        all_embeddings: List[List[float]] = []

        for i in range(0, len(truncated), _BATCH_SIZE):
            batch = truncated[i : i + _BATCH_SIZE]
            embeddings = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    # ── Chunking ──────────────────────────────────────────────────────────────

    @staticmethod
    def chunk_text(
        text: str,
        max_tokens: int = _CHUNK_TOKENS,
        stride: int = _STRIDE_TOKENS,
    ) -> List[str]:
        """
        Sliding-window chunker using tiktoken.

        Returns a list of decoded text chunks. Chunks overlap by `stride`
        tokens to preserve sentence continuity across chunk boundaries.
        """
        tokens = _TOKENIZER.encode(text)
        if not tokens:
            return []

        chunks: List[str] = []
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(_TOKENIZER.decode(chunk_tokens))
            if end == len(tokens):
                break
            start += max_tokens - stride  # slide by (max_tokens - stride)

        return chunks

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _embed_batch_with_retry(
        self,
        texts: List[str],
        max_retries: int = 4,
    ) -> List[List[float]]:
        """
        Call OpenAI embeddings API with exponential backoff.
        Backs off on RateLimitError (429) and transient APIError (5xx).
        """
        for attempt in range(max_retries):
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    encoding_format="float",
                )
                # Sort by index to guarantee output order matches input order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]

            except RateLimitError:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                logger.warning(
                    "OpenAI RateLimitError — retrying in %ds (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait)

            except APIError as exc:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "OpenAI APIError %s — retrying in %ds",
                    exc.status_code,
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(f"Embedding failed after {max_retries} retries")


# ── Sync wrapper for PySpark UDFs ────────────────────────────────────────────

def embed_texts_sync(texts: List[str]) -> List[List[float]]:
    """
    Blocking wrapper around EmbeddingClient.embed_many.
    Used inside PySpark pandas UDFs which run in synchronous Python workers.
    Creates a fresh event loop per batch — safe in forked Spark workers.
    """
    client = EmbeddingClient()
    return asyncio.run(client.embed_many(texts))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate_to_limit(text: str, max_tokens: int = _MODEL_MAX_TOKENS) -> str:
    """Silently truncate text that exceeds the model's token limit."""
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    logger.debug("Text truncated from %d to %d tokens", len(tokens), max_tokens)
    return _TOKENIZER.decode(tokens[:max_tokens])

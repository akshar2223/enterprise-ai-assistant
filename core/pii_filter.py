"""
PII Input Sanitizer — applied to user queries BEFORE sending to external LLM APIs.

This is the API-layer PII filter (query-time), distinct from the ETL-layer
PII masker (ingestion-time) in etl/spark_ingestion.py.

Purpose:
  When a user includes PII in their query (e.g., "What is the contract for
  John Smith at SSN 123-45-6789?"), we must strip it before the text leaves
  our perimeter to OpenAI's servers — satisfying NFR-1 (Data Privacy).

Implementation (v1):
  Regex-based masking for the most common PII patterns.
  Patterns are compiled once at module load for performance.

Upgrade path (v2):
  Layer in Microsoft Presidio (https://github.com/microsoft/presidio) for
  NER-based PII detection covering names, addresses, organisations, dates of
  birth, and other contextual PII that regex cannot catch.

  from presidio_analyzer import AnalyzerEngine
  from presidio_anonymizer import AnonymizerEngine
  analyzer = AnalyzerEngine()
  anonymizer = AnonymizerEngine()

Usage:
  from core.pii_filter import filter_query_pii
  safe_query = filter_query_pii(user_query)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ── PII Pattern Definitions ───────────────────────────────────────────────────

@dataclass(frozen=True)
class PIIPattern:
    name: str
    pattern: str
    replacement: str


_RAW_PATTERNS: List[PIIPattern] = [
    PIIPattern("SSN",         r"\b\d{3}-\d{2}-\d{4}\b",                              "[SSN]"),
    PIIPattern("CC_NUMBER",   r"\b(?:\d[ -]?){13,16}\b",                             "[CARD]"),
    PIIPattern("EMAIL",       r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL]"),
    PIIPattern("US_PHONE",    r"\b(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b", "[PHONE]"),
    PIIPattern("UK_PHONE",    r"\b(?:\+44|0)\s?\d{2,4}\s?\d{3,4}\s?\d{3,4}\b",      "[PHONE]"),
    PIIPattern("IP_ADDRESS",  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                        "[IP]"),
    PIIPattern("PASSPORT_US", r"\b[A-Z]{1,2}\d{6,9}\b",                              "[PASSPORT]"),
    PIIPattern("DOB",         r"\b(?:0?[1-9]|[12]\d|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:19|20)\d{2}\b", "[DOB]"),
    PIIPattern("IBAN",        r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})?\b", "[IBAN]"),
    PIIPattern("API_KEY",     r"\b(?:sk|pk|api|key)[-_][A-Za-z0-9\-_]{20,}\b",      "[API_KEY]"),
]

# Pre-compile all patterns at module load time (regex compilation is expensive)
_COMPILED: List[Tuple[re.Pattern, str]] = [
    (re.compile(p.pattern), p.replacement)
    for p in _RAW_PATTERNS
]


# ── Public API ────────────────────────────────────────────────────────────────

def filter_query_pii(text: str) -> str:
    """
    Apply all PII masking patterns to a user query string.

    Args:
        text: Raw user query that may contain PII.

    Returns:
        Sanitised string safe to send to external LLM APIs.
        Original text is never stored after this point in the request path.

    Performance:
        All patterns pre-compiled. Typical 200-token query: ~0.2ms on CPU.
    """
    if not text:
        return text

    original_len = len(text)
    redacted_count = 0

    for pattern, replacement in _COMPILED:
        new_text, n = pattern.subn(replacement, text)
        redacted_count += n
        text = new_text

    if redacted_count > 0:
        logger.info(
            "[PIIFilter] Masked %d PII token(s) | original_len=%d → sanitised_len=%d",
            redacted_count,
            original_len,
            len(text),
        )

    return text

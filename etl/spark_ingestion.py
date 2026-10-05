"""
Enterprise Data Ingestion Pipeline (PySpark).

Pipeline stages:
  1. Read JSON documents from input path
  2. Validate & filter (non-null content + required fields)
  3. PII masking (regex patterns on content before sending to external API)
  4. Semantic chunking (512-token windows, 64-token stride via tiktoken)
  5. Explode chunks → one row per chunk
  6. Batch embed chunks via OpenAI text-embedding-3-large (pandas UDF)
  7. Write to PostgreSQL enterprise_embeddings table (JDBC append)

Why PySpark for this ETL?
  See requirements.md trade-off matrix: enterprise mandate for 100M+ record
  scale-out. PySpark's pandas UDF with Arrow serialization gives us
  vectorized batch processing of embeddings without per-row Python overhead.

PII masking approach (v1):
  Regex patterns catch the most common PII forms. Production v2 should
  layer in a Presidio NER model for named-entity PII (names, orgs).

Usage:
  spark-submit etl/spark_ingestion.py <input_path> <db_url>
  
  where <db_url> is a JDBC URL:
    jdbc:postgresql://localhost:5432/enterprise_ai

Environment variables required on all Spark workers:
  OPENAI_API_KEY          — used inside the pandas UDF on each executor
  OPENAI_EMBEDDING_MODEL  — defaults to text-embedding-3-large
"""
from __future__ import annotations

import os
import re
import sys
import logging
from typing import List

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    StringType,
    StructField,
    StructType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── PII Regex Patterns ────────────────────────────────────────────────────────
# Applied before content is sent to OpenAI — satisfies NFR-1 (Data Privacy).

_PII_PATTERNS: List[tuple[str, str]] = [
    # US Social Security Numbers
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
    # Credit card numbers (Visa/MC/Amex/Discover patterns)
    (r"\b(?:\d[ -]?){13,16}\b", "[CC_REDACTED]"),
    # Email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),
    # US phone numbers
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]"),
    # IP addresses
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_REDACTED]"),
]


def mask_pii(text: str) -> str:
    """Apply all PII regex patterns sequentially."""
    for pattern, replacement in _PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


# ── Spark Session ─────────────────────────────────────────────────────────────

def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("EnterpriseDataIngestionPipeline")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # Arrow batch size — tune based on available memory per executor
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "500")
        .getOrCreate()
    )


# ── PySpark UDFs ──────────────────────────────────────────────────────────────

# Schema for the chunked output (one row per chunk)
_CHUNK_SCHEMA = StructType([
    StructField("chunk_content", StringType(), False),
    StructField("chunk_index", StringType(), False),   # stored as string for JSONB metadata
])


def _chunk_udf(content: str, max_tokens: int = 512, stride: int = 64) -> List[str]:
    """
    Python function (non-UDF) for chunking — called via pandas_udf wrapper.
    Imported inside executor scope to avoid serialization issues.
    """
    import tiktoken  # imported here so executors can find it

    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(content)
    if not tokens:
        return [content]

    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += max_tokens - stride
    return chunks


# Spark UDF returning an array of chunk strings
chunk_text_udf = F.udf(
    lambda text: _chunk_udf(text) if text else [],
    ArrayType(StringType()),
)


@F.pandas_udf(ArrayType(FloatType()))  # type: ignore[call-overload]
def embed_content_udf(texts: pd.Series) -> pd.Series:
    """
    Pandas UDF — runs on Spark executors in vectorized Arrow batches.

    Each executor calls OpenAI embeddings API in batches of 100.
    The OPENAI_API_KEY env var must be set on all workers.

    Returns a Series where each element is a list of 1536 floats.
    """
    import openai  # imported inside UDF for executor availability

    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set on Spark executor. "
            "Set it via SparkContext environment or executor environment config."
        )

    client = openai.OpenAI(api_key=api_key)
    batch_size = 100
    texts_list = texts.tolist()
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts_list), batch_size):
        batch = [str(t) if t else "" for t in texts_list[i : i + batch_size]]
        response = client.embeddings.create(model=model, input=batch, encoding_format="float")
        # Sort by index to preserve input order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([item.embedding for item in sorted_data])

    return pd.Series(all_embeddings)


# ── Main Ingestion Pipeline ───────────────────────────────────────────────────

def process_batch_documents(
    input_path: str,
    db_url: str,
    db_properties: dict,
    tenant_id: str = "default",
) -> None:
    """
    Full ingestion pipeline: read → validate → PII mask → chunk → embed → write.

    Args:
        input_path:     Path to JSON file (local or HDFS/S3).
        db_url:         JDBC URL e.g. jdbc:postgresql://localhost:5432/enterprise_ai
        db_properties:  JDBC auth dict with 'user' and 'password' keys.
        tenant_id:      Tenant identifier injected into every written row.
    """
    spark = build_spark_session()

    # ── Stage 1: Read ─────────────────────────────────────────────────────────
    logger.info("Reading documents from: %s", input_path)
    raw_df: DataFrame = spark.read.json(input_path)

    # ── Stage 2: Validate ─────────────────────────────────────────────────────
    required_cols = {"doc_id", "content"}
    available = set(raw_df.columns)
    missing = required_cols - available
    if missing:
        raise ValueError(f"Input JSON missing required columns: {missing}")

    validated_df = raw_df.filter(
        F.col("content").isNotNull() & (F.trim(F.col("content")) != "")
    )
    logger.info("Records after validation: %d", validated_df.count())

    # ── Stage 3: PII Masking ──────────────────────────────────────────────────
    pii_mask_udf = F.udf(mask_pii, StringType())
    masked_df = validated_df.withColumn("content", pii_mask_udf(F.col("content")))

    # ── Stage 4: Chunking ─────────────────────────────────────────────────────
    chunked_df = masked_df.withColumn(
        "chunks", chunk_text_udf(F.col("content"))
    ).withColumn(
        "chunk_content", F.explode(F.col("chunks"))
    )

    # ── Stage 5: Add metadata columns ─────────────────────────────────────────
    prepared_df = (
        chunked_df
        .withColumn("tenant_id", F.lit(tenant_id))
        .withColumn("created_at", F.current_timestamp())
        # Carry forward original metadata as JSON string (coalesce to empty obj)
        .withColumn(
            "metadata",
            F.coalesce(F.to_json(F.col("metadata")), F.lit("{}"))
            if "metadata" in chunked_df.columns
            else F.lit("{}")
        )
        .select("doc_id", "tenant_id", "chunk_content", "metadata", "created_at")
        # Rename for output schema clarity before embedding
        .withColumnRenamed("chunk_content", "content")
    )

    # ── Stage 6: Embed (pandas UDF — batched OpenAI calls) ───────────────────
    logger.info("Generating embeddings via OpenAI %s …", os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
    embedded_df = prepared_df.withColumn("embedding", embed_content_udf(F.col("content")))

    # ── Stage 7: Write to PostgreSQL ──────────────────────────────────────────
    logger.info("Writing to PostgreSQL: %s", db_url)
    (
        embedded_df
        .write
        .format("jdbc")
        .option("url", db_url)
        .option("dbtable", "enterprise_embeddings")
        .option("user", db_properties["user"])
        .option("password", db_properties["password"])
        .option("driver", "org.postgresql.Driver")
        # batchsize: 1000 rows per JDBC batch INSERT — reduces round-trips
        .option("batchsize", "1000")
        .mode("append")
        .save()
    )
    logger.info("✅ Ingestion complete.")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: spark-submit etl/spark_ingestion.py"
            " <input_json_path> <jdbc_db_url> [tenant_id]"
        )
        sys.exit(1)

    _input_path = sys.argv[1]
    _db_url = sys.argv[2]
    _tenant_id = sys.argv[3] if len(sys.argv) > 3 else "default"

    _db_props = {
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "change_me_in_prod"),
    }

    process_batch_documents(_input_path, _db_url, _db_props, _tenant_id)

"""
Enterprise AI Knowledge & Decision Assistant — FastAPI Application.

Startup sequence (via lifespan):
  1. Configure structured logging (JSON in prod, pretty in dev)
  2. Initialise async PostgreSQL connection pool
  3. Compile LangGraph agent DAG (once — expensive; reuse per request)
  4. Attach OpenTelemetry FastAPI instrumentation

Request flow for POST /api/v1/chat/completions:
  JWT validation → PII filter → LangGraph ainvoke → Audit log → Response

Endpoints:
  POST /api/v1/chat/completions  — main multi-agent query endpoint
  POST /api/v1/auth/token        — dev/testing JWT minting (disable in prod)
  GET  /api/v1/health            — liveness probe
  GET  /api/v1/health/db         — readiness probe (checks DB connectivity)
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agents.state import AgentState
from config import get_settings
from core.audit import write_pipeline_audit
from core.auth import create_access_token, get_current_user
from core.database import close_pool, get_pool
from core.pii_filter import filter_query_pii
from graph.orchestrator import build_agent_graph

# ── Structured Logging Setup ──────────────────────────────────────────────────

settings = get_settings()

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
        if settings.is_production
        else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO)
)
logger = logging.getLogger(__name__)

# ── Rate Limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan — runs startup logic before serving and cleanup on shutdown.
    Compiling the graph and creating the pool here means it happens ONCE,
    not on every request.
    """
    logger.info("🚀 Starting Enterprise AI Assistant | env=%s", settings.app_env)

    # Warm up DB pool (validates connectivity at startup)
    pool = await get_pool()
    app.state.db_pool = pool
    logger.info("✅ PostgreSQL connection pool ready")

    # Compile LangGraph DAG — do this once, reuse per request
    app.state.agent_graph = build_agent_graph()
    logger.info("✅ Agent graph compiled")

    yield  # ── Application is running ─────────────────────────────────────────

    logger.info("🛑 Shutting down — closing DB pool")
    await close_pool()
    logger.info("✅ Shutdown complete")


# ── App Initialisation ────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise AI Knowledge & Decision Assistant",
    description=(
        "Multi-agent RAG + SQL analytics platform with hallucination guardrails. "
        "Powered by LangGraph, pgvector, and OpenAI GPT-4o."
    ),
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,   # hide Swagger in prod
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

# Rate limiting exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS (tighten allowed_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else [],
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

# OpenTelemetry FastAPI instrumentation (traces every request span)
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry import trace

    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    logger.info("✅ OpenTelemetry tracing enabled")
except ImportError:
    logger.warning("OpenTelemetry not installed — tracing disabled")


# ── Pydantic Models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Natural language query (PII will be masked before processing)",
        examples=["What were the Q3 EMEA sales and how do they compare to our risk policy?"],
    )
    strict_validation: bool = Field(
        True,
        description=(
            "If True, REJECT responses that fail the faithfulness guardrail. "
            "Set False only in development/demo environments."
        ),
    )


class CitationModel(BaseModel):
    source_id: str
    chunk_id: str
    relevance_score: float


class ExecutionMetrics(BaseModel):
    sql_queries_run: int = 0
    retrieved_chunks: int = 0
    faithfulness_score: Optional[float] = None
    latency_ms: int = 0


class ChatResponse(BaseModel):
    conversation_id: str
    response: Optional[str] = Field(
        None,
        description="Final validated response. Null if guardrail rejected the answer."
    )
    validation_status: str = Field(description="PASS or REJECT")
    citations: List[CitationModel] = []
    execution_metrics: ExecutionMetrics


class TokenRequest(BaseModel):
    user_id: str = Field(..., examples=["usr_99823"])
    tenant_id: str = Field(..., examples=["corp_acme_01"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/chat/completions",
    response_model=ChatResponse,
    summary="Multi-agent query endpoint",
    tags=["Chat"],
)
@limiter.limit(f"{settings.api_rate_limit_per_minute}/minute")
async def chat_completions(
    request: Request,             # required by slowapi limiter
    payload: ChatRequest,
    current_user: Dict[str, str] = Depends(get_current_user),
) -> ChatResponse:
    """
    Execute the full multi-agent pipeline for a user query.

    Pipeline stages:
      1. PII filter applied to query
      2. Supervisor routes intent (DOC / SQL / HYBRID)
      3. Retrieval agents execute in parallel or sequence
      4. Synthesizer generates executive response with citations
      5. Validator gates on faithfulness score ≥ 0.96
      6. Audit log written asynchronously (non-blocking)
    """
    conversation_id = str(uuid4())
    start_time = time.time()

    tenant_id: str = current_user["tenant_id"]   # always from JWT — never request body
    user_id: str = current_user["user_id"]

    structlog.contextvars.bind_contextvars(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )

    logger.info(
        "[API] New request | conversation=%s | tenant=%s",
        conversation_id,
        tenant_id,
    )

    # ── PII filter ────────────────────────────────────────────────────────────
    safe_query = filter_query_pii(payload.query)

    # ── Build initial state ───────────────────────────────────────────────────
    initial_state: AgentState = {
        "query": safe_query,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "strict_validation": payload.strict_validation,
        "execution_metrics": {
            "pipeline_start_ms": int(start_time * 1000),
        },
    }

    # ── Run agent graph ───────────────────────────────────────────────────────
    try:
        final_state: AgentState = await app.state.agent_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error(
            "[API] ❌ Agent graph crashed | conversation=%s | error=%s",
            conversation_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal orchestration error. The incident has been logged.",
        )

    total_latency_ms = int((time.time() - start_time) * 1000)

    # ── Fire-and-forget audit log ─────────────────────────────────────────────
    asyncio.create_task(
        write_pipeline_audit(
            final_state=final_state,
            total_latency_ms=total_latency_ms,
        )
    )

    # ── Handle validation rejection ───────────────────────────────────────────
    validation_status = final_state.get("validation_status", "UNKNOWN")
    final_response = final_state.get("final_response")

    if validation_status == "REJECT" and payload.strict_validation:
        logger.warning(
            "[API] Response rejected | score=%.3f | conversation=%s",
            final_state.get("faithfulness_score", 0.0),
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": (
                    "Response rejected by the hallucination guardrail. "
                    "The generated answer could not be fully grounded in retrieved sources."
                ),
                "faithfulness_score": final_state.get("faithfulness_score"),
                "unsupported_claims": final_state.get("unsupported_claims", []),
                "conversation_id": conversation_id,
            },
        )

    # ── Build citations ───────────────────────────────────────────────────────
    raw_citations = final_state.get("citations", [])
    citations = [
        CitationModel(
            source_id=c.get("source_id", "unknown"),
            chunk_id=str(c.get("chunk_id", "")),
            relevance_score=float(c.get("relevance_score", 0.0)),
        )
        for c in raw_citations
    ]

    # ── Build metrics ─────────────────────────────────────────────────────────
    metrics_raw = final_state.get("execution_metrics", {})
    metrics = ExecutionMetrics(
        sql_queries_run=metrics_raw.get("sql_queries_run", 0),
        retrieved_chunks=metrics_raw.get("retrieved_chunks", 0),
        faithfulness_score=final_state.get("faithfulness_score"),
        latency_ms=total_latency_ms,
    )

    logger.info(
        "[API] ✅ Request complete | conversation=%s | status=%s | latency=%dms",
        conversation_id,
        validation_status,
        total_latency_ms,
    )

    return ChatResponse(
        conversation_id=conversation_id,
        response=final_response,
        validation_status=validation_status,
        citations=citations,
        execution_metrics=metrics,
    )


@app.post(
    "/api/v1/auth/token",
    response_model=TokenResponse,
    summary="Generate a dev/test JWT token",
    tags=["Auth"],
    description=(
        "⚠️ Development endpoint — mint a JWT for local testing. "
        "Disable this route in production by setting APP_ENV=production."
    ),
)
async def generate_token(body: TokenRequest) -> TokenResponse:
    """Generate a signed JWT for dev/testing. Disabled in production."""
    if settings.is_production:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token generation endpoint is disabled in production.",
        )

    token = create_access_token(user_id=body.user_id, tenant_id=body.tenant_id)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_minutes=settings.jwt_access_token_expire_minutes,
    )


@app.get(
    "/api/v1/health",
    summary="Liveness probe",
    tags=["Health"],
)
async def health_liveness() -> Dict[str, str]:
    """Returns 200 OK if the application process is alive."""
    return {"status": "ok", "env": settings.app_env}


@app.get(
    "/api/v1/health/db",
    summary="Readiness probe — checks DB connectivity",
    tags=["Health"],
)
async def health_readiness() -> Dict[str, Any]:
    """Returns 200 if DB is reachable, 503 if not. Use for k8s readiness probe."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
        return {"status": "ok", "database": "connected", "ping": result}
    except Exception as exc:
        logger.error("[Health] DB connectivity check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unreachable: {exc}",
        )


# ── Dev server entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "deployment.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )

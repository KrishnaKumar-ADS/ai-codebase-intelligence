from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from api.routes import analysis, graph, ingest, search
from api.routes.cache import router as cache_router
from api.routes.explain import router as explain_router
from api.routes.analyze import router as analyze_router
from api.routes.ask import router as ask_router
from api.routes.evaluate import router as evaluate_router
from api.routes.metrics import router as metrics_router
from api.routes.providers import router as providers_router
from api.routes.webhook import router as webhook_router
from core.logging import setup_logging, get_logger
from core.config import get_settings
from api.middleware import add_middleware
from cost_tracking.models import BudgetExceededError
from graph.neo4j_client import get_driver, close_driver, ping as neo4j_ping
from graph.schema import create_indexes

setup_logging()
settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown handler."""
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("app_startup_begin")

    # Validate LLM API keys (existing — logs warnings for missing keys)

    # Initialize Qdrant collection (existing — from Week 3)


    # ── NEW in Week 4: Initialize Neo4j ────────────────────────────────────
    try:
        get_driver()       # establishes connection pool + verifies connectivity
        create_indexes()   # creates all 9 graph indexes (idempotent)
        logger.info("neo4j_initialized")
    except Exception as e:
        # Non-fatal — app still works without Neo4j (Qdrant search still works)
        logger.warning("neo4j_startup_warning", error=str(e))

    logger.info("app_startup_complete")
    yield

    # ── Shutdown ────────────────────────────────────────────────────────────
    close_driver()
    logger.info("app_shutdown_complete")


app = FastAPI(
    title="AI Codebase Intelligence Platform",
    description="Query any GitHub repository using natural language.",
    version="0.1.0",
    lifespan=lifespan,
)

add_middleware(app)

# ── Routers ───────────────────────────────────────────────
app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(graph.router)
app.include_router(analysis.router)
app.include_router(ask_router, prefix="/api/v1")
app.include_router(analyze_router)
app.include_router(providers_router)
app.include_router(cache_router)
app.include_router(explain_router)
app.include_router(evaluate_router)
app.include_router(metrics_router)
app.include_router(webhook_router)


@app.exception_handler(BudgetExceededError)
async def budget_exceeded_handler(request, exc: BudgetExceededError):
    return JSONResponse(
        status_code=429,
        content={
            "error": "daily_budget_exceeded",
            "message": str(exc),
            "daily_limit_usd": exc.daily_limit_usd,
            "used_usd": exc.used_usd,
        },
    )


@app.get("/health", tags=["system"])
async def health():
    from reasoning.llm_router import get_available_providers
    available = [p.value for p in get_available_providers()]

    qdrant_status = "ok"
    qdrant_vectors = 0
    try:
        from embeddings.vector_store import get_collection_info
        info = get_collection_info()
        qdrant_vectors = info["vectors_count"]
    except Exception as e:
        qdrant_status = f"unavailable: {e}"

    return JSONResponse({
        "status": "ok",
        "version": "0.1.0",
        "llm_providers_available": available,
        "qdrant": {
            "status": qdrant_status,
            "total_vectors": qdrant_vectors,
        },
    })
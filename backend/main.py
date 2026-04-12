from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from api.routes import ingest, search          # ← keep both
from core.logging import setup_logging, get_logger
from core.config import get_settings
from api.middleware import add_middleware

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    # Validate API keys
    warnings = settings.validate_api_keys()
    for w in warnings:
        logger.warning("api_key_warning", message=w)

    if not warnings:
        logger.info("all_providers_configured", providers=["gemini", "deepseek", "openrouter"])

    # Ensure Qdrant collection exists
    try:
        from embeddings.vector_store import ensure_collection_exists
        ensure_collection_exists()
        logger.info("qdrant_ready")
    except Exception as e:
        logger.warning("qdrant_not_available", error=str(e))

    yield


app = FastAPI(
    title="AI Codebase Intelligence Platform",
    description="Query any GitHub repository using natural language.",
    version="0.1.0",
    lifespan=lifespan,
)

add_middleware(app)

# ── Routers ───────────────────────────────────────────────
app.include_router(ingest.router)   # ← only once
app.include_router(search.router)   # ← was missing from actual registration


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
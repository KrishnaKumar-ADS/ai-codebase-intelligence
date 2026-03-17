from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from core.logging import setup_logging, get_logger
from core.config import get_settings
from api.middleware import add_middleware
from api.routes import ingest

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    # Validate API keys at startup and log warnings
    warnings = settings.validate_api_keys()
    for w in warnings:
        logger.warning("api_key_warning", message=w)

    if not warnings:
        logger.info("all_providers_configured", providers=["gemini", "deepseek", "openrouter"])

    yield


app = FastAPI(
    title="AI Codebase Intelligence Platform",
    description="Query any GitHub repository using natural language. Powered by Gemini, DeepSeek, and OpenRouter.",
    version="0.1.0",
    lifespan=lifespan,
)

add_middleware(app)
app.include_router(ingest.router)


@app.get("/health", tags=["system"])
async def health():
    from reasoning.llm_router import get_available_providers
    available = [p.value for p in get_available_providers()]
    return JSONResponse({
        "status": "ok",
        "version": "0.1.0",
        "llm_providers_available": available,
    })
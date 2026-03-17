"""
LLM Router — selects the right provider for each task type.

Task types:
  code_qa      → DeepSeek Coder V2   (best for code understanding)
  reasoning    → Gemini 2.0 Flash    (fast, large context, free tier)
  security     → DeepSeek Reasoner   (deep chain-of-thought)
  summarize    → Gemini 2.0 Flash
  embedding    → Gemini text-embedding-004
  fallback     → OpenRouter          (if primary provider fails)
"""

from enum import Enum
from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class TaskType(str, Enum):
    CODE_QA = "code_qa"
    REASONING = "reasoning"
    SECURITY = "security"
    SUMMARIZE = "summarize"
    EMBEDDING = "embedding"


class Provider(str, Enum):
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"


# Task → primary provider + model
ROUTING_TABLE: dict[TaskType, tuple[Provider, str]] = {
    TaskType.CODE_QA:   (Provider.DEEPSEEK,  "deepseek-coder"),
    TaskType.REASONING: (Provider.GEMINI,    "gemini-2.0-flash"),
    TaskType.SECURITY:  (Provider.DEEPSEEK,  "deepseek-reasoner"),
    TaskType.SUMMARIZE: (Provider.GEMINI,    "gemini-2.0-flash"),
    TaskType.EMBEDDING: (Provider.GEMINI,    "models/text-embedding-004"),
}

# Fallback chain per task if primary is unavailable
FALLBACK_TABLE: dict[TaskType, tuple[Provider, str]] = {
    TaskType.CODE_QA:   (Provider.OPENROUTER, "deepseek/deepseek-coder"),
    TaskType.REASONING: (Provider.OPENROUTER, "google/gemini-flash-1.5"),
    TaskType.SECURITY:  (Provider.OPENROUTER, "deepseek/deepseek-r1"),
    TaskType.SUMMARIZE: (Provider.OPENROUTER, "google/gemini-flash-1.5"),
    TaskType.EMBEDDING: (Provider.GEMINI,     "models/text-embedding-004"),  # No fallback for embedding
}


def get_available_providers() -> set[Provider]:
    """Return the set of providers that have API keys configured."""
    available = set()
    if settings.gemini_api_key:
        available.add(Provider.GEMINI)
    if settings.deepseek_api_key:
        available.add(Provider.DEEPSEEK)
    if settings.openrouter_api_key:
        available.add(Provider.OPENROUTER)
    return available


def route(task: TaskType) -> tuple[Provider, str]:
    """
    Returns (provider, model_name) for a given task.
    Falls back to OpenRouter if the primary provider key is missing.
    """
    available = get_available_providers()
    provider, model = ROUTING_TABLE[task]

    if provider in available:
        logger.debug("llm_route_primary", task=task.value, provider=provider.value, model=model)
        return provider, model

    # Try fallback
    fallback_provider, fallback_model = FALLBACK_TABLE[task]
    if fallback_provider in available:
        logger.warning(
            "llm_route_fallback",
            task=task.value,
            reason=f"{provider.value} key missing",
            fallback=fallback_provider.value,
        )
        return fallback_provider, fallback_model

    raise RuntimeError(
        f"No provider available for task '{task.value}'. "
        f"Please set at least one of: GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY"
    )
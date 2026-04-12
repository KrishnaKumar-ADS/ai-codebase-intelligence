"""
OpenRouter client — fallback provider with access to 100+ models.
Also uses OpenAI-compatible API.
"""

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.config import get_settings
from core.logging import get_logger
from core.exceptions import LLMProviderError

logger = get_logger(__name__)
settings = get_settings()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# When used as fallback, which OpenRouter model mirrors each provider
DEFAULT_FALLBACK_MODEL = "google/gemini-flash-1.5"


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/your-username/ai-codebase-intelligence",
            "X-Title": "AI Codebase Intelligence Platform",
        },
    )


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
def generate(
    prompt: str,
    system_prompt: str = "",
    model: str = DEFAULT_FALLBACK_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """
    Generate a completion via OpenRouter.
    Accepts any model slug from https://openrouter.ai/models
    e.g. "google/gemini-flash-1.5", "deepseek/deepseek-coder", "anthropic/claude-3-haiku"
    """
    try:
        client = _get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = response.choices[0].message.content
        logger.debug("openrouter_response", model=model)
        return text

    except Exception as e:
        logger.error("openrouter_generate_failed", error=str(e), model=model)
        raise LLMProviderError(f"OpenRouter generation failed: {e}") from e
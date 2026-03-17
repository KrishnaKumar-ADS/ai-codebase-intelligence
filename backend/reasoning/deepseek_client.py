"""
DeepSeek API client.
DeepSeek uses an OpenAI-compatible API, so we use the openai library
with a custom base_url pointing to DeepSeek's endpoint.
"""

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.config import get_settings
from core.logging import get_logger
from core.exceptions import LLMProviderError

logger = get_logger(__name__)
settings = get_settings()

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Model aliases → actual DeepSeek model names
MODEL_MAP = {
    "deepseek-coder":    "deepseek-coder",        # DeepSeek Coder V2 — code tasks
    "deepseek-chat":     "deepseek-chat",          # DeepSeek V3 — general chat
    "deepseek-reasoner": "deepseek-reasoner",      # DeepSeek R1 — deep reasoning
}


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=DEEPSEEK_BASE_URL,
    )


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
def generate(
    prompt: str,
    system_prompt: str = "",
    model: str = "deepseek-coder",
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """
    Generate a completion using DeepSeek API.
    Best models:
      - deepseek-coder   → code explanation, bug tracing, architecture Q&A
      - deepseek-chat    → general questions
      - deepseek-reasoner → security analysis, complex multi-step reasoning
    """
    try:
        client = _get_client()
        actual_model = MODEL_MAP.get(model, model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=actual_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

        text = response.choices[0].message.content
        logger.debug("deepseek_response", model=actual_model, tokens=response.usage.total_tokens)
        return text

    except Exception as e:
        logger.error("deepseek_generate_failed", error=str(e), model=model)
        raise LLMProviderError(f"DeepSeek generation failed: {e}") from e


def generate_stream(
    prompt: str,
    system_prompt: str = "",
    model: str = "deepseek-coder",
    max_tokens: int = 4096,
):
    """
    Streaming version of generate() — yields text chunks.
    Use for the /ask endpoint with stream=True.
    """
    client = _get_client()
    actual_model = MODEL_MAP.get(model, model)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model=actual_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
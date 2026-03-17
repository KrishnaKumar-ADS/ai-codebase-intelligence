"""
Google Gemini client — handles both chat completions and embeddings.
Uses tenacity for automatic retry on rate limit errors.
"""

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.config import get_settings
from core.logging import get_logger
from core.exceptions import LLMProviderError, EmbeddingError

logger = get_logger(__name__)
settings = get_settings()


def _get_client():
    genai.configure(api_key=settings.gemini_api_key)
    return genai


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
def generate(
    prompt: str,
    system_prompt: str = "",
    model: str = "gemini-2.0-flash",
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """
    Generate a text completion using Gemini.
    Automatically retries up to 3 times with exponential backoff.
    """
    try:
        _get_client()
        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=system_prompt if system_prompt else None,
        )

        response = gemini_model.generate_content(prompt)
        text = response.text
        logger.debug("gemini_response", model=model, tokens=len(text.split()))
        return text

    except Exception as e:
        logger.error("gemini_generate_failed", error=str(e), model=model)
        raise LLMProviderError(f"Gemini generation failed: {e}") from e


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
def embed(text: str, model: str = "models/text-embedding-004") -> list[float]:
    """
    Generate a single embedding vector using Gemini text-embedding-004.
    Automatically retries on rate limit errors.
    """
    try:
        _get_client()
        result = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]

    except Exception as e:
        logger.error("gemini_embed_failed", error=str(e))
        raise EmbeddingError(f"Gemini embedding failed: {e}") from e


def embed_batch(texts: list[str], model: str = "models/text-embedding-004") -> list[list[float]]:
    """
    Embed a list of texts. Gemini free tier allows 100 req/min,
    so we process with a small delay between batches.
    """
    import time
    embeddings = []
    batch_size = 20  # Stay safely within rate limits

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            embeddings.append(embed(text, model=model))
        if i + batch_size < len(texts):
            time.sleep(0.8)  # ~75 req/min safely under 100/min limit
        logger.info("embed_batch_progress", done=min(i + batch_size, len(texts)), total=len(texts))

    return embeddings
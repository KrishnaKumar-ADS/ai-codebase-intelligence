"""Prompt helpers for quality evaluation."""

from __future__ import annotations

JUDGE_SYSTEM_PROMPT = """
You are a strict evaluator for retrieval-augmented code answers.
Score the answer from 0.0 to 1.0 on these dimensions:
- faithfulness: Is the answer supported by the provided context?
- relevance: Does the answer directly address the user's question?
- completeness: Does the answer cover the key parts needed?

Return ONLY valid JSON with keys:
{
  "faithfulness": <float 0.0-1.0>,
  "relevance": <float 0.0-1.0>,
  "completeness": <float 0.0-1.0>,
  "critique": "<short explanation>"
}
""".strip()


def _truncate_context(text: str, max_context_chars: int) -> str:
    if len(text) <= max_context_chars:
        return text
    return text[:max_context_chars] + "\n[truncated]"


def build_judge_user_message(
    question: str,
    answer: str,
    context_chunks: list[str],
    max_context_chars: int = 6000,
) -> str:
    normalized_question = (question or "").strip()
    normalized_answer = (answer or "").strip()

    if len(normalized_answer) > 3000:
        normalized_answer = normalized_answer[:3000] + "... [truncated]"

    if context_chunks:
        labeled_chunks = [f"[Chunk {index}]\n{chunk}" for index, chunk in enumerate(context_chunks, start=1)]
        context_block = "\n\n".join(labeled_chunks)
        context_block = _truncate_context(context_block, max_context_chars)
    else:
        context_block = "[No context retrieved]"

    return (
        "Question:\n"
        f"{normalized_question}\n\n"
        "Answer:\n"
        f"{normalized_answer}\n\n"
        "Retrieved Context:\n"
        f"{context_block}\n\n"
        "Return strict JSON only."
    )

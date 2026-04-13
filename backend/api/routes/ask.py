"""Ask endpoint with session-aware responses and step-level streaming SSE."""

from __future__ import annotations

import asyncio
import json as _json
import time as _time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.ask_schema import (
    AskRequest,
    AskResponse,
    DeepAskRequest,
    DeepAskResponse,
    SourceReference,
)
from conversation.history_builder import build_history_block
from conversation.session_store import SessionStore
from cost_tracking.models import BudgetExceededError
from cost_tracking.tracker import CostTracker
from core.exceptions import LLMProviderError
from core.logging import get_logger
from db.database import AsyncSessionLocal, get_db
from db.models import IngestionStatus, Repository
from decomposition.query_decomposer import QueryDecomposer
from decomposition.synthesizer import AnswerSynthesizer
from evaluation.quality_evaluator import QualityEvaluator, update_running_averages
from intent.classifier import IntentClassifier
from intent.prompt_engine import (
    get_model_for_intent,
    get_system_prompt,
    get_temperature_for_intent,
)
from reasoning.context_assembler import ContextAssembler, ContextAssemblerConfig
from reasoning.llm_router import TaskType, get_available_providers, stream_ask
from reasoning.prompt_templates import build_prompt, get_task_type_from_question
from reasoning.rag_chain import RAGChain
from semantic_cache.answer_cache import get_semantic_answer_cache

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency fallback
    tiktoken = None

logger = get_logger(__name__)
router = APIRouter(tags=["Ask"])
_rag_chain = RAGChain(default_top_k=8, include_graph=True)
_session_store = SessionStore()
_quality_evaluator = QualityEvaluator()
_semantic_cache = get_semantic_answer_cache()
_cost_tracker = CostTracker()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


async def _get_completed_repo(repo_id: str, db: AsyncSession) -> Repository:
    try:
        repo_uuid = uuid.UUID(repo_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid repository id '{repo_id}'.",
        ) from exc

    result = await db.execute(select(Repository).where(Repository.id == repo_uuid))
    repo = result.scalar_one_or_none()

    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' not found. Ingest it first with POST /api/v1/ingest.",
        )

    if repo.status != IngestionStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Repository '{repo.name}' is not yet fully indexed. "
                f"Current status: {repo.status.value}."
            ),
        )

    return repo


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a natural language question about an ingested repository.",
)
async def ask_question(request: AskRequest, db: AsyncSession = Depends(get_db)):
    repo = await _get_completed_repo(request.repo_id, db)

    task_type: TaskType | None = None
    if request.task_type:
        task_type = TaskType(request.task_type)

    logger.info(
        "ask_endpoint_called",
        repo_id=request.repo_id,
        repo_name=repo.name,
        task_type=request.task_type or "auto",
        stream=request.stream,
        question_len=len(request.question),
    )

    session_id = request.session_id or _session_store.create_session()
    if request.session_id and not _session_store.session_exists(request.session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation session '{request.session_id}' not found.",
        )

    history_turns = _session_store.get_turns(session_id)
    history_block = build_history_block(history_turns)

    _session_store.append_turn(
        session_id=session_id,
        role="user",
        content=request.question,
    )

    if request.stream:
        return StreamingResponse(
            _stream_ask_with_session(
                request=request,
                session_id=session_id,
                history_block=history_block,
                store=_session_store,
                db=db,
                task_type=task_type,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    normalized_question = request.question.strip()
    cache_similarity = 0.0
    semantic_hit = False

    try:
        from embeddings.gemini_embedder import embed_query

        question_vector = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: embed_query(normalized_question),
        )
    except Exception:
        question_vector = None

    if question_vector is not None:
        cache_result = await _semantic_cache.lookup(question_vector=question_vector, repo_id=request.repo_id)
        if cache_result.found and cache_result.cached_answer is not None:
            semantic_hit = True
            cache_similarity = cache_result.similarity
            cached_answer = cache_result.cached_answer

            answer_text = cached_answer.answer
            quality_score = cached_answer.quality_score

            _session_store.append_turn(
                session_id=session_id,
                role="assistant",
                content=answer_text,
                sources=cached_answer.sources[:5],
                provider_used=cached_answer.provider_used,
                model_used=cached_answer.model_used,
            )

            return AskResponse(
                request_id=str(uuid.uuid4())[:8],
                answer=answer_text,
                session_id=session_id,
                provider_used=cached_answer.provider_used or "semantic-cache",
                model_used=cached_answer.model_used or "semantic-cache",
                task_type=(task_type.value if task_type else get_task_type_from_question(normalized_question).value),
                sources=[
                    SourceReference(
                        file_path=str(source.get("file") or source.get("file_path") or ""),
                        function_name=str(source.get("function") or source.get("function_name") or ""),
                        start_line=int(source.get("start_line") or 0),
                        end_line=int(source.get("end_line") or 0),
                        score=float(source.get("score") or 0.0),
                        chunk_type=str(source.get("chunk_type") or ""),
                    )
                    for source in cached_answer.sources
                ],
                graph_path=[],
                context_chunks_used=len(cached_answer.sources),
                estimated_tokens=_estimate_tokens(answer_text),
                vector_search_ms=0.0,
                graph_expansion_ms=0.0,
                total_latency_ms=0.0,
                top_result_score=cache_similarity,
                quality_score=quality_score,
                cached=True,
                cache_similarity=cache_similarity,
                intent=(task_type.value if task_type else None),
            )

    try:
        rag_response = await _rag_chain.answer(
            repo_id=request.repo_id,
            question=request.question,
            task_type=task_type,
            top_k=request.top_k,
            include_graph=request.include_graph,
            language_filter=request.language_filter,
            chunk_type_filter=request.chunk_type_filter,
            history_block=history_block,
            db=db,
        )
    except LLMProviderError as exc:
        logger.error("ask_endpoint_llm_failed", error=str(exc), repo_id=request.repo_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to generate answer: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("ask_endpoint_error", error=str(exc), repo_id=request.repo_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {exc}",
        ) from exc

    context_chunks = [
        f"{source.file_path}:{source.start_line}-{source.end_line}\n{source.function_name}"
        for source in rag_response.sources[:8]
    ]
    quality = await _quality_evaluator.score(
        question=request.question,
        answer=rag_response.answer,
        context_chunks=context_chunks,
    )
    quality_payload = quality.model_dump()

    if question_vector is not None:
        await _semantic_cache.store(
            question_vector=question_vector,
            answer=rag_response.answer,
            sources=[
                {
                    "file_path": source.file_path,
                    "function_name": source.function_name,
                    "start_line": source.start_line,
                    "end_line": source.end_line,
                    "score": source.score,
                    "chunk_type": source.chunk_type,
                }
                for source in rag_response.sources
            ],
            quality_score=quality_payload,
            repo_id=request.repo_id,
            provider_used=rag_response.provider_used,
            model_used=rag_response.model_used,
        )

    asyncio.create_task(update_running_averages(quality))

    try:
        await _cost_tracker.record(
            provider=rag_response.provider_used,
            model=rag_response.model_used,
            input_tokens=max(1, rag_response.estimated_tokens // 2),
            output_tokens=_estimate_tokens(rag_response.answer),
            check_budget=True,
        )
    except BudgetExceededError:
        raise
    except Exception:
        # Cost accounting should not break successful answer responses.
        pass

    _session_store.append_turn(
        session_id=session_id,
        role="assistant",
        content=rag_response.answer,
        sources=[
            {
                "file": source.file_path,
                "function": source.function_name,
                "lines": f"{source.start_line}-{source.end_line}",
            }
            for source in rag_response.sources[:5]
        ],
        provider_used=rag_response.provider_used,
        model_used=rag_response.model_used,
    )

    return AskResponse(
        request_id=rag_response.request_id,
        answer=rag_response.answer,
        session_id=session_id,
        provider_used=rag_response.provider_used,
        model_used=rag_response.model_used,
        task_type=rag_response.task_type,
        sources=[
            SourceReference(
                file_path=source.file_path,
                function_name=source.function_name,
                start_line=source.start_line,
                end_line=source.end_line,
                score=source.score,
                chunk_type=source.chunk_type,
            )
            for source in rag_response.sources
        ],
        graph_path=rag_response.graph_path,
        context_chunks_used=rag_response.context_chunks_used,
        estimated_tokens=rag_response.estimated_tokens,
        vector_search_ms=rag_response.vector_search_ms,
        graph_expansion_ms=rag_response.graph_expansion_ms,
        total_latency_ms=rag_response.total_latency_ms,
        top_result_score=rag_response.top_result_score,
        quality_score=quality_payload,
        cached=semantic_hit,
        cache_similarity=cache_similarity,
        intent=rag_response.task_type,
    )


@router.get("/ask/providers", summary="List currently available LLM providers.")
async def list_providers() -> dict:
    providers = get_available_providers()
    return {"providers": [provider.value for provider in providers], "count": len(providers)}


@router.get("/conversation/{session_id}", summary="Get full conversation history")
async def get_conversation(session_id: str) -> dict:
    if not _session_store.session_exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation session '{session_id}' not found.",
        )

    turns = _session_store.get_turns(session_id)
    return {
        "session_id": session_id,
        "turn_count": len(turns),
        "turns": turns,
    }


@router.delete("/conversation/{session_id}", summary="Delete conversation history")
async def delete_conversation(session_id: str) -> dict:
    if not _session_store.delete_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation session '{session_id}' not found.",
        )

    return {
        "session_id": session_id,
        "deleted": True,
    }


@router.post(
    "/ask/deep",
    response_model=DeepAskResponse,
    summary="Multi-step deep ask for compound questions",
)
async def deep_ask(request: DeepAskRequest, db: AsyncSession = Depends(get_db)) -> DeepAskResponse:
    await _get_completed_repo(request.repo_id, db)

    started = _time.perf_counter()

    decomposer = QueryDecomposer()
    classifier = IntentClassifier()
    synthesizer = AnswerSynthesizer()

    sub_questions = await decomposer.decompose(request.question)
    if not sub_questions:
        sub_questions = [request.question]

    async def _run_sub_question(sub_question: str) -> tuple[str, list[dict], str]:
        intent = await classifier.classify(sub_question)
        model = get_model_for_intent(intent)
        system_prompt = get_system_prompt(intent)
        temperature = get_temperature_for_intent(intent)

        async with AsyncSessionLocal() as sub_db:
            response = await _rag_chain.answer(
                repo_id=request.repo_id,
                question=sub_question,
                task_type=None,
                top_k=8,
                include_graph=True,
                model_override=model,
                system_prompt_override=system_prompt,
                temperature_override=temperature,
                db=sub_db,
            )

        sources = [
            {
                "file": source.file_path,
                "function": source.function_name,
                "lines": f"{source.start_line}-{source.end_line}",
            }
            for source in response.sources
        ]
        return response.answer, sources, response.provider_used

    partial_answers: list[str] = []
    all_sources: list[dict] = []
    providers_used: list[str] = []

    results = await asyncio.gather(
        *[_run_sub_question(sub_question) for sub_question in sub_questions],
        return_exceptions=True,
    )

    for sub_question, result in zip(sub_questions, results):
        if isinstance(result, Exception):
            logger.warning(
                "deep_ask_sub_question_failed",
                repo_id=request.repo_id,
                sub_question=sub_question,
                error=str(result),
            )
            partial_answers.append(f"Failed to answer sub-question: {sub_question}")
            continue

        answer_text, sources, provider_used = result
        partial_answers.append(answer_text)
        all_sources.extend(sources)
        if provider_used:
            providers_used.append(provider_used)

    final_answer = await synthesizer.synthesize(
        original_question=request.question,
        sub_questions=sub_questions,
        partial_answers=partial_answers,
    )

    deduped_sources: list[dict] = []
    seen: set[str] = set()
    for source in all_sources:
        key = f"{source.get('file', '')}:{source.get('function', '')}:{source.get('lines', '')}"
        if key in seen:
            continue
        seen.add(key)
        deduped_sources.append(source)

    total_ms = (_time.perf_counter() - started) * 1000
    return DeepAskResponse(
        answer=final_answer,
        sub_questions=sub_questions,
        partial_answers=partial_answers,
        all_sources=deduped_sources,
        providers_used=list(dict.fromkeys(providers_used)),
        decomposed=len(sub_questions) > 1,
        total_ms=total_ms,
    )


async def _stream_ask_with_session(
    request: AskRequest,
    session_id: str,
    history_block: str,
    store: SessionStore,
    db: AsyncSession,
    task_type: TaskType | None,
) -> AsyncIterator[str]:
    """Full streaming pipeline with stage events and token events."""
    t_start = _time.perf_counter()
    done_marker = "data: [DONE]\\n\\n"

    def emit(event: str, payload: dict) -> str:
        return f"event: {event}\\ndata: {_json.dumps(payload)}\\n\\n"

    yield emit("step", {"stage": "searching", "message": "Searching code context..."})

    resolved_task = task_type or get_task_type_from_question(request.question)
    assembler = ContextAssembler()
    config = ContextAssemblerConfig(
        top_k=request.top_k,
        include_graph=request.include_graph,
        language_filter=request.language_filter,
        chunk_type_filter=request.chunk_type_filter,
    )

    try:
        context = await assembler.assemble(
            repo_id=request.repo_id,
            question=request.question,
            config=config,
            db=db,
        )
    except Exception as exc:
        logger.error("ask_stream_context_failed", error=str(exc))
        yield emit("error", {"error": f"Context assembly failed: {exc}"})
        yield done_marker
        return

    yield emit(
        "step",
        {
            "stage": "searching",
            "message": f"Found {context.chunks_used} relevant chunks in {int(context.vector_search_ms)}ms.",
        },
    )

    yield emit(
        "step",
        {
            "stage": "graph",
            "message": (
                f"Graph expanded to {len(context.graph.call_chain)} call-chain nodes "
                f"in {int(context.graph_expansion_ms)}ms."
            ),
        },
    )

    built_prompt = build_prompt(
        task_type=resolved_task,
        question=request.question,
        context_chunks=context.chunks,
        graph_context=context.graph,
        repo_name=context.repo_name,
        history_block=history_block,
    )

    yield emit("step", {"stage": "context", "message": "Context ready. Routing to LLM..."})
    yield emit("step", {"stage": "generating", "message": "Generating answer..."})

    provider_used = ""
    model_used = ""
    full_answer: list[str] = []

    try:
        async for chunk, provider, model in stream_ask(
            task_type=resolved_task,
            prompt=built_prompt.user_prompt,
            system_prompt=built_prompt.system_prompt,
        ):
            if provider is not None:
                provider_used = provider.value
            if model:
                model_used = model

            full_answer.append(chunk)
            yield emit("token", {"text": chunk})

    except LLMProviderError as exc:
        logger.error("ask_stream_llm_failed", error=str(exc))
        yield emit("error", {"error": f"LLM generation failed: {exc}"})
        yield done_marker
        return
    except Exception as exc:
        logger.error("ask_stream_unexpected_failed", error=str(exc))
        yield emit("error", {"error": f"Streaming failed: {exc}"})
        yield done_marker
        return

    answer_text = "".join(full_answer)
    sources = [
        {
            "file": chunk.file_path,
            "function": chunk.display_name,
            "lines": f"{chunk.start_line}-{chunk.end_line}",
        }
        for chunk in context.chunks[:5]
    ]

    store.append_turn(
        session_id=session_id,
        role="assistant",
        content=answer_text,
        sources=sources,
        provider_used=provider_used,
        model_used=model_used,
    )

    timing = {
        "search_ms": int(context.vector_search_ms),
        "graph_ms": int(context.graph_expansion_ms),
        "context_ms": 0,
        "total_ms": int((_time.perf_counter() - t_start) * 1000),
    }
    yield emit(
        "done",
        {
            "provider": provider_used,
            "model": model_used,
            "session_id": session_id,
            "timing": timing,
        },
    )
    yield done_marker

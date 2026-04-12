"""
Search endpoint — semantic code search using Gemini + Qdrant.

Endpoints:
  GET /api/v1/search
        ?q=        natural language query (required)
        &repo_id=  repository UUID (required)
        &top_k=    number of results, 1-20 (default 5)
        &language= filter by language (optional)
        &chunk_type= filter by type (optional)

  GET /api/v1/search/stats/{repo_id}
        Returns embedding progress stats for a repository.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

from db.database import get_db
from db.models import Repository
from core.logging import get_logger

router = APIRouter(prefix="/api/v1", tags=["search"])
logger = get_logger(__name__)


@router.get("/search")
async def semantic_search(
    q: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language search query",
    ),
    repo_id: UUID = Query(
        ...,
        description="Repository UUID to search within",
    ),
    top_k: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (1-20)",
    ),
    language: str | None = Query(
        default=None,
        description="Filter results by language (python, javascript, go, etc.)",
    ),
    chunk_type: str | None = Query(
        default=None,
        description="Filter by chunk type: function, class, method",
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Semantic code search.

    Converts your query to a Gemini embedding, searches Qdrant for
    the most similar code chunks in the specified repository,
    and returns ranked results with file paths and line numbers.

    Example:
        GET /api/v1/search?q=password+hashing&repo_id=...&top_k=5
    """
    # Verify the repository exists
    repo_result = await db.execute(
        select(Repository).where(Repository.id == repo_id)
    )
    repo = repo_result.scalar_one_or_none()

    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")

    if repo.status.value not in ("completed",):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Repository '{repo.name}' is not ready for search. "
                f"Current status: {repo.status.value}. "
                f"Wait for status to be 'completed'."
            ),
        )

    try:
        # Step 1: Embed the user's query using Gemini
        logger.info("search_start", query=q, repo_id=str(repo_id))

        from embeddings.gemini_embedder import embed_query
        query_vector = embed_query(q)

        # Step 2: Search Qdrant for similar vectors
        from embeddings.vector_store import search
        scored_points = search(
            query_vector=query_vector,
            repo_id=str(repo_id),
            top_k=top_k,
            language=language,
            chunk_type=chunk_type,
            score_threshold=0.2,  # lower threshold for broader results
        )

        # Step 3: Format results for the API response
        results = []
        for point in scored_points:
            p = point.payload or {}
            results.append({
                "chunk_id":       str(point.id),
                "score":          round(point.score, 4),
                "name":           p.get("name", ""),
                "display_name":   p.get("display_name", ""),
                "chunk_type":     p.get("chunk_type", ""),
                "file_path":      p.get("file_path", ""),
                "language":       p.get("language", ""),
                "start_line":     p.get("start_line", 0),
                "end_line":       p.get("end_line", 0),
                "docstring":      p.get("docstring", ""),
                "content_preview": p.get("content_preview", ""),
            })

        logger.info(
            "search_complete",
            query=q,
            repo_id=str(repo_id),
            results_count=len(results),
            top_score=results[0]["score"] if results else 0,
        )

        return {
            "query":         q,
            "repo_id":       str(repo_id),
            "repo_name":     repo.name,
            "total_results": len(results),
            "filters_applied": {
                "language":   language,
                "chunk_type": chunk_type,
            },
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("search_failed", query=q, error=str(e), repo_id=str(repo_id))
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/search/stats/{repo_id}")
async def get_search_stats(
    repo_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns embedding progress statistics for a repository.

    Use this to check:
      - How many chunks have been embedded
      - Whether Qdrant and PostgreSQL are in sync
      - Overall embedding progress percentage
    """
    repo_result = await db.execute(
        select(Repository).where(Repository.id == repo_id)
    )
    repo = repo_result.scalar_one_or_none()

    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found.")

    try:
        from embeddings.embedding_pipeline import get_embedding_stats
        stats = get_embedding_stats(str(repo_id))
    except Exception as e:
        stats = {"error": str(e)}

    return {
        "repo_id":   str(repo_id),
        "repo_name": repo.name,
        "status":    repo.status.value,
        **stats,
    }
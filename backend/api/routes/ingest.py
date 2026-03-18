from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from db.models import Repository, SourceFile, CodeChunk, IngestionStatus
import uuid

from db.database import get_db
from db.models import Repository, IngestionStatus
from api.schemas.ingest_schema import IngestRequest, IngestResponse, StatusResponse
from tasks.ingest_task import run_ingestion_task
from core.logging import get_logger

router = APIRouter(prefix="/api/v1", tags=["ingestion"])
logger = get_logger(__name__)


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_repository(request: IngestRequest, db: AsyncSession = Depends(get_db)):
    repo_name = request.github_url.rstrip("/").split("/")[-1]
    repo_id = uuid.uuid4()

    existing = await db.execute(select(Repository).where(Repository.github_url == request.github_url))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Repository already ingested.")

    repo = Repository(
        id=repo_id,
        github_url=request.github_url,
        name=repo_name,
        branch=request.branch,
        status=IngestionStatus.QUEUED,
    )
    db.add(repo)
    await db.flush()

    task = run_ingestion_task.delay(
        repo_id=str(repo_id),
        github_url=request.github_url,
        branch=request.branch,
    )

    repo.task_id = task.id
    await db.commit()

    return IngestResponse(
        repo_id=repo_id,
        task_id=task.id,
        status="queued",
        message=f"Repository '{repo_name}' queued for ingestion.",
    )


@router.get("/status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str, db: AsyncSession = Depends(get_db)):
    from celery.result import AsyncResult
    from tasks.celery_app import celery_app
    from sqlalchemy import func

    result = AsyncResult(task_id, app=celery_app)

    # ── Query 1: get repo record ──────────────────────────
    repo_result = await db.execute(
        select(Repository).where(Repository.task_id == task_id)
    )
    repo = repo_result.scalar_one_or_none()

    if repo is None and result.state == "PENDING":
        raise HTTPException(status_code=404, detail="Task not found.")

    meta = result.info or {}

    # ── Query 2: count chunks — use fresh session ─────────
    total_chunks = 0
    if repo:
        # Flush and expire before second query to avoid asyncpg conflict
        await db.commit()

        chunk_result = await db.execute(
            select(func.count(CodeChunk.id))
            .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
            .where(SourceFile.repository_id == repo.id)
        )
        total_chunks = chunk_result.scalar() or 0

    return StatusResponse(
        task_id=task_id,
        status=repo.status.value if repo else result.state.lower(),
        progress=meta.get("progress", 0) if isinstance(meta, dict) else 0,
        message=meta.get("message", "") if isinstance(meta, dict) else "",
        repo_id=str(repo.id) if repo else None,
        error=str(meta) if result.state == "FAILURE" else None,
        total_files=repo.total_files if repo else 0,
        processed_files=repo.processed_files if repo else 0,
        total_chunks=total_chunks,
    )
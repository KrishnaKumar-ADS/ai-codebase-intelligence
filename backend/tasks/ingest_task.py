import asyncio
from celery.utils.log import get_task_logger
from sqlalchemy import select

from tasks.celery_app import celery_app
from db.database import AsyncSessionLocal
from db.models import Repository, SourceFile, IngestionStatus
from ingestion.repo_loader import clone_repository
from ingestion.file_scanner import scan_repository
from core.exceptions import RepoNotFoundError

logger = get_task_logger(__name__)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="tasks.ingest_task.run_ingestion_task", max_retries=3, soft_time_limit=600)
def run_ingestion_task(self, repo_id: str, github_url: str, branch: str = "main"):
    logger.info(f"Starting ingestion: {repo_id}")

    try:
        self.update_state(state="STARTED", meta={"progress": 5, "message": "Cloning repository..."})
        _run_async(_set_status(repo_id, IngestionStatus.CLONING))

        repo_path = clone_repository(github_url, repo_id, branch)

        self.update_state(state="STARTED", meta={"progress": 30, "message": "Scanning files..."})
        _run_async(_set_status(repo_id, IngestionStatus.SCANNING))

        scanned_files = scan_repository(repo_path)

        self.update_state(state="STARTED", meta={"progress": 60, "message": f"Storing {len(scanned_files)} files..."})
        _run_async(_persist_files(repo_id, scanned_files))

        _run_async(_set_status(repo_id, IngestionStatus.COMPLETED, total=len(scanned_files), processed=len(scanned_files)))
        logger.info(f"Ingestion complete: {repo_id}, files: {len(scanned_files)}")
        return {"repo_id": repo_id, "total_files": len(scanned_files)}

    except RepoNotFoundError as e:
        _run_async(_set_status(repo_id, IngestionStatus.FAILED, error=str(e)))
        raise

    except Exception as e:
        _run_async(_set_status(repo_id, IngestionStatus.FAILED, error=str(e)))
        raise self.retry(exc=e, countdown=30)


async def _set_status(repo_id, status, total=0, processed=0, error=None):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Repository).where(Repository.id == repo_id))
        repo = result.scalar_one_or_none()
        if repo:
            repo.status = status
            if total: repo.total_files = total
            if processed: repo.processed_files = processed
            if error: repo.error_message = error
            await session.commit()


async def _persist_files(repo_id, scanned_files):
    async with AsyncSessionLocal() as session:
        for f in scanned_files:
            session.add(SourceFile(
                repository_id=repo_id,
                file_path=f.relative_path,
                language=f.language,
                size_bytes=f.size_bytes,
                line_count=f.line_count,
            ))
        await session.commit()
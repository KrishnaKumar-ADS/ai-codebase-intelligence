import sys
import os

# Ensure backend/ is on the path so all modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celery.utils.log import get_task_logger
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session

from tasks.celery_app import celery_app
from db.models import Repository, SourceFile, CodeChunk, IngestionStatus
from ingestion.repo_loader import clone_repository
from ingestion.file_scanner import scan_repository
from parsing.metadata_extractor import extract_metadata   # ← moved to top
from core.config import get_settings
from core.exceptions import RepoNotFoundError

logger = get_task_logger(__name__)
settings = get_settings()
logger = get_task_logger(__name__)
settings = get_settings()

# ── Sync engine for Celery tasks ──────────────────────────────
# Celery workers are sync — we use psycopg2 (sync) not asyncpg
sync_engine = create_engine(
    settings.sync_database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)
SyncSessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


def _get_sync_session() -> Session:
    return SyncSessionLocal()


# ── Celery Task ───────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.ingest_task.run_ingestion_task",
    max_retries=3,
    soft_time_limit=900,
)
def run_ingestion_task(self, repo_id: str, github_url: str, branch: str = "main"):
    logger.info(f"[{repo_id}] Starting ingestion pipeline")

    try:
        # ── Stage 1: Clone ─────────────────────────────────────
        _update_state(self, 5, "Cloning repository...")
        _set_status(repo_id, IngestionStatus.CLONING)

        repo_path = clone_repository(github_url, repo_id, branch)
        logger.info(f"[{repo_id}] Cloned to {repo_path}")

        # ── Stage 2: Scan ──────────────────────────────────────
        _update_state(self, 20, "Scanning source files...")
        _set_status(repo_id, IngestionStatus.SCANNING)

        scanned_files = scan_repository(repo_path)
        total_files = len(scanned_files)
        logger.info(f"[{repo_id}] Found {total_files} source files")

        if total_files == 0:
            _set_status(repo_id, IngestionStatus.COMPLETED)
            return {"repo_id": repo_id, "total_files": 0, "total_chunks": 0}

        # ── Stage 3: Parse + Store ─────────────────────────────
        _update_state(self, 30, f"Parsing {total_files} files...")
        _set_status(repo_id, IngestionStatus.PARSING)

        total_chunks = 0

        for i, scanned in enumerate(scanned_files):
            if i % 10 == 0:
                progress = 30 + int((i / total_files) * 60)
                _update_state(
                    self,
                    progress,
                    f"Parsing file {i+1}/{total_files}: {scanned.relative_path}",
                )

            try:
                from parsing.metadata_extractor import extract_metadata
                chunks = extract_metadata(
                    file_path=scanned.path,
                    relative_path=scanned.relative_path,
                    language=scanned.language,
                )
            except Exception as e:
                logger.warning(f"[{repo_id}] Failed to parse {scanned.relative_path}: {e}")
                chunks = []

            chunk_count = _store_file_and_chunks(repo_id, scanned, chunks)
            total_chunks += chunk_count

        # ── Done ───────────────────────────────────────────────
        _set_status(
            repo_id,
            IngestionStatus.COMPLETED,
            total_files=total_files,
            processed_files=total_files,
        )
        _update_state(self, 100, f"Complete. {total_files} files, {total_chunks} chunks.")

        logger.info(f"[{repo_id}] Done — {total_files} files, {total_chunks} chunks")
        return {"repo_id": repo_id, "total_files": total_files, "total_chunks": total_chunks}

    except RepoNotFoundError as e:
        logger.error(f"[{repo_id}] Repo not found: {e}")
        _set_status(repo_id, IngestionStatus.FAILED, error=str(e))
        raise

    except Exception as e:
        logger.error(f"[{repo_id}] Pipeline failed: {e}", exc_info=True)
        try:
            _set_status(repo_id, IngestionStatus.FAILED, error=str(e))
        except Exception:
            pass
        raise self.retry(exc=e, countdown=30)


# ── Sync DB helpers ───────────────────────────────────────────

def _set_status(
    repo_id: str,
    status: IngestionStatus,
    total_files: int = 0,
    processed_files: int = 0,
    error: str | None = None,
) -> None:
    """Update repository status using sync session — safe for Celery."""
    session = _get_sync_session()
    try:
        repo = session.execute(
            select(Repository).where(Repository.id == repo_id)
        ).scalar_one_or_none()

        if repo:
            repo.status = status
            if total_files:
                repo.total_files = total_files
            if processed_files:
                repo.processed_files = processed_files
            if error:
                repo.error_message = error[:500]   # truncate very long errors
            session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"_set_status failed: {e}")
    finally:
        session.close()


def _store_file_and_chunks(repo_id: str, scanned, chunks: list) -> int:
    """Store one SourceFile and its CodeChunks. Returns chunk count."""
    session = _get_sync_session()
    try:
        source_file = SourceFile(
            repository_id=repo_id,
            file_path=scanned.relative_path,
            language=scanned.language,
            size_bytes=scanned.size_bytes,
            line_count=scanned.line_count,
        )
        session.add(source_file)
        session.flush()   # get source_file.id without full commit

        for chunk in chunks:
            code_chunk = CodeChunk(
                source_file_id=source_file.id,
                chunk_type=chunk.chunk_type,
                name=chunk.name,
                display_name=chunk.display_name,
                content=chunk.content,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                docstring=chunk.docstring,
                parent_name=chunk.parent_name,
            )
            session.add(code_chunk)

        session.commit()
        return len(chunks)

    except Exception as e:
        session.rollback()
        logger.error(f"_store_file_and_chunks failed for {scanned.relative_path}: {e}")
        return 0
    finally:
        session.close()


def _update_state(task, progress: int, message: str) -> None:
    task.update_state(
        state="STARTED",
        meta={"progress": progress, "message": message},
    )
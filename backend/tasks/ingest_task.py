"""
Ingestion Task — full pipeline from GitHub URL to stored chunks.
"""

import sys
import os
from pathlib import Path

from celery.utils.log import get_task_logger
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from tasks.celery_app import celery_app
from db.models import Repository, SourceFile, CodeChunk, IngestionStatus
from ingestion.repo_loader import clone_repository
from ingestion.file_scanner import scan_repository, ScannedFile
from parsing.metadata_extractor import extract_metadata
from core.config import get_settings
from core.exceptions import RepoNotFoundError

# Ensure sibling packages like `embeddings` are always importable in worker runtime.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

logger = get_task_logger(__name__)
settings = get_settings()

sync_engine = create_engine(
    settings.sync_database_url,
    pool_pre_ping=True,
    echo=False,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)


# ✅ FIX 1: Define missing function
def _update_state(task, progress: int, message: str):
    task.update_state(
        state="STARTED",
        meta={"progress": progress, "message": message},
    )


def _set_status(
    repo_id: str,
    status: IngestionStatus,
    total_files: int = 0,
    processed_files: int = 0,
    error: str | None = None,
):
    with SyncSessionLocal() as session:
        repo = session.execute(
            select(Repository).where(Repository.id == repo_id)
        ).scalar_one_or_none()

        if not repo:
            raise RepoNotFoundError(f"Repository {repo_id} not found")

        repo.status = status

        if total_files:
            repo.total_files = total_files

        if processed_files:
            repo.processed_files = processed_files

        if error:
            repo.error_message = error

        session.commit()


def _store_file_and_chunks(repo_id: str, scanned: ScannedFile, chunks: list) -> int:
    with SyncSessionLocal() as session:
        source_file = SourceFile(
            repository_id=repo_id,
            file_path=scanned.relative_path,
            language=scanned.language,
            size_bytes=scanned.size_bytes,
            line_count=scanned.line_count,
        )

        session.add(source_file)
        session.flush()

        for chunk in chunks:
            session.add(
                CodeChunk(
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
            )

        session.commit()
        return len(chunks)


@celery_app.task(
    bind=True,
    name="tasks.ingest_task.run_ingestion_task",
    max_retries=3,
    soft_time_limit=900,
)
def run_ingestion_task(self, repo_id: str, github_url: str, branch: str = "main"):
    logger.info(f"[{repo_id}] Starting ingestion pipeline")

    try:
        # ── Stage 1: Clone ─────────────────────────────
        _update_state(self, 5, "Cloning repository...")
        _set_status(repo_id, IngestionStatus.CLONING)

        repo_path = clone_repository(github_url, repo_id, branch)
        logger.info(f"[{repo_id}] Cloned to {repo_path}")

        # ── Stage 2: Scan ─────────────────────────────
        _update_state(self, 20, "Scanning source files...")
        _set_status(repo_id, IngestionStatus.SCANNING)

        scanned_files = scan_repository(repo_path)
        total_files = len(scanned_files)

        logger.info(f"[{repo_id}] Found {total_files} files")

        if total_files == 0:
            _set_status(repo_id, IngestionStatus.COMPLETED)
            return {
                "repo_id": repo_id,
                "total_files": 0,
                "total_chunks": 0,
            }

        # ── Stage 3: Parse + Store ─────────────────────
        _update_state(self, 30, f"Parsing {total_files} files...")
        _set_status(repo_id, IngestionStatus.PARSING)

        total_chunks = 0

        for i, scanned in enumerate(scanned_files):
            if i % 10 == 0:
                progress = 30 + int((i / max(total_files, 1)) * 35)
                _update_state(
                    self,
                    progress,
                    f"Parsing {i+1}/{total_files}: {scanned.relative_path}",
                )

            try:
                chunks = extract_metadata(
                    file_path=scanned.path,
                    relative_path=scanned.relative_path,
                    language=scanned.language,
                )
            except Exception as e:
                logger.warning(f"[{repo_id}] Parse failed: {scanned.relative_path} → {e}")
                chunks = []

            total_chunks += _store_file_and_chunks(repo_id, scanned, chunks)

        logger.info(f"[{repo_id}] Stored {total_chunks} chunks")

        # ── Stage 4: Embedding ─────────────────────────
        _update_state(self, 65, f"Embedding {total_chunks} chunks...")
        _set_status(repo_id, IngestionStatus.EMBEDDING)

        total_embedded = 0
        embedding_error: str | None = None

        try:
            # Guard against worker runtime CWD/sys.path drift before dynamic import.
            os.chdir(str(BACKEND_ROOT))
            if str(BACKEND_ROOT) not in sys.path:
                sys.path.insert(0, str(BACKEND_ROOT))
            from embeddings.embedding_pipeline import embed_repository

            def embedding_progress(done, chunk_total, message):
                if chunk_total > 0:
                    progress = 65 + int((done / chunk_total) * 30)
                    _update_state(self, progress, message)

            result = embed_repository(
                repo_id=repo_id,
                progress_callback=embedding_progress,
            )

            total_embedded = result.get("embedded", 0)

            attempted_chunks = result.get("total_chunks", total_chunks) - result.get("already_embedded", 0)
            if attempted_chunks > 0 and total_embedded == 0:
                embedding_error = (
                    "Embedding produced 0 vectors. "
                    "Check Gemini embedding model/API key and Qdrant connectivity."
                )
            elif result.get("failed", 0) > 0:
                logger.warning(
                    f"[{repo_id}] Partial embedding failures: "
                    f"embedded={total_embedded}, failed={result.get('failed', 0)}"
                )

        except Exception as e:
            embedding_error = str(e)

        if embedding_error:
            logger.error(f"[{repo_id}] Embedding failed: {embedding_error}")
            failure_message = f"Failed during embedding: {embedding_error}"
            _set_status(
                repo_id,
                IngestionStatus.FAILED,
                total_files=total_files,
                processed_files=total_files,
                error=embedding_error,
            )
            _update_state(self, 100, failure_message)
            return {
                "repo_id": repo_id,
                "status": "failed",
                "error": embedding_error,
                "progress": 100,
                "message": failure_message,
                "total_files": total_files,
                "total_chunks": total_chunks,
                "total_embedded": total_embedded,
            }

        # ── Done ───────────────────────────────────────
        _set_status(
            repo_id,
            IngestionStatus.COMPLETED,
            total_files=total_files,
            processed_files=total_files,
        )

        _update_state(
            self,
            100,
            f"Done: {total_files} files, {total_chunks} chunks, {total_embedded} embedded",
        )

        logger.info(f"[{repo_id}] Pipeline completed")

        return {
            "repo_id": repo_id,
            "total_files": total_files,
            "total_chunks": total_chunks,
            "total_embedded": total_embedded,
        }

    except RepoNotFoundError as e:
        logger.error(f"[{repo_id}] Repo not found: {e}")
        raise

    except Exception as e:
        logger.error(f"[{repo_id}] FAILED: {e}")

        try:
            _set_status(repo_id, IngestionStatus.FAILED, error=str(e))
        except Exception:
            logger.error(f"[{repo_id}] Failed to update status")

        raise self.retry(exc=e)
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
import structlog


def add_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.time()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        duration = round((time.time() - start) * 1000, 2)
        structlog.get_logger().info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration,
        )
        structlog.contextvars.clear_contextvars()
        return response
import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        content_length = request.headers.get("content-length")
        query_length = int(content_length) if content_length else 0

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        status_code = response.status_code

        log_data = {
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "query_length": query_length,
        }

        if status_code >= 500:
            logger.error("request completed", extra=log_data)
        elif status_code >= 400:
            logger.warning("request completed", extra=log_data)
        else:
            logger.info("request completed", extra=log_data)

        return response

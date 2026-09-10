import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app.core.config import get_settings

API_KEY_HEADER = "x-api-key"
RATE_LIMIT_WINDOW_SECONDS = 60


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int, now: float | None = None) -> None:
        if limit <= 0:
            return
        timestamp = now if now is not None else time.monotonic()
        window_start = timestamp - RATE_LIMIT_WINDOW_SECONDS
        timestamps = self._requests[key]
        while timestamps and timestamps[0] <= window_start:
            timestamps.popleft()
        if len(timestamps) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")
        timestamps.append(timestamp)

    def reset(self) -> None:
        self._requests.clear()


chat_rate_limiter = InMemoryRateLimiter()


def require_chat_api_key(request: Request) -> str:
    settings = get_settings()
    configured_key = settings.chat_api_key
    if not configured_key:
        return "local-dev"
    supplied_key = request.headers.get(API_KEY_HEADER, "")
    if supplied_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return supplied_key


def enforce_chat_rate_limit(request: Request, identity: str) -> None:
    settings = get_settings()
    client_host = request.client.host if request.client else "unknown-client"
    limiter_key = identity if identity != "local-dev" else client_host
    chat_rate_limiter.check(limiter_key, settings.chat_rate_limit_per_minute)

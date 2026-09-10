import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolate_runtime_secrets(monkeypatch):
    from app.core.config import get_settings
    from app.core.security import chat_rate_limiter

    monkeypatch.setenv("CHAT_API_KEY", "")
    monkeypatch.setenv("INGEST_API_KEY", "")
    get_settings.cache_clear()
    chat_rate_limiter.reset()
    yield
    get_settings.cache_clear()
    chat_rate_limiter.reset()

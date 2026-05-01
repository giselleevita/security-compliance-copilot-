import os

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.integration
def test_chat_live_end_to_end_with_real_dependencies() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set.")

    client = TestClient(app)
    health = client.get("/health")
    if health.status_code != 200:
        pytest.skip("Health endpoint unavailable for live test.")
    indexed = int(health.json().get("indexed_chunks", 0))
    if indexed <= 0:
        pytest.skip("No indexed corpus found; run ingestion first.")

    response = client.post(
        "/chat",
        json={"question": "What does NIST AI RMF recommend for governance roles and accountability?"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert isinstance(payload.get("answer"), str)
    assert payload.get("answer", "").strip()
    assert payload.get("guardrail_status") in {"ok", "insufficient_context", "refused"}
    assert payload.get("confidence") in {"high", "medium", "low"}
    assert isinstance(payload.get("sources"), list)

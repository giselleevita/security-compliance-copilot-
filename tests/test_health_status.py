from fastapi.testclient import TestClient

from app.main import app


def test_health_status_shape() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["indexed_chunks"] >= 0
    assert isinstance(payload["known_sources"], list)
    assert "last_ingest_at" in payload

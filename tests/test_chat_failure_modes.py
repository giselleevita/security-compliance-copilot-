from fastapi.testclient import TestClient

from app.main import app


def test_chat_returns_runtime_error_detail(monkeypatch) -> None:
    class FailingRuntimeService:
        def answer_question(self, question: str, filters: dict[str, str] | None = None):
            raise RuntimeError("OPENAI_API_KEY is required for chat generation.")

    monkeypatch.setattr("app.api.chat.get_chat_service", lambda: FailingRuntimeService())
    client = TestClient(app)
    response = client.post("/chat", json={"question": "What does NIST recommend?"})
    assert response.status_code == 500
    assert response.json()["detail"] == "OPENAI_API_KEY is required for chat generation."


def test_chat_returns_unhandled_exception_detail(monkeypatch) -> None:
    class UnexpectedFailureService:
        def answer_question(self, question: str, filters: dict[str, str] | None = None):
            raise ValueError("Malformed retrieval payload")

    monkeypatch.setattr("app.api.chat.get_chat_service", lambda: UnexpectedFailureService())
    client = TestClient(app)
    response = client.post("/chat", json={"question": "What does NIST recommend?"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Malformed retrieval payload"


def test_chat_returns_fallback_for_unknown_exception(monkeypatch) -> None:
    class UnknownFailureService:
        def answer_question(self, question: str, filters: dict[str, str] | None = None):
            raise Exception("Unexpected chain failure")

    monkeypatch.setattr("app.api.chat.get_chat_service", lambda: UnknownFailureService())
    client = TestClient(app)
    response = client.post("/chat", json={"question": "What does NIST recommend?"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Unexpected chain failure"

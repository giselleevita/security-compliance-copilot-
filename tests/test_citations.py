from fastapi.testclient import TestClient

from app.generation.context_builder import build_context
from app.main import app
from app.models.chat import ChatResponse
from app.models.source import SourceChunk, SourceResult


def make_chunk(chunk_id: str, title: str, score: float, rerank_score: float) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        text=f"Content for {title}",
        source_id=f"src-{chunk_id}",
        title=title,
        url=f"https://example.com/{chunk_id}",
        publisher="NIST",
        source_type="html",
        framework="NIST_AI_RMF",
        section="Govern",
        chunk_index=int(chunk_id),
        score=score,
        rerank_score=rerank_score,
    )


def test_context_builder_assigns_stable_ordered_labels() -> None:
    package = build_context(
        [
            make_chunk("1", "Doc One", 0.8, 0.92),
            make_chunk("2", "Doc Two", 0.75, 0.88),
            make_chunk("3", "Doc Three", 0.7, 0.84),
        ],
        max_chars=5000,
    )

    labels = [chunk.label for chunk in package.chunks]
    assert labels == ["S1", "S2", "S3"]
    assert len(set(labels)) == len(labels)
    assert "[S1]" in package.context_text
    assert "[S2]" in package.context_text
    assert "[S3]" in package.context_text


def test_chat_endpoint_returns_labeled_sources_when_ok(monkeypatch) -> None:
    class FakeChatService:
        def answer_question(self, question: str, filters: dict[str, str] | None = None) -> ChatResponse:
            return ChatResponse(
                answer="NIST emphasizes governance as a cross-cutting function [S1].",
                sources=[
                    SourceResult(
                        label="S1",
                        title="AI RMF 1.0",
                        framework="NIST_AI_RMF",
                        url="https://example.com/ai-rmf",
                        score=0.92,
                    )
                ],
                confidence="high",
                guardrail_status="ok",
            )

    monkeypatch.setattr("app.api.chat.get_chat_service", lambda: FakeChatService())
    client = TestClient(app)

    response = client.post("/chat", json={"question": "What does governance mean?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["guardrail_status"] == "ok"
    assert payload["sources"][0]["label"] == "S1"
    assert payload["sources"][0]["score"] == 0.92

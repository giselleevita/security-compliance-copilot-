from fastapi.testclient import TestClient

from app.generation.service import ChatService
from app.guardrails.rules import GuardrailEngine
from app.main import app
from app.models.chat import ChatResponse, ConfidenceLevel, GuardrailStatus
from app.models.source import SourceChunk


def make_chunk(chunk_id: str, score: float, label: str | None = None) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        text=f"Evidence for {chunk_id}",
        source_id=f"src-{chunk_id}",
        title=f"Doc {chunk_id}",
        url=f"https://example.com/{chunk_id}",
        publisher="NIST",
        source_type="html",
        framework="NIST_AI_RMF",
        section="Govern",
        chunk_index=int(chunk_id),
        score=score,
        rerank_score=score,
        label=label,
    )


class StubRetrievalService:
    def __init__(self, chunks: list[SourceChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, question: str, filters: dict[str, str] | None = None) -> list[SourceChunk]:
        return self.chunks


class StubReranker:
    def rerank(self, chunks: list[SourceChunk], limit: int) -> list[SourceChunk]:
        return chunks[:limit]


class StubGenerationService:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, question: str, context_package) -> str:
        return self.answer


def make_chat_service(chunks: list[SourceChunk], answer: str = "Grounded answer [S1].") -> ChatService:
    return ChatService(
        retrieval_service=StubRetrievalService(chunks),
        reranker=StubReranker(),
        generation_service=StubGenerationService(answer),
        guardrails=GuardrailEngine(min_score=0.6, min_good_results=2),
        max_context_chars=5000,
        rerank_k=5,
    )


def test_chat_returns_200_for_supported_question(monkeypatch) -> None:
    service = make_chat_service(
        [make_chunk("1", 0.95, label="bad-label"), make_chunk("2", 0.88, label=None)],
        answer="NIST AI RMF emphasizes governance as an organizational function [S1].",
    )
    monkeypatch.setattr("app.api.chat.get_chat_service", lambda: service)
    client = TestClient(app)

    response = client.post("/chat", json={"question": "What does NIST AI RMF say about governance?"})
    assert response.status_code == 200
    payload = response.json()
    validated = ChatResponse.model_validate(payload)
    assert validated.guardrail_status is GuardrailStatus.OK
    assert validated.confidence in {ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH}
    assert [source.label for source in validated.sources] == ["S1", "S2"]


def test_chat_returns_valid_json_for_insufficient_context(monkeypatch) -> None:
    service = make_chat_service([make_chunk("1", 0.21)])
    monkeypatch.setattr("app.api.chat.get_chat_service", lambda: service)
    client = TestClient(app)

    response = client.post("/chat", json={"question": "Explain detailed controls for an unknown framework."})
    assert response.status_code == 200
    payload = response.json()
    validated = ChatResponse.model_validate(payload)
    assert validated.guardrail_status is GuardrailStatus.INSUFFICIENT_CONTEXT
    assert validated.confidence is ConfidenceLevel.LOW


def test_chat_returns_valid_json_for_refusal(monkeypatch) -> None:
    service = make_chat_service([make_chunk("1", 0.99), make_chunk("2", 0.97)])
    monkeypatch.setattr("app.api.chat.get_chat_service", lambda: service)
    client = TestClient(app)

    response = client.post("/chat", json={"question": "Give a direct quote from ISO 27001 Annex A."})
    assert response.status_code == 200
    payload = response.json()
    validated = ChatResponse.model_validate(payload)
    assert validated.guardrail_status is GuardrailStatus.REFUSED
    assert validated.confidence is ConfidenceLevel.LOW

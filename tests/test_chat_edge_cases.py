import pytest

from app.generation.service import ChatService
from app.guardrails.rules import GuardrailEngine
from app.models.chat import ConfidenceLevel, GuardrailStatus
from app.models.source import SourceChunk


class StubRetrievalService:
    def __init__(self, chunks: list[SourceChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, question: str, filters: dict[str, str] | None = None) -> list[SourceChunk]:
        return self._chunks


class StubReranker:
    def rerank(self, chunks: list[SourceChunk], limit: int) -> list[SourceChunk]:
        return chunks[:limit]


class StubGenerationService:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    def generate(self, question: str, context_package) -> str:
        return self._answer


def make_chunk(
    chunk_id: str,
    score: float = 0.9,
    *,
    label: str | None = None,
    title: str = "Sample title",
    framework: str = "NIST_AI_RMF",
    url: str = "https://example.com/source",
) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        text=f"Evidence text for {chunk_id}",
        source_id=f"src-{chunk_id}",
        title=title,
        url=url,
        publisher="NIST",
        source_type="html",
        framework=framework,
        section="Govern",
        chunk_index=int(chunk_id),
        score=score,
        rerank_score=score,
        label=label,
    )


def make_chat_service(chunks: list[SourceChunk], answer: str = "Answer [S1].") -> ChatService:
    return ChatService(
        retrieval_service=StubRetrievalService(chunks),
        reranker=StubReranker(),
        generation_service=StubGenerationService(answer),
        guardrails=GuardrailEngine(min_score=0.6, min_good_results=2),
        max_context_chars=5000,
        rerank_k=8,
    )


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        ("S1", "S1"),
        ("s2", "S2"),
        (" S03 ", "S3"),
        ("bad", "S1"),
        ("", "S1"),
        (None, "S1"),
    ],
)
def test_source_label_normalization_variants(raw_label: str | None, expected: str) -> None:
    service = make_chat_service([make_chunk("1")])
    normalized_sources = service._to_sources([make_chunk("1", label=raw_label)])
    assert normalized_sources[0].label == expected


def test_response_normalizes_missing_title_framework_and_url() -> None:
    service = make_chat_service(
        [
            make_chunk("1", title="", framework="", url=""),
            make_chunk("2", title="Doc 2", framework="NIST_CSF", url="https://example.com/2"),
        ]
    )
    response = service.answer_question("Give a grounded summary.")
    assert response.sources[0].title == "Untitled source"
    assert response.sources[0].framework == "unknown"
    assert response.sources[0].url == ""
    assert response.sources[1].title == "Doc 2"


def test_guardrailed_response_with_sparse_retrieval_is_valid() -> None:
    service = make_chat_service([make_chunk("1", score=0.10)])
    response = service.answer_question("Explain controls for an obscure framework.")
    assert response.guardrail_status is GuardrailStatus.INSUFFICIENT_CONTEXT
    assert response.confidence is ConfidenceLevel.LOW
    assert isinstance(response.answer, str)
    assert response.sources[0].label == "S1"

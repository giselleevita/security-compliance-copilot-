import re

from app.generation.context_builder import build_context
from app.generation.service import ChatService, GenerationService
from app.guardrails.rules import GuardrailEngine
from app.models.chat import GuardrailStatus
from app.models.source import SourceChunk


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    score: float = 0.92,
    framework: str = "NIST_AI_RMF",
    title: str = "AI RMF 1.0",
    url: str = "https://example.com/doc",
    label: str | None = None,
) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        text=text,
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


def make_chat_service(chunks: list[SourceChunk], answer: str) -> ChatService:
    return ChatService(
        retrieval_service=StubRetrievalService(chunks),
        reranker=StubReranker(),
        generation_service=StubGenerationService(answer),
        guardrails=GuardrailEngine(min_score=0.6, min_good_results=2),
        max_context_chars=8000,
        rerank_k=5,
    )


def extract_citation_labels(answer: str) -> list[str]:
    return re.findall(r"\[(S\d+)\]", answer)


def answer_claims_supported_by_context(answer: str, chunks: list[SourceChunk]) -> bool:
    support_text = " ".join(chunk.text.lower() for chunk in chunks)
    stripped = re.sub(r"\[S\d+\]", "", answer).lower()
    claims = [part.strip(" .,:;") for part in re.split(r"[.!?]", stripped) if part.strip()]
    return all(claim in support_text for claim in claims)


def test_generation_service_strips_hallucinated_citations() -> None:
    chunks = [make_chunk("1", "NIST AI RMF improves governance accountability.", label="S1")]
    context_package = build_context(chunks, max_chars=4000)
    service = GenerationService(api_key="", model="unused")

    sanitized = service._sanitize_citations("Grounded answer [S1][S9].", context_package)

    assert sanitized == "Grounded answer [S1]."


def test_every_citation_label_used_in_answer_exists_in_sources() -> None:
    chunks = [
        make_chunk("1", "The Govern function sets accountability and oversight."),
        make_chunk("2", "The Govern function is cross-cutting across the AI lifecycle."),
    ]
    service = make_chat_service(
        chunks,
        "The Govern function sets accountability and oversight and is cross-cutting across the AI lifecycle [S1][S2].",
    )

    response = service.answer_question("What is the role of the Govern function in AI risk management?")

    source_labels = {source.label for source in response.sources}
    assert response.guardrail_status is GuardrailStatus.OK
    assert source_labels == {"S1", "S2"}
    assert set(extract_citation_labels(response.answer)).issubset(source_labels)


def test_source_labels_are_unique_and_ordered_even_with_bad_input_labels() -> None:
    service = make_chat_service(
        [
            make_chunk("1", "Evidence one", label="bad"),
            make_chunk("2", "Evidence two", label="S1"),
            make_chunk("3", "Evidence three", label="s001"),
        ],
        "Supported answer [S1][S2][S3].",
    )

    response = service.answer_question("What is the purpose of NIST AI RMF?")

    assert [source.label for source in response.sources] == ["S1", "S2", "S3"]


def test_source_metadata_is_normalized_and_complete() -> None:
    service = make_chat_service(
        [
            make_chunk("1", "Evidence one", title="", framework="", url="", label=None),
            make_chunk("2", "Evidence two", title="Doc 2", framework="CISA", url="https://example.com/2"),
        ],
        "Supported answer [S1][S2].",
    )

    response = service.answer_question("What is secure by design in CISA guidance?")

    assert response.sources[0].title == "Untitled source"
    assert response.sources[0].framework == "unknown"
    assert response.sources[0].url == ""
    assert response.sources[1].framework == "CISA"


def test_supported_answer_claims_are_detected_as_supported_by_context() -> None:
    chunks = [
        make_chunk("1", "The purpose of NIST AI RMF is to help organizations manage AI risks."),
        make_chunk("2", "The framework promotes trustworthy AI outcomes and governance."),
    ]
    answer = "The purpose of NIST AI RMF is to help organizations manage AI risks [S1]."

    assert answer_claims_supported_by_context(answer, chunks) is True


def test_unsupported_answer_claims_are_detected_by_suite() -> None:
    chunks = [
        make_chunk("1", "NIST AI RMF discusses governance and risk management."),
        make_chunk("2", "The corpus does not specify exact GPU memory isolation controls."),
    ]
    answer = "NIST requires production-grade GPU memory isolation controls [S1]."

    assert answer_claims_supported_by_context(answer, chunks) is False

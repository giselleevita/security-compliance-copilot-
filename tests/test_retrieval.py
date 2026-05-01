from app.models.source import SourceChunk
from app.ranking.reranker import SimpleReranker
from app.retrieval.search import RetrievalService


class FakeEmbeddingClient:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def query(self, embedding: list[float], top_k: int, filters: dict[str, str] | None = None) -> list[SourceChunk]:
        assert embedding == [0.1, 0.2, 0.3]
        assert top_k == 3
        assert filters == {"framework": "NIST"}
        return [
            SourceChunk(
                chunk_id="1",
                text="protective controls",
                source_id="src1",
                title="NIST Protect",
                url="/tmp/doc.md",
                publisher="NIST",
                source_type="md",
                framework="NIST",
                section="Protect",
                chunk_index=0,
                score=0.8,
            )
        ]


def test_reranker_prefers_framework_labeled_chunks() -> None:
    reranker = SimpleReranker()
    chunks = [
        SourceChunk(
            chunk_id="1",
            text="generic",
            source_id="s1",
            title="Doc 1",
            url="a",
            publisher="Publisher A",
            source_type="txt",
            framework="general",
            section="Introduction",
            chunk_index=0,
            score=0.55,
        ),
        SourceChunk(
            chunk_id="2",
            text="framework",
            source_id="s2",
            title="Doc 2",
            url="b",
            publisher="Publisher B",
            source_type="md",
            framework="NIST",
            section="Protect",
            chunk_index=1,
            score=0.53,
        ),
    ]

    reranked = reranker.rerank(chunks, limit=2)
    assert reranked[0].chunk_id == "2"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score >= reranked[1].rerank_score


def test_retrieval_service_uses_embeddings_and_filters() -> None:
    service = RetrievalService(
        vector_store=FakeVectorStore(),
        embedding_client=FakeEmbeddingClient(),
        top_k=3,
    )
    result = service.retrieve("What does NIST Protect mean?", filters={"framework": "NIST"})
    assert len(result) == 1
    assert result[0].title == "NIST Protect"
    assert result[0].publisher == "NIST"


def test_retrieval_service_can_filter_by_score() -> None:
    service = RetrievalService(
        vector_store=FakeVectorStore(),
        embedding_client=FakeEmbeddingClient(),
        top_k=3,
    )
    result = service.retrieve(
        "What does NIST Protect mean?",
        filters={"framework": "NIST"},
        min_score=0.85,
    )
    assert result == []

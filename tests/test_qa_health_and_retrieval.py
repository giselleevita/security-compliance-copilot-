from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import _get_last_ingest_at, _load_known_sources, get_health_status
from app.main import app
from app.models.source import SourceChunk
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.search import RetrievalService


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.inputs.append(text)
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self) -> None:
        self.count_value = 42
        self.fail_with: Exception | None = None
        self.queries: list[dict] = []

    def count(self) -> int:
        if self.fail_with:
            raise self.fail_with
        return self.count_value

    def query(self, embedding: list[float], top_k: int, filters: dict[str, str] | None = None) -> list[SourceChunk]:
        self.queries.append({"embedding": embedding, "top_k": top_k, "filters": filters})
        return [
            SourceChunk(
                chunk_id="1",
                text="NIST AI RMF helps organizations manage AI risk.",
                source_id="src-1",
                title="AI RMF 1.0",
                url="https://example.com/1",
                publisher="NIST",
                source_type="html",
                framework="NIST_AI_RMF",
                section="Govern",
                chunk_index=0,
                score=0.9,
            )
        ]


def test_query_rewriter_expands_known_acronyms_and_normalizes_whitespace() -> None:
    rewritten = QueryRewriter().rewrite("  What does AI RMF and  CSF  say about   RAG? ")
    assert rewritten == (
        "What does NIST AI RMF and NIST Cybersecurity Framework (CSF) say about "
        "retrieval-augmented generation (RAG)?"
    )


def test_retrieval_service_uses_rewritten_question_for_embeddings() -> None:
    embedding_client = FakeEmbeddingClient()
    service = RetrievalService(vector_store=FakeVectorStore(), embedding_client=embedding_client, top_k=4)

    result = service.retrieve("What does AI RMF say about RAG?", filters={"framework": "NIST_AI_RMF"})

    assert len(result) == 1
    assert embedding_client.inputs == [
        "What does NIST AI RMF say about retrieval-augmented generation (RAG)?"
    ]


def test_health_helpers_return_plausible_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    (raw_dir / "doc.metadata.json").write_text('{"framework": "CISA"}', encoding="utf-8")
    processed_file = processed_dir / "chunk.json"
    processed_file.write_text("{}", encoding="utf-8")

    known_sources = _load_known_sources(raw_dir)
    last_ingest_at = _get_last_ingest_at(processed_dir)

    assert known_sources == [{"framework": "CISA", "count": 1}]
    assert isinstance(last_ingest_at, str)
    assert "T" in last_ingest_at


def test_health_endpoint_shape() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {"status", "indexed_chunks", "known_sources", "last_ingest_at"}


def test_health_degrades_instead_of_crashing_when_store_fails(monkeypatch, tmp_path: Path) -> None:
    store = FakeVectorStore()
    store.fail_with = RuntimeError("chroma unavailable")
    monkeypatch.setattr("app.core.dependencies.get_vector_store", lambda: store)
    monkeypatch.setattr(
        "app.core.dependencies.get_settings",
        lambda: type(
            "Settings",
            (),
            {"data_raw_dir": tmp_path / "raw", "data_processed_dir": tmp_path / "processed"},
        )(),
    )
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()

    status = get_health_status()

    assert status["status"] != "ok"
    assert status["indexed_chunks"] == 0


@pytest.mark.integration
def test_health_values_are_plausible_when_index_exists() -> None:
    client = TestClient(app)
    response = client.get("/health")
    if response.status_code != 200:
        pytest.skip("Health endpoint unavailable.")

    payload = response.json()
    if payload["indexed_chunks"] <= 0:
        pytest.skip("No indexed corpus available for plausibility check.")

    assert payload["status"] == "ok"
    assert payload["indexed_chunks"] > 0
    assert isinstance(payload["known_sources"], list)

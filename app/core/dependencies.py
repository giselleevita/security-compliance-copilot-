import json
import logging
from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.generation.service import ChatService, GenerationService
from app.guardrails.rules import GuardrailEngine
from app.ingestion.pipeline import IngestionPipeline
from app.ranking.reranker import SimpleReranker
from app.retrieval.embeddings import LocalEmbeddingClient
from app.retrieval.search import RetrievalService
from app.retrieval.vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    settings = get_settings()
    return ChromaVectorStore(
        persist_directory=str(settings.chroma_dir),
        collection_name=settings.chroma_collection,
        raw_dir=str(settings.data_raw_dir),
    )


@lru_cache
def get_embedding_client() -> LocalEmbeddingClient:
    return LocalEmbeddingClient(model="sentence-transformers/all-mpnet-base-v2")


@lru_cache
def get_retrieval_service() -> RetrievalService:
    settings = get_settings()
    return RetrievalService(
        vector_store=get_vector_store(),
        embedding_client=get_embedding_client(),
        top_k=settings.top_k,
    )


@lru_cache
def get_generation_service() -> GenerationService:
    settings = get_settings()
    return GenerationService(
        api_key=settings.gemini_api_key,
        model="gemini-2.0-flash",
    )


@lru_cache
def get_guardrails() -> GuardrailEngine:
    settings = get_settings()
    return GuardrailEngine(
        min_score=settings.min_retrieval_score,
        min_good_results=settings.min_good_results,
    )


@lru_cache
def get_chat_service() -> ChatService:
    settings = get_settings()
    return ChatService(
        retrieval_service=get_retrieval_service(),
        reranker=SimpleReranker(),
        generation_service=get_generation_service(),
        guardrails=get_guardrails(),
        max_context_chars=settings.max_context_chars,
        rerank_k=settings.rerank_k,
    )


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    settings = get_settings()
    return IngestionPipeline(
        raw_dir=settings.data_raw_dir,
        processed_dir=settings.data_processed_dir,
        vector_store=get_vector_store(),
        embedding_client=get_embedding_client(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def get_health_status() -> dict:
    settings = get_settings()
    status = "ok"
    try:
        store = get_vector_store()
        count = store.count()
    except ModuleNotFoundError:
        count = 0
        status = "degraded"
    except Exception:
        logger.exception("Health check failed while querying vector store")
        count = 0
        status = "degraded"
    return {
        "status": status,
        "indexed_chunks": count,
        "known_sources": _load_known_sources(settings.data_raw_dir),
        "last_ingest_at": _get_last_ingest_at(settings.data_processed_dir),
    }


def _load_known_sources(raw_dir: Path) -> list[dict]:
    counts: Counter[str] = Counter()
    for sidecar_path in raw_dir.glob("*.metadata.json"):
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        framework = str(payload.get("framework") or "unknown")
        counts[framework] += 1
    return [
        {"framework": framework, "count": count}
        for framework, count in sorted(counts.items(), key=lambda item: item[0])
    ]


def _get_last_ingest_at(processed_dir: Path) -> str | None:
    processed_files = [path for path in processed_dir.glob("*.json") if path.is_file()]
    if not processed_files:
        return None
    latest_mtime = max(path.stat().st_mtime for path in processed_files)
    return datetime.fromtimestamp(latest_mtime, tz=UTC).isoformat()

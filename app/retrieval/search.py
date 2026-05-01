import logging

from app.models.source import SourceChunk
from app.retrieval.embeddings import OpenAIEmbeddingClient
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_client: OpenAIEmbeddingClient,
        top_k: int = 8,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.top_k = top_k
        self.query_rewriter = query_rewriter or QueryRewriter()

    def rewrite_question(self, question: str) -> str:
        return self.query_rewriter.rewrite(question)

    def retrieve(
        self,
        question: str,
        filters: dict[str, str] | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[SourceChunk]:
        resolved_top_k = top_k or self.top_k
        rewritten_question = self.rewrite_question(question)
        embedding = self.embedding_client.embed_query(rewritten_question)
        chunks = self.vector_store.query(embedding=embedding, top_k=resolved_top_k, filters=filters)
        if min_score is not None:
            chunks = [chunk for chunk in chunks if chunk.score >= min_score]

        logger.info(
            "Retrieval query=%r rewritten=%r filters=%s returned=%s top_titles=%s",
            question,
            rewritten_question,
            filters,
            len(chunks),
            [f"{chunk.title} ({chunk.score:.3f})" for chunk in chunks[:5]],
        )
        return chunks

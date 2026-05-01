from app.models.source import SourceChunk


TRUSTED_FRAMEWORK_BONUSES = {
    "NIST_AI_RMF": 0.08,
    "NIST_CSF": 0.08,
    "CISA": 0.07,
    "FTC": 0.04,
}

TRUSTED_PUBLISHER_KEYWORDS = ("nist", "cisa", "ftc", "fbi", "nsa")


class SimpleReranker:
    def rerank(self, chunks: list[SourceChunk], limit: int) -> list[SourceChunk]:
        scored = [self._with_rerank_score(chunk) for chunk in chunks]
        ranked = sorted(
            scored,
            key=lambda chunk: (
                chunk.rerank_score or 0.0,
                chunk.score,
                -chunk.chunk_index,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def _with_rerank_score(self, chunk: SourceChunk) -> SourceChunk:
        metadata_bonus = 0.0
        if chunk.framework in TRUSTED_FRAMEWORK_BONUSES:
            metadata_bonus += TRUSTED_FRAMEWORK_BONUSES[chunk.framework]
        if chunk.publisher and any(keyword in chunk.publisher.lower() for keyword in TRUSTED_PUBLISHER_KEYWORDS):
            metadata_bonus += 0.03
        if chunk.section and chunk.section.lower() not in {"introduction", "unknown section"}:
            metadata_bonus += 0.03
        if chunk.title and chunk.url:
            metadata_bonus += 0.02
        if chunk.source_type in {"md", "html", "pdf"}:
            metadata_bonus += 0.02

        rerank_score = round(chunk.score + metadata_bonus, 4)
        return chunk.model_copy(update={"rerank_score": rerank_score})

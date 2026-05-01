from dataclasses import dataclass

from app.models.source import SourceChunk


@dataclass
class ContextPackage:
    context_text: str
    chunks: list[SourceChunk]


def build_context(chunks: list[SourceChunk], max_chars: int) -> ContextPackage:
    parts: list[str] = []
    labeled_chunks: list[SourceChunk] = []
    total = 0

    for index, chunk in enumerate(chunks, start=1):
        labeled_chunk = chunk.model_copy(update={"label": f"S{index}"})
        block = (
            f"[{labeled_chunk.label}]\n"
            f"Title: {labeled_chunk.title}\n"
            f"Framework: {labeled_chunk.framework}\n"
            f"Section: {labeled_chunk.section}\n"
            f"URL: {labeled_chunk.url}\n"
            f"Content:\n"
            f"{labeled_chunk.text.strip()}\n"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        labeled_chunks.append(labeled_chunk)
        total += len(block)

    return ContextPackage(context_text="\n".join(parts), chunks=labeled_chunks)

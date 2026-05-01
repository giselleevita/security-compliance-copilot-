from pydantic import BaseModel


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    source_id: str
    title: str
    url: str
    publisher: str
    source_type: str
    framework: str
    section: str
    chunk_index: int
    score: float
    rerank_score: float | None = None
    label: str | None = None


class SourceResult(BaseModel):
    label: str
    title: str
    framework: str
    url: str
    score: float

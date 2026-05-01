from enum import Enum

from pydantic import BaseModel, Field

from app.models.source import SourceResult


class GuardrailStatus(str, Enum):
    OK = "ok"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    REFUSED = "refused"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ChatRequest(BaseModel):
    question: str = Field(min_length=3)
    filters: dict[str, str] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    confidence: ConfidenceLevel
    guardrail_status: GuardrailStatus

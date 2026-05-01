import logging
import re

from app.generation.context_builder import ContextPackage, build_context
from app.generation.prompts import SYSTEM_PROMPT
from app.guardrails.rules import GuardrailDecision, GuardrailEngine
from app.models.chat import ChatResponse, ConfidenceLevel, GuardrailStatus
from app.models.source import SourceChunk, SourceResult
from app.ranking.reranker import SimpleReranker
from app.retrieval.search import RetrievalService

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


class GenerationService:
    def __init__(self, api_key: str, model: str, base_url: str = "") -> None:
        self.model = model
        if api_key:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        else:
            self.client = None

    def generate(self, question: str, context_package: ContextPackage) -> str:
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY is required for chat generation.")
        logger.info("Generating answer from retrieved context (%s chars)", len(context_package.context_text))
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context_package.context_text}\n\n"
            f"Allowed citations: {', '.join(chunk.label or '' for chunk in context_package.chunks)}"
        )
        response = self.client.generate_content(prompt, stream=False)
        answer = response.text if response and hasattr(response, 'text') else ""
        return self._sanitize_citations(answer.strip(), context_package)

    def _sanitize_citations(self, answer: str, context_package: ContextPackage) -> str:
        allowed_labels = {chunk.label for chunk in context_package.chunks if chunk.label}

        def replace(match: re.Match[str]) -> str:
            label = match.group(1)
            return f"[{label}]" if label in allowed_labels else ""

        sanitized = CITATION_PATTERN.sub(replace, answer)
        sanitized = re.sub(r" +([.,;:])", r"\1", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
        return sanitized


class ChatService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        reranker: SimpleReranker,
        generation_service: GenerationService,
        guardrails: GuardrailEngine,
        max_context_chars: int,
        rerank_k: int,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.reranker = reranker
        self.generation_service = generation_service
        self.guardrails = guardrails
        self.max_context_chars = max_context_chars
        self.rerank_k = rerank_k

    def answer_question(self, question: str, filters: dict[str, str] | None = None) -> ChatResponse:
        response, _ = self.answer_question_with_trace(question=question, filters=filters)
        return response

    def answer_question_with_trace(
        self, question: str, filters: dict[str, str] | None = None
    ) -> tuple[ChatResponse, dict]:
        if hasattr(self.retrieval_service, "rewrite_question"):
            rewritten_question = self.retrieval_service.rewrite_question(question)
        else:
            rewritten_question = question
        retrieved = self.retrieval_service.retrieve(question, filters=filters)
        reranked = self.reranker.rerank(retrieved, limit=self.rerank_k)
        decision = self.guardrails.evaluate(question, reranked)
        context_package = build_context(reranked, max_chars=self.max_context_chars)

        logger.info(
            "Chat retrieval question=%r retrieved=%s reranked=%s context_sources=%s guardrail=%s",
            question,
            len(retrieved),
            len(reranked),
            len(context_package.chunks),
            decision.status,
        )
        if decision.status != GuardrailStatus.OK:
            response = self._guardrailed_response(decision, context_package.chunks)
            return response, {
                "rewritten_query": rewritten_question,
                "top_retrieval_count": len(retrieved),
                "detection_flags": decision.detection_flags,
            }

        answer = self.generation_service.generate(question=question, context_package=context_package)
        response = ChatResponse(
            answer=self._normalize_answer(answer),
            sources=self._to_sources(context_package.chunks),
            confidence=self._normalize_confidence(self.guardrails.estimate_confidence(reranked)),
            guardrail_status=GuardrailStatus.OK,
        )
        self._log_response_debug(response)
        return response, {
            "rewritten_query": rewritten_question,
            "top_retrieval_count": len(retrieved),
            "detection_flags": decision.detection_flags,
        }

    def _guardrailed_response(self, decision: GuardrailDecision, chunks: list[SourceChunk]) -> ChatResponse:
        response = ChatResponse(
            answer=self._normalize_answer(decision.message),
            sources=self._to_sources(chunks),
            confidence=ConfidenceLevel.LOW,
            guardrail_status=self._normalize_guardrail_status(decision.status),
        )
        self._log_response_debug(response)
        return response

    def _to_sources(self, chunks: list[SourceChunk]) -> list[SourceResult]:
        sources: list[SourceResult] = []
        for index, chunk in enumerate(chunks, start=1):
            label = self._normalize_source_label(chunk.label, index)
            score = chunk.rerank_score if chunk.rerank_score is not None else chunk.score
            sources.append(
                SourceResult(
                    label=label,
                    title=chunk.title or "Untitled source",
                    framework=chunk.framework or "unknown",
                    url=chunk.url or "",
                    score=round(score, 4),
                )
            )
        return sources

    def _normalize_answer(self, answer: str | None) -> str:
        return (answer or "").strip()

    def _normalize_guardrail_status(self, status: GuardrailStatus | str | None) -> GuardrailStatus:
        if isinstance(status, GuardrailStatus):
            return status
        value = (status or "").strip().lower()
        aliases = {
            "ok": GuardrailStatus.OK,
            "insufficient_context": GuardrailStatus.INSUFFICIENT_CONTEXT,
            "insufficient-context": GuardrailStatus.INSUFFICIENT_CONTEXT,
            "refused": GuardrailStatus.REFUSED,
        }
        return aliases.get(value, GuardrailStatus.INSUFFICIENT_CONTEXT)

    def _normalize_confidence(self, confidence: ConfidenceLevel | str | None) -> ConfidenceLevel:
        if isinstance(confidence, ConfidenceLevel):
            return confidence
        value = (confidence or "").strip().lower()
        aliases = {
            "high": ConfidenceLevel.HIGH,
            "medium": ConfidenceLevel.MEDIUM,
            "low": ConfidenceLevel.LOW,
        }
        return aliases.get(value, ConfidenceLevel.LOW)

    def _normalize_source_label(self, label: str | None, index: int) -> str:
        candidate = (label or "").strip()
        if not candidate:
            return f"S{index}"
        if candidate.lower().startswith("s") and candidate[1:].isdigit():
            return f"S{int(candidate[1:])}"
        return f"S{index}"

    def _log_response_debug(self, response: ChatResponse) -> None:
        logger.debug(
            "Chat response prepared answer_len=%s guardrail_status=%s confidence=%s source_labels=%s",
            len(response.answer),
            response.guardrail_status.value,
            response.confidence.value,
            [source.label for source in response.sources],
        )

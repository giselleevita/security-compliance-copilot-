import logging
import re
from typing import Protocol

from app.generation.context_builder import ContextPackage, build_context
from app.generation.prompts import SYSTEM_PROMPT
from app.guardrails.rules import GuardrailDecision, GuardrailEngine
from app.models.chat import ChatResponse, ConfidenceLevel, GuardrailStatus
from app.models.source import SourceChunk, SourceResult
from app.retrieval.search import RetrievalService

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
SENTENCE_PATTERN = re.compile(r"[^.!?\n]+[.!?]?")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[SourceChunk], limit: int) -> list[SourceChunk]: ...


class GroundingValidator:
    def validate(self, answer: str, context_package: ContextPackage) -> GuardrailDecision:
        unsafe_output = self._detect_unsafe_output(answer)
        if unsafe_output:
            return GuardrailDecision(
                status=GuardrailStatus.REFUSED,
                message=(
                    "I cannot help with bypassing safeguards, exposing internal prompts/configuration, or "
                    "extracting sensitive content. Ask a specific question about public NIST or CISA guidance."
                ),
                detection_flags=unsafe_output,
            )

        if not context_package.chunks or not context_package.context_text.strip():
            return GuardrailDecision(
                status=GuardrailStatus.INSUFFICIENT_CONTEXT,
                message=(
                    "I do not have enough retrieved evidence to answer this reliably from the indexed sources. "
                    "This is not legal or compliance advice."
                ),
                detection_flags=["empty_context"],
            )

        citation_labels = set(CITATION_PATTERN.findall(answer))
        allowed_labels = {chunk.label for chunk in context_package.chunks if chunk.label}
        if not citation_labels:
            return self._unsupported("The generated answer did not cite retrieved evidence.", "missing_citation")
        if citation_labels - allowed_labels:
            return self._unsupported("The generated answer cited sources outside the retrieved context.", "invalid_citation")

        label_to_text = {chunk.label: chunk.text.lower() for chunk in context_package.chunks if chunk.label}
        checked_claims = 0
        supported_claims = 0
        uncited_claims = 0
        for sentence in self._claim_sentences(answer):
            sentence_labels = CITATION_PATTERN.findall(sentence)
            if not sentence_labels:
                uncited_claims += 1
                continue
            cited_text = " ".join(label_to_text.get(label, "") for label in sentence_labels)
            checked_claims += 1
            if self._sentence_supported(sentence, cited_text):
                supported_claims += 1

        if checked_claims == 0:
            return self._unsupported("No cited factual claims could be checked against retrieved evidence.", "missing_citation")

        support_ratio = supported_claims / checked_claims
        if support_ratio < 0.25:
            return self._unsupported(
                "The generated answer was not sufficiently supported by the cited retrieved chunks.",
                "unsupported_claim",
            )
        if uncited_claims > supported_claims:
            return self._unsupported("Too many factual claims lacked citations to retrieved evidence.", "uncited_claim")

        return GuardrailDecision(status=GuardrailStatus.OK, message="", detection_flags=[])

    def citation_precision(self, answer: str, context_package: ContextPackage) -> float:
        labels = CITATION_PATTERN.findall(answer)
        if not labels:
            return 0.0
        allowed_labels = {chunk.label for chunk in context_package.chunks if chunk.label}
        valid = sum(1 for label in labels if label in allowed_labels)
        return round(valid / len(labels), 4)

    def faithfulness_score(self, answer: str, context_package: ContextPackage) -> float:
        sentences = self._claim_sentences(answer)
        if not sentences:
            return 0.0
        label_to_text = {chunk.label: chunk.text.lower() for chunk in context_package.chunks if chunk.label}
        supported = 0
        for sentence in sentences:
            cited_text = " ".join(label_to_text.get(label, "") for label in CITATION_PATTERN.findall(sentence))
            if self._sentence_supported(sentence, cited_text):
                supported += 1
        return round(supported / len(sentences), 4)

    def _unsupported(self, message: str, flag: str) -> GuardrailDecision:
        return GuardrailDecision(
            status=GuardrailStatus.INSUFFICIENT_CONTEXT,
            message=f"{message} I cannot answer reliably from the retrieved context. This is not legal or compliance advice.",
            detection_flags=[flag],
        )

    def _claim_sentences(self, answer: str) -> list[str]:
        sentences: list[str] = []
        protected_answer = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", answer)
        for match in SENTENCE_PATTERN.finditer(protected_answer):
            sentence = match.group(0).replace("<DOT>", ".").strip()
            if not sentence:
                continue
            lowered = sentence.lower()
            if "not legal or compliance advice" in lowered:
                continue
            if "provided context" in lowered or "retrieved context" in lowered:
                continue
            if lowered.startswith(("it is essential to consult", "consult the actual")):
                continue
            sentences.append(sentence)
        return sentences

    def _sentence_supported(self, sentence: str, support_text: str) -> bool:
        tokens = self._content_tokens(CITATION_PATTERN.sub("", sentence))
        if not tokens:
            return True
        support_tokens = set(self._content_tokens(support_text))
        overlap = sum(1 for token in tokens if token in support_tokens)
        return overlap >= min(3, len(tokens)) or (overlap / len(tokens)) >= 0.25

    def _content_tokens(self, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9][a-z0-9-]{2,}", text.lower())
            if token not in STOP_WORDS
        ]

    def _detect_unsafe_output(self, answer: str) -> list[str]:
        normalized = re.sub(r"[^a-z0-9]+", " ", answer.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        compact = normalized.replace(" ", "")
        flags: set[str] = set()

        if any(
            pattern in normalized or pattern in compact
            for pattern in (
                "i am ignoring previous instructions",
                "i m ignoring previous instructions",
                "i am ignoring the previous rules",
                "i m ignoring the previous rules",
                "ignorepreviousinstructions",
                "ignoringpreviousinstructions",
                "ignoringthepreviousrules",
            )
        ):
            flags.add("prompt_injection_compliance")

        prompt_leak_patterns = (
            "here is the developer instruction",
            "here are the developer instructions",
            "here is the developer message",
            "here is the hidden prompt",
            "here is the internal prompt",
            "here is the system message",
            "here is the system prompt",
            "system prompt revealed",
            "developer message revealed",
            "hidden prompt revealed",
            "internal prompt revealed",
        )
        if any(pattern in normalized for pattern in prompt_leak_patterns):
            flags.add("prompt_leak_output")

        if any(
            pattern in normalized
            for pattern in (
                "api key is",
                "groq api key is",
                "password is",
                "token is",
                "environment secret is",
                "here is the api key",
                "here are the api keys",
                "here is the password",
                "here is the token",
            )
        ):
            flags.add("sensitive_content_output")

        return sorted(flags)


class GenerationService:
    def __init__(self, api_key: str, model: str, base_url: str = "", provider: str = "gemini") -> None:
        self.model = model
        self.provider = provider.lower()
        self.base_url = base_url
        if api_key:
            if self.provider == "groq":
                from openai import OpenAI

                self.client = OpenAI(api_key=api_key, base_url=base_url)
            elif self.provider == "gemini":
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        else:
            self.client = None

    def generate(self, question: str, context_package: ContextPackage) -> str:
        if not context_package.chunks or not context_package.context_text.strip():
            raise ValueError("Cannot generate an answer without retrieved context.")
        if not self.client:
            required_key = "GROQ_API_KEY" if self.provider == "groq" else "GEMINI_API_KEY"
            raise RuntimeError(f"{required_key} is required for chat generation.")
        logger.info("Generating answer from retrieved context (%s chars)", len(context_package.context_text))
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context_package.context_text}\n\n"
            f"Allowed citations: {', '.join(chunk.label or '' for chunk in context_package.chunks)}"
        )
        if self.provider == "groq":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
            except Exception as exc:
                raise RuntimeError("LLM provider request failed.") from exc
            answer = response.choices[0].message.content if response.choices else ""
        else:
            try:
                response = self.client.generate_content(prompt, stream=False)
            except Exception as exc:
                raise RuntimeError("LLM provider request failed.") from exc
            answer = response.text if response and hasattr(response, "text") else ""
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
        reranker: Reranker,
        generation_service: GenerationService,
        guardrails: GuardrailEngine,
        max_context_chars: int,
        rerank_k: int,
        grounding_validator: GroundingValidator | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.reranker = reranker
        self.generation_service = generation_service
        self.guardrails = guardrails
        self.max_context_chars = max_context_chars
        self.rerank_k = rerank_k
        self.grounding_validator = grounding_validator or GroundingValidator()

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
        input_decision = self.guardrails.evaluate_input(question)
        if input_decision:
            response = self._guardrailed_response(input_decision, [])
            return response, {
                "rewritten_query": rewritten_question,
                "top_retrieval_count": 0,
                "detection_flags": input_decision.detection_flags,
            }
        retrieved = self.retrieval_service.retrieve(question, filters=filters)
        reranked = self.reranker.rerank(query=rewritten_question, chunks=retrieved, limit=self.rerank_k)
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
        if not context_package.chunks:
            empty_context = GuardrailDecision(
                status=GuardrailStatus.INSUFFICIENT_CONTEXT,
                message=(
                    "I do not have enough retrieved evidence to answer this reliably from the indexed sources. "
                    "This is not legal or compliance advice."
                ),
                detection_flags=["empty_context"],
            )
            response = self._guardrailed_response(empty_context, context_package.chunks)
            return response, {
                "rewritten_query": rewritten_question,
                "top_retrieval_count": len(retrieved),
                "detection_flags": empty_context.detection_flags,
            }

        answer = self.generation_service.generate(question=question, context_package=context_package)
        grounding_decision = self.grounding_validator.validate(answer, context_package)
        if grounding_decision.status != GuardrailStatus.OK:
            response = self._guardrailed_response(grounding_decision, context_package.chunks)
            return response, {
                "rewritten_query": rewritten_question,
                "top_retrieval_count": len(retrieved),
                "detection_flags": grounding_decision.detection_flags,
            }
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

from dataclasses import dataclass

from app.models.chat import ConfidenceLevel, GuardrailStatus
from app.models.source import SourceChunk

PROPRIETARY_QUOTE_KEYWORDS = ["quote", "exact text", "verbatim", "direct quote"]
PROPRIETARY_FRAMEWORKS = ["soc 2", "iso 27001", "iso27001"]
UNSAFE_PATTERNS: list[tuple[str, str]] = [
    ("ignore previous instructions", "prompt_injection_attempt"),
    ("system prompt", "prompt_leak_request"),
    ("developer message", "prompt_leak_request"),
    ("bypass rules", "guardrail_bypass_attempt"),
    ("jailbreak", "jailbreak_request"),
    ("prompt injection", "prompt_injection_attempt"),
    ("internal documents", "internal_document_request"),
    ("private documents", "private_document_request"),
]
BROAD_PATTERNS: list[tuple[str, str]] = [
    ("dump all files", "broad_data_dump_request"),
    ("show all files", "broad_data_dump_request"),
    ("print all files", "broad_data_dump_request"),
    ("all files in the index", "broad_data_dump_request"),
    ("all documents", "broad_data_dump_request"),
    ("dump the index", "index_dump_request"),
]
SENSITIVE_TERM_PATTERNS = ["config", "secret", "password", "token", "api key", "exfiltrate"]
SENSITIVE_ACTION_PATTERNS = ["show", "dump", "reveal", "leak", "give", "extract", "exfiltrate"]


@dataclass
class GuardrailDecision:
    status: GuardrailStatus
    message: str
    detection_flags: list[str]


class GuardrailEngine:
    def __init__(self, min_score: float, min_good_results: int = 2) -> None:
        self.min_score = min_score
        self.min_good_results = min_good_results

    def evaluate(self, question: str, chunks: list[SourceChunk]) -> GuardrailDecision:
        lowered = question.lower()
        unsafe_flags = self._detect_unsafe_flags(lowered)
        if unsafe_flags:
            return GuardrailDecision(
                status=GuardrailStatus.REFUSED,
                message=(
                    "I cannot help with bypassing safeguards, exposing internal prompts/configuration, or "
                    "extracting sensitive content. Ask a specific question about public NIST or CISA guidance."
                ),
                detection_flags=unsafe_flags,
            )

        broad_flags = self._detect_broad_flags(lowered)
        if broad_flags:
            return GuardrailDecision(
                status=GuardrailStatus.INSUFFICIENT_CONTEXT,
                message=(
                    "That request is too broad for a grounded response. Ask a narrower question tied to a "
                    "specific NIST or CISA topic. This is not legal or compliance advice."
                ),
                detection_flags=broad_flags,
            )

        if self._requests_proprietary_quote(lowered):
            return GuardrailDecision(
                status=GuardrailStatus.REFUSED,
                message=(
                    "I can only use available public source material here and cannot provide or fabricate "
                    "exact proprietary standards text. This is not legal or compliance advice."
                ),
                detection_flags=["proprietary_text_request"],
            )

        good_chunks = [chunk for chunk in chunks if chunk.score >= self.min_score]
        if not good_chunks:
            return GuardrailDecision(
                status=GuardrailStatus.INSUFFICIENT_CONTEXT,
                message=(
                    "I do not have enough strong retrieved evidence to answer this reliably from the indexed "
                    "sources. This is not legal or compliance advice."
                ),
                detection_flags=[],
            )

        if len(good_chunks) < self.min_good_results:
            return GuardrailDecision(
                status=GuardrailStatus.INSUFFICIENT_CONTEXT,
                message=(
                    "The retrieved evidence is too thin to support a grounded answer. I can cite the source I "
                    "found, but I would treat this as incomplete evidence. This is not legal or compliance advice."
                ),
                detection_flags=[],
            )
        return GuardrailDecision(status=GuardrailStatus.OK, message="", detection_flags=[])

    def estimate_confidence(self, chunks: list[SourceChunk]) -> ConfidenceLevel:
        good_chunks = [chunk for chunk in chunks if chunk.score >= self.min_score]
        if len(good_chunks) >= 4:
            return ConfidenceLevel.HIGH
        if len(good_chunks) >= 2:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _requests_proprietary_quote(self, lowered_question: str) -> bool:
        asks_for_quote = any(keyword in lowered_question for keyword in PROPRIETARY_QUOTE_KEYWORDS)
        mentions_proprietary = any(keyword in lowered_question for keyword in PROPRIETARY_FRAMEWORKS) or (
            "proprietary standard" in lowered_question and "full text" in lowered_question
        )
        return asks_for_quote and mentions_proprietary

    def _detect_unsafe_flags(self, lowered_question: str) -> list[str]:
        flags = [flag for pattern, flag in UNSAFE_PATTERNS if pattern in lowered_question]

        asks_for_sensitive_content = any(term in lowered_question for term in SENSITIVE_TERM_PATTERNS) and any(
            action in lowered_question for action in SENSITIVE_ACTION_PATTERNS
        )
        if asks_for_sensitive_content:
            flags.append("sensitive_content_request")

        asks_for_proprietary_full_text = (
            "full text of iso" in lowered_question
            or "full text of proprietary standards" in lowered_question
            or ("full text" in lowered_question and "iso" in lowered_question)
        )
        if asks_for_proprietary_full_text:
            flags.append("proprietary_text_request")

        return sorted(set(flags))

    def _detect_broad_flags(self, lowered_question: str) -> list[str]:
        return sorted({flag for pattern, flag in BROAD_PATTERNS if pattern in lowered_question})

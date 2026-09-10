import re
from dataclasses import dataclass

from app.models.chat import ConfidenceLevel, GuardrailStatus
from app.models.source import SourceChunk

PROPRIETARY_QUOTE_KEYWORDS = ["quote", "exact text", "exact wording", "verbatim", "direct quote", "full text"]
PROPRIETARY_COPY_KEYWORDS = [
    "copy",
    "provide",
    "show",
    "give",
    "print",
    "full standard",
    "whole standard",
    "entire standard",
]
PROPRIETARY_FRAMEWORKS = [
    "soc 2",
    "soc2",
    "iso 27001",
    "iso27001",
    "pci dss",
    "pcidss",
    "paid standard",
    "proprietary standard",
]
PROMPT_LEAK_TERMS = [
    "chain of thought",
    "developer instruction",
    "developer instructions",
    "developer message",
    "hidden instruction",
    "hidden instructions",
    "hidden policy",
    "hidden prompt",
    "initial prompt",
    "initialization text",
    "instruction hierarchy",
    "internal instruction",
    "internal instructions",
    "internal policy",
    "internal prompt",
    "policy text",
    "private instruction",
    "private instructions",
    "reasoning trace",
    "scratchpad",
    "system instruction",
    "system instructions",
    "system message",
    "system prompt",
]
PROMPT_LEAK_ACTIONS = ["display", "dump", "print", "reveal", "show", "tell", "what is", "write"]
UNSAFE_PATTERNS: list[tuple[str, str]] = [
    ("ignore previous instructions", "prompt_injection_attempt"),
    ("ignore previous rules", "prompt_injection_attempt"),
    ("ignorepreviousinstructions", "prompt_injection_attempt"),
    ("ignorepreviousrules", "prompt_injection_attempt"),
    ("all previous rules", "prompt_injection_attempt"),
    ("disregard previous instructions", "prompt_injection_attempt"),
    ("disregard earlier directions", "prompt_injection_attempt"),
    ("forget your instructions", "prompt_injection_attempt"),
    ("override instructions", "prompt_injection_attempt"),
    ("system prompt", "prompt_leak_request"),
    ("hidden prompt", "prompt_leak_request"),
    ("initial prompt", "prompt_leak_request"),
    ("internal prompt", "prompt_leak_request"),
    ("hidden policy", "prompt_leak_request"),
    ("initialization text", "prompt_leak_request"),
    ("developer message", "prompt_leak_request"),
    ("developer instructions", "prompt_leak_request"),
    ("system message", "prompt_leak_request"),
    ("system instructions", "prompt_leak_request"),
    ("instruction hierarchy", "prompt_leak_request"),
    ("chain of thought", "prompt_leak_request"),
    ("scratchpad", "prompt_leak_request"),
    ("reasoning trace", "prompt_leak_request"),
    ("bypass rules", "guardrail_bypass_attempt"),
    ("bypass safeguards", "guardrail_bypass_attempt"),
    ("disable guardrails", "guardrail_bypass_attempt"),
    ("jailbreak", "jailbreak_request"),
    ("do anything now", "jailbreak_request"),
    ("dan mode", "jailbreak_request"),
    ("prompt injection", "prompt_injection_attempt"),
    ("internal documents", "internal_document_request"),
    ("private documents", "private_document_request"),
]
BROAD_ACTION_PATTERNS = ["display", "dump", "enumerate", "export", "list", "print", "show"]
BROAD_SCOPE_PATTERNS = [
    "all chunk",
    "all citation",
    "all document",
    "all evidence",
    "all file",
    "all retrieved",
    "all source",
    "entire corpus",
    "every chunk",
    "every citation",
    "every document",
    "every source",
    "full corpus",
    "raw context",
    "retrieved context",
    "whole corpus",
]
BROAD_PATTERNS: list[tuple[str, str]] = [
    ("dump all files", "broad_data_dump_request"),
    ("show all files", "broad_data_dump_request"),
    ("print all files", "broad_data_dump_request"),
    ("all files in the index", "broad_data_dump_request"),
    ("all documents", "broad_data_dump_request"),
    ("dump all documents", "broad_data_dump_request"),
    ("dump every document", "broad_data_dump_request"),
    ("dump every chunk", "broad_data_dump_request"),
    ("dump all chunks", "broad_data_dump_request"),
    ("all chunks", "broad_data_dump_request"),
    ("all sources", "broad_data_dump_request"),
    ("all citations", "broad_data_dump_request"),
    ("dump the index", "index_dump_request"),
]
RECENCY_PATTERNS: list[tuple[str, str]] = [
    ("latest", "current_events_request"),
    ("last week", "current_events_request"),
    ("today", "current_events_request"),
    ("yesterday", "current_events_request"),
    ("breaking", "current_events_request"),
    ("recent enforcement", "current_events_request"),
]
SENSITIVE_TERM_PATTERNS = [
    "config",
    "credential",
    "database credential",
    "env",
    "environment secret",
    "environment variable",
    ".env",
    "secret",
    "key",
    "keys",
    "password",
    "token",
    "api key",
    "groq api key",
    "exfiltrate",
]
SENSITIVE_ACTION_PATTERNS = ["show", "dump", "print", "reveal", "leak", "give", "extract", "exfiltrate"]


@dataclass
class GuardrailDecision:
    status: GuardrailStatus
    message: str
    detection_flags: list[str]


class GuardrailEngine:
    def __init__(self, min_score: float, min_good_results: int = 2) -> None:
        self.min_score = min_score
        self.min_good_results = min_good_results

    def evaluate_input(self, question: str) -> GuardrailDecision | None:
        normalized = self._normalize(question)
        unsafe_flags = self._detect_unsafe_flags(normalized)
        if unsafe_flags:
            return GuardrailDecision(
                status=GuardrailStatus.REFUSED,
                message=(
                    "I cannot help with bypassing safeguards, exposing internal prompts/configuration, or "
                    "extracting sensitive content. Ask a specific question about public NIST or CISA guidance."
                ),
                detection_flags=unsafe_flags,
            )

        broad_flags = self._detect_broad_flags(normalized)
        if broad_flags:
            return GuardrailDecision(
                status=GuardrailStatus.REFUSED,
                message=(
                    "I cannot dump broad corpus or index contents. Ask a narrower question tied to a specific "
                    "NIST or CISA topic. This is not legal or compliance advice."
                ),
                detection_flags=broad_flags,
            )

        if self._requests_proprietary_quote(normalized):
            return GuardrailDecision(
                status=GuardrailStatus.REFUSED,
                message=(
                    "I can only use available public source material here and cannot provide or fabricate "
                    "exact proprietary standards text. This is not legal or compliance advice."
                ),
                detection_flags=["proprietary_text_request"],
            )
        recency_flags = self._detect_recency_flags(normalized)
        if recency_flags:
            return GuardrailDecision(
                status=GuardrailStatus.INSUFFICIENT_CONTEXT,
                message=(
                    "I do not have live or current-events coverage in the indexed corpus. Ask about the public "
                    "NIST or CISA guidance that is included in the local index. This is not legal or compliance advice."
                ),
                detection_flags=recency_flags,
            )
        return None

    def evaluate(self, question: str, chunks: list[SourceChunk]) -> GuardrailDecision:
        input_decision = self.evaluate_input(question)
        if input_decision:
            return input_decision

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

    def _normalize(self, question: str) -> str:
        original = question.lower()
        original = re.sub(r"[^a-z0-9]+", " ", original)
        original = re.sub(r"\s+", " ", original).strip()
        deobfuscated = original.translate(str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"}))
        compact = original.replace(" ", "")
        compact_deobfuscated = deobfuscated.replace(" ", "")
        return f"{original} {deobfuscated} {compact} {compact_deobfuscated}"

    def _requests_proprietary_quote(self, normalized_question: str) -> bool:
        asks_for_quote = any(keyword in normalized_question for keyword in PROPRIETARY_QUOTE_KEYWORDS)
        asks_to_copy_standard = any(keyword in normalized_question for keyword in PROPRIETARY_COPY_KEYWORDS) and any(
            standard_term in normalized_question
            for standard_term in ("standard", "requirement", "requirements", "control", "controls")
        )
        mentions_proprietary = any(keyword in normalized_question for keyword in PROPRIETARY_FRAMEWORKS)
        return (asks_for_quote or asks_to_copy_standard) and mentions_proprietary

    def _detect_unsafe_flags(self, normalized_question: str) -> list[str]:
        flags = [flag for pattern, flag in UNSAFE_PATTERNS if pattern in normalized_question]

        asks_for_prompt_leak = any(term in normalized_question for term in PROMPT_LEAK_TERMS) and any(
            action in normalized_question for action in PROMPT_LEAK_ACTIONS
        )
        if asks_for_prompt_leak:
            flags.append("prompt_leak_request")

        asks_for_sensitive_content = any(term in normalized_question for term in SENSITIVE_TERM_PATTERNS) and any(
            action in normalized_question for action in SENSITIVE_ACTION_PATTERNS
        )
        if asks_for_sensitive_content:
            flags.append("sensitive_content_request")

        asks_for_proprietary_full_text = (
            "full text of iso" in normalized_question
            or "full text of proprietary standards" in normalized_question
            or ("full text" in normalized_question and "iso" in normalized_question)
            or ("full text" in normalized_question and "pci dss" in normalized_question)
            or ("exact wording" in normalized_question and "pci dss" in normalized_question)
        )
        if asks_for_proprietary_full_text:
            flags.append("proprietary_text_request")

        return sorted(set(flags))

    def _detect_broad_flags(self, normalized_question: str) -> list[str]:
        flags = {flag for pattern, flag in BROAD_PATTERNS if pattern in normalized_question}
        asks_for_broad_export = any(action in normalized_question for action in BROAD_ACTION_PATTERNS) and any(
            scope in normalized_question for scope in BROAD_SCOPE_PATTERNS
        )
        if asks_for_broad_export:
            flags.add("broad_data_dump_request")
        return sorted(flags)

    def _detect_recency_flags(self, normalized_question: str) -> list[str]:
        return sorted({flag for pattern, flag in RECENCY_PATTERNS if pattern in normalized_question})

import re


class QueryRewriter:
    def __init__(self) -> None:
        self.patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"\bAI RMF\b", flags=re.IGNORECASE), "NIST AI RMF"),
            (
                re.compile(r"\bCSF\b", flags=re.IGNORECASE),
                "NIST Cybersecurity Framework (CSF)",
            ),
            (
                re.compile(r"\bSSDF\b", flags=re.IGNORECASE),
                "NIST Secure Software Development Framework (SSDF)",
            ),
            (
                re.compile(r"\bGenAI\b", flags=re.IGNORECASE),
                "generative AI",
            ),
            (
                re.compile(r"\bLLM\b", flags=re.IGNORECASE),
                "large language model (LLM)",
            ),
            (
                re.compile(r"\bRAG\b", flags=re.IGNORECASE),
                "retrieval-augmented generation (RAG)",
            ),
        ]

    def rewrite(self, question: str) -> str:
        rewritten = " ".join(question.split()).strip()
        for pattern, replacement in self.patterns:
            rewritten = pattern.sub(replacement, rewritten)
        return rewritten

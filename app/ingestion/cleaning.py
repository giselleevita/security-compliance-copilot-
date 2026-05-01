import re
from pathlib import Path


FRAMEWORK_KEYWORDS = {
    "nist": "NIST",
    "csf": "NIST CSF",
    "soc 2": "SOC 2",
    "iso 27001": "ISO 27001",
    "hipaa": "HIPAA",
    "gdpr": "GDPR",
    "ai rmf": "NIST AI RMF",
    "owasp": "OWASP",
}


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def infer_framework(text: str, path: Path) -> str:
    corpus = f"{path.name} {text[:2000]}".lower()
    for keyword, framework in FRAMEWORK_KEYWORDS.items():
        if keyword in corpus:
            return framework
    return "general"


def infer_title(text: str, path: Path) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if len(line) > 3:
            return line[:160]
    return path.stem.replace("_", " ").replace("-", " ").title()

import json
from pathlib import Path


SUPPORTED_SUFFIXES = {".md", ".txt", ".html", ".htm", ".pdf"}


def list_supported_files(raw_dir: Path) -> list[Path]:
    return sorted(
        path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup
        from markdownify import markdownify as html_to_markdown

        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return html_to_markdown(str(soup))
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_sidecar_metadata(path: Path) -> dict:
    sidecar_path = Path(f"{path}.metadata.json")
    if not sidecar_path.exists():
        return {}
    return json.loads(sidecar_path.read_text(encoding="utf-8"))

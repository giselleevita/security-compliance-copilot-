from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    section: str
    chunk_index: int


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    lines = [line.rstrip() for line in text.splitlines()]
    sections: list[tuple[str, str]] = []
    current_section = "Introduction"
    buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if buffer:
                sections.append((current_section, "\n".join(buffer).strip()))
                buffer = []
            current_section = stripped.lstrip("#").strip() or current_section
            continue
        buffer.append(line)

    if buffer:
        sections.append((current_section, "\n".join(buffer).strip()))

    chunks: list[Chunk] = []
    chunk_index = 0
    for section, section_text in sections:
        content = section_text.strip()
        if not content:
            continue
        start = 0
        while start < len(content):
            end = min(len(content), start + chunk_size)
            split = content[start:end]
            if end < len(content):
                last_break = max(split.rfind("\n"), split.rfind(". "))
                if last_break > int(chunk_size * 0.6):
                    end = start + last_break + 1
                    split = content[start:end]
            chunks.append(Chunk(text=split.strip(), section=section, chunk_index=chunk_index))
            if end >= len(content):
                break
            start = max(end - chunk_overlap, start + 1)
            chunk_index += 1
        chunk_index += 1
    return chunks

from app.ingestion.chunker import chunk_text


def test_chunker_preserves_sections() -> None:
    text = "# Intro\nThis is a short intro.\n# Protect\nProtective controls include MFA and logging."
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)

    assert chunks
    assert chunks[0].section == "Intro"
    assert any(chunk.section == "Protect" for chunk in chunks)
    assert all(chunk.text for chunk in chunks)

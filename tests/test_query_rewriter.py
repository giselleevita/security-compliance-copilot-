from app.retrieval.query_rewriter import QueryRewriter


def test_query_rewriter_expands_csf() -> None:
    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite("What does CSF say?")
    assert "NIST Cybersecurity Framework (CSF)" in rewritten


def test_query_rewriter_trims_and_expands_ai_rmf() -> None:
    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite("   What does AI RMF say about governance?   ")
    assert rewritten.startswith("What does NIST AI RMF")
    assert rewritten.endswith("?")

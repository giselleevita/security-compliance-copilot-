import argparse
import textwrap

from app.core.dependencies import get_retrieval_service
from app.ranking.reranker import SimpleReranker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare retrieval results before and after reranking.")
    parser.add_argument("query", help="Question to run against the retrieval index.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of chunks to retrieve.")
    parser.add_argument("--top-n", type=int, default=6, help="Number of reranked chunks to keep.")
    return parser


def preview(text: str, width: int = 160) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder="...")


def print_chunks(title: str, chunks: list) -> None:
    print(title)
    print("-" * len(title))
    if not chunks:
        print("No chunks\n")
        return
    for index, chunk in enumerate(chunks, start=1):
        rerank_suffix = f" rerank={chunk.rerank_score:.4f}" if chunk.rerank_score is not None else ""
        print(f"[{index}] score={chunk.score:.4f}{rerank_suffix} title={chunk.title}")
        print(f"    framework={chunk.framework} section={chunk.section}")
        print(f"    preview={preview(chunk.text)}")
    print()


def main() -> None:
    args = build_parser().parse_args()
    retrieval_service = get_retrieval_service()
    reranker = SimpleReranker()
    rewritten_query = retrieval_service.rewrite_question(args.query)

    retrieved = retrieval_service.retrieve(args.query, top_k=args.top_k)
    reranked = reranker.rerank(retrieved, limit=args.top_n)

    print(f"Query: {args.query}\n")
    print(f"Rewritten query: {rewritten_query}\n")
    print_chunks("Original retrieval", retrieved)
    print_chunks("Reranked", reranked)


if __name__ == "__main__":
    main()

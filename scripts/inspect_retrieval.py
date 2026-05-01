import argparse
import textwrap

from app.core.dependencies import get_retrieval_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect top retrieved chunks from the Chroma index.")
    parser.add_argument("query", help="Question to run against the retrieval index.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum score threshold applied after retrieval.",
    )
    return parser


def preview(text: str, width: int = 240) -> str:
    compact = " ".join(text.split())
    return textwrap.shorten(compact, width=width, placeholder="...")


def main() -> None:
    args = build_parser().parse_args()
    retrieval_service = get_retrieval_service()
    rewritten_query = retrieval_service.rewrite_question(args.query)
    chunks = retrieval_service.retrieve(
        args.query,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    print(f"Query: {args.query}")
    print(f"Rewritten query: {rewritten_query}")
    print(f"Retrieved chunks: {len(chunks)}")
    print()

    if not chunks:
        print("No retrieval results.")
        return

    for index, chunk in enumerate(chunks, start=1):
        print(f"[{index}] score={chunk.score:.4f} title={chunk.title}")
        print(f"    framework={chunk.framework} publisher={chunk.publisher}")
        print(f"    section={chunk.section} chunk_index={chunk.chunk_index} source_id={chunk.source_id}")
        print(f"    url={chunk.url}")
        print(f"    preview={preview(chunk.text)}")
        print()


if __name__ == "__main__":
    main()

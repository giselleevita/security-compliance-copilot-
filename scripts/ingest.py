from app.core.dependencies import get_ingestion_pipeline


def main() -> None:
    result = get_ingestion_pipeline().run()
    print(f"Ingested documents: {result.documents_processed}")
    print(f"Stored chunks: {result.chunks_stored}")


if __name__ == "__main__":
    main()

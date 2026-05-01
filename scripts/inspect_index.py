from app.core.dependencies import get_vector_store


def main() -> None:
    store = get_vector_store()
    print(f"Collection count: {store.count()}")


if __name__ == "__main__":
    main()

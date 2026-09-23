from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"


def main():
    print("=" * 60)
    print("EMBEDDING MODEL TEST")
    print("=" * 60)

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    test_texts = [
        "Transformers use self-attention mechanisms.",
        "Retrieval augmented generation combines retrieval with generation.",
        "LoRA reduces the number of trainable parameters.",
    ]

    embeddings = model.encode(
        test_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("\nEmbedding test successful.")
    print(f"Number of texts: {len(test_texts)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
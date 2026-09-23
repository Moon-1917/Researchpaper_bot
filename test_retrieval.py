from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


VECTORSTORE_PATH = Path(
    "vectorstore/chroma_bge_m3"
)

MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "research_papers_bge_m3"


def main():

    print("=" * 70)
    print("SEMANTIC RETRIEVAL TEST")
    print("=" * 70)

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_PATH)
    )

    collection = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Collection contains: "
        f"{collection.count():,} chunks"
    )

    queries = [
        "What architecture does the Transformer use?",
        "How does BERT perform pre-training?",
        "What is Retrieval-Augmented Generation?",
        "How does LoRA reduce the number of trainable parameters?",
        "What is Chain-of-Thought prompting?",
        "How is RLHF used to train language models?",
        "What is SELF-RAG?",
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=3,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):

            print(f"\n--- Result {rank} ---")

            print(
                f"Paper: "
                f"{metadata['paper_title']}"
            )

            print(
                f"Page: "
                f"{metadata['page']}"
            )

            print(
                f"Chunk: "
                f"{metadata['chunk_id']}"
            )

            print(
                f"Cosine distance: "
                f"{distance:.4f}"
            )

            print(
                "\nText:"
            )

            print(
                document[:700]
                .replace("\n", " ")
            )


if __name__ == "__main__":
    main()
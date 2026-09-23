from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("outputs/chunks.json")
VECTORSTORE_PATH = Path("vectorstore/chroma_bge_m3")

MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "research_papers_bge_m3"


def load_chunks():
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("=" * 60)
    print("BUILDING CHROMADB VECTOR STORE")
    print("=" * 60)

    print(f"Loading chunks from: {CHUNKS_PATH}")

    chunks = load_chunks()

    print(f"Chunks loaded: {len(chunks):,}")

    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding chunks...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=True,
    )

    print(f"Embedding shape: {embeddings.shape}")

    print("\nCreating ChromaDB...")

    VECTORSTORE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_PATH)
    )

    # Delete an existing collection so this script is reproducible.
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Existing collection removed.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "embedding_model": MODEL_NAME,
            "distance_metric": "cosine",
        },
    )

    ids = [
        chunk["metadata"]["chunk_id"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print("\nVector store created successfully.")

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents: {collection.count()}")
    print(f"Location: {VECTORSTORE_PATH}")


if __name__ == "__main__":
    main()
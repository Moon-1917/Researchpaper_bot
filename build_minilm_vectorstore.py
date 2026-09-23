import json
import os

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "outputs/chunks.json"

VECTORSTORE_PATH = "vectorstore/chroma_minilm"

COLLECTION_NAME = "research_papers_minilm"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32


# ============================================================
# LOAD CHUNKS
# ============================================================

print("=" * 92)
print("# BUILD MINILM VECTOR STORE")
print("=" * 92)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"\nLoaded {len(chunks)} chunks")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading MiniLM...")

model = SentenceTransformer(MODEL_NAME)

print(
    f"Embedding dimension: "
    f"{model.get_embedding_dimension()}"
)


# ============================================================
# PREPARE DATA
# ============================================================

texts = [
    chunk["text"]
    for chunk in chunks
]

ids = [
    chunk["metadata"]["chunk_id"]
    for chunk in chunks
]

metadatas = [
    chunk["metadata"]
    for chunk in chunks
]


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("\nCreating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

print(
    f"Embedding shape: {embeddings.shape}"
)


# ============================================================
# CREATE CHROMA DATABASE
# ============================================================

print("\nCreating ChromaDB...")

client = chromadb.PersistentClient(
    path=VECTORSTORE_PATH
)

# Delete existing collection for reproducibility
try:
    client.delete_collection(
        name=COLLECTION_NAME
    )
    print("Deleted existing collection")
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={
        "embedding_model": MODEL_NAME,
        "embedding_dimension": 384,
    },
)


# ============================================================
# INSERT IN BATCHES
# ============================================================

print("\nAdding documents to ChromaDB...")

for start in range(
    0,
    len(chunks),
    BATCH_SIZE,
):

    end = min(
        start + BATCH_SIZE,
        len(chunks),
    )

    collection.add(
        ids=ids[start:end],
        documents=texts[start:end],
        metadatas=metadatas[start:end],
        embeddings=embeddings[start:end].tolist(),
    )

    print(
        f"Added {end}/{len(chunks)} chunks"
    )


# ============================================================
# VERIFY
# ============================================================

print("\n")
print("=" * 92)
print("## VERIFICATION")
print("=" * 92)

print(
    f"Collection count: "
    f"{collection.count()}"
)

print(
    f"Vector store: "
    f"{VECTORSTORE_PATH}"
)

print(
    "\nMiniLM vector store built successfully."
)
from pathlib import Path
import json
import math

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("outputs/chunks.json")
VECTORSTORE_PATH = Path("vectorstore/chroma_bge_m3")

MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "research_papers_bge_m3"


def load_chunks():
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    denominator = (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def dense_retrieval(
    query_embedding,
    collection,
    top_k=3,
):
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    output = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(
            {
                "text": document,
                "metadata": metadata,
                "score": 1.0 - distance,
            }
        )

    return output


def mmr_retrieval(
    query_embedding,
    collection,
    chunks,
    top_k=3,
    fetch_k=10,
    lambda_mult=0.7,
):
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=fetch_k,
        include=[
            "documents",
            "metadatas",
            "embeddings",
        ],
    )

    candidate_documents = results["documents"][0]
    candidate_metadata = results["metadatas"][0]
    candidate_embeddings = np.asarray(
        results["embeddings"][0]
    )

    selected = []
    selected_indices = []

    for _ in range(
        min(top_k, len(candidate_documents))
    ):

        best_index = None
        best_score = -float("inf")

        for i in range(len(candidate_documents)):

            if i in selected_indices:
                continue

            relevance = cosine_similarity(
                query_embedding,
                candidate_embeddings[i],
            )

            if not selected_indices:
                diversity_penalty = 0.0
            else:
                similarities = [
                    cosine_similarity(
                        candidate_embeddings[i],
                        candidate_embeddings[j],
                    )
                    for j in selected_indices
                ]

                diversity_penalty = max(similarities)

            mmr_score = (
                lambda_mult * relevance
                - (1 - lambda_mult)
                * diversity_penalty
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = i

        selected_indices.append(best_index)

        selected.append(
            {
                "text": candidate_documents[best_index],
                "metadata": candidate_metadata[best_index],
                "score": best_score,
            }
        )

    return selected


def build_bm25(chunks):
    tokenized_corpus = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_corpus)


def bm25_retrieval(
    query,
    bm25,
    chunks,
    top_k=3,
):
    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "text": chunks[index]["text"],
                "metadata": chunks[index]["metadata"],
                "score": float(scores[index]),
            }
        )

    return results


def normalize_scores(results):
    if not results:
        return results

    scores = [
        result["score"]
        for result in results
    ]

    min_score = min(scores)
    max_score = max(scores)

    if math.isclose(max_score, min_score):
        for result in results:
            result["normalized_score"] = 1.0

        return results

    for result in results:
        result["normalized_score"] = (
            result["score"] - min_score
        ) / (max_score - min_score)

    return results


def hybrid_retrieval(
    query,
    query_embedding,
    bm25,
    chunks,
    collection,
    top_k=3,
    candidate_k=10,
    alpha=0.5,
):
    # Dense candidates
    dense_results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=candidate_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    dense_candidates = {}

    for document, metadata, distance in zip(
        dense_results["documents"][0],
        dense_results["metadatas"][0],
        dense_results["distances"][0],
    ):

        chunk_id = metadata["chunk_id"]

        dense_candidates[chunk_id] = {
            "text": document,
            "metadata": metadata,
            "dense_score": 1.0 - distance,
        }

    # BM25 candidates
    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_top_indices = np.argsort(
        bm25_scores
    )[::-1][:candidate_k]

    candidates = {}

    for index in bm25_top_indices:

        chunk = chunks[index]
        chunk_id = chunk["metadata"]["chunk_id"]

        candidates[chunk_id] = {
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "bm25_score": float(
                bm25_scores[index]
            ),
        }

    # Merge dense + BM25 candidate sets
    for chunk_id, item in dense_candidates.items():

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "text": item["text"],
                "metadata": item["metadata"],
                "bm25_score": 0.0,
            }

        candidates[chunk_id]["dense_score"] = (
            item["dense_score"]
        )

    for item in candidates.values():

        if "dense_score" not in item:
            item["dense_score"] = 0.0

        if "bm25_score" not in item:
            item["bm25_score"] = 0.0

    candidate_list = list(
        candidates.values()
    )

    # Normalize each scoring system
    dense_scores = [
        item["dense_score"]
        for item in candidate_list
    ]

    bm25_scores_list = [
        item["bm25_score"]
        for item in candidate_list
    ]

    def min_max(values):

        minimum = min(values)
        maximum = max(values)

        if math.isclose(
            minimum,
            maximum,
        ):
            return [1.0] * len(values)

        return [
            (value - minimum)
            / (maximum - minimum)
            for value in values
        ]

    dense_normalized = min_max(
        dense_scores
    )

    bm25_normalized = min_max(
        bm25_scores_list
    )

    for item, dense_score, bm25_score in zip(
        candidate_list,
        dense_normalized,
        bm25_normalized,
    ):

        item["hybrid_score"] = (
            alpha * dense_score
            + (1 - alpha) * bm25_score
        )

    candidate_list.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )

    results = []

    for item in candidate_list[:top_k]:

        results.append(
            {
                "text": item["text"],
                "metadata": item["metadata"],
                "score": item["hybrid_score"],
            }
        )

    return results


def print_results(
    method,
    results,
):
    print(
        f"\n{'-' * 60}"
    )
    print(method)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        metadata = result["metadata"]

        print(
            f"\n#{rank}"
        )

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
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            "Text: "
            + result["text"][:300]
            .replace("\n", " ")
        )


def main():

    print("=" * 70)
    print("RETRIEVAL STRATEGY EXPERIMENT")
    print("=" * 70)

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks):,} chunks."
    )

    print("\nLoading BGE-M3...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_PATH)
    )

    collection = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Vector store: "
        f"{collection.count():,} chunks"
    )

    print("\nBuilding BM25 index...")

    bm25 = build_bm25(chunks)

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

        print("\n\n")
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        dense_results = dense_retrieval(
            query_embedding,
            collection,
            top_k=3,
        )

        mmr_results = mmr_retrieval(
            query_embedding,
            collection,
            chunks,
            top_k=3,
            fetch_k=10,
            lambda_mult=0.7,
        )

        bm25_results = bm25_retrieval(
            query,
            bm25,
            chunks,
            top_k=3,
        )

        hybrid_results = hybrid_retrieval(
            query,
            query_embedding,
            bm25,
            chunks,
            collection,
            top_k=3,
            candidate_k=10,
            alpha=0.5,
        )

        print_results(
            "DENSE COSINE",
            dense_results,
        )

        print_results(
            "MMR",
            mmr_results,
        )

        print_results(
            "BM25",
            bm25_results,
        )

        print_results(
            "HYBRID BM25 + DENSE",
            hybrid_results,
        )


if __name__ == "__main__":
    main()
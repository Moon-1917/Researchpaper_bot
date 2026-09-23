import json
import os

import chromadb
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "outputs/chunks.json"
EVAL_PATH = "outputs/retrieval_eval.json"
VECTORSTORE_PATH = "vectorstore/chroma_bge_m3"
COLLECTION_NAME = "research_papers_bge_m3"

MODEL_NAME = "BAAI/bge-m3"

TOP_K = 3
FETCH_K = 10

# Hybrid weighting
ALPHA = 0.5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 92)
print("# RETRIEVAL EVALUATION")
print("=" * 92)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_questions = json.load(f)

print(f"\nLoaded {len(chunks)} chunks")
print(f"Loaded {len(eval_questions)} evaluation questions")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},
)


# ============================================================
# LOAD CHROMA
# ============================================================

print("Loading ChromaDB...")

client = chromadb.PersistentClient(path=VECTORSTORE_PATH)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"Chroma collection count: {collection.count()}")


# ============================================================
# BM25
# ============================================================

print("Building BM25 index...")

corpus = [
    chunk["text"].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(corpus)


# ============================================================
# DENSE RETRIEVAL
# ============================================================

def dense_search(query, top_k=TOP_K):

    query_embedding = embeddings.embed_query(query)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    output = []

    for metadata, distance in zip(
        result["metadatas"][0],
        result["distances"][0],
    ):
        output.append(
            {
                "metadata": metadata,
                "score": 1 - distance,
            }
        )

    return output


# ============================================================
# MMR RETRIEVAL
# ============================================================

def mmr_search(query, top_k=TOP_K, fetch_k=FETCH_K, lambda_mult=0.7):

    query_embedding = embeddings.embed_query(query)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["embeddings", "metadatas", "documents", "distances"],
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    embeddings_list = result["embeddings"][0]

    selected = []
    remaining = list(range(len(documents)))

    # Query similarity
    query_similarities = []

    for emb in embeddings_list:
        dot = sum(
            a * b
            for a, b in zip(query_embedding, emb)
        )
        query_similarities.append(dot)

    for _ in range(min(top_k, len(remaining))):

        if not selected:
            best_idx = max(
                remaining,
                key=lambda i: query_similarities[i],
            )

        else:
            def mmr_score(i):

                relevance = query_similarities[i]

                max_similarity = max(
                    sum(
                        a * b
                        for a, b in zip(
                            embeddings_list[i],
                            embeddings_list[j],
                        )
                    )
                    for j in selected
                )

                return (
                    lambda_mult * relevance
                    - (1 - lambda_mult) * max_similarity
                )

            best_idx = max(
                remaining,
                key=mmr_score,
            )

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [
        {
            "metadata": metadatas[i],
            "score": query_similarities[i],
        }
        for i in selected
    ]


# ============================================================
# BM25 RETRIEVAL
# ============================================================

def bm25_search(query, top_k=TOP_K):

    scores = bm25.get_scores(
        query.lower().split()
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [
        {
            "metadata": chunks[i]["metadata"],
            "score": float(scores[i]),
        }
        for i in ranked_indices
    ]


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def min_max_normalize(scores):

    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0 for _ in scores]

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def hybrid_search(query, top_k=TOP_K, candidate_k=FETCH_K):

    query_embedding = embeddings.embed_query(query)

    # Dense candidates
    dense_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["metadatas", "distances"],
    )

    dense_metadatas = dense_result["metadatas"][0]
    dense_distances = dense_result["distances"][0]

    dense_scores = [
        1 - distance
        for distance in dense_distances
    ]

    # BM25 candidates
    bm25_scores_all = bm25.get_scores(
        query.lower().split()
    )

    bm25_indices = sorted(
        range(len(bm25_scores_all)),
        key=lambda i: bm25_scores_all[i],
        reverse=True,
    )[:candidate_k]

    # Merge candidates
    candidates = {}

    for metadata, score in zip(
        dense_metadatas,
        dense_scores,
    ):
        chunk_id = metadata["chunk_id"]

        candidates[chunk_id] = {
            "metadata": metadata,
            "dense_score": float(score),
            "bm25_score": 0.0,
        }

    for i in bm25_indices:

        metadata = chunks[i]["metadata"]
        chunk_id = metadata["chunk_id"]

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "metadata": metadata,
                "dense_score": 0.0,
                "bm25_score": 0.0,
            }

        candidates[chunk_id]["bm25_score"] = float(
            bm25_scores_all[i]
        )

    candidate_list = list(candidates.values())

    # Normalize scores
    dense_values = [
        item["dense_score"]
        for item in candidate_list
    ]

    bm25_values = [
        item["bm25_score"]
        for item in candidate_list
    ]

    normalized_dense = min_max_normalize(
        dense_values
    )

    normalized_bm25 = min_max_normalize(
        bm25_values
    )

    # Combine scores
    for item, dense_norm, bm25_norm in zip(
        candidate_list,
        normalized_dense,
        normalized_bm25,
    ):
        item["score"] = (
            ALPHA * dense_norm
            + (1 - ALPHA) * bm25_norm
        )

    candidate_list.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidate_list[:top_k]

# ============================================================
# METRICS
# ============================================================

def calculate_metrics(results, expected_papers):

    ranks = []

    for rank, result in enumerate(results, start=1):

        paper_id = result["metadata"]["paper_id"]

        if paper_id in expected_papers:
            ranks.append(rank)

    hit1 = 1.0 if 1 in ranks else 0.0

    hit3 = 1.0 if ranks else 0.0

    mrr = (
        1.0 / min(ranks)
        if ranks
        else 0.0
    )

    return hit1, hit3, mrr


# ============================================================
# EVALUATION
# ============================================================

methods = {
    "Dense": dense_search,
    "MMR": mmr_search,
    "BM25": bm25_search,
    "Hybrid": hybrid_search,
}

all_results = {}

for method_name, search_function in methods.items():

    print(f"\nRunning {method_name} evaluation...")

    method_results = []

    for question in eval_questions:

        query = question["question"]

        expected_papers = question["expected_papers"]

        results = search_function(
            query,
            TOP_K,
        )

        hit1, hit3, mrr = calculate_metrics(
            results,
            expected_papers,
        )

        method_results.append(
            {
                "question_id": question["id"],
                "question": query,
                "expected_papers": expected_papers,
                "results": results,
                "hit_at_1": hit1,
                "hit_at_3": hit3,
                "mrr": mrr,
            }
        )

    all_results[method_name] = method_results


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 92)
print("## RESULTS")
print("=" * 92)

summary = {}

for method_name, results in all_results.items():

    hit1 = sum(
        r["hit_at_1"]
        for r in results
    ) / len(results)

    hit3 = sum(
        r["hit_at_3"]
        for r in results
    ) / len(results)

    mrr = sum(
        r["mrr"]
        for r in results
    ) / len(results)

    summary[method_name] = {
        "Hit@1": hit1,
        "Hit@3": hit3,
        "MRR": mrr,
    }

    print(
        f"{method_name:<10}"
        f"Hit@1={hit1:.3f} "
        f"Hit@3={hit3:.3f} "
        f"MRR={mrr:.3f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output = {
    "configuration": {
        "embedding_model": MODEL_NAME,
        "top_k": TOP_K,
        "fetch_k": FETCH_K,
        "hybrid_alpha": ALPHA,
    },
    "summary": summary,
    "detailed_results": all_results,
}

OUTPUT_PATH = "outputs/retrieval_evaluation_all_methods.json"

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(
    f"\nDetailed results saved to: {OUTPUT_PATH}"
)
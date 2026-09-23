import json

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "outputs/chunks.json"
EVAL_PATH = "outputs/retrieval_eval.json"

VECTORSTORE_PATH = "vectorstore/chroma_minilm"
COLLECTION_NAME = "research_papers_minilm"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 92)
print("# MINILM RETRIEVAL EVALUATION")
print("=" * 92)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_questions = json.load(f)

print(f"\nLoaded {len(chunks)} chunks")
print(f"Loaded {len(eval_questions)} evaluation questions")


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
# LOAD CHROMA
# ============================================================

print("Loading ChromaDB...")

client = chromadb.PersistentClient(
    path=VECTORSTORE_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"Chroma collection count: "
    f"{collection.count()}"
)


# ============================================================
# RETRIEVAL
# ============================================================

def search(query, top_k=TOP_K):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "metadatas",
            "distances",
        ],
    )

    results = []

    for metadata, distance in zip(
        result["metadatas"][0],
        result["distances"][0],
    ):

        results.append(
            {
                "metadata": metadata,
                "score": 1 - distance,
            }
        )

    return results


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    results,
    expected_papers,
):

    ranks = []

    for rank, result in enumerate(
        results,
        start=1,
    ):

        paper_id = result["metadata"]["paper_id"]

        if paper_id in expected_papers:
            ranks.append(rank)

    hit1 = (
        1.0
        if 1 in ranks
        else 0.0
    )

    hit3 = (
        1.0
        if ranks
        else 0.0
    )

    mrr = (
        1.0 / min(ranks)
        if ranks
        else 0.0
    )

    return hit1, hit3, mrr


# ============================================================
# EVALUATION
# ============================================================

print("\nRunning MiniLM evaluation...")

all_results = []

for question in eval_questions:

    query = question["question"]

    expected_papers = (
        question["expected_papers"]
    )

    results = search(
        query,
        TOP_K,
    )

    hit1, hit3, mrr = calculate_metrics(
        results,
        expected_papers,
    )

    all_results.append(
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


# ============================================================
# SUMMARY
# ============================================================

hit1 = sum(
    result["hit_at_1"]
    for result in all_results
) / len(all_results)

hit3 = sum(
    result["hit_at_3"]
    for result in all_results
) / len(all_results)

mrr = sum(
    result["mrr"]
    for result in all_results
) / len(all_results)


print("\n")
print("=" * 92)
print("## RESULTS")
print("=" * 92)

print(
    f"MiniLM    "
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
        "embedding_dimension": 384,
        "top_k": TOP_K,
    },
    "metrics": {
        "Hit@1": hit1,
        "Hit@3": hit3,
        "MRR": mrr,
    },
    "detailed_results": all_results,
}

OUTPUT_PATH = "outputs/minilm_evaluation.json"

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
    f"\nDetailed results saved to: "
    f"{OUTPUT_PATH}"
)
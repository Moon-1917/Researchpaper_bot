import json

from sentence_transformers import CrossEncoder
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "outputs/chunks.json"
EVAL_PATH = "outputs/retrieval_eval.json"

VECTORSTORE_PATH = "vectorstore/chroma_bge_m3"
COLLECTION_NAME = "research_papers_bge_m3"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

CANDIDATE_K = 10
TOP_K = 3


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 92)
print("# RERANKER EVALUATION")
print("=" * 92)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_questions = json.load(f)

print(f"\nLoaded {len(chunks)} chunks")
print(f"Loaded {len(eval_questions)} evaluation questions")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading BGE-M3 embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={
        "normalize_embeddings": True
    },
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
    f"Chroma collection count: {collection.count()}"
)


# ============================================================
# LOAD RERANKER
# ============================================================

print("\nLoading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker loaded successfully")


# ============================================================
# DENSE CANDIDATE RETRIEVAL
# ============================================================

def retrieve_candidates(query, candidate_k=CANDIDATE_K):

    query_embedding = embeddings.embed_query(
        query
    )

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=[
            "documents",
            "metadatas",
        ],
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    return [
        {
            "text": document,
            "metadata": metadata,
        }
        for document, metadata in zip(
            documents,
            metadatas,
        )
    ]


# ============================================================
# RERANK
# ============================================================

def rerank(query, candidates, top_k=TOP_K):

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked = []

    for candidate, score in zip(
        candidates,
        scores,
    ):
        reranked.append(
            {
                "metadata": candidate["metadata"],
                "score": float(score),
            }
        )

    reranked.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return reranked[:top_k]


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

all_results = []

print("\nRunning reranker evaluation...")

for question in eval_questions:

    query = question["question"]

    expected_papers = (
        question["expected_papers"]
    )

    candidates = retrieve_candidates(
        query,
        CANDIDATE_K,
    )

    results = rerank(
        query,
        candidates,
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
    f"Reranker  "
    f"Hit@1={hit1:.3f} "
    f"Hit@3={hit3:.3f} "
    f"MRR={mrr:.3f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = {
    "configuration": {
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
    },
    "metrics": {
        "Hit@1": hit1,
        "Hit@3": hit3,
        "MRR": mrr,
    },
    "detailed_results": all_results,
}

OUTPUT_PATH = (
    "outputs/reranker_evaluation.json"
)

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
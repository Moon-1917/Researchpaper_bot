import json
import os


# ============================================================
# EXPERIMENT SUMMARY
# Research Paper Answer Bot
# ============================================================

OUTPUT_PATH = "outputs/experiment_summary.json"


summary = {
    "project": "Research Paper Answer Bot",

    "dataset": {
        "number_of_papers": 9,
        "number_of_pages": 319,
        "number_of_chunks": 1291,
        "chunk_size": 1000,
        "chunk_overlap": 150
    },

    "evaluation": {
        "number_of_questions": 10,
        "metrics": [
            "Hit@1",
            "Hit@3",
            "MRR"
        ]
    },

    "embedding_comparison": [
        {
            "model": "BAAI/bge-m3",
            "dimension": 1024,
            "retrieval_method": "Dense",
            "Hit@1": 1.000,
            "Hit@3": 1.000,
            "MRR": 1.000,
            "selected_for_final_pipeline": True
        },
        {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "retrieval_method": "Dense",
            "Hit@1": 1.000,
            "Hit@3": 1.000,
            "MRR": 1.000,
            "selected_for_final_pipeline": False
        }
    ],

    "retrieval_comparison": [
        {
            "method": "BGE-M3 Dense",
            "Hit@1": 1.000,
            "Hit@3": 1.000,
            "MRR": 1.000
        },
        {
            "method": "BGE-M3 MMR",
            "Hit@1": 1.000,
            "Hit@3": 1.000,
            "MRR": 1.000
        },
        {
            "method": "BGE-M3 + Reranker",
            "Hit@1": 1.000,
            "Hit@3": 1.000,
            "MRR": 1.000
        },
        {
            "method": "Hybrid BM25 + Dense",
            "Hit@1": 0.900,
            "Hit@3": 0.900,
            "MRR": 0.900
        },
        {
            "method": "BM25",
            "Hit@1": 0.500,
            "Hit@3": 0.700,
            "MRR": 0.600
        }
    ],

    "final_retrieval_configuration": {
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimension": 1024,
        "initial_retrieval": "Dense vector retrieval",
        "candidate_k": 10,
        "reranker": "BAAI/bge-reranker-v2-m3",
        "final_top_k": 3
    },

    "llm": {
        "provider": "Google Gemini",
        "model": "gemini-2.5-flash",
        "temperature": 0
    },

    "selection_reason": (
        "BGE-M3 was selected for the final pipeline because it "
        "achieved perfect retrieval metrics on the 10-question "
        "evaluation set and provides a stronger semantic retrieval "
        "baseline than the smaller MiniLM model. The reranker also "
        "maintained perfect retrieval performance and is used as a "
        "second-stage relevance ranking component."
    ),

    "limitations": [
        "The retrieval evaluation contains only 10 questions.",
        "Perfect retrieval metrics on this evaluation set do not "
        "establish universal superiority.",
        "Answer quality and groundedness require separate evaluation.",
        "The reranker maintained retrieval performance but was not "
        "shown to improve Hit@1 over the already-perfect dense baseline."
    ]
}


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# DISPLAY
# ============================================================

print("=" * 92)
print("EXPERIMENT SUMMARY")
print("=" * 92)

print("\nDataset:")
print("  Papers :", summary["dataset"]["number_of_papers"])
print("  Pages  :", summary["dataset"]["number_of_pages"])
print("  Chunks :", summary["dataset"]["number_of_chunks"])

print("\nEmbedding comparison:")

for item in summary["embedding_comparison"]:
    print(
        f"  {item['model']}: "
        f"Hit@1={item['Hit@1']:.3f}, "
        f"Hit@3={item['Hit@3']:.3f}, "
        f"MRR={item['MRR']:.3f}"
    )

print("\nRetrieval comparison:")

for item in summary["retrieval_comparison"]:
    print(
        f"  {item['method']:<25} "
        f"Hit@1={item['Hit@1']:.3f} "
        f"Hit@3={item['Hit@3']:.3f} "
        f"MRR={item['MRR']:.3f}"
    )

print("\nFinal configuration:")
print(
    f"  Embedding : "
    f"{summary['final_retrieval_configuration']['embedding_model']}"
)

print(
    f"  Retrieval : "
    f"{summary['final_retrieval_configuration']['initial_retrieval']}"
)

print(
    f"  Reranker  : "
    f"{summary['final_retrieval_configuration']['reranker']}"
)

print(
    f"  Top-K     : "
    f"{summary['final_retrieval_configuration']['final_top_k']}"
)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)
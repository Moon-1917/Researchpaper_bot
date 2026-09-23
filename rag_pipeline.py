import json
import os

import chromadb
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "outputs/chunks.json"

VECTORSTORE_PATH = "vectorstore/chroma_bge_m3"
COLLECTION_NAME = "research_papers_bge_m3"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

LLM_MODEL = "gemini-2.5-flash"

CANDIDATE_K = 10
TOP_K = 3


# ============================================================
# LOAD CHUNKS
# ============================================================

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8",
) as f:
    chunks = json.load(f)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading BGE-M3...")

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


# ============================================================
# LOAD RERANKER
# ============================================================

print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)


# ============================================================
# LOAD GEMINI
# ============================================================

print("Loading Gemini...")

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0,
)


# ============================================================
# RETRIEVE CANDIDATES
# ============================================================

def retrieve_candidates(
    query,
    candidate_k=CANDIDATE_K,
):

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

    candidates = []

    for document, metadata in zip(
        result["documents"][0],
        result["metadatas"][0],
    ):

        candidates.append(
            {
                "text": document,
                "metadata": metadata,
            }
        )

    return candidates


# ============================================================
# RERANK CANDIDATES
# ============================================================

def rerank_candidates(
    query,
    candidates,
    top_k=TOP_K,
):

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
                "text": candidate["text"],
                "metadata": candidate["metadata"],
                "reranker_score": float(score),
            }
        )

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    return reranked[:top_k]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(passages):

    context_parts = []

    for i, passage in enumerate(
        passages,
        start=1,
    ):

        metadata = passage["metadata"]

        title = metadata.get(
            "paper_title",
            "Unknown paper",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        text = passage["text"]

        context_parts.append(
            f"""
PASSAGE {i}
Paper: {title}
Page: {page}

{text}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    passages,
):

    context = build_context(
        passages
    )

    prompt = f"""
You are a research-paper question answering assistant.

Answer the user's question using ONLY the supplied
research-paper passages.

Do not use outside knowledge.

If the passages do not contain enough information to
answer the question, clearly say:

"Insufficient information in the provided research papers."

Do not invent facts, citations, page numbers, or claims.

Keep the answer concise but informative.

After the answer, provide a "Sources" section.

For every source include:
- Paper title
- Page number
- The supporting passage

USER QUESTION:
{query}

RESEARCH PAPER CONTEXT:
{context}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def answer_question(query):

    candidates = retrieve_candidates(
        query,
        CANDIDATE_K,
    )

    passages = rerank_candidates(
        query,
        candidates,
        TOP_K,
    )

    answer = generate_answer(
        query,
        passages,
    )

    return {
        "question": query,
        "answer": answer,
        "retrieved_passages": passages,
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 92)
    print("RESEARCH PAPER ANSWER BOT")
    print("=" * 92)

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        print("No question provided.")
        raise SystemExit

    print("\nRetrieving relevant passages...")

    result = answer_question(
        question
    )

    print("\n")
    print("=" * 92)
    print("ANSWER")
    print("=" * 92)

    print(result["answer"])

    print("\n")
    print("=" * 92)
    print("RETRIEVED SUPPORTING PASSAGES")
    print("=" * 92)

    for i, passage in enumerate(
        result["retrieved_passages"],
        start=1,
    ):

        metadata = passage["metadata"]

        print(
            f"\n[{i}] "
            f"{metadata['paper_title']} "
            f"(Page {metadata['page']})"
        )

        print(
            f"Reranker score: "
            f"{passage['reranker_score']:.4f}"
        )

        print(
            f"\n{passage['text']}"
        )
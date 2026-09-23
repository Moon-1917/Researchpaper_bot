import sys
from pathlib import Path

from tavily import TavilyClient


# Add src directory to Python path
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import retrieve_candidates, rerank_candidates


def get_tavily_client():
    """
    Create the Tavily client using the configured API key.
    """
    import os

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured."
        )

    return TavilyClient(api_key=api_key)


def evaluate_local_retrieval(passages):
    """
    Lightweight corrective-RAG decision.

    The local retrieval is considered strong when the top
    reranked passages have sufficiently high relevance scores.
    """

    if not passages:
        return {
            "decision": "web_search",
            "reason": "No local passages were retrieved.",
        }

    scores = [
        float(p.get("reranker_score", 0))
        for p in passages
    ]

    top_score = max(scores)

    # Conservative threshold for triggering corrective search.
    if top_score >= 0.5:
        return {
            "decision": "local_rag",
            "reason": (
                f"Strong local evidence found "
                f"(top reranker score={top_score:.4f})."
            ),
        }

    return {
        "decision": "web_search",
        "reason": (
            f"Local evidence appears weak "
            f"(top reranker score={top_score:.4f})."
        ),
    }


def web_search(query, max_results=3):
    """
    Search the web with Tavily when local evidence is weak.
    """

    client = get_tavily_client()

    response = client.search(
        query,
        max_results=max_results,
        search_depth="advanced",
    )

    results = []

    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
            }
        )

    return results


def crag_retrieve(query, candidate_k=10, top_k=3):
    """
    Corrective RAG retrieval.

    1. Retrieve from the research-paper knowledge base.
    2. Rerank local passages.
    3. Evaluate retrieval quality.
    4. If evidence is weak, perform Tavily web search.
    """

    candidates = retrieve_candidates(
        query,
        candidate_k=candidate_k,
    )

    local_passages = rerank_candidates(
        query,
        candidates,
        top_k=top_k,
    )

    evaluation = evaluate_local_retrieval(
        local_passages
    )

    result = {
        "query": query,
        "decision": evaluation["decision"],
        "reason": evaluation["reason"],
        "local_passages": local_passages,
        "web_results": [],
    }

    if evaluation["decision"] == "web_search":
        result["web_results"] = web_search(query)

    return result


def print_result(result):
    print("\n" + "=" * 80)
    print("CRAG RESULT")
    print("=" * 80)

    print(f"\nQuery: {result['query']}")
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")

    print("\nLocal research-paper passages:")
    print("-" * 80)

    for i, passage in enumerate(
        result["local_passages"],
        start=1,
    ):
        metadata = passage.get(
            "metadata",
            {},
        )

        print(
            f"\n[{i}] "
            f"{metadata.get('paper_title', 'Unknown')} "
            f"(Page {metadata.get('page', '?')})"
        )

        print(
            "Reranker score:",
            passage.get("reranker_score"),
        )

    if result["web_results"]:

        print("\nTavily web results:")
        print("-" * 80)

        for i, item in enumerate(
            result["web_results"],
            start=1,
        ):
            print(
                f"\n[{i}] {item['title']}"
            )

            print(
                f"URL: {item['url']}"
            )

            print(
                f"Score: {item['score']}"
            )

            print(
                item["content"][:400]
            )


def main():

    print("=" * 80)
    print("CORRECTIVE RAG TEST")
    print("=" * 80)

    print(
        "\nThis test evaluates local retrieval and "
        "uses Tavily only when evidence is weak."
    )

    print("\nType 'exit' to stop.\n")

    while True:

        query = input("Query: ").strip()

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        if not query:
            continue

        try:

            result = crag_retrieve(query)

            print_result(result)

        except Exception as e:

            print(
                f"\nCRAG error: {e}"
            )


if __name__ == "__main__":
    main()
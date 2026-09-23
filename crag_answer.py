import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crag import crag_retrieve
from rag_pipeline import llm


def build_local_context(passages):
    parts = []

    for i, passage in enumerate(passages, start=1):
        metadata = passage.get("metadata", {})

        parts.append(
            f"[Source {i}]\n"
            f"Paper: {metadata.get('paper_title', 'Unknown')}\n"
            f"Page: {metadata.get('page', '?')}\n"
            f"Passage: {passage.get('text', '')}"
        )

    return "\n\n".join(parts)


def build_web_context(results):
    parts = []

    for i, result in enumerate(results, start=1):
        parts.append(
            f"[Web Source {i}]\n"
            f"Title: {result.get('title', 'Unknown')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {result.get('content', '')}"
        )

    return "\n\n".join(parts)


def generate_crag_answer(query, result):
    if result["decision"] == "local_rag":
        context = build_local_context(
            result["local_passages"]
        )

        source_instruction = (
            "Cite the supporting research paper title "
            "and page number for relevant claims."
        )

    else:
        context = build_web_context(
            result["web_results"]
        )

        source_instruction = (
            "Cite the supporting web source title and URL "
            "for relevant claims."
        )

    prompt = f"""
You are a research assistant using Corrective Retrieval-Augmented Generation.

Answer the user's question using ONLY the supplied evidence.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge.
3. If the evidence is insufficient, clearly say that the evidence is insufficient.
4. Keep the answer concise and directly relevant.
5. {source_instruction}

User question:
{query}

Evidence:
{context}
"""

    response = llm.invoke(prompt)

    return response.content.strip()


def answer_with_crag(query):
    result = crag_retrieve(query)

    if result["decision"] == "local_rag":
        answer = generate_crag_answer(
            query,
            result,
        )
    else:
        answer = generate_crag_answer(
            query,
            result,
        )

    return {
        "query": query,
        "decision": result["decision"],
        "reason": result["reason"],
        "answer": answer,
        "local_passages": result["local_passages"],
        "web_results": result["web_results"],
    }


def main():
    print("=" * 80)
    print("CRAG RESEARCH PAPER ANSWER BOT")
    print("=" * 80)

    print("\nType 'exit' to stop.\n")

    while True:
        query = input("Query: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            result = answer_with_crag(query)

            print("\nDecision:")
            print(result["decision"])

            print("\nReason:")
            print(result["reason"])

            print("\nAnswer:")
            print(result["answer"])

            if result["decision"] == "local_rag":
                print("\nResearch-paper sources:")

                for i, passage in enumerate(
                    result["local_passages"],
                    start=1,
                ):
                    metadata = passage.get(
                        "metadata",
                        {},
                    )

                    print(
                        f"[{i}] "
                        f"{metadata.get('paper_title', 'Unknown')} "
                        f"- Page {metadata.get('page', '?')}"
                    )

            else:
                print("\nWeb sources:")

                for i, source in enumerate(
                    result["web_results"],
                    start=1,
                ):
                    print(
                        f"[{i}] "
                        f"{source.get('title', 'Unknown')}"
                    )

                    print(
                        source.get("url", "")
                    )

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
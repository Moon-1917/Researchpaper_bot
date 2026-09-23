import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import answer_question


def build_retrieval_query(question, chat_history):
    """
    Build a retrieval query using the current question and
    recent conversation history.

    No additional LLM/API call is used here.
    """

    if not chat_history:
        return question

    recent_turns = chat_history[-4:]

    history_text = "\n".join(
        f"{turn['role']}: {turn['content']}"
        for turn in recent_turns
    )

    return (
        "Conversation context:\n"
        f"{history_text}\n\n"
        "Current question:\n"
        f"{question}"
    )


def conversational_answer(question, chat_history):
    """
    Answer the current question while using previous conversation
    turns as additional retrieval context.
    """

    retrieval_query = build_retrieval_query(
        question,
        chat_history,
    )

    result = answer_question(retrieval_query)

    return {
        "question": question,
        "retrieval_query": retrieval_query,
        "answer": result["answer"],
        "retrieved_passages": result["retrieved_passages"],
    }


def print_sources(passages):
    print("\nSupporting passages:")
    print("-" * 80)

    for i, passage in enumerate(passages, start=1):
        metadata = passage.get("metadata", {})

        print(
            f"\n[{i}] "
            f"{metadata.get('paper_title', 'Unknown paper')} "
            f"(Page {metadata.get('page', '?')})"
        )

        print(
            f"Reranker score: "
            f"{passage.get('reranker_score', 'N/A')}"
        )

        print(passage.get("text", "")[:500])


def main():
    print("=" * 80)
    print("CONVERSATIONAL RESEARCH PAPER ANSWER BOT")
    print("=" * 80)

    print("\nType 'exit' to stop.")
    print(
        "Conversation history is included in retrieval "
        "without an additional LLM call.\n"
    )

    chat_history = []

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = conversational_answer(
                question,
                chat_history,
            )

            print("\nRetrieval query:")
            print(result["retrieval_query"])

            print("\nBot:")
            print(result["answer"])

            print_sources(
                result["retrieved_passages"]
            )

            chat_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            chat_history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                }
            )

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
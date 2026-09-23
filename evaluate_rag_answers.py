import json
from pathlib import Path

from rag_pipeline import answer_question


BASE_DIR = Path(__file__).resolve().parents[1]

EVAL_FILE = BASE_DIR / "outputs" / "retrieval_eval.json"
OUTPUT_FILE = BASE_DIR / "outputs" / "rag_answer_evaluation.json"


def main():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    print("=" * 80)
    print("RAG ANSWER EVALUATION")
    print("=" * 80)

    for item in questions:
        question_id = item["id"]
        question = item["question"]

        print(f"\n[{question_id}/10] {question}")

        try:
            result = answer_question(question)

            results.append(
                {
                    "id": question_id,
                    "question": question,
                    "expected_papers": item["expected_papers"],
                    "answer": result["answer"],
                    "retrieved_passages": result["retrieved_passages"],
                }
            )

            print("Status: SUCCESS")

        except Exception as e:
            print(f"Status: FAILED - {e}")

            results.append(
                {
                    "id": question_id,
                    "question": question,
                    "expected_papers": item["expected_papers"],
                    "answer": None,
                    "retrieved_passages": [],
                    "error": str(e),
                }
            )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("Evaluation completed.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
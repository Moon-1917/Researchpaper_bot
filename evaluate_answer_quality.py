import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "outputs" / "rag_answer_evaluation.json"
OUTPUT_FILE = BASE_DIR / "outputs" / "answer_quality_evaluation.json"


# Scores:
# relevance, accuracy/completeness, groundedness, citation/source quality
MANUAL_SCORES = {
    1: {
        "relevance": 5,
        "accuracy": 5,
        "groundedness": 5,
        "citation_quality": 5,
        "failure_case": "",
    },
    2: {
        "relevance": 5,
        "accuracy": 5,
        "groundedness": 5,
        "citation_quality": 5,
        "failure_case": "",
    },
    3: {
        "relevance": 5,
        "accuracy": 4,
        "groundedness": 5,
        "citation_quality": 3,
        "failure_case": (
            "The answer was grounded in SELF-RAG, but the original RAG paper "
            "(data3) was not retrieved/cited even though it was an expected "
            "supporting paper."
        ),
    },
    4: {
        "relevance": 5,
        "accuracy": 5,
        "groundedness": 5,
        "citation_quality": 5,
        "failure_case": "",
    },
    5: {
        "relevance": 5,
        "accuracy": 5,
        "groundedness": 5,
        "citation_quality": 5,
        "failure_case": "",
    },
    6: {
        "relevance": 5,
        "accuracy": 4,
        "groundedness": 5,
        "citation_quality": 5,
        "failure_case": (
            "The answer was correct but slightly incomplete: it stopped after "
            "the reward-model stage and omitted the final RL fine-tuning step."
        ),
    },
    7: {
        "relevance": 5,
        "accuracy": 5,
        "groundedness": 5,
        "citation_quality": 5,
        "failure_case": "",
    },
    8: {
        "relevance": 5,
        "accuracy": 5,
        "groundedness": 5,
        "citation_quality": 5,
        "failure_case": "",
    },
    9: {
        "relevance": 5,
        "accuracy": 5,
        "groundedness": 5,
        "citation_quality": 5,
        "failure_case": "",
    },
    10: {
        "relevance": 5,
        "accuracy": 5,
        "groundedness": 5,
        "citation_quality": 5,
        "failure_case": "",
    },
}


def get_retrieved_papers(result):
    """Return unique paper IDs retrieved for a question."""
    papers = set()

    for passage in result.get("retrieved_passages", []):
        metadata = passage.get("metadata", {})
        paper_id = metadata.get("paper_id")

        if paper_id:
            papers.add(paper_id)

    return sorted(papers)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    evaluated_results = []

    for result in results:
        question_id = result["id"]

        scores = MANUAL_SCORES[question_id]

        expected_papers = result.get("expected_papers", [])
        retrieved_papers = get_retrieved_papers(result)

        missing_expected_papers = [
            paper
            for paper in expected_papers
            if paper not in retrieved_papers
        ]

        full_paper_coverage = len(missing_expected_papers) == 0

        evaluation = {
            **result,
            "evaluation": {
                "relevance": scores["relevance"],
                "accuracy_completeness": scores["accuracy"],
                "groundedness": scores["groundedness"],
                "citation_source_quality": scores["citation_quality"],
                "failure_case": scores["failure_case"],
                "retrieved_papers": retrieved_papers,
                "expected_papers_retrieved": [
                    paper
                    for paper in expected_papers
                    if paper in retrieved_papers
                ],
                "missing_expected_papers": missing_expected_papers,
                "full_expected_paper_coverage": full_paper_coverage,
            },
        }

        evaluated_results.append(evaluation)

    dimensions = {
        "relevance": "relevance",
        "accuracy_completeness": "accuracy_completeness",
        "groundedness": "groundedness",
        "citation_source_quality": "citation_source_quality",
    }

    summary = {}

    for output_name, field_name in dimensions.items():
        scores = [
            item["evaluation"][field_name]
            for item in evaluated_results
        ]

        average = sum(scores) / len(scores)
        percentage = (average / 5) * 100

        summary[output_name] = {
            "average_score": round(average, 2),
            "percentage": round(percentage, 2),
            "max_score": 5,
        }

    all_scores = []

    for item in evaluated_results:
        evaluation = item["evaluation"]

        all_scores.extend(
            [
                evaluation["relevance"],
                evaluation["accuracy_completeness"],
                evaluation["groundedness"],
                evaluation["citation_source_quality"],
            ]
        )

    overall_average = sum(all_scores) / len(all_scores)

    documented_limitations = [
        {
            "id": item["id"],
            "question": item["question"],
            "failure_case": item["evaluation"]["failure_case"],
        }
        for item in evaluated_results
        if item["evaluation"]["failure_case"]
    ]

    coverage_count = sum(
        item["evaluation"]["full_expected_paper_coverage"]
        for item in evaluated_results
    )

    output = {
        "evaluation_type": "Manual answer-quality evaluation",
        "scale": {
            "1": "Incorrect / major issue",
            "2": "Poor",
            "3": "Moderate",
            "4": "Good / minor issue",
            "5": "Excellent",
        },
        "dimensions": {
            "relevance": "Does the answer directly address the question?",
            "accuracy_completeness": (
                "Are the claims accurate and sufficiently complete "
                "for the question?"
            ),
            "groundedness": (
                "Are the answer's claims supported by the retrieved "
                "research-paper passages?"
            ),
            "citation_source_quality": (
                "Are the cited paper/page sources appropriate and "
                "sufficient for the answer?"
            ),
        },
        "summary": {
            "num_questions": len(evaluated_results),
            "metrics": summary,
            "overall_average_score": round(overall_average, 2),
            "overall_percentage": round(
                (overall_average / 5) * 100, 2
            ),
            "questions_with_full_expected_paper_coverage": coverage_count,
            "expected_paper_coverage_percentage": round(
                (coverage_count / len(evaluated_results)) * 100,
                2,
            ),
            "documented_limitation_count": len(
                documented_limitations
            ),
        },
        "documented_limitations": documented_limitations,
        "results": evaluated_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print("ANSWER QUALITY EVALUATION")
    print("=" * 80)

    print(f"\nQuestions evaluated: {len(evaluated_results)}")

    for name, metric in summary.items():
        print(
            f"{name}: "
            f"{metric['average_score']}/5 "
            f"({metric['percentage']}%)"
        )

    print(
        f"\nOverall: "
        f"{round(overall_average, 2)}/5 "
        f"({round((overall_average / 5) * 100, 2)}%)"
    )

    print(
        f"Full expected-paper coverage: "
        f"{coverage_count}/{len(evaluated_results)} "
        f"({round((coverage_count / len(evaluated_results)) * 100, 2)}%)"
    )

    print(
        f"Documented limitations: "
        f"{len(documented_limitations)}"
    )

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
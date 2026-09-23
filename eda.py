from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = Path("outputs/chunks.json")
OUTPUT_DIR = Path("outputs/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_chunks():
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():

    chunks = load_chunks()

    records = []

    for chunk in chunks:
        metadata = chunk["metadata"]

        records.append(
            {
                "paper_id": metadata["paper_id"],
                "paper_title": metadata["paper_title"],
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
                "text": chunk["text"],
                "char_length": len(chunk["text"]),
                "word_count": len(chunk["text"].split()),
            }
        )

    df = pd.DataFrame(records)

    print("=" * 60)
    print("RAG DATASET EDA")
    print("=" * 60)

    print(f"Total chunks: {len(df):,}")
    print(f"Total papers: {df['paper_id'].nunique()}")
    print(f"Total pages: {df[['paper_id', 'page']].drop_duplicates().shape[0]}")

    print("\nChunks per paper:")
    print(
        df.groupby(
            ["paper_id", "paper_title"]
        ).size().reset_index(name="chunks")
        .to_string(index=False)
    )

    print("\nChunk statistics:")
    print(
        df[["char_length", "word_count"]].describe().round(2)
    )

    # ---------------------------------------------------------
    # 1. Pages per paper
    # ---------------------------------------------------------

    pages_per_paper = (
        df.groupby(["paper_id", "paper_title"])["page"]
        .nunique()
        .reset_index(name="pages")
    )

    plt.figure(figsize=(12, 6))
    plt.bar(
        pages_per_paper["paper_id"],
        pages_per_paper["pages"],
    )
    plt.xlabel("Paper")
    plt.ylabel("Number of Pages")
    plt.title("Number of Pages per Research Paper")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "pages_per_paper.png",
        dpi=150,
    )
    plt.close()

    # ---------------------------------------------------------
    # 2. Chunks per paper
    # ---------------------------------------------------------

    chunks_per_paper = (
        df.groupby(["paper_id", "paper_title"])
        .size()
        .reset_index(name="chunks")
    )

    plt.figure(figsize=(12, 6))
    plt.bar(
        chunks_per_paper["paper_id"],
        chunks_per_paper["chunks"],
    )
    plt.xlabel("Paper")
    plt.ylabel("Number of Chunks")
    plt.title("Number of Chunks per Research Paper")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "chunks_per_paper.png",
        dpi=150,
    )
    plt.close()

    # ---------------------------------------------------------
    # 3. Chunk length distribution
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))
    plt.hist(
        df["word_count"],
        bins=40,
    )
    plt.xlabel("Words per Chunk")
    plt.ylabel("Number of Chunks")
    plt.title("Chunk Word Count Distribution")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "chunk_word_distribution.png",
        dpi=150,
    )
    plt.close()

    # ---------------------------------------------------------
    # 4. Average chunk length by paper
    # ---------------------------------------------------------

    avg_length = (
        df.groupby("paper_id")["word_count"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    plt.bar(
        avg_length["paper_id"],
        avg_length["word_count"],
    )
    plt.xlabel("Paper")
    plt.ylabel("Average Words per Chunk")
    plt.title("Average Chunk Length by Paper")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "average_chunk_length.png",
        dpi=150,
    )
    plt.close()

    # ---------------------------------------------------------
    # 5. Pages vs chunks
    # ---------------------------------------------------------

    paper_stats = (
        df.groupby("paper_id")
        .agg(
            pages=("page", "nunique"),
            chunks=("chunk_id", "count"),
        )
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        paper_stats["pages"],
        paper_stats["chunks"],
    )

    for _, row in paper_stats.iterrows():
        plt.annotate(
            row["paper_id"],
            (row["pages"], row["chunks"]),
        )

    plt.xlabel("Number of Pages")
    plt.ylabel("Number of Chunks")
    plt.title("Pages vs. Chunks per Paper")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "pages_vs_chunks.png",
        dpi=150,
    )
    plt.close()

    # ---------------------------------------------------------
    # 6. Word count by paper
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    df.boxplot(
        column="word_count",
        by="paper_id",
        grid=False,
    )

    plt.xlabel("Paper")
    plt.ylabel("Words per Chunk")
    plt.title("Chunk Size Distribution by Paper")
    plt.suptitle("")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "chunk_size_by_paper.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------
    # Save statistics
    # ---------------------------------------------------------

    summary = {
        "total_papers": int(df["paper_id"].nunique()),
        "total_pages": int(
            df[["paper_id", "page"]]
            .drop_duplicates()
            .shape[0]
        ),
        "total_chunks": int(len(df)),
        "average_words_per_chunk": float(
            df["word_count"].mean()
        ),
        "median_words_per_chunk": float(
            df["word_count"].median()
        ),
        "min_words_per_chunk": int(
            df["word_count"].min()
        ),
        "max_words_per_chunk": int(
            df["word_count"].max()
        ),
    }

    with (
        OUTPUT_DIR / "eda_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nEDA files saved to:")
    print(OUTPUT_DIR)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
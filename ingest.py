from pathlib import Path
import json
import re

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATASET_PATH = Path(
    r"C:\Users\mugee\Downloads\AV_Capstone\capstone_dataset"
)

OUTPUT_PATH = Path("outputs/chunks.json")


# Known titles from the provided research-paper dataset.
PAPER_TITLES = {
    "data1": "Attention Is All You Need",
    "data2": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "data3": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "data4": "Language Models are Few-Shot Learners",
    "data5": "LoRA: Low-Rank Adaptation of Large Language Models",
    "data6": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    "data7": "Training Language Models to Follow Instructions with Human Feedback",
    "data8": "LLaMA: Open and Efficient Foundation Language Models",
    "data9": "SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
}


def clean_text(text: str) -> str:
    """Basic cleanup while preserving the actual paper content."""

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def extract_pages(dataset_path: Path):
    """Extract text page-by-page from every PDF."""

    documents = []

    pdf_files = sorted(dataset_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {dataset_path}"
        )

    for pdf_path in pdf_files:
        paper_id = pdf_path.stem

        if paper_id not in PAPER_TITLES:
            raise ValueError(
                f"No title mapping found for {paper_id}"
            )

        paper_title = PAPER_TITLES[paper_id]

        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = clean_text(text)

            if not text:
                continue

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "paper_id": paper_id,
                        "source": pdf_path.name,
                        "paper_title": paper_title,
                        "page": page_number,
                    },
                }
            )

    return documents


def chunk_documents(documents):
    """Split each page into chunks while preserving page metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for document in documents:
        page_chunks = splitter.split_text(document["text"])

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            metadata = document["metadata"].copy()

            metadata["chunk_id"] = (
                f"{metadata['paper_id']}"
                f"_p{metadata['page']}"
                f"_c{chunk_number}"
            )

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )

    return chunks


def save_chunks(chunks, output_path: Path):
    """Save chunks as JSON for inspection and reproducibility."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":

    print("=" * 60)
    print("RESEARCH PAPER INGESTION + CHUNKING")
    print("=" * 60)

    pages = extract_pages(DATASET_PATH)

    print(f"Pages extracted: {len(pages)}")

    chunks = chunk_documents(pages)

    print(f"Chunks created: {len(chunks)}")

    save_chunks(chunks, OUTPUT_PATH)

    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFirst chunk:")
    print("-" * 60)
    print(chunks[0]["text"][:700])

    print("\nFirst chunk metadata:")
    print("-" * 60)
    print(chunks[0]["metadata"])

    print("\nLast chunk metadata:")
    print("-" * 60)
    print(chunks[-1]["metadata"])

    print("\nChunking successful.")
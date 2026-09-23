import sys
from pathlib import Path

import streamlit as st


# Add src directory to Python path
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import answer_question


st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📚",
    layout="wide",
)


st.title("📚 Research Paper Answer Bot")
st.markdown(
    "Ask questions about the research papers in the knowledge base."
)

st.caption(
    "RAG: BGE-M3 → ChromaDB → BGE Reranker → Gemini 2.5 Flash"
)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar
with st.sidebar:
    st.header("About")

    st.write(
        """
        This application uses Retrieval-Augmented Generation (RAG)
        to answer questions from a collection of research papers.
        """
    )

    st.subheader("Pipeline")

    st.write("1. PDF page extraction")
    st.write("2. Page-aware chunking")
    st.write("3. BGE-M3 embeddings")
    st.write("4. ChromaDB vector retrieval")
    st.write("5. BGE reranking")
    st.write("6. Gemini answer generation")

    st.subheader("Knowledge Base")

    st.write("• 9 research papers")
    st.write("• 319 pages")
    st.write("• 1,291 chunks")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            passages = message.get("passages", [])

            if passages:

                with st.expander(
                    "📖 Supporting passages"
                ):

                    for i, passage in enumerate(
                        passages,
                        start=1,
                    ):

                        metadata = passage.get(
                            "metadata",
                            {},
                        )

                        paper_title = metadata.get(
                            "paper_title",
                            "Unknown paper",
                        )

                        page = metadata.get(
                            "page",
                            "?",
                        )

                        score = passage.get(
                            "reranker_score",
                            "N/A",
                        )

                        st.markdown(
                            f"**[{i}] {paper_title} — "
                            f"Page {page}**"
                        )

                        st.caption(
                            f"Reranker score: {score}"
                        )

                        st.write(
                            passage.get(
                                "text",
                                "",
                            )
                        )

                        if i < len(passages):
                            st.divider()


# Chat input
question = st.chat_input(
    "Ask a question about the research papers..."
)


if question:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the research papers..."
        ):

            try:

                result = answer_question(
                    question
                )

                answer = result["answer"]
                passages = result[
                    "retrieved_passages"
                ]

                st.markdown(answer)

                with st.expander(
                    "📖 Supporting passages"
                ):

                    for i, passage in enumerate(
                        passages,
                        start=1,
                    ):

                        metadata = passage.get(
                            "metadata",
                            {},
                        )

                        paper_title = metadata.get(
                            "paper_title",
                            "Unknown paper",
                        )

                        page = metadata.get(
                            "page",
                            "?",
                        )

                        score = passage.get(
                            "reranker_score",
                            "N/A",
                        )

                        st.markdown(
                            f"**[{i}] {paper_title} — "
                            f"Page {page}**"
                        )

                        st.caption(
                            f"Reranker score: {score}"
                        )

                        st.write(
                            passage.get(
                                "text",
                                "",
                            )
                        )

                        if i < len(passages):
                            st.divider()


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "passages": passages,
                    }
                )

            except Exception as e:

                error_message = (
                    "Unable to generate an answer right now.\n\n"
                    f"Error: `{e}`"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "passages": [],
                    }
                )
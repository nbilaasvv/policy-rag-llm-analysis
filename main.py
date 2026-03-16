"""
Entry point for running RAG queries.

This script loads the vector database,
initializes the retriever and LLM,
and runs example queries through the RAG pipeline.
"""

from vectorstore import load_vectorstore, get_retriever
from embedding_pipeline import load_embedding_model
from rag_query_engine import run_rag_query
from llm_loader import load_llm


VECTORSTORE_PATH = "vectorstore_reformasi_2045"


def initialize_rag():
    """Load embeddings, vectorstore, retriever, and LLM."""

    embeddings = load_embedding_model()

    vectorstore = load_vectorstore(
        VECTORSTORE_PATH,
        embeddings
    )

    retriever = get_retriever(vectorstore)

    llm = load_llm()

    return retriever, llm


def main():

    queries = [
        "Apa visi reformasi birokrasi digital Indonesia menuju 2045?",
        "Bagaimana konsep agile governance diterapkan dalam reformasi administrasi publik?",
        "Apa tantangan utama transformasi digital birokrasi di Indonesia?",
    ]

    retriever, llm = initialize_rag()

    for query in queries:

        answer = run_rag_query(query, retriever, llm)

        print("\n==============================")
        print("QUERY:")
        print(query)

        print("\nRESPONSE:")
        print(answer)
        print("==============================\n")


if __name__ == "__main__":
    main()

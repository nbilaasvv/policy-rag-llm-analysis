RAG query execution script.

This script runs one or multiple user queries through the
Retrieval-Augmented Generation (RAG) pipeline.

It performs the following steps:
- loads the vector database and retriever
- initializes the LLM model
- sends queries to the RAG query engine
- prints the generated analytical responses

This file acts as the main entry point for testing
semantic queries over embedded policy documents.
"""

from rag_query_engine import load_vectorstore, run_rag_query
from llm_loader import load_llm


def main():

    queries = [
        "Apa visi reformasi birokrasi digital Indonesia menuju 2045?",
        "Bagaimana konsep agile governance diterapkan dalam reformasi administrasi publik?",
        "Apa tantangan utama transformasi digital birokrasi di Indonesia?"
    ]

    # load retriever
    _, retriever = load_vectorstore()

    # load LLM
    llm = load_llm()

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

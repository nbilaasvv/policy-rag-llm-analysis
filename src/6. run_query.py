"""
RAG query execution script.

Runs semantic queries over the policy RAG system.
"""

import logging
from rag_query_engine import load_vectorstore, load_llm, run_rag_query


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


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

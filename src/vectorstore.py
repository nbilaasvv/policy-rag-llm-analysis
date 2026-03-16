"""
Vectorstore management module.

Handles:
- creating FAISS vectorstore
- saving vectorstore
- loading vectorstore
"""

import logging
from langchain_community.vectorstores import FAISS


def create_vectorstore(documents, embeddings):
    """
    Create FAISS vectorstore from document chunks.
    """

    logging.info("Creating FAISS vectorstore...")

    vectorstore = FAISS.from_documents(documents, embeddings)

    logging.info("Vectorstore created with %s documents", len(documents))

    return vectorstore


def save_vectorstore(vectorstore, path):
    """
    Save FAISS vectorstore locally.
    """

    vectorstore.save_local(path)

    logging.info("Vectorstore saved at %s", path)


def load_vectorstore(path, embeddings):
    """
    Load FAISS vectorstore from disk.
    """

    logging.info("Loading vectorstore from %s", path)

    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def create_retriever(vectorstore, k=5):
    """
    Create retriever for similarity search.
    """

    logging.info("Retriever created with top %s results", k)

    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

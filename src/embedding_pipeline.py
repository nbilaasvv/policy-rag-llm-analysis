"""
Embedding pipeline module.

Handles:
- loading embedding model
- creating FAISS vectorstore
- saving vector database
"""

import logging

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_PATH = "vectorstore_reformasi_2045"


def load_embedding_model():
    """Load embedding model."""

    logging.info("Loading embedding model")

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )


def build_vectorstore(documents):
    """Create and save FAISS vectorstore."""

    embeddings = load_embedding_model()

    logging.info("Creating FAISS vectorstore")

    vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(VECTORSTORE_PATH)

    logging.info("Vectorstore successfully saved")

    return vectorstore

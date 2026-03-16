"""
Build FAISS vector database for the RAG pipeline.

The script loads policy documents (PDF) and supporting news data (CSV),
cleans the text, splits it into chunks, generates embeddings,
and stores everything in a FAISS vector database.
"""

import logging

from pdf_parser import extract_pdf_text, process_csv, clean_text
from text_chunking import create_chunks
from embedding_pipeline import load_embedding_model
from vectorstore import create_vectorstore, save_vectorstore


PDF_PATH = "data/reformasi2045.pdf"
CSV_PATH = "data/scraping_news.csv"

VECTORSTORE_PATH = "vectorstore_reformasi_2045"


def build_database():
    logging.info("Building vector database...")

    # load and clean policy document
    logging.info("Reading policy document")
    pdf_text = clean_text(extract_pdf_text(PDF_PATH))

    # load supporting news data
    logging.info("Loading news dataset")
    csv_texts = process_csv(CSV_PATH)

    # combine all sources
    all_texts = [pdf_text] + csv_texts

    # split documents into chunks
    logging.info("Splitting documents into chunks")
    documents = []

    for text in all_texts:
        chunks = create_chunks(text, source="policy_source")
        documents.extend(chunks)

    logging.info("Chunks created: %d", len(documents))

    # embedding model
    logging.info("Loading embedding model")
    embeddings = load_embedding_model()

    # create FAISS index
    logging.info("Generating FAISS index")
    vectorstore = create_vectorstore(documents, embeddings)

    # save locally
    save_vectorstore(vectorstore, VECTORSTORE_PATH)

    logging.info("Vector database ready")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    build_database()

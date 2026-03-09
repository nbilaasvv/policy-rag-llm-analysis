"""
Text chunking module for RAG pipeline.
Splits documents into overlapping chunks.
"""

import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

#Create text splitter
def create_splitter():
      return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

#Split text into chunks with metadata
def create_chunks(text: str, label: str):
    splitter = create_splitter()
    chunks = splitter.split_text(text)
    documents = [
        Document(
            page_content=chunk,
            metadata={"kategori": label}
        )
        for chunk in chunks
    ]

    logging.info("Created %s chunks for %s", len(documents), label)

    return documents

#Return basic statistics about chunks
def chunk_statistics(documents):
    lengths = [len(doc.page_content) for doc in documents]
    stats = {
        "num_chunks": len(documents),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0
    }

    logging.info("Chunk statistics: %s", stats)

    return stats

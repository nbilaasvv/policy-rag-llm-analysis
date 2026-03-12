"""
RAG query engine module.

This module handles:
- loading the FAISS vector database
- retrieving relevant document chunks
- building contextual prompts
- generating responses using a Large Language Model (LLM)

The module acts as the core component of the Retrieval-Augmented
Generation (RAG) pipeline, enabling semantic search and
context-aware policy analysis from embedded documents.
"""

import os
import logging
import re
import pandas as pd
import pdfplumber

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

# load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_PATH = "vectorstore_reformasi_2045"
LLM_MODEL = "gemini-2.5-flash"

def load_vectorstore():
    """
    Load FAISS vectorstore and initialize retriever.
    """

    logging.info("Initializing embedding model")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )

    logging.info("Loading FAISS vectorstore")

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    logging.info("Vectorstore successfully loaded")

    return vectorstore, retriever

    def load_llm():
    """
    Initialize Gemini LLM model.
    """

    logging.info("Initializing Gemini LLM")

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0
    )

    return llm

    if __name__ == "__main__":

    print("Testing configuration...")

    vectorstore, retriever = load_vectorstore()
    llm = load_llm()

    response = llm.invoke("Halo, ini tes koneksi.")

    print(response.content)

"""
PDF and CSV text extraction module.

This module handles:
- PDF text extraction
- CSV text aggregation
- basic text cleaning
for the RAG document processing pipeline.
"""

import os
import re
import logging
import pandas as pd
import pdfplumber
from PyPDF2 import PdfReader


def extract_pdf_text(file_path: str) -> str:

    if not os.path.exists(file_path):
        logging.error("File not found: %s", file_path)
        return ""

    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        if text.strip():
            logging.info("Text extracted using pdfplumber: %s", file_path)
            return text

    except Exception as e:
        logging.warning("pdfplumber failed: %s", e)

    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""

        if text.strip():
            logging.info("Text extracted using PyPDF2: %s", file_path)
            return text

    except Exception as e:
        logging.warning("PyPDF2 failed: %s", e)

    logging.error("Text extraction failed")
    return ""


def clean_text(text: str) -> str:

    if not text:
        return ""

    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def process_csv(file_path: str):

    df = pd.read_csv(file_path)

    logging.info("Available columns: %s", df.columns.tolist())

    possible_columns = ["text", "content", "article", "body"]

    text_column = None

    for col in possible_columns:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        raise ValueError("No valid text column found in CSV")

    df = df[[text_column]].dropna().drop_duplicates()

    logging.info("Number of articles after cleaning: %s", len(df))

    return df[text_column].astype(str).tolist()

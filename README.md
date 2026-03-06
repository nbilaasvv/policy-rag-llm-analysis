# policy-rag-llm-analysis
This project demonstrates an AI-powered pipeline for semantic analysis of policy documents using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). The system processes policy PDFs, converts them into vector embeddings, retrieves relevant context using semantic search, and generates structured thematic insights using an LLM.

The goal of this project is to illustrate how AI/ML models, vector databases, and LLM orchestration frameworks can be integrated to support policy analysis and decision-making.

## Key Features
- End-to-end AI pipeline for document analysis
- Semantic search using FAISS vector database
- Text embeddings using Sentence Transformers
- Prompt engineering for context-aware LLM responses
- Automated thematic analysis output

## Tech Stack
- Python
- LangChain
- Sentence Transformers
- FAISS
- pdfplumber
- Pandas
- LLM APIs

## Pipeline Architecture
<img width="1024" height="1536" alt="pipeline" src="https://github.com/user-attachments/assets/928f63e5-623d-41df-a6f0-f86953eaa806" />
The system processes policy documents through several stages including document extraction, embedding generation, vector storage, semantic retrieval, and LLM-based synthesis.

# policy-rag-llm-analysis

This project demonstrates an **AI-powered pipeline for semantic analysis of policy documents** using **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**. 

The system processes policy PDFs, converts them into vector embeddings, retrieves relevant context using semantic search, and generates structured thematic insights using an LLM.

The goal of this project is to illustrate how **AI/ML models, vector databases, and LLM orchestration frameworks** can be integrated to support **policy analysis and decision-making**.

---

## Key Features

- End-to-end AI pipeline for document analysis
- Semantic search using **FAISS vector database**
- Text embeddings using **Sentence Transformers**
- Prompt engineering for context-aware LLM responses
- Automated **thematic analysis generation**

---

## Tech Stack

- **Python**
- **LangChain**
- **Sentence Transformers**
- **FAISS**
- **pdfplumber**
- **Pandas**
- **LLM APIs**

---
## Requirements

This project was developed using **Python 3.10+**.

Install all dependencies using:

```bash
pip install -r requirements.txt
```
---

## Pipeline Architecture

<p align="center">
  <img src="images/pipeline.png" width="650">
</p>

The system processes policy documents through several stages:

1. **PDF Parsing** – Extract text from policy documents.  
2. **Text Chunking** – Split long documents into manageable semantic units.  
3. **Embedding Generation** – Convert text chunks into vector embeddings.  
4. **Vector Storage** – Store embeddings in a **FAISS vector database**.  
5. **Semantic Retrieval** – Retrieve the most relevant document chunks based on user queries.  
6. **Prompt Engineering** – Combine retrieved context with analytical questions.  
7. **LLM Inference** – Generate structured insights using a large language model.  
8. **Thematic Analysis Output** – Produce synthesized findings for policy analysis.

---

## Use Case

This project demonstrates how **LLM-based retrieval systems** can support:

- Policy and governance analysis  
- Knowledge extraction from unstructured documents  
- AI-assisted decision support systems

---

## Future Improvements

- Deploy the pipeline using **FastAPI** for production APIs
- Integrate scalable vector databases such as **Pinecone**
- Implement **LLM fine-tuning for domain-specific policy analysis**
- Add **MLOps pipeline for monitoring and deployment**

---


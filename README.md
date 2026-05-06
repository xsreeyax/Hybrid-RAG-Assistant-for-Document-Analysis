# Hybrid-RAG-Assistant-for-Document-Analysis

A Streamlit-based application that enables users to query both structured (CSV) and unstructured (PDF) data using natural language. The system combines Pandas-based computation with a Retrieval-Augmented Generation (RAG) pipeline to deliver accurate and context-aware responses.

Features
Upload CSV and PDF files
Automatic query classification (structured vs unstructured)
Fast numerical analysis using Pandas
Semantic search using embeddings and FAISS
Context-aware answers using an LLM via Ollama
Hybrid pipeline combining machine learning and rule-based logic
Interactive user interface built with Streamlit
System Architecture

The system follows a hybrid pipeline:

Data Input
CSV → Structured Data (DataFrame)
PDF → Extracted Text
Preprocessing
Text chunking with overlap
Conversion to embeddings
Query Classification
Structured → Pandas engine
Unstructured → RAG pipeline
Retrieval (RAG)
Embeddings generated using SentenceTransformers
Stored and searched using FAISS
Response Generation
Context passed to LLM via Ollama
Output generated based on retrieved data

Tech Stack
Frontend: Streamlit
Data Processing: Pandas
PDF Parsing: PyPDF2
Embeddings: SentenceTransformers
Vector Search: FAISS
LLM Runtime: Ollama

Key Advantages
Combines exact computation with AI-based reasoning
Reduces hallucination using context grounding
Supports multiple data formats
Scalable for large datasets using vector indexing

Limitations
Keyword-based query classification
Performance depends on PDF text extraction quality
Requires local LLM setup via Ollama

Future Improvements
Advanced query classification using machine learning
Support for additional file formats (Excel, DOCX)
Improved user interface and visualization
Cloud deployment support

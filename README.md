# College Notes RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain that answers questions from college PDF notes.

## Features

- Load college notes from a PDF
- Split the PDF into smaller chunks
- Generate embeddings using Gemini
- Store embeddings in ChromaDB
- Retrieve relevant information using a retriever
- Generate answers using Gemini

## Architecture

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Splitter
 ↓
Gemini Embeddings
 ↓
ChromaDB
 ↓
Retriever
 ↓
Relevant Chunks
 ↓
Prompt
 ↓
Gemini LLM
 ↓
Answer
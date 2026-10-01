# 📄 AskPDF AI — Backend

The Python backend for **AskPDF**, an AI-powered document assistant that uses **Retrieval-Augmented Generation (RAG)** to answer questions, explain concepts, and solve problems using information from uploaded PDF documents.

## Features

* 📄 PDF document processing
* 🔎 Semantic search using embeddings
* 🧠 Retrieval-Augmented Generation (RAG)
* 🤖 Google Gemini integration
* 💬 Conversational question answering
* 📝 Clear and friendly explanations
* 🧮 Problem solving and calculations
* 🔐 Secure environment-based API configuration

## How It Works

```text
                  User Question
                       │
                       ▼
                Query Embedding
                       │
                       ▼
                 Vector Search
                       │
                       ▼
              Relevant PDF Chunks
                       │
                       ▼
                  RAG Prompt
                       │
                       ▼
                 Google Gemini
                       │
                       ▼
                Helpful Answer
```

AskPDF retrieves relevant information from the uploaded PDF and provides it to Gemini so the assistant can answer questions using the document's content.

The assistant can also **explain concepts, work through problems, perform calculations, and provide step-by-step solutions** when the necessary information is available.

## Tech Stack

* **Python**
* **Google Gemini API**
* **Embeddings**
* **Vector Search**
* **RAG**
* **PDF Processing**
* **REST API**

## RAG Pipeline

AskPDF follows a Retrieval-Augmented Generation pipeline:

1. **Upload PDF** — The document is processed and its text is extracted.
2. **Chunking** — The extracted text is split into smaller chunks.
3. **Embeddings** — Chunks are converted into vector embeddings.
4. **Retrieval** — Relevant chunks are retrieved based on the user's question.
5. **Prompt Generation** — Retrieved information is provided to Gemini.
6. **Answer Generation** — Gemini generates a natural and helpful response.
7. **Response** — The answer is returned to the React frontend.


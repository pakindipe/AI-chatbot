# AI Document Assistant

An AI-powered document assistant built with **Streamlit** that allows users to upload documents, generate intelligent summaries, detect document types, and chat with a knowledge base using Retrieval-Augmented Generation (RAG).

---

## Features

### Document Summarizer
- Upload **PDF, DOCX, or TXT** files
- Automatic **document type detection** (e.g., Resume/CV, Article, Notes)
- Transformer-based **multi-chunk summarization**
- Download extracted raw text

### RAG Chat (Knowledge Base)
- Ask questions against a **FAISS vector database**
- Uses sentence embeddings for semantic search
- Displays **sources used** in responses

### User Interface
- Clean **Streamlit** interface
- Custom fonts and icons
- Tab-based navigation

---
## Setup & Installation

1️⃣ Clone the repository

git clone https://github.com/pakindipe/AI-chatbot.git
cd AI-chatbot

2️⃣ Create & activate a virtual environment

python -m venv venv

Windows:

venv\Scripts\activate

macOS / Linux

source venv/bin/activate

3️⃣ Install dependencies

pip install -r requirements.txt

▶️ Running the App

```text
streamlit run app.py

Then open your browser at:

http://localhost:8501
```
---

## Tech Stack

- **Python**
- **Streamlit**
- **Hugging Face Transformers**
- **FAISS**
- **Sentence Transformers**
- **PyPDF**
- **python-docx**

---

## Project Structure

```text
AI-chatbot/
├── app.py              # Streamlit UI
├── summarizer.py       # Text extraction, doc type detection, summarization
├── chat.py             # RAG chat logic
├── ingest.py           # FAISS indexing pipeline
├── faiss_index.bin     # Vector index
├── documents.pkl       # Stored documents
├── requirements.txt
└── README.md


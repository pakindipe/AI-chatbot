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

```bash
git clone https://github.com/pakindipe/AI-chatbot.git
cd AI-chatbot
```

2️⃣ Create & activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux

```bash
source venv/bin/activate
```

3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

▶️ Running the App

```bash
streamlit run app.py
```

Then open your browser at:

```bash
http://localhost:8501
```
---

## Privacy & Security
- Uploaded documents are processed locally
- No files are stored remotely
- Sensitive documents are excluded from Git history

---

## Author
Philip Akindipe
Computer Engineering — Queen’s University
LinkedIn: https://www.linkedin.com/in/philip-akindipe/

---

## Future Improvements
- Streaming summaries
- GPU acceleration
- Per-document chat
- Multi-document comparison
- Export summaries as PDF

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


"""
ingest.py
-------------

This script processes text documents in a ``data`` folder and builds a vector
store for retrieval.  It performs the following steps:

1. Collect all ``.txt`` files in the ``data`` directory.
2. Split each document into smaller chunks using ``RecursiveCharacterTextSplitter``.
3. Compute embeddings for each chunk using a SentenceTransformer model.
4. Create a FAISS index from the embeddings.
5. Save the FAISS index and the list of text chunks to disk for later use.

Run this script from the root of your project with::

    python ingest.py

It will output ``faiss_index.bin`` and ``documents.pkl`` in the current
working directory.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

def load_and_split_documents(data_dir: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs: list[Document] = []

    # TXT
    txt_pattern = os.path.join(data_dir, "*.txt")
    for filepath in glob.glob(txt_pattern):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = splitter.split_text(text)
        for c in chunks:
            docs.append(Document(page_content=c, metadata={"source": filename}))

    # PDF
    pdf_pattern = os.path.join(data_dir, "*.pdf")
    for filepath in glob.glob(pdf_pattern):
        filename = os.path.basename(filepath)
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        for page in pages:
            chunks = splitter.split_text(page.page_content)
            for c in chunks:
                docs.append(Document(page_content=c, metadata={"source": filename}))

    return docs


def build_faiss_index(embeddings):
    """Build a FAISS L2 index from the given embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def main() -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist. Create it and add .txt files."
        )

    print("Loading and splitting documents...")
    documents = load_and_split_documents(data_dir)
    if not documents:
        raise ValueError(
            f"No text files found in {data_dir}. Please add some .txt files to build the index."
        )
    print(f"Loaded {len(documents)} document chunks.")

    print("Computing embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [d.page_content for d in documents]
    embeddings = model.encode(texts, show_progress_bar=True)


    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Save index and documents
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index.bin")
    docs_path = os.path.join(os.path.dirname(__file__), "documents.pkl")
    print(f"Saving index to {index_path}")
    faiss.write_index(index, index_path)
    print(f"Saving documents list to {docs_path}")
    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
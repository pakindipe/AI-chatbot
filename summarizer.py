# summarizer.py
from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@lru_cache(maxsize=1)
def get_summarizer():
    """
    Loads a summarization model once and reuses it.
    flan-t5-base works but can be a bit weak; still fine for this project.
    """
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def split_text(text: str) -> List[str]:
    # Chunking avoids the "583 > 512" warning and prevents truncation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,   # characters (not tokens)
        chunk_overlap=150
    )
    chunks = splitter.split_text(text)
    # remove tiny/empty chunks
    return [c.strip() for c in chunks if c.strip()]


def _summarize_one(chunk: str, tokenizer, model, max_new_tokens: int = 140) -> str:
    prompt = (
        "Summarize the following text clearly and concisely.\n"
        "If it looks like a resume, say whose resume it is.\n\n"
        f"{chunk}\n\nSummary:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def summarize_document(text: str) -> str:
    """
    Map-reduce summarization:
    1) split text into chunks
    2) summarize each chunk
    3) summarize the summaries (final)
    """
    tokenizer, model = get_summarizer()

    chunks = split_text(text)
    if not chunks:
        return "No text found to summarize."

    # MAP: summarize each chunk
    chunk_summaries = [_summarize_one(c, tokenizer, model) for c in chunks]

    # If document is short, just return first pass summary
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # REDUCE: summarize the summaries
    combined = "\n".join(f"- {s}" for s in chunk_summaries)

    final_prompt = (
        "You are summarizing an entire document based on section summaries.\n"
        "Write a clean overall summary in 2-5 sentences.\n"
        "If it's a resume, say whose resume it is and what it highlights.\n\n"
        f"{combined}\n\nFinal Summary:"
    )

    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    outputs = model.generate(**inputs, max_new_tokens=180)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

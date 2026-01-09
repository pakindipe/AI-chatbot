import io
import re
from typing import List, Dict

import docx
from pypdf import PdfReader
from transformers import pipeline


# -----------------------------
# Text extraction
# -----------------------------

def extract_text(filename: str, file_bytes: bytes) -> str:
    filename = filename.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported file type")


# -----------------------------
# Chunking (simple, word-based)
# -----------------------------

def chunk_text(text: str, max_words: int = 260) -> List[str]:
    # Smaller chunks = safer for model max length
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks


# -----------------------------
# Doc type detection (heuristics)
# -----------------------------

def detect_doc_type(text: str) -> str:
    t = text.lower()

    # Resume/CV signals
    resume_signals = [
        "education", "experience", "skills", "projects", "certifications",
        "linkedin.com", "github.com", "curriculum vitae", "resume"
    ]
    resume_hits = sum(1 for s in resume_signals if s in t)

    # Email/letter signals
    letter_signals = ["dear ", "sincerely", "to whom it may concern", "regards,"]
    letter_hits = sum(1 for s in letter_signals if s in t)

    # Academic/article signals
    article_signals = ["abstract", "introduction", "method", "results", "references", "bibliography"]
    article_hits = sum(1 for s in article_signals if s in t)

    # Simple scoring
    if resume_hits >= 3:
        return "Resume/CV"
    if letter_hits >= 2:
        return "Cover letter / Letter"
    if article_hits >= 3:
        return "Article / Paper"
    return "General document"


def try_extract_person_name(text: str) -> str | None:
    # quick heuristic: look for a line that looks like a name near the top
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    top = lines[:25]

    # common email/phone patterns to ignore
    email_re = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
    phone_re = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")

    for ln in top:
        if email_re.search(ln) or phone_re.search(ln):
            continue
        # looks like 2-4 capitalized words
        words = ln.split()
        if 2 <= len(words) <= 4 and all(w[:1].isalpha() for w in words):
            # reject section headers like EXPERIENCE
            if ln.isupper() and len(ln) < 20:
                continue
            # if it contains mostly letters and spaces, accept
            if sum(ch.isalpha() for ch in ln) >= max(6, int(0.7 * len(ln))):
                return ln
    return None


# -----------------------------
# Summarization (fast + stable)
# -----------------------------

# Use a smaller model than bart-large-cnn for speed on CPU
_SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"


def summarize_document(text: str) -> Dict[str, str]:
    text = (text or "").strip()
    if not text:
        return {"doc_type": "Empty", "summary": "No text found in the document."}

    doc_type = detect_doc_type(text)
    person = try_extract_person_name(text)

    summarizer = pipeline("summarization", model=_SUMMARY_MODEL)

    chunks = chunk_text(text)
    summaries = []

    # summarize each chunk
    for chunk in chunks:
        # keep lengths conservative to avoid warnings / bad outputs
        result = summarizer(
            chunk,
            max_length=120,
            min_length=35,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    # final compression pass (only if multiple chunks)
    combined = " ".join(summaries).strip()
    if len(summaries) > 1:
        final = summarizer(
            combined,
            max_length=140,
            min_length=60,
            do_sample=False
        )[0]["summary_text"]
    else:
        final = combined

    # add a short “label” line when it’s clearly a resume
    if doc_type == "Resume/CV":
        if person:
            final = f"This appears to be {person}'s resume/CV.\n\n{final}"
        else:
            final = f"This appears to be a resume/CV.\n\n{final}"

    return {"doc_type": doc_type, "summary": final}

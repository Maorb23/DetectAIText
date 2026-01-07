# scripts/convert.py
# Gets text from various file formats -> raw text string

from __future__ import annotations
from pathlib import Path
from pypdf import PdfReader


def load_text_from_txt(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding, errors="replace", newline=None) as f:
        return f.read()


def load_text_from_docx(path: str) -> str:
    """
    Extract text from a .docx file while preserving paragraph structure.
    """
    from docx import Document  # python-docx

    document = Document(path)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(paragraphs)


def load_text_from_pdf(path: str) -> str:
    """
    Extract text from a text-based PDF.
    Will fail loudly on scanned/image-only PDFs or malformed files.
    """
    reader = PdfReader(path)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(pages)


def load_text_format(path: str) -> str:
    """
    Load .txt, .docx, or .pdf into a raw text string.
    """
    suffix = Path(path).suffix.lower()

    if suffix == ".txt":
        return load_text_from_txt(path)
    if suffix == ".docx":
        return load_text_from_docx(path)
    if suffix == ".pdf":
        return load_text_from_pdf(path)

    raise ValueError(f"Unsupported file type: {suffix}")

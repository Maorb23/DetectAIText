# scripts/preprocess.py
# Gets raw text string -> normalized, windowed text examples

from __future__ import annotations

import re
import unicodedata
import hashlib
from typing import Any, Dict, List, Optional
import logging 
logger = logging.getLogger(__name__) # set up a logger
logger.setLevel(logging.DEBUG) # set logging level to DEBUG. Now we can use logger.debug(), logger.info(), etc.



# --- Whitespace / newline patterns ---
SPACE_OR_TAB_RUN = re.compile(r"[ \t]+")
THREE_OR_MORE_NEWLINES = re.compile(r"\n{3,}")

# A simple sentence splitter (English-ish). You can replace later if needed.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


# --- Newline variants we want to normalize into '\n' ---
# These appear in copy/pasted text, PDFs, web pages, etc.
NONSTANDARD_NEWLINE_CHARS = {
    "\r\n": "\n",  # Windows
    "\r": "\n",    # old Mac
    "\u2028": "\n",  # Unicode Line Separator
    "\u2029": "\n",  # Unicode Paragraph Separator
    "\u0085": "\n",  # NEL (Next Line)
    "\x0b": "\n",    # VT (Vertical Tab)
    "\x0c": "\n",    # FF (Form Feed)
}


def load_text_file(path: str, encoding: str = "utf-8") -> str:
    """
    Read a text file in a way that preserves/normalizes line endings.
    newline=None enables universal newline handling in Python.
    """
    with open(path, "r", encoding=encoding, errors="replace", newline=None) as f:
        return f.read()


def normalize_newlines(text: str, convert_literal_backslash_n: bool = True) -> str:
    """
    Make *all* line breaks consistent as '\n'.

    Handles:
    - Windows/Mac line endings
    - Unicode line/paragraph separators (\\u2028, \\u2029)
    - NEL (\\u0085), VT, FF
    - Optional: literal two-character sequences '\\\\n' and '\\\\r\\\\n'
      that sometimes appear when text was serialized/escaped.
    """
    if not text:
        return ""

    # Optional: convert literal escaped newlines into real ones
    if convert_literal_backslash_n:
        # turn "\\r\\n" into "\n" first, then "\\n" into "\n"
        text = text.replace("\\r\\n", "\n")
        text = text.replace("\\n", "\n")

    # Normalize known newline variants
    # Do CRLF first so we don't turn it into double newlines.
    text = text.replace("\r\n", "\n")
    for bad_newline, good_newline in NONSTANDARD_NEWLINE_CHARS.items():
        if bad_newline == "\r\n":
            continue
        text = text.replace(bad_newline, good_newline)

    return text


def normalize_text(text: str) -> str:
    """
    Minimal normalization that keeps structure intact for detection:
    - Unicode normalize (NFKC)
    - Robust newline normalization
    - Strip outer whitespace
    - Collapse repeated spaces/tabs (preserve newlines)
    - Reduce 3+ newlines to 2 to keep paragraph structure but avoid pathological spacing
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text) # normalization using NFKC form
    text = normalize_newlines(text, convert_literal_backslash_n=True)
    text = text.strip()

    # Collapse runs of spaces/tabs but keep newlines
    text = SPACE_OR_TAB_RUN.sub(" ", text)

    # Preserve paragraphs: allow up to 2 consecutive newlines
    text = THREE_OR_MORE_NEWLINES.sub("\n\n", text)

    return text


def compute_stable_text_id(normalized_text: str) -> str:
    """Deterministic ID for caching features and avoiding duplicates."""
    return hashlib.sha1(normalized_text.encode("utf-8")).hexdigest() # sha1 uses 


def estimate_token_count(text: str) -> int:
    """
    Cheap approximation for English-ish text:
    ~4 chars/token (rough heuristic).
    """
    if not text:
        return 0
    return (len(text) + 3) // 4


def should_keep_text(text: str, min_estimated_tokens: int = 80) -> bool:
    """Conservative filter: keep only non-empty texts of reasonable length."""
    normalized = normalize_text(text)
    if not normalized:
        return False
    if estimate_token_count(normalized) < min_estimated_tokens:
        return False
    return True

def flush_window(current_paragraphs, current_token_budget, windows):
    if current_paragraphs:
        windows.append("\n\n".join(current_paragraphs).strip())
    return [], 0



def split_text_into_windows(
    text: str,
    max_estimated_tokens: int,
    overlap_estimated_tokens: int = 0,
) -> List[str]:
    """
    Split long text into roughly token-sized windows while preserving structure:
    - Prefer paragraph boundaries
    - Fall back to sentence boundaries if a paragraph is huge
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    if estimate_token_count(normalized) <= max_estimated_tokens:
        return [normalized]

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    windows: List[str] = []

    current_paragraphs: List[str] = []
    current_token_budget = 0



    for paragraph in paragraphs:
        paragraph_tokens = estimate_token_count(paragraph)

        # If one paragraph is too large, split by sentences
        if paragraph_tokens > max_estimated_tokens:
            # Flush any existing window first
            current_paragraphs, current_token_budget = flush_window(current_paragraphs, current_token_budget, windows)
            sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(paragraph) if s.strip()]

            current_sentences: List[str] = []
            current_sent_tokens = 0

            for sentence in sentences:
                sentence_tokens = estimate_token_count(sentence)
                if current_sent_tokens + sentence_tokens > max_estimated_tokens and current_sentences:
                    windows.append(" ".join(current_sentences).strip())
                    current_sentences = []
                    current_sent_tokens = 0
                current_sentences.append(sentence)
                current_sent_tokens += sentence_tokens

            if current_sentences:
                windows.append(" ".join(current_sentences).strip())
            continue

        # Pack paragraphs into windows
        if current_token_budget + paragraph_tokens > max_estimated_tokens and current_paragraphs:
            current_paragraphs, current_token_budget = flush_window(current_paragraphs, current_token_budget, windows)

        current_paragraphs.append(paragraph)
        current_token_budget += paragraph_tokens

    current_paragraphs, current_token_budget = flush_window(current_paragraphs, current_token_budget, windows)

    # Optional overlap (approx chars-based)
    if overlap_estimated_tokens > 0 and len(windows) > 1:
        overlap_chars = overlap_estimated_tokens * 4
        overlapped_windows: List[str] = []
        previous_window = ""
        for window in windows:
            tail = previous_window[-overlap_chars:] if previous_window else ""
            overlapped_windows.append((tail + "\n\n" + window).strip() if tail else window)
            previous_window = window
        windows = overlapped_windows

    return [w for w in windows if w] # it returns non empty windows in a list


def prepare_examples(
    examples: List[Dict[str, Any]],
    max_estimated_tokens: Optional[int] = None,
    overlap_estimated_tokens: int = 0,
    min_estimated_tokens: int = 80,
    normalize: bool = False,
) -> List[Dict[str, Any]]:
    """
    Input example format:
      {"text": "...", "label": 0/1 or "human"/"ai", "meta": {...}}

    Output: normalized + optionally windowed examples with stable IDs.
    """
    prepared: List[Dict[str, Any]] = []

    for example in examples:
        raw_text = example.get("text", "")
        label = example.get("label", None)
        metadata = dict(example.get("meta", {}))

        if normalize:
            print("Normalizing text for example.")
            normalized_text = normalize_text(raw_text)
        else:
            normalized_text = raw_text
        if not should_keep_text(normalized_text, min_estimated_tokens=min_estimated_tokens):
            logger.debug(f"Skipping example due to insufficient length after normalization. Estimated tokens: {estimate_token_count(normalized_text)}")
            continue

        metadata["text_id"] = compute_stable_text_id(normalized_text)
        metadata["estimated_tokens"] = estimate_token_count(normalized_text)

        if max_estimated_tokens is None:
            prepared.append({"text": normalized_text, "label": label, "meta": metadata})
        else:
            windows = split_text_into_windows(
                normalized_text,
                max_estimated_tokens=max_estimated_tokens,
                overlap_estimated_tokens=overlap_estimated_tokens,
            )
            for window_index, window_text in enumerate(windows):
                window_meta = dict(metadata)
                window_meta["window_index"] = window_index
                window_meta["window_text_id"] = compute_stable_text_id(window_text)
                window_meta["window_estimated_tokens"] = estimate_token_count(window_text)
                prepared.append({"text": window_text, "label": label, "meta": window_meta})

    return prepared

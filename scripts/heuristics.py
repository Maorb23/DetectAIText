# ai_detect/tools/heuristics.py
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


_STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","for","from","has","have","he","her","his","i",
    "if","in","into","is","it","its","me","my","not","of","on","or","our","she","so","that","the",
    "their","them","there","they","this","to","was","we","were","what","when","where","which","who",
    "will","with","you","your"
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WS_RE = re.compile(r"\s+")
_WINDOW_SEPARATOR = "\n\n--- WINDOW ---\n\n"


@dataclass(frozen=True)
class HeuristicFeatures:
    # repetition
    distinct_3gram_ratio: float
    longest_repeat_ngram: int
    repeated_sentence_ratio: float

    # burstiness / variability
    sentence_len_var: float
    punctuation_entropy: float
    stopword_ratio_var: float

    # entropy collapse proxies
    char_trigram_entropy: float
    word_entropy: float

    # convenience
    n_sentences: int
    n_words: int
    n_chars: int


def compute_heuristics(
    examples: List[Dict[str, Any]],
    aggregate_doc: bool = True,
    window_separator: str = _WINDOW_SEPARATOR,
) -> List[Dict[str, Any]]:
    """
    Compute heuristic features for prepared examples from `prepare_examples`.

    Input example (prepared):
      {"text": "...", "label": ..., "meta": {"text_id": "...", "window_index": ..., ...}}

    Output (per example):
      {
        "label": ...,
        "meta": {...},
        "features": {...},
        "heuristics_score": float in [0,1]
      }

    If aggregate_doc=True, also adds a document-level record per text_id:
      meta["level"] = "doc"
      meta["text_id"] = ...
      features are mean across windows; scores averaged.
    """
    per_window: List[Dict[str, Any]] = []
    for ex in examples:
        text = ex.get("text", "") or ""
        meta = dict(ex.get("meta", {}))
        label = ex.get("label", None)

        feats = extract_features(text)
        score = heuristics_score(feats)

        per_window.append(
            {
                "label": label,
                "meta": meta,
                "features": feats.__dict__,
                "heuristics_score": score,
            }
        )

    if not aggregate_doc:
        return per_window

    by_text_id: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_window:
        tid = str(row.get("meta", {}).get("text_id", ""))
        by_text_id.setdefault(tid, []).append(row)

    doc_rows: List[Dict[str, Any]] = []
    for tid, rows in by_text_id.items():
        if not tid:
            continue
        doc_rows.append(_aggregate_doc_rows(tid, rows, window_separator=window_separator))

    return per_window + doc_rows


def extract_features(text: str) -> HeuristicFeatures:
    sentences = split_sentences(text)
    tokens = tokenize_words(text)
    n_chars = len(text)

    distinct_3 = distinct_ngram_ratio(tokens, n=3)
    longest_rep = longest_repeated_ngram_len(tokens, max_n=12)
    rep_sent = repeated_sentence_ratio(sentences)

    sent_lens = [len(tokenize_words(s)) for s in sentences if s.strip()]
    sent_len_var = variance(sent_lens)

    punct_entropy = punctuation_entropy(text)

    stop_ratios = [stopword_ratio(s) for s in sentences if s.strip()]
    stop_var = variance(stop_ratios)

    char_tri_ent = char_trigram_entropy(text)
    word_ent = word_entropy(tokens)

    return HeuristicFeatures(
        distinct_3gram_ratio=distinct_3,
        longest_repeat_ngram=longest_rep,
        repeated_sentence_ratio=rep_sent,
        sentence_len_var=sent_len_var,
        punctuation_entropy=punct_entropy,
        stopword_ratio_var=stop_var,
        char_trigram_entropy=char_tri_ent,
        word_entropy=word_ent,
        n_sentences=len([s for s in sentences if s.strip()]),
        n_words=len(tokens),
        n_chars=n_chars,
    )


def heuristics_score(f: HeuristicFeatures) -> float:
    """
    A small, stable scoring function:
    - more repetition => higher score (more AI-like)
    - lower variability/entropy => higher score (more AI-like)

    Returns probability-like score in [0,1].
    """
    rep = 0.0
    rep += clamp01((0.80 - f.distinct_3gram_ratio) / 0.30)            # low distinctness -> AI-ish
    rep += clamp01((f.longest_repeat_ngram - 6.0) / 10.0)            # long repeats -> AI-ish
    rep += clamp01((f.repeated_sentence_ratio - 0.02) / 0.15)        # duplicate sentences -> AI-ish
    rep /= 3.0

    burst = 0.0
    burst += clamp01((8.0 - f.sentence_len_var) / 12.0)              # low variance -> AI-ish
    burst += clamp01((1.6 - f.punctuation_entropy) / 1.6)            # low punct entropy -> AI-ish
    burst += clamp01((0.020 - f.stopword_ratio_var) / 0.020)         # low stopword variance -> AI-ish
    burst /= 3.0

    ent = 0.0
    ent += clamp01((3.5 - f.char_trigram_entropy) / 2.0)             # low char entropy -> AI-ish
    ent += clamp01((6.5 - f.word_entropy) / 3.0)                     # low word entropy -> AI-ish
    ent /= 2.0

    # down-weight when text is short (heuristics less reliable)
    length_gate = clamp01((f.n_words - 80) / 200.0)

    raw = 0.50 * rep + 0.30 * burst + 0.20 * ent
    raw = (0.60 * raw + 0.40 * 0.50) if length_gate < 0.25 else raw  # pull toward 0.5 if too short
    raw = 0.50 + length_gate * (raw - 0.50)

    return clamp01(raw)


def _aggregate_doc_rows(text_id: str, rows: List[Dict[str, Any]], window_separator: str) -> Dict[str, Any]:
    feats_list = [r["features"] for r in rows]
    scores = [float(r["heuristics_score"]) for r in rows]

    mean_feats = {k: mean([float(d[k]) for d in feats_list]) for k in feats_list[0].keys()}
    doc_meta = dict(rows[0].get("meta", {}))
    doc_meta["level"] = "doc"
    doc_meta["text_id"] = text_id
    doc_meta.pop("window_index", None)
    doc_meta.pop("window_text_id", None)
    doc_meta.pop("window_estimated_tokens", None)

    return {
        "label": rows[0].get("label", None),
        "meta": doc_meta,
        "features": mean_feats,
        "heuristics_score": float(mean(scores)),
    }


def split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def tokenize_words(text: str) -> List[str]:
    clean = _PUNCT_RE.sub(" ", text.lower())
    clean = _WS_RE.sub(" ", clean).strip()
    if not clean:
        return []
    return clean.split(" ")


def distinct_ngram_ratio(tokens: List[str], n: int = 3) -> float:
    if len(tokens) < n + 1:
        return 1.0
    total = len(tokens) - n + 1
    uniq = len({tuple(tokens[i : i + n]) for i in range(total)})
    return safe_div(uniq, total, default=1.0)


def longest_repeated_ngram_len(tokens: List[str], max_n: int = 12) -> int:
    """
    Proxy for "longest repeated substring" using repeated n-grams.
    Returns the maximum n such that at least one n-gram repeats.
    """
    best = 0
    for n in range(2, max_n + 1):
        if len(tokens) < n * 2:
            break
        seen = set()
        repeated = False
        for i in range(0, len(tokens) - n + 1):
            ng = tuple(tokens[i : i + n])
            if ng in seen:
                repeated = True
                break
            seen.add(ng)
        if repeated:
            best = n
        else:
            # if no repeats at n, likely no repeats at larger n
            break
    return best


def repeated_sentence_ratio(sentences: List[str]) -> float:
    norm = [normalize_sentence(s) for s in sentences if s.strip()]
    if len(norm) <= 1:
        return 0.0
    counts = Counter(norm)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return safe_div(dup, len(norm), default=0.0)


def normalize_sentence(s: str) -> str:
    s = s.lower().strip()
    s = _WS_RE.sub(" ", s)
    s = s.strip(" .,!?:;\"'()[]{}")
    return s


def punctuation_entropy(text: str) -> float:
    punct = re.findall(r"[.,;:!?()\[\]\"'—\-]", text)
    if not punct:
        return 0.0
    counts = Counter(punct)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return shannon_entropy(probs)


def stopword_ratio(sentence: str) -> float:
    toks = tokenize_words(sentence)
    if not toks:
        return 0.0
    sw = sum(1 for t in toks if t in _STOPWORDS)
    return safe_div(sw, len(toks), default=0.0)


def char_trigram_entropy(text: str) -> float:
    s = text.lower()
    s = _WS_RE.sub(" ", s).strip()
    if len(s) < 20:
        return 0.0
    trigs = [s[i : i + 3] for i in range(0, len(s) - 2)]
    counts = Counter(trigs)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return shannon_entropy(probs)


def word_entropy(tokens: List[str]) -> float:
    if len(tokens) < 20:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return shannon_entropy(probs)


def shannon_entropy(probs: Iterable[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p, 2)
    return h


def mean(xs: List[float]) -> float:
    return safe_div(sum(xs), len(xs), default=0.0)


def variance(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

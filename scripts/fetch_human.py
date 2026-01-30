# scripts/fetch_human_50k.py
"""
Robust human-paragraph fetcher (50k) with:
- separate connect/read timeouts
- retries + jitter
- per-source failure counters (don't drop source on first hiccup)
- periodic checkpointing + resume
- Gutenberg streaming size guard
- dynamic quota reallocation if sources die
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import random
import logging
import hashlib
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Any, List

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import HTTPError, RequestException


# -------------------- Paths / imports --------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_files.preprocess import trim_front_matter, normalize_text  # noqa


# -------------------- Config --------------------

UA = {"User-Agent": "ai-detect-corpus-builder/1.0 (research; contact: you@example.com)"}

# timeouts: (connect, read)
TIMEOUT_JSON = (10, 120)
TIMEOUT_TEXT = (10, 180)

# throttling / politeness
SLEEP_BASE = 0.15
SLEEP_JITTER = 0.45

# checkpointing
CHECKPOINT_EVERY = 2000
OUT_PATH_DEFAULT = "data/raw/human_paragraphs2_0.jsonl"
CHECKPOINT_PATH_DEFAULT = "data/raw/human_paragraphs2_0.checkpoint.json"

# robustness
MAX_FAIL_PER_SOURCE = 25
MAX_TEXT_BYTES = 5_000_000  # ~5MB per downloaded text; prevents huge Gutenberg pulls


WS = re.compile(r"[ \t]+")
PARA_SPLIT = re.compile(r"\n\s*\n+")
WIKI_TEMPL = re.compile(r"\{\{.*?\}\}", re.DOTALL)
WIKI_REF = re.compile(r"<ref.*?>.*?</ref>", re.DOTALL)
WIKI_TAGS = re.compile(r"</?[^>]+>")
WIKI_LINKS = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
WIKI_BRACKETS = re.compile(r"\[http[^\s\]]+\s*([^\]]*)\]")
NON_TEXT = re.compile(r"[^\S\n]+")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_human_50k")


# -------------------- Requests session with retries/backoff --------------------

def make_session() -> requests.Session:
    retry = Retry(
        total=10,
        connect=10,
        read=10,
        status=10,
        backoff_factor=0.9,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    s = requests.Session()
    s.headers.update(UA)
    s.mount("http://", HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50))
    return s

SESSION = make_session()

# CourtListener API token via env
CL_KEY = os.getenv("COURTLISTENER_API_KEY") or os.getenv("COURTLISTENER_API_TOKEN")
if CL_KEY:
    SESSION.headers.update({"Authorization": f"Token {CL_KEY}"})


def polite_sleep(mult: float = 1.0) -> None:
    time.sleep(mult * (SLEEP_BASE + random.random() * SLEEP_JITTER))


def get_json(url: str, params: dict) -> dict:
    r = SESSION.get(url, params=params, timeout=TIMEOUT_JSON)
    if r.status_code >= 400:
        # Let caller decide if to skip/disable
        raise HTTPError(f"HTTP {r.status_code} for {url}", response=r)
    return r.json()


def get_text(url: str) -> str:
    r = SESSION.get(url, timeout=TIMEOUT_TEXT)
    if r.status_code >= 400:
        raise HTTPError(f"HTTP {r.status_code} for {url}", response=r)
    return r.text


def get_text_limited(url: str, max_bytes: int = MAX_TEXT_BYTES) -> str:
    """Stream download and abort if too large (prevents huge Gutenberg texts)."""
    with SESSION.get(url, timeout=TIMEOUT_TEXT, stream=True) as r:
        if r.status_code >= 400:
            raise HTTPError(f"HTTP {r.status_code} for {url}", response=r)
        chunks: List[bytes] = []
        total = 0
        for ch in r.iter_content(chunk_size=65536):
            if not ch:
                continue
            total += len(ch)
            if total > max_bytes:
                raise RuntimeError(f"Response too large ({total} bytes) for {url}")
            chunks.append(ch)
    return b"".join(chunks).decode("utf-8", errors="ignore")


# -------------------- Text utils --------------------

def to_paragraphs(text: str, min_chars: int = 200, max_chars: int = 2000) -> List[str]:
    text = normalize_text(text)
    paras = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    out = []
    for p in paras:
        p = NON_TEXT.sub(" ", p).strip()
        if min_chars <= len(p) <= max_chars:
            out.append(p)
    return out


def wiki_to_text(wikitext: str) -> str:
    t = WIKI_REF.sub(" ", wikitext)
    t = WIKI_TEMPL.sub(" ", t)
    t = WIKI_TAGS.sub(" ", t)
    t = WIKI_LINKS.sub(r"\1", t)
    t = WIKI_BRACKETS.sub(r"\1", t)
    t = re.sub(r"''+", "", t)
    t = re.sub(r"\n\*+", "\n", t)
    return normalize_text(t)


def stable_para_id(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


# -------------------- Sources --------------------

def fetch_gutenberg_paras(target_paras: int) -> Iterator[dict]:
    base = "https://gutendex.com/books"
    got = 0
    page = 1
    while got < target_paras:
        js = get_json(base, {"page": page})
        for b in js.get("results", []):
            if got >= target_paras:
                break

            formats = b.get("formats", {}) or {}
            txt_url = (
                formats.get("text/plain; charset=utf-8")
                or formats.get("text/plain; charset=us-ascii")
                or formats.get("text/plain")
            )
            if not txt_url:
                continue

            # Stream + size guard
            raw = get_text_limited(txt_url, max_bytes=MAX_TEXT_BYTES)
            raw = trim_front_matter(raw)
            paras = to_paragraphs(raw)

            for p in paras:
                yield {
                    "text": p,
                    "meta": {"source": "book_gutenberg", "gutenberg_id": b.get("id"), "title": b.get("title")},
                }
                got += 1
                if got >= target_paras:
                    break

            polite_sleep()
        if js.get("next") is None:
            break
        page += 1


def mw_random_titles(api: str, n: int, namespace: int = 0) -> List[str]:
    js = get_json(
        api,
        {"action": "query", "list": "random", "rnnamespace": namespace, "rnlimit": n, "format": "json"},
    )
    return [x["title"] for x in js["query"]["random"]]


def mw_old_revision_wikitext(api: str, title: str, rvstart: str) -> str:
    js = get_json(
        api,
        {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvslots": "main",
            "rvprop": "timestamp|content",
            "rvlimit": 1,
            "rvstart": rvstart,
            "rvdir": "older",
            "format": "json",
            "formatversion": "2",
        },
    )
    page = js["query"]["pages"][0]
    if "missing" in page:
        return ""
    revs = page.get("revisions", [])
    if not revs:
        return ""
    return revs[0]["slots"]["main"].get("content", "")


def fetch_wikipedia_old_paras(target_paras: int, cutoff_iso: str = "2018-12-31T23:59:59Z") -> Iterator[dict]:
    api = "https://en.wikipedia.org/w/api.php"
    got = 0
    while got < target_paras:
        titles = mw_random_titles(api, 20, namespace=0)
        for t in titles:
            wikitext = mw_old_revision_wikitext(api, t, cutoff_iso)
            if not wikitext:
                continue
            text = wiki_to_text(wikitext)
            paras = to_paragraphs(text)
            for p in paras:
                yield {"text": p, "meta": {"source": "wikipedia_pre2019", "title": t, "cutoff": cutoff_iso}}
                got += 1
                if got >= target_paras:
                    break
            polite_sleep(mult=0.6)


def fetch_wikinews_old_paras(target_paras: int, cutoff_iso: str = "2018-12-31T23:59:59Z") -> Iterator[dict]:
    api = "https://en.wikinews.org/w/api.php"
    got = 0
    while got < target_paras:
        titles = mw_random_titles(api, 20, namespace=0)
        for t in titles:
            wikitext = mw_old_revision_wikitext(api, t, cutoff_iso)
            if not wikitext:
                continue
            text = wiki_to_text(wikitext)
            paras = to_paragraphs(text)
            for p in paras:
                yield {"text": p, "meta": {"source": "news_wikinews_pre2019", "title": t, "cutoff": cutoff_iso}}
                got += 1
                if got >= target_paras:
                    break
            polite_sleep(mult=0.6)


def fetch_arxiv_abstract_paras(target_paras: int, until_year: int = 2018) -> Iterator[dict]:
    base = "http://export.arxiv.org/api/query"
    got = 0
    start = 0
    batch = 100

    while got < target_paras:
        # keep query simple + stable
        q = "all:the"
        url = f"{base}?search_query={q}&start={start}&max_results={batch}&sortBy=submittedDate&sortOrder=ascending"
        xml = get_text(url)
        root = ET.fromstring(xml)
        ns = {"a": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("a:entry", ns):
            if got >= target_paras:
                break
            published = entry.findtext("a:published", default="", namespaces=ns)
            y = int(published[:4]) if published[:4].isdigit() else 9999
            if y > until_year:
                return
            abstract = entry.findtext("a:summary", default="", namespaces=ns).strip()
            for p in to_paragraphs(abstract, min_chars=120, max_chars=2000):
                yield {"text": p, "meta": {"source": "academic_arxiv_abstract", "published": published}}
                got += 1
                if got >= target_paras:
                    break

        start += batch
        polite_sleep(mult=1.2)


def fetch_federal_register_paras(target_paras: int, until_date: str = "2018-12-31") -> Iterator[dict]:
    base = "https://www.federalregister.gov/api/v1/documents.json"
    got = 0
    page = 1
    while got < target_paras:
        js = get_json(
            base,
            {
                "per_page": 100,
                "page": page,
                "order": "oldest",
                "conditions[publication_date][lte]": until_date,
            },
        )
        for doc in js.get("results", []):
            if got >= target_paras:
                break
            body = doc.get("abstract") or doc.get("title") or ""
            for p in to_paragraphs(body, min_chars=120, max_chars=2000):
                yield {
                    "text": p,
                    "meta": {
                        "source": "gov_federal_register",
                        "document_number": doc.get("document_number"),
                        "publication_date": doc.get("publication_date"),
                    },
                }
                got += 1
                if got >= target_paras:
                    break
        if not js.get("results"):
            break
        page += 1
        polite_sleep()


def fetch_courtlistener_opinion_paras(target_paras: int, until_date: str = "2018-12-31") -> Iterator[dict]:
    base = "https://www.courtlistener.com/api/rest/v4/opinions/"
    got = 0
    url = base + f"?filed__lte={until_date}&order_by=filed"

    while got < target_paras:
        try:
            js = get_json(url, {})
        except HTTPError as e:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            if status == 401:
                log.error("CourtListener 401 Unauthorized — set COURTLISTENER_API_KEY to enable. Skipping.")
                return
            # transient or other http errors -> stop this source (orchestrator may disable after repeats)
            raise
        for op in js.get("results", []):
            if got >= target_paras:
                break
            text = op.get("plain_text") or op.get("html_with_citations") or ""
            text = re.sub(r"<[^>]+>", " ", text)
            for p in to_paragraphs(text, min_chars=200, max_chars=2000):
                yield {
                    "text": p,
                    "meta": {"source": "legal_courtlistener", "id": op.get("id"), "filed": op.get("filed")},
                }
                got += 1
                if got >= target_paras:
                    break

        url = js.get("next")
        if not url:
            break
        polite_sleep()


# -------------------- Checkpointing / IO --------------------

def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: str, records: List[dict]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_checkpoint(path: str, state: dict) -> None:
    ensure_parent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------- Orchestration --------------------

def redistribute_quotas(quotas: Dict[str, int], counts: Dict[str, int], active: List[str]) -> Dict[str, int]:
    """
    If some sources are disabled, redistribute remaining needed paragraphs
    across active sources proportional to their original quotas.
    """
    total_target = sum(quotas.values())
    got_total = sum(counts.values())
    remaining = max(0, total_target - got_total)
    if remaining == 0 or not active:
        return quotas

    # compute weights from original quotas
    w_sum = sum(quotas[k] for k in active)
    if w_sum == 0:
        return quotas

    new_quotas = dict(quotas)
    for k in active:
        # target for k = already got + share of remaining
        share = int(round(remaining * (quotas[k] / w_sum)))
        new_quotas[k] = counts[k] + share

    # fix rounding drift
    drift = total_target - sum(new_quotas.values())
    if drift != 0 and active:
        # add/subtract drift to largest quota source
        k0 = max(active, key=lambda kk: quotas[kk])
        new_quotas[k0] += drift

    return new_quotas


def build_50k(
    out_path: str = OUT_PATH_DEFAULT,
    checkpoint_path: str = CHECKPOINT_PATH_DEFAULT,
    total: int = 50_000,
    resume: bool = True,
) -> None:
    quotas = {
        "book_gutenberg": 12_000,
        "news_wikinews_pre2019": 8_000,
        "academic_arxiv_abstract": 8_000,
        "gov_federal_register": 7_000,
        "legal_courtlistener": 7_000,
        "wikipedia_pre2019": 8_000,
    }

    gens = {
        "book_gutenberg": fetch_gutenberg_paras(quotas["book_gutenberg"]),
        "news_wikinews_pre2019": fetch_wikinews_old_paras(quotas["news_wikinews_pre2019"]),
        "academic_arxiv_abstract": fetch_arxiv_abstract_paras(quotas["academic_arxiv_abstract"]),
        "gov_federal_register": fetch_federal_register_paras(quotas["gov_federal_register"]),
        "legal_courtlistener": fetch_courtlistener_opinion_paras(quotas["legal_courtlistener"]),
        "wikipedia_pre2019": fetch_wikipedia_old_paras(quotas["wikipedia_pre2019"]),
    }

    counts = {k: 0 for k in gens}
    failures = {k: 0 for k in gens}
    active = list(gens.keys())
    seen: set[str] = set()

    # resume support (best-effort; we store seen hashes to avoid duplicates across restarts)
    if resume:
        ck = load_checkpoint(checkpoint_path)
        if ck:
            counts.update(ck.get("counts", {}))
            failures.update({k: 0 for k in failures})
            active = [k for k in active if k not in ck.get("disabled_sources", [])]
            # seen can be huge; store a bloom or partial? Here we store hashes list up to a cap.
            seen_list = ck.get("seen_hashes", [])
            seen.update(seen_list)
            log.info("Resumed from checkpoint: total=%d", ck.get("total_records", 0))

    # start fresh output file if not resuming
    if not resume and os.path.exists(out_path):
        os.remove(out_path)

    buffer: List[dict] = []
    total_written = 0

    while active and total_written < total:
        # If some sources were disabled, reallocate quotas so we still reach total
        quotas = redistribute_quotas(quotas, counts, active)

        k = active.pop(0)
        if counts.get(k, 0) >= quotas.get(k, 0):
            # quota reached for this source
            continue

        try:
            rec = next(gens[k], None)
            failures[k] = 0
        except Exception as e:
            failures[k] += 1
            log.warning("Source %s error (%d/%d): %s", k, failures[k], MAX_FAIL_PER_SOURCE, repr(e))

            if failures[k] >= MAX_FAIL_PER_SOURCE:
                log.error("Disabling source %s after %d failures.", k, failures[k])
                gens.pop(k, None)
                counts.pop(k, None)
                failures.pop(k, None)
                active = [a for a in active if a != k]
            else:
                # try again later
                active.append(k)

            polite_sleep(mult=1.5)
            continue

        if rec is None:
            # exhausted -> don't re-add
            continue

        t = rec["text"]
        h = stable_para_id(t)
        if h not in seen:
            seen.add(h)
            buffer.append({"text": t, "label": "human", "meta": rec["meta"]})
            counts[k] += 1
            total_written += 1

        # round-robin continue
        active.append(k)

        if total_written % 500 == 0:
            log.info("total=%d %s", total_written, " ".join([f"{kk}={counts[kk]}" for kk in counts]))

        if len(buffer) >= 500:
            append_jsonl(out_path, buffer)
            buffer.clear()

        if total_written % CHECKPOINT_EVERY == 0:
            disabled = [src for src in gens.keys() if src not in active and counts.get(src, 0) < quotas.get(src, 0)]
            # store only a capped set of seen hashes (avoid huge checkpoint files)
            seen_cap = 200_000
            seen_snapshot = list(seen)[:seen_cap]
            save_checkpoint(
                checkpoint_path,
                {
                    "total_records": total_written,
                    "counts": counts,
                    "disabled_sources": [k for k in quotas.keys() if k not in gens],
                    "seen_hashes": seen_snapshot,
                    "seen_hashes_capped": seen_cap,
                    "note": "Seen hashes capped; duplicates across restarts are still unlikely but not impossible.",
                },
            )
            log.info("Checkpoint saved: %s", checkpoint_path)

        polite_sleep()

    if buffer:
        append_jsonl(out_path, buffer)
        buffer.clear()

    log.info("DONE total=%d %s", total_written, " ".join([f"{kk}={counts[kk]}" for kk in counts]))
    save_checkpoint(
        checkpoint_path,
        {
            "total_records": total_written,
            "counts": counts,
            "disabled_sources": [k for k in quotas.keys() if k not in gens],
            "seen_hashes": [],  # final: omit hashes to keep small
            "final": True,
        },
    )


if __name__ == "__main__":
    build_50k()

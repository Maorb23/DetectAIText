# scripts/modules/similarity.py
"""
Lightweight semantic-coherence / embedding-drift feature extractor.

Features (per-text-window):
- n_sentences
- mean_adjacent_cosine, std_adjacent_cosine
- mean_pairwise_cosine, std_pairwise_cosine
- embedding_variance (mean variance across embedding dims)
- centroid_cosine (mean cosine similarity to centroid)
- cluster_inertia (KMeans inertia if sklearn available)
- silhouette (if sklearn available and applicable)

Design goals:
- Use `sentence-transformers` if available (fast and simple API).
- Otherwise fall back to HuggingFace AutoModel + mean-pooling.
- Minimal external deps; clustering/silhouette are used only if sklearn exists.
- Returns per-window dicts compatible with other featurizers (meta/features/label keys).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import math
import numpy as np

# re-use sentence splitting and utilities from heuristics
from scripts.modules.heuristics import split_sentences, clamp01

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


class _EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._use_st = False
        self._loaded = False
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._loaded:
            return
        # Use sentence-transformers as the primary backend (no fallback or exception suppression)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._use_st = True
        self._loaded = True

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self._load()
        # Use sentence-transformers backend for encoding (no fallback)
        embs = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=batch_size)
        return embs


def _pairwise_cosines(embs: np.ndarray) -> np.ndarray:
    if embs.shape[0] <= 1:
        return np.array([])
    embs_n = _l2_normalize(embs)
    sim = embs_n @ embs_n.T
    # take upper triangle excluding diagonal
    i, j = np.triu_indices(sim.shape[0], k=1)
    return sim[i, j]


def compute_coherence_features(
    text: str,
    model: Optional[_EmbeddingModel] = None,
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
    batch_size: int = 32,
    clustering_k: int = 2,
) -> Dict[str, float]:
    """Compute sentence-embedding coherence features for a text window.

    Returns a flat dict with numeric features.
    """
    sents = split_sentences(text)
    n = len(sents)

    if n == 0:
        return {
            "n_sentences": 0,
            "mean_adjacent_cosine": 0.0,
            "std_adjacent_cosine": 0.0,
            "mean_pairwise_cosine": 0.0,
            "std_pairwise_cosine": 0.0,
            "embedding_variance": 0.0,
            "centroid_cosine": 0.0,
            "cluster_inertia": float("nan"),
            "silhouette": float("nan"),
        }

    if model is None:
        model = _EmbeddingModel(model_name=model_name, device=device)

    # compute sentence embeddings
    embs = model.encode(sents, batch_size=batch_size)
    embs = np.asarray(embs, dtype=float)

    # normalize
    embs_n = _l2_normalize(embs)

    # adjacency similarities
    if n >= 2:
        prev_embs = embs_n[:-1]         # embeddings for sentences 0..(n-2)
        next_embs = embs_n[1:]          # embeddings for sentences 1..(n-1)

        # element-wise dot products -> an array of length (n-1)
        adjacency_dots = np.sum(prev_embs * next_embs, axis=1)

        # ensure float dtype to match previous behavior
        adj = adjacency_dots.astype(float)
        mean_adj = float(np.mean(adj))
        std_adj = float(np.std(adj, ddof=0))
    else:
        mean_adj = 1.0
        std_adj = 0.0

    # pairwise similarities
    pairwise = _pairwise_cosines(embs)
    if pairwise.size:
        mean_pw = float(pairwise.mean())
        std_pw = float(pairwise.std(ddof=0))
    else:
        mean_pw = 1.0
        std_pw = 0.0

    # embedding variance (mean variance across dimensions)
    emb_var = float(np.var(embs, axis=0).mean())

    # centroid consistency: mean cosine of each sentence to centroid vector
    centroid = embs_n.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    centroid_cos = float(np.dot(embs_n, centroid).mean())

    # clustering / topical consistency
    k = min(clustering_k, n) # cannot have more clusters than points
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(embs_n)
    cluster_inertia = float(km.inertia_) # The inertia: Sum of squared distances to closest cluster center,
                                         # Adds a useful measure of how tight the clusters are.
    silhouette = float("nan")
    if n >= 3 and k >= 2:
        silhouette = float(silhouette_score(embs_n, labels)) # Silhouette is a metric that measures how similar an object is to its own cluster
                                                            # Adds a useful measure of how well-separated the clusters are.

    return {
        "n_sentences": int(n),
        "mean_adjacent_cosine": mean_adj,
        "std_adjacent_cosine": std_adj,
        "mean_pairwise_cosine": mean_pw,
        "std_pairwise_cosine": std_pw,
        "embedding_variance": emb_var,
        "centroid_cosine": centroid_cos,
        "cluster_inertia": cluster_inertia,
        "silhouette": silhouette,
    }


from scripts.features import aggregate_doc_from_windows


def featurize_texts_sim(
    examples: List[Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
    batch_size: int = 32,
    clustering_k: int = 2,
    aggregate_doc: bool = True,
) -> List[Dict[str, Any]]:
    """Featurize prepared examples and optionally aggregate to document level.

    Input example: {"text": str, "meta": {...}, "label": ...}
    Output per-window: {"label": ..., "meta": {...}, "features": { ... }}

    If `aggregate_doc=True`, returns per-window rows plus document-level aggregated rows
    (meta["level"] == "doc") similar to `compute_heuristics`.
    """
    if not examples:
        return []

    model = _EmbeddingModel(model_name=model_name, device=device)

    window_rows: List[Dict[str, Any]] = []
    for ex in examples:
        text = ex.get("text", "") or ""
        meta = dict(ex.get("meta", {}))
        label = ex.get("label", None)

        feats = compute_coherence_features(
            text, model=model, model_name=model_name, device=device, batch_size=batch_size, clustering_k=clustering_k
        )

        window_rows.append({"label": label, "meta": meta, "features": feats})

    if not aggregate_doc:
        return window_rows

    # aggregate per-doc (mean of numeric features and scores)
    by_doc = {}
    for r in window_rows:
        tid = str(r.get("meta", {}).get("text_id", r.get("meta", {}).get("doc_id", r.get("meta", {}).get("text_id", ""))))
        if not tid:
            continue
        by_doc.setdefault(tid, []).append(r)

    doc_rows: List[Dict[str, Any]] = []
    for tid, rows in by_doc.items():
        feats_list = [rr.get("features", {}) for rr in rows]
        keys = sorted({k for d in feats_list for k in d.keys()})
        agg: Dict[str, Any] = {}
        for k in keys:
            vals = np.array([d[k] for d in feats_list if k in d and isinstance(d[k], (int, float))], dtype=float)
            if vals.size == 0:
                continue
            agg[f"mean_{k}"] = float(vals.mean())
            agg[f"std_{k}"] = float(vals.std(ddof=0)) if vals.size > 1 else 0.0

        doc_meta = dict(rows[0].get("meta", {}))
        doc_meta["level"] = "doc"
        doc_meta["text_id"] = tid

        doc_rows.append({"label": rows[0].get("label"), "meta": doc_meta, "features": agg})

    return window_rows + doc_rows


# convenience: small CLI test when run directly
if __name__ == "__main__":
    sample = """
    Alice went to the market. She bought apples. They were bright and shiny.

    Suddenly, a dragon appeared. The dragon breathed fire and the market panicked.
    """
    print("Model:", "all-MiniLM-L6-v2")
    feats = compute_coherence_features(sample)
    for k, v in feats.items():
        print(k, v)

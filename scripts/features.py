# features.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Any, Dict, List
import numpy as np

def aggregate_doc_from_windows(
    window_rows: List[Dict[str, Any]],
    feature_key: str,
    doc_id_key: str = "text_id",
) -> List[Dict[str, Any]]:
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for r in window_rows:
        meta = r.get("meta", {}) or {}
        doc_id = meta.get(doc_id_key) or meta.get("doc_id") or meta.get("text_id") or "unknown"
        by_doc.setdefault(str(doc_id), []).append(r)

    doc_rows: List[Dict[str, Any]] = []
    for doc_id, rows in by_doc.items():
        feats_list = [rr.get(feature_key, {}) for rr in rows if isinstance(rr.get(feature_key, {}), dict)]
        if not feats_list:
            continue

        keys = sorted({k for d in feats_list for k in d.keys()})
        agg: Dict[str, Any] = {}
        for k in keys:
            vals = np.array([d[k] for d in feats_list if k in d and isinstance(d[k], (int, float))], dtype=float)
            if vals.size == 0:
                continue
            agg[f"mean_{k}"] = float(vals.mean())
            agg[f"std_{k}"] = float(vals.std(ddof=0)) if vals.size > 1 else 0.0

        doc_rows.append(
            {
                "label": rows[0].get("label", None),
                "meta": {"level": "doc", doc_id_key: doc_id},
                feature_key: agg,
            }
        )
    return doc_rows

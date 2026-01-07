# scripts/score.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.features import aggregate_doc_from_windows
from scripts.heuristics import compute_heuristics
from scripts.logits import LogitsFeatureExtractor   


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(json.loads(line))
    return examples


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n") # write json combined with newline


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parse = argparse.ArgumentParser()
    parse.add_argument("--prepared_path", type=str, required=True, help="Path to prepared examples JSONL")
    parse.add_argument("--out_dir", type=str, default=None, help="Output directory (default: same as input)")
    parse.add_argument("--aggregate_doc", action="store_true", help="Also compute document-level aggregates")

    # Logits (Step 3)
    parse.add_argument("--with_logits", action="store_true", help="Also compute logits features (Step 3)")
    parse.add_argument("--logits_model_id", type=str, default="Qwen/Qwen3-0.6B", help="HF model id for logits")
    parse.add_argument("--logits_max_tokens", type=int, default=512, help="Max tokens per window for logits scoring")
    parse.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda). Default: auto")

    args = parse.parse_args()

    prepared_path = Path(args.prepared_path)
    out_dir = Path(args.out_dir) if args.out_dir else prepared_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = read_jsonl(prepared_path)
    base = prepared_path.stem.replace("_prepared", "")

    # -------------------------
    # Step 1: Heuristics
    # -------------------------
    scored_rows = compute_heuristics(examples, aggregate_doc=args.aggregate_doc)
    heur_window_rows = [r for r in scored_rows if r.get("meta", {}).get("level") != "doc"]
    heur_doc_rows = [r for r in scored_rows if r.get("meta", {}).get("level") == "doc"]

    heur_windows_out = out_dir / f"{base}_heuristics_windows.jsonl"
    heur_doc_out = out_dir / f"{base}_heuristics_doc.json"
    write_jsonl(heur_windows_out, heur_window_rows)

    if heur_doc_rows:
        doc_obj = heur_doc_rows[0] if len(heur_doc_rows) == 1 else {"docs": heur_doc_rows}
        write_json(heur_doc_out, doc_obj)

    print(f"Saved window heuristics: {heur_windows_out}")
    if heur_doc_rows:
        print(f"Saved doc heuristics: {heur_doc_out}")

    # -------------------------
    # Step 3: Logits features
    # -------------------------
    if args.with_logits:
        extractor = LogitsFeatureExtractor(
            model_id=args.logits_model_id,
            device=args.device,
            max_tokens=args.logits_max_tokens,
        )

        logits_rows = extractor.featurize_examples(examples)  # returns [{label, meta, features}]
        # rename "features" -> "logits_features" to avoid collisions later
        logits_window_rows: List[Dict[str, Any]] = []
        for r in logits_rows:
            logits_window_rows.append(
                {
                    "label": r.get("label", None),
                    "meta": r.get("meta", {}),
                    "logits_features": r.get("features", {}),
                }
            )

        logits_windows_out = out_dir / f"{base}_logits_windows.jsonl"
        write_jsonl(logits_windows_out, logits_window_rows)
        print(f"Saved window logits: {logits_windows_out}")

        if args.aggregate_doc:
            logits_doc_rows = aggregate_doc_from_windows(logits_window_rows, feature_key="logits_features")
            logits_doc_out = out_dir / f"{base}_logits_doc.json"
            doc_obj = logits_doc_rows[0] if len(logits_doc_rows) == 1 else {"docs": logits_doc_rows}
            write_json(logits_doc_out, doc_obj)
            print(f"Saved doc logits: {logits_doc_out}")


if __name__ == "__main__":
    main()
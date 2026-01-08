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
from scripts.modules.heuristics import compute_heuristics
from scripts.modules.logits import LogitsFeatureExtractor
from scripts.modules.binocular import BinocularsTool, BinocularsConfig


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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parse = argparse.ArgumentParser()
    parse.add_argument("--prepared_path", type=str, required=True, help="Path to prepared examples JSONL")
    parse.add_argument("--out_dir", type=str, default=None, help="Output directory (default: same as input)")
    parse.add_argument("--aggregate_doc", action="store_true", help="Also compute document-level aggregates")

    # Step 1: Heuristics
    parse.add_argument("--with_heuristics", action="store_true", help="Also compute heuristic features (Step 1)")

    # Step 2: Binoculars
    parse.add_argument("--with_binoculars", action="store_true", help="Also compute binoculars features (Step 2)")
    parse.add_argument("--bino_observer_id", type=str, default="Qwen/Qwen2.5-1.5B", help="HF model id for observer (base)")
    parse.add_argument("--bino_performer_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model id for performer (instruct)")
    parse.add_argument("--bino_max_tokens", type=int, default=512, help="Max tokens per window for binoculars")
    parse.add_argument("--bino_mode", type=str, default="low-fpr", choices=["low-fpr", "accuracy"], help="Binoculars threshold mode")
    parse.add_argument("--bino_use_bfloat16", action="store_true", help="Use bfloat16 for binoculars (if supported)")
    parse.add_argument("--bino_device_1", type=str, default=None, help="Observer device (e.g., cuda:0). Default: auto")
    parse.add_argument("--bino_device_2", type=str, default=None, help="Performer device (e.g., cuda:1). Default: auto")
    
    # Step 3: Logits
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
    if args.with_heuristics:
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
    # Step 2: Binoculars features
    # -------------------------
    if args.with_binoculars:
        tool = BinocularsTool(
            BinocularsConfig(
                observer_model_id=args.bino_observer_id,
                performer_model_id=args.bino_performer_id,
                max_tokens=args.bino_max_tokens,
                mode=args.bino_mode,
                use_bfloat16=args.bino_use_bfloat16,
                device_1=args.bino_device_1,
                device_2=args.bino_device_2,
            )
        )

        texts = [r.get("text", "") for r in examples]
        bino_feats = tool.featurize_texts(texts)  # list of {"binoculars_features": {...}}

        bino_window_rows: List[Dict[str, Any]] = []
        for rr, feat in zip(examples, bino_feats):
            bino_window_rows.append(
                {
                    "label": rr.get("label", None),
                    "meta": rr.get("meta", {}),
                    "binoculars_features": feat.get("binoculars_features", {}),
                }
            )

        bino_windows_out = out_dir / f"{base}_binoculars_windows.jsonl"
        write_jsonl(bino_windows_out, bino_window_rows)
        print(f"Saved window binoculars: {bino_windows_out}")

        if args.aggregate_doc:
            bino_doc_rows = aggregate_doc_from_windows(bino_window_rows, feature_key="binoculars_features")
            bino_doc_out = out_dir / f"{base}_binoculars_doc.json"
            doc_obj = bino_doc_rows[0] if len(bino_doc_rows) == 1 else {"docs": bino_doc_rows}
            write_json(bino_doc_out, doc_obj)
            print(f"Saved doc binoculars: {bino_doc_out}")

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

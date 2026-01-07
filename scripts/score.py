# scripts/score.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.heuristics import compute_heuristics


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
    parse.add_argument("--prepared_path", type=str, help="Path to prepared examples JSONL")
    parse.add_argument("--out_dir", type=str, default=None, help="Output directory (default: same as input)")
    parse.add_argument("--aggregate_doc", action="store_true", help="Also compute document-level aggregates")
    args = parse.parse_args()

    prepared_path = Path(args.prepared_path)
    out_dir = Path(args.out_dir) if args.out_dir else prepared_path.parent

    examples = read_jsonl(prepared_path)

    # compute_heuristics returns: window rows + doc rows (level='doc')
    scored_rows = compute_heuristics(examples, aggregate_doc=args.aggregate_doc)

    window_rows = [r for r in scored_rows if r.get("meta", {}).get("level") != "doc"]
    doc_rows = [r for r in scored_rows if r.get("meta", {}).get("level") == "doc"]

    base = prepared_path.stem.replace("_prepared", "")
    windows_out = out_dir / f"{base}_heuristics_windows.jsonl"
    doc_out = out_dir / f"{base}_heuristics_doc.json"

    write_jsonl(windows_out, window_rows)

    if doc_rows: 
        # usually one doc row per text_id. if multiple, store list.
        doc_obj = doc_rows[0] if len(doc_rows) == 1 else {"docs": doc_rows}
        write_json(doc_out, doc_obj)

    print(f"Saved window heuristics: {windows_out}")
    if doc_rows:
        print(f"Saved doc heuristics: {doc_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run `get_text.py` on every JSONL row that has a `text` field.

Writes a raw `.raw.txt` file for each row and (by default) calls
`scripts/get_text.py` to produce a normalized `.txt` file.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slugify(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:maxlen] if s else "row"


def main() -> None:
    p = argparse.ArgumentParser(description="Run get_text.py on each JSONL text row")
    p.add_argument("--jsonl_path", type=str, default="data/raw/human_paragraphs2_0.jsonl")
    p.add_argument("--output_dir", type=str, default="data/processed/human_paragraphs2_0_gettext")
    p.add_argument("--call_get_text", action="store_true", help="Invoke scripts/get_text.py for normalization")
    p.add_argument("--max_rows", type=int, default=0, help="If >0, only process this many rows")
    p.add_argument("--max_estimated_tokens", type=int, default=None)
    p.add_argument("--overlap_estimated_tokens", type=int, default=0)
    args = p.parse_args()

    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if args.max_rows and i >= args.max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                print(f"Skipping invalid JSON at row {i}")
                continue

            text = obj.get("text") or obj.get("content")
            if not text:
                print(f"No text field at row {i}, skipping")
                continue

            meta = obj.get("meta", {}) or {}
            parts = []
            if isinstance(meta, dict) and meta.get("title"):
                parts.append(str(meta.get("title")))
            elif isinstance(meta, dict) and meta.get("gutenberg_id"):
                parts.append(str(meta.get("gutenberg_id")))
            parts.append(str(i))
            base = slugify("_".join(parts))

            raw_path = out_dir / f"{base}.raw.txt"
            with raw_path.open("w", encoding="utf-8", newline="\n") as rf:
                rf.write(text)

            normalized_path = out_dir / f"{base}.txt"

            if args.call_get_text:
                cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "get_text.py"),
                       "--input_path", str(raw_path),
                       "--output_path", str(normalized_path)]
                if args.max_estimated_tokens is not None:
                    cmd += ["--max_estimated_tokens", str(args.max_estimated_tokens),
                            "--overlap_estimated_tokens", str(args.overlap_estimated_tokens)]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
            else:
                # if not calling get_text.py, just copy raw to normalized (no normalization)
                with normalized_path.open("w", encoding="utf-8", newline="\n") as wf:
                    wf.write(text)

    print(f"All done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()

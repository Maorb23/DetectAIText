# scripts/get_text.py
# gets text from various file formats and normalizes it

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.convert import load_text_format
from scripts.preprocess import normalize_text, split_text_into_windows, prepare_examples
import logging
import json
logger = logging.getLogger(__name__) # set up a logger
logger.setLevel(logging.DEBUG)


def write_text(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

def write_jsonl(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .pdf/.docx/.txt into normalized .txt")
    parser.add_argument("--input_path", type=str, help="Path to input file (.pdf, .docx, or .txt)")
    parser.add_argument("--output_path",type=str,default=None, help="Where to save the normalized text")
    parser.add_argument("--max_estimated_tokens",type=int,default=None, help="Split into windows of ~this many tokens and join with a separator.")
    parser.add_argument("--overlap_estimated_tokens",type=int,default=0, help="Optional overlap between windows (approx tokens). Only used if max_estimated_tokens is set.")
    parser.add_argument("--window_separator",type=str,default="\n\n--- WINDOW ---\n\n", help="Separator used when joining multiple windows into a single output file.") 
    parser.add_argument("--print",action="store_true", help="Print the normalized text to stdout instead of writing to a file.")
    parser.add_argument("--prepare_examples",action="store_true", help="Prepare examples instead of just normalizing text (for future use).")
    parser.add_argument("--normalize",action="store_true", help="Whether to normalize the text after loading.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_text = load_text_format(str(input_path))
    normalized_text = normalize_text(raw_text)

    if args.max_estimated_tokens is not None:
        windows = split_text_into_windows(
            normalized_text,
            max_estimated_tokens=args.max_estimated_tokens,
            overlap_estimated_tokens=args.overlap_estimated_tokens,
        )
        normalized_text = args.window_separator.join(windows)

    if args.print:
        print(normalized_text)
        return

    if args.output_path is None:
        output_path = input_path.with_suffix(".txt")
    else:
        output_path = Path(args.output_path)

    write_text(output_path, normalized_text)
    print(f"Saved: {output_path}")
    if args.prepare_examples:
        examples = [{"text": raw_text, "label": 0, "meta": {}}] 
        prepared = prepare_examples(
            examples,
            max_estimated_tokens=args.max_estimated_tokens,
            overlap_estimated_tokens=args.overlap_estimated_tokens,
            normalize=args.normalize,
        ) ## Remember to pass normalize arg
        print(f"Prepared {len(prepared)} example(s).")
        # save prepaed examples
        prep_output_path = Path(str(output_path).replace(".txt", "_prepared.jsonl"))
        write_jsonl(prep_output_path, prepared)
        print(f"Saved prepared examples to: {prep_output_path}")


if __name__ == "__main__":
    main()

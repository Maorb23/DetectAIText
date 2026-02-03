#!/usr/bin/env python3
# scripts/data_files/generate_synthetic.py
"""
Create synthetic windows by rewriting windows from input documents
using a ChatGPT-style model (e.g. gpt-5.2). Outputs a JSONL with:
{source_file, window_idx, original_tokens, original_text, synthetic_tokens, synthetic_text}
"""
from __future__ import annotations
import os
import time
import json
import argparse
from pathlib import Path
from typing import List

import openai
from transformers import AutoTokenizer
from tqdm import tqdm

from scripts.data_files.convert import load_text_format


def get_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained("gpt2", use_fast=True)


def split_into_windows(text: str, tokenizer, window_size: int, stride: int = None):
    if stride is None:
        stride = window_size
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    windows = []
    for start in range(0, max(1, len(token_ids)), stride):
        chunk = token_ids[start : start + window_size]
        if not chunk:
            break
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True).strip()
        if chunk_text:
            windows.append((start, chunk, chunk_text))
        if start + window_size >= len(token_ids):
            break
    return windows


def call_chatgpt_rewrite(prompt_text: str, model: str, approx_tokens: int, temperature: float = 0.7, max_retries: int = 3, sleep_base: float = 1.0):
    system_msg = (
        "You are a professional writer. Rewrite the user's text to be as natural and human-like as possible. "
        "Preserve the meaning and factual content but rephrase and vary structure and wording. "
        "Output only the rewritten text (no headers or explanations). "
        f"Try to produce approximately {approx_tokens} tokens (allow small deviations)."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            # older/newer clients: get content
            content = resp["choices"][0]["message"]["content"]
            return content.strip()
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(sleep_base * attempt)
    raise RuntimeError("Unreachable")


def process_file(path: Path, out_f, model: str, tokenizer, window_size: int, stride: int, temperature: float, api_key_env: str):
    text = load_text_format(str(path))
    windows = split_into_windows(text, tokenizer, window_size, stride)
    for idx, (start_token, token_ids, original_text) in enumerate(windows):
        try:
            synthetic = call_chatgpt_rewrite(original_text, model=model, approx_tokens=window_size, temperature=temperature)
        except Exception as e:
            synthetic = ""
        original_tokens = len(token_ids)
        synthetic_token_ids = tokenizer.encode(synthetic, add_special_tokens=False)
        out = {
            "source_file": str(path),
            "window_idx": idx,
            "original_tokens": original_tokens,
            "original_text": original_text,
            "synthetic_tokens": len(synthetic_token_ids),
            "synthetic_text": synthetic,
        }
        out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
        out_f.flush()


def main():
    p = argparse.ArgumentParser(description="Generate synthetic windows with ChatGPT-style model")
    p.add_argument("inputs", nargs="+", help="Input files or directories (.pdf, .docx, .jsonl, .txt)")
    p.add_argument("--out", default="synthetic_windows.jsonl", help="Output JSONL path")
    p.add_argument("--model", default="gpt-5.2", help="Chat model name (OpenAI).")
    p.add_argument("--tokenizer", default=None, help="Tokenizer model id (HF) - defaults to model or gpt2 fallback")
    p.add_argument("--window_size", type=int, default=512, help="Window size in tokens (approx)")
    p.add_argument("--stride", type=int, default=None, help="Stride in tokens between windows (default=window_size)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--openai_key_env", default="OPENAI_API_KEY", help="Env var with OpenAI key")
    args = p.parse_args()

    api_key = os.getenv(args.openai_key_env)
    if not api_key:
        raise RuntimeError(f"Please set env var {args.openai_key_env} to your OpenAI API key")
    openai.api_key = api_key

    tokenizer_name = args.tokenizer or args.model
    tokenizer = get_tokenizer(tokenizer_name)

    input_paths: List[Path] = []
    for inp in args.inputs:
        pth = Path(inp)
        if pth.is_dir():
            for f in pth.rglob("*"):
                if f.suffix.lower() in {".pdf", ".docx", ".jsonl", ".txt"}:
                    input_paths.append(f)
        elif pth.exists():
            input_paths.append(pth)
        else:
            print(f"Warning: {inp} does not exist, skipping")

    with open(args.out, "w", encoding="utf-8") as out_f:
        for path in tqdm(input_paths, desc="Files"):
            process_file(path, out_f, model=args.model, tokenizer=tokenizer, window_size=args.window_size, stride=args.stride or args.window_size, temperature=args.temperature, api_key_env=args.openai_key_env)


if __name__ == "__main__":
    main()
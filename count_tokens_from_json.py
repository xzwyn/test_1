# scripts/count_tokens_from_json.py
"""
Count tiktoken tokens for each text segment in English and German Azure Document Intelligence JSONs.

Usage:
  python scripts/count_tokens_from_json.py --english-json input/67.pdf.json --german-json input/67g.pdf.json --tiktoken-model text-embedding-3-large --output output/token_counts.csv

What it does:
- Loads structured content from the provided JSONs using your existing parser (process_document_json).
- Uses tiktoken to count tokens per text segment.
- Prints total tokens per language and grand total.
- Optionally writes per-segment counts to CSV or JSONL.

Notes:
- The default tokenizer is for "text-embedding-3-large". If unavailable in your tiktoken install, it falls back to "cl100k_base".
- Add "tiktoken" to your requirements.txt if not already present.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any

import tiktoken
import pandas as pd

from src.processing.json_parser import process_document_json


def get_encoding(model_name: str = "text-embedding-3-large"):
    """
    Get tiktoken encoding for a given model, with fallback.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, enc) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def to_rows(content: List[Dict[str, Any]], enc, language: str) -> List[Dict[str, Any]]:
    rows = []
    for idx, item in enumerate(content):
        text = item.get("text", "") or ""
        tokens = count_tokens(text, enc)
        rows.append({
            "language": language,
            "index": idx,
            "page": item.get("page", None),
            "type": item.get("type", ""),
            "tokens": tokens,
            "text": text,
        })
    return rows


def save_output(rows: List[Dict[str, Any]], output_path: Path):
    if output_path.suffix.lower() == ".jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for r in rows:
                # Minimal manual JSON to avoid importing json for speed
                import json
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved JSONL to {output_path.resolve()}")
    else:
        # Default to CSV (supports .csv and other tabular formats via pandas)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Saved CSV to {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Count tokens for each text segment in English and German JSONs.")
    parser.add_argument("--english-json", required=True, type=str, help="Path to English Azure Document Intelligence JSON.")
    parser.add_argument("--german-json", required=True, type=str, help="Path to German Azure Document Intelligence JSON.")
    parser.add_argument("--tiktoken-model", type=str, default="text-embedding-3-large", help="Model name for tiktoken encoding.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save per-segment counts (CSV or JSONL).")
    args = parser.parse_args()

    eng_path = Path(args.english_json)
    ger_path = Path(args.german_json)

    if not eng_path.exists():
        raise FileNotFoundError(f"English JSON not found: {eng_path}")
    if not ger_path.exists():
        raise FileNotFoundError(f"German JSON not found: {ger_path}")

    enc = get_encoding(args.tiktoken_model)

    # Use your existing JSON processor to get structured content
    eng_content = process_document_json(eng_path)
    ger_content = process_document_json(ger_path)

    eng_rows = to_rows(eng_content, enc, language="English")
    ger_rows = to_rows(ger_content, enc, language="German")

    # Totals
    total_eng_tokens = sum(r["tokens"] for r in eng_rows)
    total_ger_tokens = sum(r["tokens"] for r in ger_rows)
    total_tokens_all = total_eng_tokens + total_ger_tokens

    # Print totals (mirroring your requested print style)
    print(f"The total number of tokens for all strings is {total_tokens_all}.")
    print(f"English total tokens: {total_eng_tokens}")
    print(f"German total tokens:  {total_ger_tokens}")

    # Optional output
    if args.output:
        all_rows = eng_rows + ger_rows
        save_output(all_rows, Path(args.output))


if __name__ == "__main__":
    main()

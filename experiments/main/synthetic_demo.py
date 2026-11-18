"""Generate artificial test fixtures and exercise offline IO, never paper results."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .offline import select_file, rerank_file


def make_fixture(directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "pool.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query", "doc_id", "doc_title", "doc_text", "bm25_rank", "bm25_score"])
        for i in range(1, 1001):
            title = "Amber Observatory" if i == 1 else f"Synthetic record {i}"
            text = "A fictional link to Amber Observatory." if i == 21 else "Artificial test content."
            writer.writerow(["synthetic-q1", "Artificial interface test", f"synthetic-d{i}", title, text, i, 1001-i])
    scores = {f"synthetic-d{i}": float((i * 17) % 101) for i in range(1, 1001)}
    (directory / "scores.jsonl").write_text(json.dumps({"query_id": "synthetic-q1", "scores": scores}) + "\n")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="A new directory for synthetic fixtures")
    args = parser.parse_args()
    d = make_fixture(args.out)
    select_file(d / "pool.csv", d / "selection.jsonl")
    rerank_file(d / "selection.jsonl", d / "scores.jsonl", d / "top10.jsonl")
    print("SYNTHETIC INTERFACE CHECK ONLY: created fixtures and Top-10; no research result reproduced.")


if __name__ == "__main__":
    main()

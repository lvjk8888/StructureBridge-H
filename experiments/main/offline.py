"""Release-only offline IO wrapper; scientific functions retain their source logic.

Accept externally prepared Top-1000 pools and reranker scores. No retrieval,
models, downloads, HTTP, or API calls are performed by this entry point.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from sbh.selection.candidates import choose
from sbh.selection.input import stream_query_groups
from sbh.selection.ranking import rank_one, union_for
from sbh.selection.scoring import validate_group
from sbh.structure.adapter import AdapterConfig, build_query_hypergraph

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "frozen_main.json"
COLUMNS = {"query_id", "query", "doc_id", "doc_title", "doc_text", "bm25_rank", "bm25_score"}


def select_file(pool: Path, output: Path):
    cfg = json.loads(CONFIG.read_text())
    with pool.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        if len(header) != len(COLUMNS) or set(header) != COLUMNS:
            raise ValueError("Pool must contain exactly seven unique documented columns, without labels/answers")
        if next(reader, None) is None:
            raise ValueError("Pool has no candidate rows")
    seen = set()
    with output.open("x", encoding="utf-8") as out:
        for q, group in stream_query_groups(pool, 100000):
            if q in seen or len(set(map(str, group["query"]))) != 1:
                raise ValueError("Repeated query group or inconsistent query text")
            seen.add(q)
            edges, stats = build_query_hypergraph(group, AdapterConfig(max_edge_size=cfg["max_edge_size"]))
            docs = [{"doc_id": str(r.doc_id), "title": str(r.doc_title), "text": str(r.doc_text),
                     "rank": int(r.bm25_rank), "retrieval_score": float(r.bm25_score)}
                    for r in group.itertuples(index=False)]
            validate_group(q, [(d["doc_id"], d["rank"], d["retrieval_score"]) for d in docs],
                           {"query_id": q, "candidate_doc_ids": [d["doc_id"] for d in docs],
                            "hyperedges": edges}, cfg["top_k"])
            cells = {str(m): {name: choose(docs, edges, m, name, lam, cfg["rounds"])
                             for name, lam in cfg["methods"].items()} for m in cfg["budgets"]}
            for m, methods in cells.items():
                if any(len(set(c["selected"])) != int(m) for c in methods.values()):
                    raise ValueError("Selector did not produce exact unique M")
            record = {"query_id": q, "query": str(group["query"].iloc[0]),
                      "candidates": [{k: d[k] for k in ("doc_id", "rank", "retrieval_score")} for d in docs],
                      "hyperedges": edges, "adapter_stats": stats, "by_budget": cells}
            out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return len(seen)


def rerank_file(records: Path, scores: Path, output: Path, kind="qwen"):
    cfg = json.loads(CONFIG.read_text())
    maps = {}
    with scores.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = str(row["query_id"])
            if q in maps:
                raise ValueError("Duplicate score query")
            vals = {str(d): float(v) for d, v in row["scores"].items()}
            if not all(math.isfinite(v) for v in vals.values()):
                raise ValueError("Nonfinite reranker score")
            maps[q] = vals
    if not maps:
        raise ValueError("Empty reranker score input")
    with records.open(encoding="utf-8") as check:
        if not check.readline().strip():
            raise ValueError("Empty selection records")
    seen = set()
    with records.open(encoding="utf-8") as src, output.open("x", encoding="utf-8") as out:
        for line in src:
            record = json.loads(line)
            q = str(record["query_id"])
            if q in seen:
                raise ValueError("Duplicate selection query")
            seen.add(q)
            required = {d["doc_id"] for d in union_for(record)}
            if q not in maps or not required <= maps[q].keys():
                raise ValueError("Missing selected-document reranker scores; no fallback scores are allowed")
            rankings = rank_one(record, maps[q], kind)
            top = {b: {m: ids[:cfg["top_n"]] for m, ids in ms.items()} for b, ms in rankings.items()}
            out.write(json.dumps({"query_id": q, "reranker_kind": kind, "top10": top}, allow_nan=False) + "\n")
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("select", help="Apply the frozen selectors to an external Top-1000 pool")
    a.add_argument("--pool", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("rerank", help="Order externally supplied scores and emit Top-10")
    b.add_argument("--records", type=Path, required=True)
    b.add_argument("--scores", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--kind", choices=["qwen", "erank"], default="qwen")
    args = parser.parse_args()
    count = (select_file(args.pool, args.output) if args.command == "select" else
             rerank_file(args.records, args.scores, args.output, args.kind))
    print(f"Processed {count} query groups; no model inference performed.")


if __name__ == "__main__":
    main()

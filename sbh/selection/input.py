"""Historical CSV parser and normalization used by the frozen selector.

The release's frozen SBH method is anchor_ic_max, lambda 0.75;
the earlier title-score method is not included.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
import pandas as pd

def stream_query_groups(path: Path, chunksize: int) -> Iterator[Tuple[str, pd.DataFrame]]:
    usecols = [
        "query_id",
        "query",
        "doc_id",
        "doc_title",
        "doc_text",
        "bm25_rank",
        "bm25_score",
    ]
    dtypes = {
        "query_id": "string",
        "query": "string",
        "doc_id": "string",
        "doc_title": "string",
        "doc_text": "string",
        "bm25_rank": "int64",
        "bm25_score": "float64",
    }
    carry = None
    finalized = set()

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtypes,
        chunksize=chunksize,
        keep_default_na=False,
    ):
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)

        last_qid = str(chunk["query_id"].iloc[-1])
        mask = chunk["query_id"].astype(str) == last_qid
        complete = chunk.loc[~mask]
        carry = chunk.loc[mask].copy()

        for qid, group in complete.groupby("query_id", sort=False):
            qid = str(qid)
            if qid in finalized:
                raise RuntimeError(f"Query {qid} reappeared; pool must be query-contiguous.")
            finalized.add(qid)
            yield qid, group.sort_values("bm25_rank", kind="mergesort")

    if carry is not None and len(carry):
        qid = str(carry["query_id"].iloc[0])
        if qid in finalized:
            raise RuntimeError(f"Query {qid} reappeared; pool must be query-contiguous.")
        yield qid, carry.sort_values("bm25_rank", kind="mergesort")


def normalize_bm25(scores: Dict[str, float]) -> Dict[str, float]:
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {doc_id: 1.0 for doc_id in scores}
    den = hi - lo
    return {doc_id: (score - lo) / den for doc_id, score in scores.items()}




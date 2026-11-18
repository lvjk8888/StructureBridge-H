"""Frozen ordering of externally supplied reranker scores.

rank_one returns every selected document in order. Take [:10] per condition for
the frozen Top-10 output. It performs no model inference.
"""
from __future__ import annotations

def union_for(r):
    chosen=set()
    for ms in r['by_budget'].values():
        for cell in ms.values():chosen.update(cell['selected'])
    return [d for d in r['candidates'] if d['doc_id'] in chosen]


def rank_one(r,score_map,kind):
    docs=union_for(r);by={d['doc_id']:d for d in docs}
    lo=min(d['retrieval_score'] for d in docs);hi=max(d['retrieval_score'] for d in docs)
    norm={d['doc_id']:(d['retrieval_score']-lo)/(hi-lo) if hi-lo>1e-12 else 0.0 for d in docs}
    result={}
    for b,ms in r['by_budget'].items():
        result[b]={}
        for m,c in ms.items():
            if kind=='erank':key=lambda d:(-(score_map[d]+1e-6*norm[d]),by[d]['rank'],d)
            else:key=lambda d:(-score_map[d],-by[d]['retrieval_score'],by[d]['rank'],d)
            result[b][m]=sorted(c['selected'],key=key)
    return result


"""Frozen candidate selector with unadopted method branches omitted."""
from __future__ import annotations
from .validation import need, finite
from . import input as v1
from . import scoring as core

def choose(docs,edges,m,method,lam,rounds,custom=None):
    order=[x['doc_id'] for x in docs];ranks={x['doc_id']:x['rank'] for x in docs}
    rel=v1.normalize_bm25({x['doc_id']:x['retrieval_score'] for x in docs})
    seeds=order[:m];need(len(seeds)==m,'Not enough documents for exact-M')
    trace=[];visited=set(seeds);seen={tuple(sorted(seeds)):0}
    score={d:0.0 for d in seeds};scope=list(seeds)
    for t in range(1,(0 if method=='raw' else rounds)+1):
        touched=core.touched_edges(edges,set(seeds));allowed=set(seeds)
        for e in touched:allowed.update(e['members'])
        scope=[d for d in order if d in allowed];visited.update(scope)
        if custom is not None:
            score=custom(list(scope),[dict(e) for e in touched],set(seeds),dict(rel),len(order))
            need(isinstance(score,dict) and set(score)==set(scope),'Selector plugin must return one structural score per scope document')
            score={d:finite(v) for d,v in score.items()}
        else:score=core.structural(method,scope,touched,set(seeds),rel,len(order))
        nxt=core.select(scope,rel,score,ranks,m,lam);key=tuple(sorted(nxt));repeat=seen.get(key)
        trace.append({'round':t,'scope_size':len(scope),'cumulative_visited':len(visited),'touched_edges':len(touched),
                      'changed_from_previous':len(set(nxt)-set(seeds)),'repeat_of_round':repeat})
        if key not in seen:seen[key]=t
        seeds=nxt
    return {'selected':seeds,'scope':scope,'normalized_relevance':{d:rel[d] for d in seeds},
            'structural_score':{d:score.get(d,0.0) for d in seeds},
            'final_score':{d:rel[d]+lam*score.get(d,0.0) for d in seeds},'trace':trace}


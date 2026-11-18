"""Frozen research functions; source identity is recorded in the release provenance."""
import itertools
from .answers_hotpot import parse_answer,answer_scores
from ._support import need
METHODS=("raw","edge_mean_touched","anchor_ic_max")
BUDGETS=(20,50,100)

def score_output(text,finish_reason,gold):
    p=parse_answer(text);answer=p['answer'] if p['answer_parse_ok'] and finish_reason=='stop' else ''
    em,f1=answer_scores(answer,gold)
    return {'prediction':answer,'em':em,'f1':f1,'answer_parse_ok':p['answer_parse_ok'],'citation_parse_ok':p['citation_parse_ok'],'format_ok':p['format_ok'],'finish_reason':finish_reason,'usable_answer':bool(p['answer_parse_ok'] and finish_reason=='stop')}

def paired_stats(per_condition):
    import numpy as np
    grouped={(m,b):{} for m in METHODS for b in BUDGETS}
    for r in per_condition:grouped[r['method'],r['budget']][r['query_id']]=r
    out=[]
    for b in BUDGETS:
        for m1,m2 in itertools.combinations(METHODS,2):
            qids=sorted(grouped[m1,b]);need(set(qids)==set(grouped[m2,b]),'Unpaired qids')
            for metric in ['em','f1']:
                d=np.asarray([grouped[m2,b][q][metric]-grouped[m1,b][q][metric] for q in qids],dtype=np.float64)
                seed=20261108;rg=np.random.default_rng(seed);means=[]
                for _ in range(100):
                    idx=rg.integers(0,len(d),size=(100,len(d)));means.extend(d[idx].mean(axis=1).tolist())
                lo,hi=np.quantile(means,[.025,.975],method='linear')
                out.append({'budget':b,'contrast':m2+' minus '+m1,'metric':metric,'n':len(qids),'mean_difference':float(d.mean()),'ci95_lower':float(lo),'ci95_upper':float(hi),'positive_differences':int((d>0).sum()),'negative_differences':int((d<0).sum()),'ties':int((d==0).sum()),'units':'fraction (0..1); multiply by 100 for percentage points','bootstrap':'paired by qid; percentile; 10000 resamples; numpy PCG64','seed':seed,'analysis_version':'new_local_bf16_paired_v1; does not replace old/cloud statistics'})
    return out

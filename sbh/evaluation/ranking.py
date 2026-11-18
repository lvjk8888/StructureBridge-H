"""Frozen research functions; source identity is recorded in the release provenance."""
import math
import numpy as np

def metrics(ranked, gold, n=10):
    gold=set(gold)
    if not gold: raise ValueError('Missing/empty gold: do not silently delete query')
    top=list(ranked[:n])
    if len(top)!=len(set(top)): raise ValueError('Duplicate output ID')
    hits=[int(d in gold) for d in top]
    gain=sum(x/math.log2(i+2) for i,x in enumerate(hits))
    ideal=sum(1/math.log2(i+2) for i in range(min(n,len(gold))))
    return {'CER':float(gold<=set(top)), 'Recall':len(gold&set(top))/len(gold),
            'nDCG':gain/ideal if ideal else 0.,
            'MRR':next((1/(i+1) for i,x in enumerate(hits) if x),0.)}

def evidence_flow(pool, exposed, selected, final, gold):
    """Product identity is applicable only when the specified stages are nested."""
    sets=[set(x) for x in (pool,exposed,selected,final)]; gold=set(gold)
    nested=all(sets[i+1]<=sets[i] for i in range(3))
    return {'nested':nested,'complete':[int(gold<=s) for s in sets],
            'retention_product_applicable':nested}

def paired_summary(a, b, seed=20261012, draws=10000):
    """a/b = [query, budget]; whole-query resampling, no pseudo-replication."""
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite paired query×budget arrays required')
    d=(a-b).mean(axis=1); rng=np.random.default_rng(seed); vals=[]
    for start in range(0,draws,100):
        take=min(100,draws-start)
        vals.extend(d[rng.integers(0,len(d),size=(take,len(d)))].mean(axis=1).tolist())
    return {'n_queries':len(d),'budgets_per_query':a.shape[1],
            'delta_pp':100*float(d.mean()),
            'ci95_pp':[100*float(x) for x in np.quantile(vals,[.025,.975])],
            'query_mean_wins':int((d>1e-12).sum()),'query_mean_losses':int((d< -1e-12).sum()),
            'query_mean_ties':int((np.abs(d)<=1e-12).sum()),'bootstrap_draws':draws,'seed':seed}

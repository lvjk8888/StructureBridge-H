"""Frozen research functions; source identity is recorded in the release provenance."""
import numpy as np
from ._support import need

def paired(a,b,seed,draws):
    x=np.asarray(a,float)-np.asarray(b,float);need(x.ndim==2 and x.shape[1]==3 and np.isfinite(x).all(),'Bad query-budget paired data');d=x.mean(axis=1);rng=np.random.default_rng(seed);v=[]
    for start in range(0,draws,100):
        ix=rng.integers(0,len(d),size=(min(100,draws-start),len(d)));v.extend(d[ix].mean(axis=1).tolist())
    return {'n_queries':len(d),'budgets_per_query':3,'delta_pp':float(d.mean()*100),'ci95_pp':[float(t*100) for t in np.quantile(v,[.025,.975])],'wins':int((d>1e-12).sum()),'losses':int((d< -1e-12).sum()),'ties':int((abs(d)<=1e-12).sum()),'draws':draws,'seed':seed,'multiplicity':'unadjusted; compare primary and supplemental identities'}

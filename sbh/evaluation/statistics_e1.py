"""Frozen research functions; source identity is recorded in the release provenance."""
import numpy as np
from ._support import need
SEED=20261020

def paired_numeric(a,b,seed=SEED,draws=10000,ratio=False):
    a=np.asarray(a,float);b=np.asarray(b,float)
    need(a.shape==b.shape and a.ndim==2 and np.isfinite(a).all() and np.isfinite(b).all(),'Paired query-budget arrays invalid')
    x=a.mean(axis=1);y=b.mean(axis=1);rng=np.random.default_rng(seed);vals=[]
    for start in range(0,draws,100):
        ix=rng.integers(0,len(x),size=(min(100,draws-start),len(x)))
        z=x[ix].mean(axis=1)/y[ix].mean(axis=1) if ratio else (x-y)[ix].mean(axis=1)
        vals.extend(z.tolist())
    v=x.mean()/y.mean() if ratio else (x-y).mean()
    return {'n_queries':len(x),'budgets_per_query':a.shape[1],'estimate':float(v),'ci95':list(map(float,np.quantile(vals,[.025,.975]))),
            'bootstrap_draws':draws,'seed':seed,'type':'ratio_of_means' if ratio else 'mean_difference','multiplicity':'unadjusted descriptive supplemental intervals'}

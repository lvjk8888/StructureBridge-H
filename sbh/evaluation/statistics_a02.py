"""Frozen research functions; source identity is recorded in the release provenance."""
import numpy as np

def boot(d: np.ndarray, seed: int, draws: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    vals = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 100):
        stop = min(start+100, draws)
        ix = rng.integers(0, len(d), size=(stop-start, len(d)))
        vals[start:stop] = d[ix].mean(axis=1)
    ci = np.quantile(vals, [.025, .975], method='linear')
    return dict(n_queries=len(d), delta_pp=float(d.mean()*100),
                ci95_pp=(ci*100).tolist(), seed=seed, draws=draws,
                wins=int((d>1e-12).sum()), losses=int((d< -1e-12).sum()),
                ties=int((np.abs(d)<=1e-12).sum()))

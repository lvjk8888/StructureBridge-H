#!/usr/bin/env python3
"""CPU-only reanalysis of saved F1 values. Never reruns models or edits a source.
This creates a NEW post-hoc per-budget analysis; it cannot establish the
provenance of the three historical rounded per-budget confidence intervals.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, platform, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json(p: Path, obj: object) -> None:
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent-zip', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--member-prefix', required=True, help='Exact relative evaluation directory inside the supplied archive')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing to overwrite an existing output directory.')
    zbytes = a.parent_zip.read_bytes()
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failure')
        names = z.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive paths')
        matches = [n for n in names if n == a.member_prefix.rstrip('/') + '/SUCCESS.json']
        if len(matches) != 1:
            raise ValueError('Expected exactly one formal parent evaluation')
        prefix = matches[0][:-len('SUCCESS.json')]
        success = json.loads(z.read(matches[0]))
        if success.get('passed') is not True:
            raise ValueError('Evaluation is not marked passed')
        seal=[]
        raw={}
        for rel, expected in success['files_sha256'].items():
            b=z.read(prefix+rel)
            actual=sha(b)
            if actual != expected:
                raise ValueError('Hash mismatch: '+rel)
            seal.append(dict(member=prefix+rel,sha256=actual,bytes=len(b),passed=True))
            raw[rel]=b
        rows=list(csv.DictReader(io.StringIO(raw['per_condition.csv'].decode('utf-8-sig'))))
        keys=[(r['query_id'],r['method'],int(r['budget'])) for r in rows]
        if len(keys)!=45000 or len(set(keys))!=45000:
            raise ValueError('Wrong row count or duplicated condition')
        qs=sorted({k[0] for k in keys}); budgets=[20,50,100]
        methods=sorted({k[1] for k in keys})
        expected_methods={'raw','edge_mean_touched','anchor_ic_max','gar_cp','ore_hybrid_cp_bge_terrier_qreset_c100'}
        if len(qs)!=3000 or set(methods)!=expected_methods:
            raise ValueError('Wrong question or method set')
        if set(keys)!={(q,m,b) for q in qs for m in methods for b in budgets}:
            raise ValueError('Incomplete condition grid')
        lookup={k:r for k,r in zip(keys,rows)}
        arr={m:np.array([[float(lookup[q,m,b]['Answer_F1']) for b in budgets] for q in qs]) for m in methods}
        if not all(np.isfinite(v).all() and (v>=0).all() and (v<=1).all() for v in arr.values()):
            raise ValueError('Invalid saved F1 values')
        delta=arr['anchor_ic_max']-arr['edge_mean_touched']
        registered=json.loads(raw['paired_comparisons.json'])['V2_minus_edge_mean_touched']['Answer_F1']
        if registered['seed']!=20261024 or registered['draws']!=10000:
            raise ValueError('Unexpected primary protocol')
        formal=boot(delta.mean(axis=1),registered['seed'],registered['draws'])
        for field in ['delta_pp','ci95_pp','wins','losses','ties']:
            if not np.allclose(formal[field], registered[field], rtol=0, atol=1e-12):
                raise ValueError('Formal primary mismatch: '+field)
        # Parameters selected to reproduce the independent checking protocol in
        # the user's Nov-5 checklist, NOT to fit the historical rounded intervals.
        seed=20261024; draws=10000
        new={str(b):boot(delta[:,j],seed,draws) for j,b in enumerate(budgets)}
        protocol=dict(analysis_id='A02_budget_F1_posthoc_reanalysis_20261105_v1',
          created_utc=datetime.now(timezone.utc).isoformat(),
          status='NEW_POSTHOC_ANALYSIS_NOT_RECOVERED_ORIGINAL',
          purpose='Reproduce formal primary; independently define per-budget CIs',
          source_archive=a.parent_zip.name, source_archive_sha256=sha(zbytes),
          source_member=prefix+'per_condition.csv', source_member_sha256=sha(raw['per_condition.csv']),
          source_metric='saved Answer_F1; NOT rescoring answers against gold',
          question_order='lexicographic sorted unique query_id',
          question_count=3000, budgets=budgets,comparison='anchor_ic_max minus edge_mean_touched',
          rng='numpy.random.default_rng (PCG64)',seed=seed,draws=draws,
          rng_lifetime='fresh RNG reset to same seed separately for each budget',
          sampling='paired query resampling with replacement; no excluded questions',
          statistic='mean per-question F1 difference at one fixed budget',
          interval='percentile [0.025,0.975], numpy.quantile method=linear',
          output_units='percentage points (F1 proportions multiplied by 100)',
          multiplicity='unadjusted exploratory intervals; all three budgets reported',
          original_primary_unchanged=True, legacy_CI_provenance='UNRESOLVED',
          source_written=False, API_calls=0, GPU_calls=0,
          software=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),
          script_sha256=sha(Path(__file__).read_bytes()))
        a.out.mkdir(parents=True,exist_ok=False)
        write_json(a.out/'PROTOCOL.json',protocol)
        write_json(a.out/'FORMAL_PRIMARY_REPRODUCTION.json',dict(passed=True,recomputed=formal,registered=registered))
        write_json(a.out/'NEW_PER_BUDGET_CI.json',new)
        write_json(a.out/'SOURCE_HASH_CHECKS.json',seal)
        with (a.out/'PER_QUERY_F1_DIFFERENCES.csv').open('x',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['query_id','delta_M20','delta_M50','delta_M100','mean_over_budgets'])
            for q,v in zip(qs,delta):w.writerow([q,*v.tolist(),float(v.mean())])
        for rel in ['paired_comparisons.json','PLAN.json','SUCCESS.json']:
            data=z.read(prefix+rel)
            (a.out/('PARENT_'+rel)).write_bytes(data)
        print(json.dumps(dict(formal_primary_passed=True,new_per_budget=new,out=str(a.out)),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

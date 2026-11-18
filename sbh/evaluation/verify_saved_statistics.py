#!/usr/bin/env python3
"""Verify registered saved-metric statistics; CPU only, no model imports.
Input folders C, SO, E1, REPEAT are read-only copies of existing exports.
Does not certify original answer scoring or recover missing historical scripts.
"""
import argparse,csv,hashlib,json,sys,platform
from pathlib import Path
import numpy as np
BUDGETS=(20,50,100)
def loadcsv(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def boot_diff(d,seed,draws=10000,scale=100):
    rng=np.random.default_rng(seed);vals=np.empty(draws,float)
    for st in range(0,draws,100):
        en=min(draws,st+100)
        ix=rng.integers(0,len(d),size=(en-st,len(d)))
        vals[st:en]=d[ix].mean(axis=1)
    return dict(estimate=float(d.mean()*scale),
                ci95=(np.quantile(vals,[.025,.975],method="linear")*scale).tolist(),
                wins=int((d>1e-12).sum()),losses=int((d< -1e-12).sum()),
                ties=int((abs(d)<=1e-12).sum()))
def paired_numeric(a,b,seed=20261020,draws=10000,ratio=False):
    x=a.mean(axis=1);y=b.mean(axis=1);rng=np.random.default_rng(seed);vals=[]
    for st in range(0,draws,100):
        ix=rng.integers(0,len(x),size=(min(100,draws-st),len(x)))
        z=x[ix].mean(axis=1)/y[ix].mean(axis=1) if ratio else (x-y)[ix].mean(axis=1)
        vals.extend(z.tolist())
    return dict(estimate=float(x.mean()/y.mean() if ratio else (x-y).mean()),
                ci95=np.quantile(vals,[.025,.975],method="linear").tolist())
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise SystemExit("Output exists; refusing overwrite.")
    records=[];hashes=[];tables={};new_budget={}
    for panel in ("C","SO"):
        d=args.sources/panel;csvp=d/"per_condition.csv"
        rows=loadcsv(csvp);lk={(r["query_id"],r["method"],int(r["budget"])):r for r in rows}
        qs=sorted({r["query_id"] for r in rows});ms=sorted({r["method"] for r in rows})
        assert len(qs)==3000 and len(lk)==len(rows)==3000*len(ms)*3
        assert set(lk)=={(q,m,b) for q in qs for m in ms for b in BUDGETS}
        tables[panel]=lk
        jp=d/("paired_comparisons.json" if panel=="C" else "paired_comparisons_SO.json")
        reg=json.loads(jp.read_text())
        arr=lambda m,k:np.array([[float(lk[q,m,b][k]) for b in BUDGETS] for q in qs],float)
        for comp,metrics in reg.items():
            m1,m2=comp.split("_minus_",1);m1="anchor_ic_max" if m1=="V2" else m1
            for metric,r in metrics.items():
                delta=(arr(m1,metric)-arr(m2,metric)).mean(axis=1)
                got=boot_diff(delta,r["seed"],r["draws"],100)
                expected=dict(estimate=r["delta_pp"],ci95=r["ci95_pp"],wins=r["wins"],losses=r["losses"],ties=r["ties"])
                diffs={k:float(np.max(abs(np.asarray(got[k])-np.asarray(expected[k])))) for k in expected}
                ok=all(v<=1e-11 for v in diffs.values())
                records.append(dict(panel=panel,comparison=comp,metric=metric,passed=ok,recomputed=got,registered=expected,max_abs_error=max(diffs.values())))
        if panel=="C":
            a=arr("anchor_ic_max","Answer_F1")-arr("edge_mean_touched","Answer_F1")
            new_budget={str(b):boot_diff(a[:,j],20261024,10000,100) for j,b in enumerate(BUDGETS)}
        hashes.extend([dict(file=str(p),sha256=sha(p)) for p in [csvp,jp,d/"PLAN.json",d/"SUCCESS.json"]])
    parent=tables["C"];so=tables["SO"]
    same_parent=all(all(so[k].get(f)==v for f,v in r.items()) for k,r in parent.items())
    som="ore_scored_only_posthoc_same_live_trajectory";orem="ore_hybrid_cp_bge_terrier_qreset_c100"
    so100=all(all(so[q,som,100][f]==so[q,orem,100][f] for f in ("request_sha256","answer","Answer_EM","Answer_F1")) for q in qs)
    for panel,filename,pairfilename,n in [
        ("E1","per_query_means.csv","paired_comparisons.json",3000),
        ("REPEAT","same200_three_repeats_per_query.csv","same200_paired_comparisons.json",200)
    ]:
        d=args.sources/panel;rs=loadcsv(d/filename)
        lk={(r["method"],r["query_id"],int(r["budget"])):r for r in rs}
        qs=sorted({r["query_id"] for r in rs});assert len(qs)==n and len(lk)==n*6*3
        arr=lambda m,k:np.array([[float(lk[m,q,b][k]) for b in BUDGETS] for q in qs],float)
        reg=json.loads((d/pairfilename).read_text())
        for comp,metrics in reg.items():
            other=comp.removeprefix("V2_minus_")
            for metric,r in metrics.items():
                field="CER_at_10" if metric=="CER_pp" else "post_pool_loaded_latency_ms"
                scale=100 if metric=="CER_pp" else 1
                got=paired_numeric(scale*arr("anchor_ic_max",field),scale*arr(other,field),
                                   r["seed"],r["bootstrap_draws"],metric=="latency_ratio_V2_over_other")
                err=max(abs(got["estimate"]-r["estimate"]),max(abs(np.array(got["ci95"])-r["ci95"])))
                records.append(dict(panel=panel,comparison=comp,metric=metric,passed=bool(err<=1e-10),
                                    recomputed=got,registered=dict(estimate=r["estimate"],ci95=r["ci95"]),max_abs_error=float(err)))
        hashes.extend([dict(file=str(p),sha256=sha(p)) for p in [d/filename,d/pairfilename]])
    result=dict(analysis_id="S02_S03_registered_statistics_recheck_20261106_v1",
                checks=len(records),passed=all(r["passed"] for r in records),
                parent_conditions_identical=same_parent,so_M100_identical=so100,
                records=records,sources=hashes,
                scope="Recompute stored metric statistics, not rescore original answers or original corpus.",
                original_CI_provenance_recovered=False,source_written=False,API_calls=0,GPU_calls=0,
                software=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),
                script_sha256=sha(Path(__file__)))
    args.out.mkdir(parents=True)
    (args.out/"REGISTERED_STATS_CHECK.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (args.out/"A02_20261105_REPRODUCTION.json").write_text(json.dumps(
        dict(identity="Reproduction of A02_budget_F1_posthoc_reanalysis_20261105_v1; NOT recovered original H11:H13",
             seed=20261024,draws=10000,rng="fresh PCG64 per budget",quantile="linear",
             query_order="lexicographic",multiplicity="unadjusted",unit="percentage points",
             results=new_budget),ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["checks","passed","parent_conditions_identical","so_M100_identical"]}))
    if not result["passed"] or not same_parent or not so100:raise SystemExit(2)
if __name__=="__main__":main()

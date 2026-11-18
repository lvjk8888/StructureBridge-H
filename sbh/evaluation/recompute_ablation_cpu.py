#!/usr/bin/env python3
"""NEW CPU-only post-hoc statistics from saved binary A/B controls.
NOT a recovery of historical 20261005 CIs/p-values; never runs models.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,platform,sys,zipfile
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
METHODS=("raw","edge_mean_touched","anchor_ic_max","no_ic_same_lambda",
         "allow_self_same_lambda","bounded_or_same_lambda","diffusion_same_lambda")
BUDGETS=(20,50,100)
BASE="anchor_ic_max"
def sha(data):return hashlib.sha256(data).hexdigest()
def load_source(path):
    path=path.resolve()
    if path.is_dir():
        p=path/"controls_per_query.csv"
        if not p.is_file():raise ValueError("Need controls_per_query.csv directly under source directory")
        return p.read_bytes(),{"path":str(p),"kind":"file"}
    if path.suffix.lower()==".zip":
        with zipfile.ZipFile(path) as z:
            if len(set(z.namelist()))!=len(z.namelist()):raise ValueError("Duplicate ZIP member names")
            matches=[n for n in z.namelist() if n.endswith("/controls_per_query.csv") or n=="controls_per_query.csv"]
            if len(matches)!=1:raise ValueError("Expected exactly one controls_per_query.csv")
            inf=z.getinfo(matches[0])
            if inf.file_size>128*1024*1024:raise ValueError("Unexpected input size; inspect manually")
            data=z.read(matches[0])
        return data,{"path":str(path),"member":matches[0],"archive_sha256":sha(path.read_bytes()),"kind":"zip"}
    return path.read_bytes(),{"path":str(path),"kind":"file"}
def seed_for(seed,stage,method,operation):
    return int.from_bytes(hashlib.sha256(f"{seed}|{stage}|{method}|{operation}".encode()).digest()[:8],"big")
def boot(k,seed,draws):
    # k_q is the integer difference in number of complete budgets, in [-3,3].
    # Resample whole questions. All three budgets remain together.
    rng=np.random.default_rng(seed);vals=np.empty(draws,float);n=len(k)
    for s in range(0,draws,100):
        e=min(draws,s+100);ix=rng.integers(0,n,size=(e-s,n))
        vals[s:e]=k[ix].sum(axis=1)/(3*n)*100
    return np.quantile(vals,[.025,.975],method="linear").tolist()
def signflip(k,seed,draws):
    # Under a paired label-swap null each query's full 3-budget difference flips sign.
    # Group equal |k_q|: sum of +/-j is j*(2*Binomial(n_j,.5)-n_j).
    # This is the same null distribution as independent signs per nonzero query,
    # not a normal approximation. Grouping reduces computation substantially.
    rng=np.random.default_rng(seed);null=np.zeros(draws,np.int64)
    groups={}
    for j in (1,2,3):
        n=int((abs(k)==j).sum());groups[str(j)]=n
        if n:null+=j*(2*rng.binomial(n,.5,size=draws)-n)
    observed=abs(int(k.sum()));extreme=int((abs(null)>=observed).sum())
    return {"p":(extreme+1)/(draws+1),"extreme":extreme,"draws":draws,
            "minimum_reportable_p":1/(draws+1),"magnitude_counts":groups}
def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);adjusted=[None]*len(values)
    running=values[order[0]]*0
    for rank,i in enumerate(order):
        running=max(running,min(1,values[i]*(len(values)-rank)));adjusted[i]=running
    return adjusted
def exact_mcnemar(wins,losses):
    n=wins+losses
    if n==0:return Fraction(1,1)
    return min(Fraction(1,1),Fraction(2*sum(math.comb(n,i) for i in range(min(wins,losses)+1)),2**n))
def csv_write(path,rows):
    with path.open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--expected-queries",type=int,default=3000)
    ap.add_argument("--seed",type=int,default=20261106)
    ap.add_argument("--bootstrap-draws",type=int,default=20000)
    ap.add_argument("--permutation-draws",type=int,default=100000)
    a=ap.parse_args()
    if a.out.exists():raise SystemExit("Refusing existing output directory")
    if min(a.expected_queries,a.bootstrap_draws,a.permutation_draws)<1:raise SystemExit("Invalid count")
    data,source=load_source(a.source)
    reader=csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    required={"query_id","method","budget","A_CER","B_CER","B_score_coverage_complete"}
    if not required<=set(reader.fieldnames or []):raise ValueError("Unexpected schema; do not guess fields: "+str(reader.fieldnames))
    rows=list(reader);lookup={}
    for r in rows:
        key=(r["query_id"],r["method"],int(r["budget"]))
        if key in lookup:raise ValueError("Duplicate query/method/budget")
        if float(r["B_score_coverage_complete"])!=1:raise ValueError("Unscored condition present")
        if any(float(r[k]) not in (0,1) for k in ("A_CER","B_CER")):raise ValueError("CER is not per-condition binary")
        if float(r["B_CER"])>float(r["A_CER"]):raise ValueError("B complete without A complete in controls")
        lookup[key]=r
    qs=sorted({k[0] for k in lookup})
    if len(qs)!=a.expected_queries:raise ValueError("Unexpected query count")
    if set(lookup)!={(q,m,b) for q in qs for m in METHODS for b in BUDGETS}:raise ValueError("Incomplete grid or wrong method identities")
    protocol={"analysis_id":"A05_saved_metrics_posthoc_reanalysis_20261106_v1",
      "identity":"NEW_ANALYSIS_NOT_HISTORICAL_STATISTICS_RECOVERY",
      "created_utc":datetime.now(timezone.utc).isoformat(),"source":source,
      "controls_per_query_sha256":sha(data),"query_count":len(qs),"condition_count":len(rows),
      "query_order":"lexicographic","budgets":list(BUDGETS),"methods":list(METHODS),
      "seed":a.seed,"rng":"numpy.default_rng/PCG64",
      "seed_derivation":"first 8 bytes SHA256(base_seed|stage|method|operation), big endian",
      "bootstrap_draws":a.bootstrap_draws,"bootstrap_unit":"whole query with all three budgets",
      "interval":"marginal percentile [0.025,0.975]; numpy.quantile linear",
      "mean_statistic":"mean of per-query three-budget mean binary CER difference",
      "mean_test":"two-sided Monte Carlo paired full-query label-swap; grouped absolute integer magnitudes; add-one correction",
      "permutation_draws":a.permutation_draws,"mean_Holm_family":"12 tests = 2 stages A/B x six comparators vs V2",
      "single_budget_test":"two-sided exact McNemar, binomial p=.5; discordant query pairs",
      "single_budget_Holm_family":"36 tests = 2 stages x 3 budgets x six comparators",
      "CI_multiplicity":"marginal CIs NOT simultaneous, NOT Holm-adjusted CIs",
      "source_metric":"stored A_CER/B_CER; NOT gold/ranking rescoring",
      "no_seed_search":True,"all_contrasts_reported":True,
      "API_calls":0,"GPU_calls":0,"source_written":False,
      "script_sha256":sha(Path(__file__).read_bytes()),"numpy":np.__version__,
      "python":sys.version,"platform":platform.platform()}
    a.out.mkdir(parents=True)
    (a.out/"PROTOCOL.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    arr={s:{m:np.array([[int(float(lookup[q,m,b][s+"_CER"])) for b in BUDGETS] for q in qs],dtype=np.int64)
            for m in METHODS} for s in ("A","B")}
    summary=[];meanrows=[];budgetrows=[];rawp=[];details={}
    for m in METHODS:
        for j,b in enumerate(BUDGETS):
            summary.append(dict(method=m,budget=b,num_queries=len(qs),
                     A_complete=int(arr["A"][m][:,j].sum()),B_complete=int(arr["B"][m][:,j].sum()),
                     A_CER=float(arr["A"][m][:,j].mean()),B_CER=float(arr["B"][m][:,j].mean())))
    for stage in ("A","B"):
        for m in METHODS:
            if m==BASE:continue
            d=arr[stage][BASE]-arr[stage][m];k=d.sum(axis=1)
            bootseed=seed_for(a.seed,stage,m,"bootstrap");pseed=seed_for(a.seed,stage,m,"signflip")
            ci=boot(k,bootseed,a.bootstrap_draws);p=signflip(k,pseed,a.permutation_draws)
            meanrows.append(dict(stage=stage,comparison=BASE+" minus "+m,n_queries=len(qs),
                    delta_pp=float(k.sum()/(3*len(qs))*100),ci_low_pp=ci[0],ci_high_pp=ci[1],
                    paired_swap_p=p["p"],Holm12_p=None,bootstrap_seed=bootseed,permutation_seed=pseed,
                    permutation_resolution=p["minimum_reportable_p"],identity=protocol["analysis_id"]))
            details[stage+"_"+m]=p
            for j,b in enumerate(BUDGETS):
                w=int((d[:,j]>0).sum());l=int((d[:,j]<0).sum());p_exact=exact_mcnemar(w,l);rawp.append(p_exact)
                budgetrows.append(dict(stage=stage,budget=b,comparison=BASE+" minus "+m,n_queries=len(qs),
                    delta_pp=float(d[:,j].mean()*100),wins=w,losses=l,ties=len(qs)-w-l,
                    mcnemar_p=float(p_exact),Holm36_p=None,
                    p_numerator=str(p_exact.numerator),p_denominator=str(p_exact.denominator),
                    p_underflows_float=(p_exact>0 and float(p_exact)==0)))
    for r,p in zip(meanrows,holm([r["paired_swap_p"] for r in meanrows])):r["Holm12_p"]=float(p)
    for r,p in zip(budgetrows,holm(rawp)):r["Holm36_p"]=float(p)
    csv_write(a.out/"SUMMARY_EXACT.csv",summary)
    csv_write(a.out/"PAIRED_MEAN_NEW.csv",meanrows)
    csv_write(a.out/"PAIRED_BUDGET_NEW.csv",budgetrows)
    (a.out/"SIGNFLIP_DETAILS.json").write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding="utf-8")
    np.savez_compressed(a.out/"SOURCE_METRIC_ARRAYS.npz",query_ids=np.array(qs),
                       **{stage+"__"+m:arr[stage][m] for stage in ("A","B") for m in METHODS})
    files={p.name:sha(p.read_bytes()) for p in a.out.iterdir() if p.is_file()}
    (a.out/"SUCCESS.json").write_text(json.dumps({"new_analysis_complete":True,
       "original_statistics_provenance_recovered":False,"files_sha256":files,
       "API_calls":0,"GPU_calls":0},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"NEW_POSTHOC_ANALYSIS_COMPLETE","mean_contrasts":len(meanrows),
                      "budget_contrasts":len(budgetrows),"out":str(a.out)},ensure_ascii=False))
if __name__=="__main__":main()

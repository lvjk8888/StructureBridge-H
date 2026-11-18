"""Frozen research functions; source identity is recorded in the release provenance."""
from ._support import need

def evidence_metrics(r,gold,selected,scope,m):
    pool={d['doc_id'] for d in r['candidates']};base={d['doc_id'] for d in r['candidates'][:m]};sel=set(selected);scope=set(scope)
    need(gold and len(sel)==m,'Gold/M invalid')
    rc=len(sel&gold)/len(gold);complete=int(gold<=sel);raw=int(gold<=base);pc=int(gold<=pool);sc=int(gold<=scope)
    return {'gold_count':len(gold),'CER':complete,'Recall':rc,'AnyEvidence':int(bool(sel&gold)),
       'relevant_found':len(sel&gold),'raw_complete':raw,'pool_complete':pc,'scope_complete':sc,
       'pool_recoverable':int(not raw and pc),'scope_recoverable':int(not raw and sc),
       'recovered':int(not raw and complete),'harmed':int(raw and not complete),
       'substitutions':len(sel-base),'scope_size':len(scope)}

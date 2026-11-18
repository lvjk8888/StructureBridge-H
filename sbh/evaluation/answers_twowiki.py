"""Frozen research functions; source identity is recorded in the release provenance."""
import re,string
from ._support import jlines,need

def normalized_answer(s):
    s=s.lower();s=''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())

def answer_score(pred,gold):
    from collections import Counter
    p,g=normalized_answer(pred),normalized_answer(gold);em=float(p==g)
    if (p in ('yes','no','noanswer') or g in ('yes','no','noanswer')) and p!=g:return em,0.0
    a,b=p.split(),g.split();common=sum((Counter(a)&Counter(b)).values())
    if common==0:return em,0.0
    pr=common/len(a);rc=common/len(b);return em,2*pr*rc/(pr+rc)

def answers(ds,qids):
    out={};wanted=set(qids)
    for r in jlines(ds['answers']):
        q=str(r['query_id'])
        if q not in wanted:continue
        need(q not in out,'Duplicate answer query');need(isinstance(r.get('answer'),str) and r['answer'].strip(),'Expected a nonempty string gold answer')
        out[q]=r
    need(set(out)==wanted,'Missing answer; cannot silently remove a query')
    if ds['answer_metric']=='twowiki_alias':
        need('aliases' in ds,'2Wiki requires source id_aliases.json JSONL')
        ids={r.get('answer_id') for r in out.values() if isinstance(r.get('answer_id'),str)};am={}
        for r in jlines(ds['aliases']):
            key=r.get('Q_id')
            if key not in ids:continue
            need(key not in am,'Duplicate relevant alias Q_id')
            al=r.get('aliases',[])+r.get('demonyms',[]);need(all(isinstance(x,str) for x in al),'Alias values must be strings');am[key]=set(al)
        return {q:sorted({r['answer']}|am.get(r.get('answer_id'),set())) for q,r in out.items()}
    need(ds['answer_metric']=='hotpot','Unsupported answer metric; add an explicit dataset evaluator')
    return {q:[r['answer']] for q,r in out.items()}

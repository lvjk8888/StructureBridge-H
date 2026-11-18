"""Frozen answer parsing/scoring functions only. No I/O, model, API or bootstrap. Not executed during handoff."""
import re,string,collections

def parse_answer(text):
    """Fixed non-gold-aware extraction. Never shorten/correct a generated answer."""
    lines = [x.strip() for x in (text or '').splitlines() if x.strip()]
    a = [x[len('ANSWER:'):].strip() for x in lines if x.startswith('ANSWER:')]
    c = [x[len('CITATIONS:'):].strip() for x in lines if x.startswith('CITATIONS:')]
    answer_ok = len(a) == 1 and bool(a[0]) and '<think' not in (text or '').lower()
    cites = []
    cites_ok = False
    if len(c) == 1 and re.fullmatch(r'(?:10|[1-9])(?:\s*,\s*(?:10|[1-9]))*', c[0]):
        cites = [int(v.strip()) for v in c[0].split(',')]
        cites_ok = len(cites) == len(set(cites))
    format_ok = answer_ok and cites_ok and len(lines) == 2 and lines[0].startswith('ANSWER:') and lines[1].startswith('CITATIONS:')
    return {'answer_parse_ok': answer_ok, 'answer': a[0] if answer_ok else '',
            'citation_parse_ok': cites_ok, 'citations': cites if cites_ok else [], 'format_ok': format_ok}

def norm(x):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in x.lower() if c not in string.punctuation)).split())

def score(pred,golds,usable=True):
    # Project execution policy retains rejected/invalid conditions and explicitly scores them zero.
    if not usable:return 0.,0.
    pp=norm(pred);pt=pp.split();ems=[];f1s=[]
    for g in golds:
        gg=norm(g);gt=gg.split();ems.append(float(pp==gg))
        common=sum((collections.Counter(pt)&collections.Counter(gt)).values())
        if not pt or not gt:f1s.append(float(pt==gt))
        elif not common:f1s.append(0.)
        else:f1s.append(2*common/(len(pt)+len(gt)))
    return max(ems),max(f1s)

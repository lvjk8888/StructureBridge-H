"""Exact AST-selected frozen HotpotQA functions; no 2Wiki aliases or prompt changes."""

from collections import Counter

import re,string

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

def response_info(body, expected_model):
    """Even invalid responses are retained as terminal results; no answer-based retries."""
    if not isinstance(body, dict):
        body = {}
    choices = body.get('choices') or []
    ch = choices[0] if len(choices) == 1 and isinstance(choices[0], dict) else {}
    msg = ch.get('message') or {}
    text = msg.get('content') if isinstance(msg.get('content'), str) else ''
    parsed = parse_answer(text)
    normal = (len(choices) == 1 and ch.get('finish_reason') == 'stop'
              and not msg.get('tool_calls') and not msg.get('refusal') and not msg.get('reasoning_content'))
    model_ok = body.get('model') == expected_model
    usable = bool(normal and model_ok and parsed['answer_parse_ok'])
    usage = body.get('usage') or {}
    known = all(type(usage.get(k)) is int and usage[k] >= 0
                for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    details = usage.get('prompt_tokens_details') or {}
    return {**parsed, 'answer': parsed['answer'] if usable else '', 'usable_answer': usable,
            'normal_stop': bool(normal), 'model_match': model_ok,
            'finish_reason': ch.get('finish_reason'), 'provider_response_model': body.get('model'),
            'response_id': body.get('id'), 'system_fingerprint': body.get('system_fingerprint'),
            'usage_known': known, 'prompt_tokens': usage.get('prompt_tokens') if known else None,
            'completion_tokens': usage.get('completion_tokens') if known else None,
            'total_tokens': usage.get('total_tokens') if known else None,
            'cached_prompt_tokens': details.get('cached_tokens'), 'output_text': text}

def normalize_answer(text):
    text=text.lower()
    text=''.join(c for c in text if c not in set(string.punctuation))
    text=re.sub(r'\b(a|an|the)\b',' ',text)
    return ' '.join(text.split())

def answer_scores(prediction,gold):
    p,g=normalize_answer(prediction),normalize_answer(gold)
    em=float(p==g)
    if (p in ('yes','no','noanswer') or g in ('yes','no','noanswer')) and p!=g:
        return em,0.0
    pt,gt=p.split(),g.split()
    common=sum((Counter(pt)&Counter(gt)).values())
    if not common:
        return em,0.0
    precision,recall=common/len(pt),common/len(gt)
    return em,2*precision*recall/(precision+recall)

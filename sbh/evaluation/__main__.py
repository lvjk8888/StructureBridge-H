"""Explicit local JSONL adapter for the frozen answer/evidence functions.

This adapter is release packaging, not a historical experiment driver.
The evidence task requires scored-only ranking membership. For ORE mixed
output, use sbh.bef output-rules; do not force mixed rankings into this schema.
"""
import argparse,json
from pathlib import Path
from ._support import jlines,need

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=('answers','evidence'))
    parser.add_argument('--dataset',choices=('hotpot','twowiki','musique'))
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    need(not args.output.exists(),'Output exists; refuse overwrite')
    if args.task=='answers':need(args.dataset is not None,'Answer evaluation requires explicit dataset')
    result=[];seen=set()
    for row in jlines(args.input):
        key=(row['query_id'],row.get('method'),row.get('budget'))
        need(key not in seen,'Duplicate query/method/budget');seen.add(key)
        identity={k:row[k] for k in ('query_id','method','budget') if k in row}
        if args.task=='answers':
            need(isinstance(row.get('prediction'),str),'prediction must be a string')
            golds=row['golds'];need(isinstance(golds,list) and golds and all(isinstance(g,str) and g.strip() for g in golds),'golds must contain nonempty strings')
            need(type(row.get('usable_answer')) is bool,'usable_answer must be explicit')
            prediction=row['prediction'] if row['usable_answer'] else ''
            if args.dataset=='musique':
                from .answers_musique import score
                em,f1=score(row['prediction'],golds,row['usable_answer'])
            else:
                if args.dataset=='hotpot':
                    from .answers_hotpot import answer_scores
                    need(len(golds)==1,'Frozen Hotpot evaluator requires one gold answer')
                else:
                    from .answers_twowiki import answer_score as answer_scores
                scores=[answer_scores(prediction,g) for g in golds]
                em=max(x[0] for x in scores);f1=max(x[1] for x in scores)
            result.append({**identity,'Answer_EM':em,'Answer_F1':f1})
        else:
            from .evidence import evidence_metrics
            from .ranking import metrics
            gold=set(row['gold']);selected=row['selected'];ranking=row['ranking'];budget=row['budget']
            need(len(gold)==len(row['gold']) and bool(gold),'Gold IDs must be nonempty and unique')
            need(len(selected)==len(set(selected))==budget,'Selected IDs must be distinct and exactly M')
            need(len(ranking)==len(set(ranking)) and set(ranking)==set(selected),'Ranking must permute the scored selection')
            a=evidence_metrics(row,gold,selected,row['scope'],budget)
            b=metrics(ranking,gold,10)
            result.append({**identity,**a,'A_CER':a['CER'],'B_CER':b['CER'],**{k+'_at_10':v for k,v in b.items()}})
    need(result,'Empty evaluation input')
    with args.output.open('x',encoding='utf-8') as handle:
        for row in result:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

if __name__=='__main__':main()

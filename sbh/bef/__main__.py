"""Offline adapter for supplied frozen BEF projections; no policy execution."""
import argparse,json
from pathlib import Path
from sbh.evaluation._support import jlines,need
from .record_checks import ledger,check_saved
from .output_rules import compare_output_rules

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('ledger','output-rules','saved-trace'),required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();need(not args.output.exists(),'Output exists; refuse overwrite')
    results=[]
    for row in jlines(args.input):
        if args.mode=='saved-trace':
            result=check_saved(row['projection'],row['pool'])
        elif args.mode=='output-rules':
            result=compare_output_rules(row['pool'],row['gold'],row['scored_indices'],row['ranking_indices'],row['budget'])
        else:
            result=ledger(row['pool'],row['gold'],row['scored_indices'],row['ranking_indices'][:10],row['budget'])
        results.append({'query_id':row['query_id'],'budget':row['budget'],**result})
    need(results,'Empty BEF input')
    with args.output.open('x',encoding='utf-8') as handle:
        for result in results:handle.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')

if __name__=='__main__':main()

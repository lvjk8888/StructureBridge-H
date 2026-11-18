"""Original E2 strata and saved-answer aggregates, with explicit caller-supplied inputs.
C values are stored metrics; these functions do not rescore generated answers.
"""
import json,math,statistics
from .record_checks import canon,need
METHODS=('raw','edge_mean_touched','anchor_ic_max')
BUDGETS=(20,50,100)
LABEL={'raw':'Raw','edge_mean_touched':'EdgeMean','anchor_ic_max':'SBH'}
BUCKETS=['pool_incomplete','raw_already_complete','missing_gold_outside_touched_scope','reachable_but_SBH_not_complete','SBH_A_rescue_B_incomplete','SBH_A_rescue_B_complete']
def ratio(a,b):return a/b if b else ''

def strata_rows(frozen_lines, gold, cm, old):
    rows=[];seen=set();metric_checks=0
    for line in frozen_lines:
        f=json.loads(line);q=f['query_id'];need(q not in seen,'Duplicate frozen query');seen.add(q);g=gold[q];need(g,'Empty qrels: never drop')
        pool=set(f['pool_ids']);pool_complete=g<=pool
        for budget in BUDGETS:
            cells=f['cells'][str(budget)];seeds=set(f['pool_ids'][:budget]);raw=cells['raw'];sbh=cells['anchor_ic_max'];missing=g-set(raw['selected']);scope=set(sbh['scope'])
            need(seeds==set(raw['selected']),'Raw seed identity');reach=missing&scope;sbsel=set(sbh['selected']);sbA=g<=sbsel;sbB=g<=set(sbh['top10']);rawA=not missing
            if not pool_complete:bucket='pool_incomplete'
            elif rawA:bucket='raw_already_complete'
            elif not missing<=scope:bucket='missing_gold_outside_touched_scope'
            elif not sbA:bucket='reachable_but_SBH_not_complete'
            elif not sbB:bucket='SBH_A_rescue_B_incomplete'
            else:bucket='SBH_A_rescue_B_complete'
            r={'query_id':q,'budget':budget,'repeat':0,'panel':'2Wiki_Hybrid_E1_main','gold_count':len(g),'gold_doc_ids':canon(sorted(g)),'pool_size':len(pool),'pool_complete':int(pool_complete),'raw_missing_gold_count':len(missing),'raw_missing_gold_ids':canon(sorted(missing)),'missing_gold_in_touched_scope_count':len(reach),'missing_gold_in_touched_scope_ids':canon(sorted(reach)),'all_missing_gold_touched':int(missing<=scope) if missing else '', 'any_missing_gold_touched':int(bool(reach)) if missing else '', 'missing_gold_not_touched_ids':canon(sorted(missing-scope)),'SBH_selected_missing_gold_count':len(missing&sbsel),'SBH_selected_missing_gold_ids':canon(sorted(missing&sbsel)),'SBH_dropped_original_gold_ids':canon(sorted((g&seeds)-sbsel)),'SBH_scope_size':len(scope),'stratum':bucket,'frozen_block_origin_plan':f['sources']['block_plan_sha256'],'session_id':f['sources']['session_id'],'source_A_shard':f['sources']['A_shard'],'source_A_sha256':f['sources']['A_file_sha256'],'source_block':f['sources']['block'],'source_block_sha256':f['sources']['block_sha256']}
            for method in METHODS:
                label=LABEL[method];cell=cells[method];ac=g<=set(cell['selected']);bc=g<=set(cell['top10']);need(not bc or ac,'B complete without A complete')
                r[label+'_A_complete']=int(ac);r[label+'_B_complete']=int(bc)
                hist=old[(q,budget,method)];need(int(hist['scored_set_complete'])==int(ac) and float(hist['CER_at_10'])==int(bc) and int(hist['pool_complete'])==int(pool_complete),'Independent set metric disagrees with stored metric')
                metric_checks+=1
                c=cm.get((q,budget,method));r[label+'_C_available']=int(c is not None)
                if c:
                    need(c['request_sha256']==cell['C_rid'],'C rid mismatch')
                    for name in ['Answer_EM','Answer_F1']:
                        val=float(c[name]);need(math.isfinite(val) and 0<=val<=1,'C value range');r[label+'_'+name+'_stored']=val
                    r[label+'_C_request_id']=c['request_sha256'];r[label+'_C_usable_answer']=int(c['usable_answer']=='True')
                else:
                    r[label+'_Answer_EM_stored']='';r[label+'_Answer_F1_stored']='';r[label+'_C_request_id']='';r[label+'_C_usable_answer']=''
            for label in ['EdgeMean','SBH']:
                for stage in ['A','B']:
                    rawv=r['Raw_'+stage+'_complete'];v=r[label+'_'+stage+'_complete'];r[label+'_'+stage+'_rescue_vs_Raw']=int(not rawv and v);r[label+'_'+stage+'_loss_vs_Raw']=int(rawv and not v)
                r[label+'_stored_F1_delta_vs_Raw']=r[label+'_Answer_F1_stored']-r['Raw_Answer_F1_stored'] if r[label+'_C_available'] and r['Raw_C_available'] else ''
            rows.append(r)
    return rows

def strata_summary(rows):
    summary=[]
    for budget in [20,50,100,'ALL']:
        base=[r for r in rows if budget=='ALL' or r['budget']==budget]
        for bucket in ['ALL']+BUCKETS:
            subset=[r for r in base if bucket=='ALL' or r['stratum']==bucket];n=len(subset)
            for method in METHODS:
                label=LABEL[method];ac=sum(r[label+'_A_complete'] for r in subset);bc=sum(r[label+'_B_complete'] for r in subset)
                sr={'budget':budget,'stratum':bucket,'method':label,'n_query_budget':n,'n_unique_queries':len({r['query_id'] for r in subset}),'pool_complete_n':sum(r['pool_complete'] for r in subset),'A_complete_n':ac,'A_denominator':n,'A_CER':ratio(ac,n),'B_complete_n':bc,'B_denominator':n,'B_CER':ratio(bc,n)}
                for stage in ['A','B']:
                    rescued=sum(int(not r['Raw_'+stage+'_complete'] and r[label+'_'+stage+'_complete']) for r in subset);lost=sum(int(r['Raw_'+stage+'_complete'] and not r[label+'_'+stage+'_complete']) for r in subset)
                    sr[stage+'_rescue_vs_Raw_n']=rescued;sr[stage+'_loss_vs_Raw_n']=lost;sr[stage+'_net_vs_Raw_n']=rescued-lost
                available=[r for r in subset if r[label+'_C_available']];paired=[r for r in available if r['Raw_C_available']]
                sr['C_stored_metric_denominator']=len(available);sr['C_missing_n']=n-len(available);sr['C_usable_answer_n']=sum(r[label+'_C_usable_answer'] for r in available)
                for name in ['Answer_EM','Answer_F1']:sr[name+'_stored_mean']=statistics.mean(r[label+'_'+name+'_stored'] for r in available) if available else ''
                sr['C_paired_with_Raw_denominator']=len(paired);sr['F1_delta_vs_Raw_stored_mean']=statistics.mean(r[label+'_Answer_F1_stored']-r['Raw_Answer_F1_stored'] for r in paired) if paired else ''
                summary.append(sr)
    return summary

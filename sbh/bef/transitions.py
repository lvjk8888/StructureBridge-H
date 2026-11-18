"""Original paired E2 transition, rescue/loss and opportunity accounting."""
import itertools
from .record_checks import need
METHODS=('raw','edge_mean_touched','anchor_ic_max')
BUDGETS=(20,50,100)
LABELS={'raw':'Raw','edge_mean_touched':'EdgeMean','anchor_ic_max':'SBH'}

def events(rows, predicate):
    pairs = sorted((x['query_id'],x['budget']) for x in rows if predicate(x))
    return {'condition_count':len(pairs), 'independent_qid_count':len({q for q,b in pairs}), 'pairs':[{'query_id':q,'budget':b} for q,b in pairs]}

def transition_rows(records, original):
    rows=[];seen=set()
    for r in records:
        q=r['query_id'];need(q not in seen,'duplicate qid');seen.add(q)
        ids=r['ordered_pool_ids'];gold=set(r['complete_gold_ids'])
        need(gold and len(gold)==len(r['complete_gold_ids']),'complete gold identity')
        pool=set(ids);need(len(pool)==1000,'pool identity')
        for m in BUDGETS:
            cells=r['budgets'][str(m)];o=original[(q,m)]
            row={'query_id':q,'budget':m,'repeat':0,'pool_complete':int(gold<=pool),'gold_count':len(gold)}
            for method in METHODS:
                c=cells['methods'][method];S={ids[d] for d in c['S_indices']};T={ids[d] for d in c['T10_indices']}
                need(len(S)==m and len(T)==10 and T<=S,'scored-only identity '+q)
                a,b=int(gold<=S),int(gold<=T);ell,j=a*(1-b),(1-a)*b
                need(b<=a and b==a-ell+j and j==0,'BEF failure '+q)
                label=LABELS[method]
                for name,v in [('A',a),('B',b),('ell',ell),('j',j)]:row[label+'_'+name]=v
                need(a==int(o[label+'_A_complete']) and b==int(o[label+'_B_complete']),'prior E2 mismatch '+q)
                if method=='raw':need(c['S_indices']==cells['seed_indices']==list(range(m)),'Raw seed order')
            rawset={ids[d] for d in cells['methods']['raw']['S_indices']}
            missing=gold-rawset;scope={ids[d] for d in cells['methods']['anchor_ic_max']['scope_indices']}
            reachable=missing<=scope
            row['raw_missing_gold_count']=len(missing)
            row['missing_gold_touched_count']=len(missing&scope)
            row['all_missing_gold_touched']=int(reachable) if missing else ''
            if not row['pool_complete']:stage='pool_incomplete'
            elif row['Raw_A']:stage='raw_already_complete'
            elif not reachable:stage='missing_gold_outside_scope'
            elif not row['SBH_A']:stage='reachable_not_selected_complete'
            elif not row['SBH_B']:stage='A_rescued_B_incomplete'
            else:stage='A_rescued_B_complete'
            row['opportunity_stratum']=stage
            row['SBH_A_loss_with_Raw_A_complete']=int(row['Raw_A']==1 and row['SBH_A']==0)
            for ref in ('Raw','EdgeMean'):
                for stage in ('A','B'):
                    row['SBH_'+stage+'_rescue_vs_'+ref]=int(row[ref+'_'+stage]==0 and row['SBH_'+stage]==1)
                    row['SBH_'+stage+'_loss_vs_'+ref]=int(row[ref+'_'+stage]==1 and row['SBH_'+stage]==0)
            rows.append(row)
    return rows

def transition_summary(rows):
    transitions=[];account=[];opportunity=[];overlap={}
    for budget in (*BUDGETS,'ALL'):
        sub=[r for r in rows if budget=='ALL' or r['budget']==budget];den=len(sub);nq=len({r['query_id'] for r in sub})
        for ref in ('Raw','EdgeMean'):
            for ar,as_,br,bs in itertools.product((0,1),repeat=4):
                hit=[r for r in sub if (r[ref+'_A'],r['SBH_A'],r[ref+'_B'],r['SBH_B'])==(ar,as_,br,bs)]
                transitions.append({'budget':budget,'reference':ref,'A_reference':ar,'A_SBH':as_,'B_reference':br,'B_SBH':bs,'conditions':len(hit),'independent_qids':len({r['query_id'] for r in hit}),'denominator_conditions':den,'denominator_qids':nq,'fraction':len(hit)/den,'scored_only_possible':int(br<=ar and bs<=as_)})
        totals={label:{k:sum(r[label+'_'+k] for r in sub) for k in ('A','B','ell','j')} for label in LABELS.values()}
        for name,tot in list(totals.items())+[(f'SBH-minus-{ref}',{k:totals['SBH'][k]-totals[ref][k] for k in ('A','B','ell','j')}) for ref in ('Raw','EdgeMean')]:
            need(tot['B']==tot['A']-tot['ell']+tot['j'],'aggregate identity')
            account.append({'budget':budget,'method_or_contrast':name,'denominator_conditions':den,'independent_qids':nq,**{k+'_count':v for k,v in tot.items()},**{k+'_rate':v/den for k,v in tot.items()},'identity_residual':tot['B']-tot['A']+tot['ell']-tot['j']})
        kinds=('pool_incomplete','raw_already_complete','missing_gold_outside_scope','reachable_not_selected_complete','A_rescued_B_incomplete','A_rescued_B_complete')
        for kind in kinds:
            hit=[r for r in sub if r['opportunity_stratum']==kind]
            opportunity.append({'budget':budget,'stratum':kind,'conditions':len(hit),'independent_qids':len({r['query_id'] for r in hit}),'denominator_conditions':den,'denominator_qids':nq,'SBH_A_losses_vs_Raw':sum(r['SBH_A_loss_vs_Raw'] for r in hit),'SBH_B_losses_vs_Raw':sum(r['SBH_B_loss_vs_Raw'] for r in hit)})
        al=lambda r:r['SBH_A_loss_vs_Raw']==1
        bl=lambda r:r['SBH_B_loss_vs_Raw']==1
        overlap[str(budget)]={
            'denominator_conditions':den,'denominator_qids':nq,
            'A_loss':events(sub,al),'B_loss':events(sub,bl),
            'intersection':events(sub,lambda r:al(r) and bl(r)),
            'only_A_loss':events(sub,lambda r:al(r) and not bl(r)),
            'only_B_loss':events(sub,lambda r:bl(r) and not al(r)),
            'extra_B_rescue_both_A_complete':events(sub,lambda r:r['Raw_A']==r['SBH_A']==1 and r['Raw_B']==0 and r['SBH_B']==1),
            'pool_complete_Raw_incomplete':events(sub,lambda r:r['pool_complete']==1 and r['Raw_A']==0),
            'A_rescue_B_complete':events(sub,lambda r:r['SBH_A_rescue_vs_Raw']==1 and r['SBH_B']==1),
            'A_rescue_B_incomplete':events(sub,lambda r:r['SBH_A_rescue_vs_Raw']==1 and r['SBH_B']==0)
        }
    return {'transitions':transitions,'accounting':account,'opportunity':opportunity,'loss_overlap':overlap}

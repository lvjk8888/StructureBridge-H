"""Frozen research functions; source identity is recorded in the release provenance."""
import math
from .record_checks import need,digest

def compare_numbers(actual,old):
    if isinstance(old,list):
        need(isinstance(actual,list) and len(actual)==len(old),'numeric shape mismatch')
        pairs=[compare_numbers(a,b) for a,b in zip(actual,old)]
        return all(x[0] for x in pairs),all(x[1] for x in pairs),max([x[2] for x in pairs]+[0.])
    if isinstance(old,(int,float)):
        need(isinstance(actual,(int,float)) and math.isfinite(actual),'numeric value')
        return actual==old,math.isclose(actual,old,rel_tol=1e-12,abs_tol=1e-12),abs(actual-old)
    return actual==old,actual==old,0.

class Validator:
    def __init__(self,c,record):self.c=c;self.r=record;self.numeric_exact=True;self.numeric_tolerance=True;self.max_diff=0.;self.states=[]
    def state(self,s):
        need(s['state_origin']=='RECONSTRUCTED_THIS_RUN','state origin label')
        self.states.append({'stage':s['stage'],'sha256':s['state_sha256']})
        if s['stage']=='gar_after_feedback':
            n=s['scoring_event_count_so_far'];es=self.r['events'][:n]
            expected=[[d,v] for e in es for d,v in zip(e['doc_ids'],e['scores'])]
            need(s['ordered_scored_so_far']==expected,'GAR reconstructed observed state differs')
        if s['stage']=='ore_after_update':
            t=s['saved_trace_comparable'];old=self.r['trace'][t['round']]
            for k in ('round','mode','heads','doc_ids','feedback_neighbor_nonzero','rm3_nonzero'):need(t[k]==old[k],'ORE discrete trace differs '+k)
            for k in ('selected_estimates','selected_features','weights_before','weights_after'):
                exact,tol,diff=compare_numbers(t[k],old[k]);self.numeric_exact &= exact;self.numeric_tolerance &= tol;self.max_diff=max(self.max_diff,diff)
                need(tol,'ORE numerical divergence '+k)
        return {'ok':True}
    def final(self,p):
        need(p.get('state_origin')=='RECONSTRUCTED_THIS_RUN','output origin label')
        need(p['output_payload_sha256']==digest({k:v for k,v in p.items() if k!='output_payload_sha256'}),'policy output seal')
        for k in ('method','query_id','budget','repeat'):need(p[k]==self.r[k],'output condition identity '+k)
        for k in ('ranking','scored_ids','scored_scores','scores'):need(p[k]==self.r[k],'full output differs '+k)
        need(p['ranking'][:10]==self.r['ranking'][:10] and len(p['scored_ids'])==len(set(p['scored_ids']))==self.c['budget'],'Top10/S/budget')
        need(p['state_hashes']==self.states,'state stream/output seal')
        need(len(p['observed_events'])==len(self.r['events']),'event boundary count')
        for a,b in zip(p['observed_events'],self.r['events']):
            for k in ('step','doc_ids','scores','logical_evaluations_cumulative'):need(a[k]==b[k],'saved event mismatch '+k)
        if p['method']=='gar_cp':need(p['author_tie_ranking']==self.r['author_tie_ranking'],'GAR author tie diagnostic')
        else:
            need(p['scored_only_ranking']==[d for d in p['ranking'] if d in p['scored_ids']]==self.r['scored_only_ranking'],'derived SO output')
            for k in ('iterations','estimated_only_ids'):need(p[k]==self.r[k],'ORE final output source '+k)
        need(p['actual_new_model_calls']==0 and p['auxiliary_computations_independently_reexecuted'],'execution identity')
        return {'ok':True,'status':'PASS_POLICY_TRACE_EXACT','ordered_events_exact':True,'score_values_exact':True,
          'ordered_S_exact':True,'full_ranking_exact':True,'top10_exact':True,'author_tie_ranking_exact':p['method']=='gar_cp',
          'numeric_exact':self.numeric_exact,'numeric_within_frozen_tolerance':self.numeric_tolerance,'max_numeric_abs_diff':self.max_diff,
          'hidden_historical_state_bitwise_verified':False,'full_snapshots':p['full_state_snapshots']}

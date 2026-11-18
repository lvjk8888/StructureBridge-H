"""Pure checks of SAVED observations; never selects, scores, fits or replays a policy."""
import hashlib,json,math

GAR='gar_cp'
ORE='ore_hybrid_cp_bge_terrier_qreset_c100'
SO='ore_scored_only_posthoc_same_live_trajectory'
BASE=['raw','edge_mean_touched','anchor_ic_max']
METHODS=BASE+[GAR,ORE,SO]
BUDGETS=[20,50,100]

def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)

def ledger(pool,gold,s,t,m):
    need(len(pool)==len(set(pool))==1000,'pool uniqueness')
    need(all(isinstance(i,int) and 0<=i<len(pool) for i in s+t),'index domain')
    S={pool[i] for i in s};T={pool[i] for i in t};G=set(gold);U=set(pool)
    need(bool(G) and len(G)==len(gold),'full gold nonempty unique')
    a=int(G<=S);b=int(G<=T);ell=a*(1-b);j=(1-a)*b
    return {'a':a,'b':b,'ell':ell,'j':j,'unscored_final_count':len(T-S),
      'identity_residual':b-a+ell-j,'T_subset_S':int(T<=S),'S_subset_allowed':int(S<=U),
      'T_subset_allowed':int(T<=U),'exact_M':int(len(s)==len(S)==m),
      'top10_length_ok':int(len(t)==len(T)==10),'gold_count':len(G),'gold_outside_pool_count':len(G-U)}

def project_record(r,idx):
    # Original field hashes and hashes of this projection are explicitly different.
    keep=['method','budget','repeat','session_id','A_record_sha256','input_payload_sha256',
          'payload_sha256','order_index','logical_phi_evaluations','feedback_batches',
          'adaptive_feedback_rounds','new_gpu_pairs','score_cache_pairs','forward_calls',
          'input_tokens_nonpad','input_tokens_padded','sum_padded_length_squared',
          'physical_microbatch','qrels_read','answers_read','cache_bypassed','scores_are_live',
          'final_unscored_docs','feature_calls']
    p={k:r[k] for k in keep}
    for k in ['scored_ids','ranking','estimated_only_ids','scored_only_ranking']:
        p[k]=[idx[d] for d in r[k]]
    p['scored_scores']=[r['scored_scores'][d] for d in r['scored_ids']]
    p['scores']=[r['scores'][d] for d in r['ranking']]
    if 'iterations' in r:p['iterations']=[r['iterations'][d] for d in r['ranking']]
    if 'weights_final' in r:p['weights_final']=r['weights_final']
    p['original_events_sha256']=digest(r['events'])
    p['original_feedback_trace_sha256']=digest(r['trace'])
    p['original_scored_sequence_sha256']=digest([[d,r['scored_scores'][d]] for d in r['scored_ids']])
    p['events']=[]
    for e in r['events']:
        if 'execution_usage' in e:need(e['execution_usage']==e['physical_usage'],'duplicate usage differs')
        z={k:v for k,v in e.items() if k not in ('doc_ids','execution_usage','physical_usage')}
        z['doc_ids']=[idx[d] for d in e['doc_ids']]
        u=dict(e['physical_usage']);u['physical_batches']=[]
        for b in e['physical_usage']['physical_batches']:
            c=dict(b);c['doc_ids']=[idx[d] for d in b['doc_ids']];u['physical_batches'].append(c)
        z['physical_usage']=u;p['events'].append(z)
    p['trace']=[]
    for t in r['trace']:
        z=dict(t);z['doc_ids']=[idx[d] for d in t['doc_ids']]
        z['heads']=[[idx[d],v] for d,v in t['heads']];p['trace'].append(z)
    p['projection_sha256']=digest(p)
    return p

def check_saved(p,pool,rendered_hashes=None):
    need(p['projection_sha256']==digest({k:v for k,v in p.items() if k!='projection_sha256'}),'projection hash')
    m=p['budget'];method=p['method'];S=p['scored_ids'];rank=p['ranking'];U=set(range(len(pool)))
    original_events=[]
    for e in p['events']:
        z={k:v for k,v in e.items() if k not in ('doc_ids','physical_usage')}
        z['doc_ids']=[pool[d] for d in e['doc_ids']]
        u=dict(e['physical_usage']);u['physical_batches']=[]
        for b in e['physical_usage']['physical_batches']:
            c=dict(b);c['doc_ids']=[pool[d] for d in b['doc_ids']];u['physical_batches'].append(c)
        z['physical_usage']=u
        if method==ORE:z['execution_usage']=u
        original_events.append(z)
    need(digest(original_events)==p['original_events_sha256'],'original event field hash reconstructed')
    original_trace=[]
    for t in p['trace']:
        z=dict(t);z['doc_ids']=[pool[d] for d in t['doc_ids']]
        z['heads']=[[pool[d],v] for d,v in t['heads']];original_trace.append(z)
    need(digest(original_trace)==p['original_feedback_trace_sha256'],'original feedback field hash reconstructed')
    need(len(S)==len(set(S))==m and set(S)<=U,'scored domain/count')
    need(len(rank)==len(set(rank)) and set(rank)<=U,'ranking domain')
    need(p['repeat']==0 and not p['qrels_read'] and not p['answers_read'],'online label declaration')
    need(p['cache_bypassed'] and p['scores_are_live'] and p['physical_microbatch']==5,'live configuration')
    smap=dict(zip(S,p['scored_scores']));omap=dict(zip(rank,p['scores']))
    need(len(smap)==len(p['scored_scores']) and len(omap)==len(p['scores']),'score alignment')
    need(all(math.isfinite(x) and 0<=x<=1 for x in p['scored_scores']),'actual probability range')
    need(all(math.isfinite(x) for x in p['scores']),'output finite')
    need(all(omap[d]==s for d,s in smap.items()),'actual output score altered')
    totals={k:0 for k in ('new_gpu_pairs','score_cache_pairs','forward_calls','input_tokens_nonpad','input_tokens_padded','sum_padded_length_squared')}
    flat=[];short=0;inputhashcount=0
    for step,e in enumerate(p['events']):
        ids=e['doc_ids'];values=e['scores'];u=e['physical_usage'];phys=u['physical_batches']
        need(e['step']==step and len(ids)==len(values)==e['logical_evaluations'] and len(ids)>0,'event boundary')
        need(not set(ids)&set(flat) and len(ids)==len(set(ids)),'repeated score request')
        need(all(smap[d]==s for d,s in zip(ids,values)),'event/raw score equality')
        flat+=ids;need(e['logical_evaluations_cumulative']==len(flat),'disclosure cumulative')
        need(u['new_gpu_pairs']==len(ids) and u['score_cache_pairs']==0,'physical live pair count')
        need([d for b in phys for d in b['doc_ids']]==ids,'physical request order')
        for b in phys:
            n=b['batch_size'];ls=b['effective_lengths'];pad=b['padded_length'];hs=b['rendered_text_sha256']
            need(1<=n<=5 and len(b['doc_ids'])==len(ls)==len(hs)==n,'physical batch capacity')
            need(0<pad<=2048 and all(0<x<=pad for x in ls),'recorded token lengths')
            need(all(len(h)==64 for h in hs),'render hash format')
            if rendered_hashes is not None:need(hs==[rendered_hashes[d] for d in b['doc_ids']],'actual rendered text hashes')
            short+=int(n<5);inputhashcount+=n
        need(u['forward_calls']==len(phys),'forward calls')
        need(u['input_tokens_nonpad']==sum(sum(b['effective_lengths']) for b in phys),'nonpadding token counter')
        need(u['input_tokens_padded']==sum(b['batch_size']*b['padded_length'] for b in phys),'padding token counter')
        need(u['sum_padded_length_squared']==sum(b['batch_size']*b['padded_length']**2 for b in phys),'length squared counter')
        for k in totals:totals[k]+=u[k]
    need(flat==S and p['logical_phi_evaluations']==m,'ordered expensive requests')
    need(p['feedback_batches']==len(p['events']),'actual scoring event count')
    need(all(p[k]==v for k,v in totals.items()),'physical totals')
    need(digest([[pool[d],smap[d]] for d in S])==p['original_scored_sequence_sha256'],'raw score sequence hash')
    headrefs=0;estimate_count=0;feedback='UNKNOWN';state='UNKNOWN'
    if method==GAR:
        need(set(rank)==set(S) and not p['estimated_only_ids'] and not p['scored_only_ranking'],'GAR scored-only output')
        need(p['trace']==[],'unexpected GAR trace; inspect before claiming completeness')
    elif method==ORE:
        need(len(rank)==100 and set(p['estimated_only_ids'])==set(rank)-set(S),'ORE estimated domain')
        need(p['scored_only_ranking']==[d for d in rank if d in smap],'saved SO ordered filter')
        need(len(p['trace'])==m//5+int(m<100),'ORE saved trace length')
        seen={};insertion={};it=dict(zip(rank,p['iterations']));event_idx=0;prevw=[.2]*4
        for round_idx,t in enumerate(p['trace']):
            need(t['round']==round_idx and t['weights_before']==prevw,'ORE weight/round continuity')
            need(len(t['weights_before'])==len(t['weights_after'])==4,'weight dimensions')
            need(all(math.isfinite(x) and 0<=x<=1 for x in t['weights_after']),'saved weight bound')
            need(len(t['heads'])==min(10,len(seen)),'head length')
            need(len({d for d,v in t['heads']})==len(t['heads']),'head uniqueness')
            for d,v in t['heads']:
                need(d in seen and v==seen[d] and v[1]<round_idx,'future or different-score feedback')
                headrefs+=1
            # Check saved head membership/order against already observed score tuples;
            # this never picks a new scoring candidate or executes the original policy.
            headkeys=[(-v[0],-v[1],insertion[d]) for d,v in t['heads']]
            need(all(a<=b for a,b in zip(headkeys,headkeys[1:])),'saved head order')
            if headkeys:
                hs={d for d,v in t['heads']}
                need(all((-v[0],-v[1],insertion[d])>=headkeys[-1] for d,v in seen.items() if d not in hs),'saved head top membership')
            ids=t['doc_ids'];est=t['selected_estimates'];features=t['selected_features']
            need(len(ids)==len(est)==len(features) and not set(ids)&set(seen),'selected observation alignment')
            need(all(len(x)==4 and all(math.isfinite(v) for v in x) for x in features),'saved feature dimensions')
            # Local arithmetic on SAVED selected features only; no fit or candidate selection.
            dims=2 if round_idx==0 else 4
            need(all(math.isclose(sum(x[k]*t['weights_before'][k] for k in range(dims)),z,rel_tol=1e-12,abs_tol=1e-12) for x,z in zip(features,est)),'saved selected estimate arithmetic')
            if t['mode']=='expensive':
                e=p['events'][event_idx];need(ids==e['doc_ids'] and len(ids)==5,'ORE event/trace alignment')
                need(t['fit'] is not None and t['fit']['status']>0,'saved fit status')
                vals=e['scores'];event_idx+=1
            else:
                need(t['mode']=='estimated_only' and round_idx==len(p['trace'])-1 and event_idx==len(p['events']),'estimated terminal boundary')
                need(set(ids)==set(p['estimated_only_ids']) and t['fit'] is None and t['weights_after']==t['weights_before'],'estimated-only no score/fit')
                vals=est;estimate_count+=len(ids)
            for d,v in zip(ids,vals):
                need(omap[d]==v and it[d]==round_idx,'saved output provenance')
                insertion[d]=len(insertion);seen[d]=[v,round_idx]
            if round_idx:
                u=t['rm3_usage'];need(u['cache_hit'] is False and u['logical_RM3_calls']==1,'saved RM3 usage')
                need(u['physical_usage']['candidate_scoring_count']==1000 and u['physical_usage']['returned_domain']=='registered U only','saved RM3 domain declaration')
            else:need(t['heads']==[] and not t['rm3_usage'],'first round reset')
            prevw=t['weights_after']
        need(prevw==p['weights_final'] and event_idx==len(p['events']),'saved final weight and event total')
        keys=[(-omap[d],-it[d],insertion[d]) for d in rank]
        need(all(a<=b for a,b in zip(keys,keys[1:])),'saved ORE ranking order')
        feedback='VERIFIED_SAVED_HEADS_PRIOR_ACTUAL_EVENTS'
    else:raise ValueError('not a trace method')
    origins=['actual' if d in smap else 'estimated' if d in p['estimated_only_ids'] else 'UNKNOWN' for d in rank[:10]]
    need('UNKNOWN' not in origins,'missing output origin')
    need(p['final_unscored_docs']==origins.count('estimated'),'saved final unscored counter')
    return {'distinct_scored_documents':len(set(flat)),'score_request_occurrences':len(flat),
      'repeated_score_requests':len(flat)-len(set(flat)),'scoring_event_count':len(p['events']),
      'declared_adaptive_feedback_rounds':p['adaptive_feedback_rounds'],
      'declared_rounds_equal_event_count':int(p['adaptive_feedback_rounds']==len(p['events'])),
      'physical_forward_calls':totals['forward_calls'],'short_physical_batches':short,
      'rendered_hash_occurrences':inputhashcount,'feedback_head_references':headrefs,
      'estimated_output_documents':estimate_count,'feedback_provenance':feedback,
      'complete_policy_state_or_replay_premises':state,'top10_origins':origins}

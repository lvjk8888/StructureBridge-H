"""Frozen research functions; source identity is recorded in the release provenance."""
import time
from .record_checks import canon,digest

class OracleRejection(Exception):
    def __init__(self,code,detail):super().__init__(code);self.code=code;self.detail=detail

class HistoricalObservationOracle:
    def __init__(self,condition,record,log=None):
        self.c=condition;self.r=record;self.events=record['events'];self.i=0;self.used=[];self.closed=False;self.log=log
    def reject(self,code,detail):
        self.closed=True;raise OracleRejection(code,detail)
    def request(self,req):
        received=time.monotonic_ns()
        if self.closed:self.reject('CLOSED_CONDITION',{})
        if req.get('op','score')!='score':self.reject('FUTURE_OR_UNSUPPORTED_ACCESS',{'operation':req.get('op')})
        for k in ('condition_id','query_id','method','budget','repeat','block_sha256','original_input_payload_sha256','whitelist_payload_sha256'):
            if req.get(k)!=self.c[k]:self.reject('INPUT_OR_CONDITION_IDENTITY_MISMATCH',{'field':k,'received':req.get(k),'expected':self.c[k]})
        if req.get('event_index')!=self.i:self.reject('EVENT_BOUNDARY_OR_FUTURE_ACCESS',{'received':req.get('event_index'),'expected':self.i})
        ids=req.get('doc_ids',[])
        if not ids or len(ids)!=len(set(ids)) or set(ids)&set(self.used):self.reject('DUPLICATE_OR_EMPTY_REQUEST',{'doc_ids':ids})
        if not set(ids)<=set(self.r['pool_document_ids']):self.reject('OUTSIDE_ALLOWED_DOMAIN',{'doc_ids':ids})
        if len(self.used)+len(ids)>self.c['budget']:self.reject('BUDGET_EXCEEDED',{'used':len(self.used),'requested':len(ids)})
        if self.i>=len(self.events):self.reject('NO_BOUND_HISTORICAL_EVENT',{})
        expected=self.events[self.i]
        if ids!=expected['doc_ids']:self.reject('ORDERED_REQUEST_DIVERGED',{'actual':ids,'historical':expected['doc_ids'],'event_index':self.i})
        phys=expected['physical_usage']['physical_batches'];rh=[h for p in phys for h in p['rendered_text_sha256']]
        if req.get('rendered_text_sha256')!=rh:self.reject('RENDERED_INPUT_IDENTITY_MISMATCH',{'received':req.get('rendered_text_sha256'),'expected':rh})
        raw=list(expected['scores'])
        if len(raw)!=len(ids):self.reject('HISTORICAL_SCORE_MISSING',{})
        key={k:self.c[k] for k in ('condition_id','query_id','method','budget','repeat','block_sha256','original_input_payload_sha256')}
        key.update({'event_index':self.i,'ordered_doc_ids':ids,'rendered_text_sha256':rh,'historical_physical_batch_sha256':digest(phys)})
        row={'kind':'SCORE_DISCLOSURE','condition_id':self.c['condition_id'],'request_received_ns':received,'disclosure_recorded_ns':time.monotonic_ns(),
          'request':req,'request_identity_sha256':digest(key),'historical_physical_batch_sha256':digest(phys),
          'scores':raw,'logical_evaluations_cumulative':len(self.used)+len(ids),'historical_provenance':expected['physical_usage'],
          'actual_new_model_calls':0,'record_written_before_reply':True}
        if self.log:self.log.write(canon(row)+'\n');self.log.flush()
        self.i+=1;self.used+=ids
        return {'ok':True,'scores':raw}
    def complete(self):return self.i==len(self.events) and len(self.used)==len(set(self.used))==self.c['budget']

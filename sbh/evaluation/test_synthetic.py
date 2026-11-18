"""Synthetic fixtures only. These checks do not reproduce paper results."""
import io,json,unittest
from fractions import Fraction
import numpy as np
from .answers_hotpot import answer_scores,parse_answer
from .answers_twowiki import answer_score
from .answers_musique import score
from .evidence import evidence_metrics
from .statistics_e1 import paired_numeric
from .statistics_c import paired
from .statistics_a02 import boot
from .recompute_ablation_cpu import exact_mcnemar,holm,signflip
from sbh.bef.record_checks import ledger
from sbh.bef.output_rules import compare_output_rules
from sbh.bef.oracle import HistoricalObservationOracle,OracleRejection
from sbh.bef.validator import compare_numbers
from sbh.bef.strata import strata_rows,strata_summary
from sbh.bef.transitions import transition_rows,transition_summary

class SyntheticEvaluation(unittest.TestCase):
    def test_frozen_answer_rules(self):
        self.assertEqual(answer_scores('The RED, fox!','red fox'),(1.,1.))
        self.assertEqual(answer_score('yes please','yes'),(0.,0.))
        self.assertEqual(answer_scores('',''),(1.,0.))
        self.assertEqual(score('yes please',['yes']),(0.,2/3))
        self.assertEqual(score('same',['same'],False),(0.,0.))
        self.assertEqual(answer_scores('cat cat','cat'),(0.,2/3))
        self.assertTrue(parse_answer('ANSWER: red\nCITATIONS: 1, 2')['format_ok'])
        self.assertFalse(parse_answer('ANSWER: red\nCITATIONS: 1, 1')['citation_parse_ok'])

    def test_evidence_and_output_rules(self):
        pool=['synthetic-'+str(i) for i in range(1000)]
        r={'candidates':[{'doc_id':d} for d in pool]}
        m=evidence_metrics(r,{pool[0],pool[999]},pool[:19]+[pool[999]],pool,20)
        self.assertEqual((m['CER'],m['Recall'],m['recovered'],m['substitutions']),(1,1.,1,1))
        ranking=[999]+list(range(100))
        z=compare_output_rules(pool,[pool[0],pool[999]],list(range(20)),ranking,20)
        self.assertEqual((z['mixed']['a'],z['mixed']['b'],z['mixed']['j']),(0,1,1))
        self.assertEqual((z['scored_only']['b'],z['b_SO_minus_b_mix'],z['identity_residual']),(0,-1,0))
        self.assertFalse(z['SO_is_selection_strategy'])
        with self.assertRaises(ValueError):ledger(pool,[],list(range(20)),list(range(10)),20)

    def test_statistics(self):
        a=np.ones((3,3));b=np.zeros((3,3))
        self.assertEqual(paired_numeric(a,b,draws=16)['ci95'],[1.,1.])
        self.assertEqual(paired(a,b,7,16)['ci95_pp'],[100.,100.])
        self.assertEqual(boot(np.ones(3),7,16)['ci95_pp'],[100.,100.])
        self.assertEqual(exact_mcnemar(3,0),Fraction(1,4))
        self.assertEqual(holm([.01,.03,.04]),[.03,.06,.06])
        self.assertEqual(signflip(np.zeros(3,dtype=np.int64),7,16)['p'],1.)

    def test_oracle_rejects_future_and_preserves_scores(self):
        c=dict(condition_id='synthetic',query_id='q',method='ore',budget=2,repeat=0,
               block_sha256='synthetic',original_input_payload_sha256='synthetic',whitelist_payload_sha256='synthetic')
        event={'doc_ids':['a','b'],'scores':[.2,.8],'physical_usage':{'physical_batches':[{'rendered_text_sha256':['h1','h2']}]}}
        r={'events':[event],'pool_document_ids':['a','b']}
        req={**c,'event_index':0,'doc_ids':['a','b'],'rendered_text_sha256':['h1','h2']}
        sink=io.StringIO();o=HistoricalObservationOracle(c,r,sink)
        self.assertEqual(o.request(req),{'ok':True,'scores':[.2,.8]})
        self.assertTrue(o.complete());self.assertIn('"record_written_before_reply":true',sink.getvalue())
        with self.assertRaises(OracleRejection):HistoricalObservationOracle(c,r).request({**req,'event_index':1})
        with self.assertRaises(OracleRejection):HistoricalObservationOracle(c,r).request({**req,'doc_ids':['b','a']})
        self.assertEqual(compare_numbers([1.],[1.]),(True,True,0.))
        self.assertEqual(compare_numbers(1.+1e-13,1.)[:2],(False,True))

    def test_native_strata_transition_blocks(self):
        pool=['synthetic-'+str(i) for i in range(1000)];gold={'q':{pool[0],pool[999]}}
        f={'query_id':'q','pool_ids':pool,'cells':{},'sources':{k:'synthetic' for k in ('block_plan_sha256','session_id','A_shard','A_file_sha256','block','block_sha256')}}
        cm={};old={};review={'query_id':'q','ordered_pool_ids':pool,'complete_gold_ids':sorted(gold['q']),'budgets':{}};original={}
        for b in (20,50,100):
            f['cells'][str(b)]={};review['budgets'][str(b)]={'seed_indices':list(range(b)),'methods':{}}
            original['q',b]={}
            for method,label in (('raw','Raw'),('edge_mean_touched','EdgeMean'),('anchor_ic_max','SBH')):
                ids=list(range(b-1))+[999] if method=='anchor_ic_max' else list(range(b))
                top=[0,999]+list(range(1,9)) if method=='anchor_ic_max' else list(range(10))
                f['cells'][str(b)][method]={'selected':[pool[i] for i in ids],'top10':[pool[i] for i in top],'scope':pool,'C_rid':'synthetic'}
                v=int(method=='anchor_ic_max');old['q',b,method]={'scored_set_complete':v,'CER_at_10':v,'pool_complete':1}
                cm['q',b,method]={'request_sha256':'synthetic','Answer_EM':v,'Answer_F1':v,'usable_answer':'True'}
                review['budgets'][str(b)]['methods'][method]={'S_indices':ids,'T10_indices':top,'scope_indices':list(range(1000))}
                original['q',b][label+'_A_complete']=v;original['q',b][label+'_B_complete']=v
        rows=strata_rows([json.dumps(f)],gold,cm,old)
        self.assertEqual(len(rows),3);self.assertTrue(all(x['stratum']=='SBH_A_rescue_B_complete' for x in rows))
        s=next(x for x in strata_summary(rows) if x['budget']=='ALL' and x['stratum']=='ALL' and x['method']=='SBH')
        self.assertEqual((s['n_query_budget'],s['n_unique_queries'],s['F1_delta_vs_Raw_stored_mean']),(3,1,1.))
        t=transition_summary(transition_rows([review],original))
        self.assertEqual(t['loss_overlap']['ALL']['A_rescue_B_complete']['condition_count'],3)
        self.assertTrue(all(x['identity_residual']==0 for x in t['accounting']))

if __name__=='__main__':unittest.main()

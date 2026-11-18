"""ORE scored-only is an output rule on the same scored trajectory."""
from .record_checks import ledger,need

def compare_output_rules(pool,gold,scored_ids,ranking,budget):
    """Indices refer to the frozen ordered pool, as in the original trace audit."""
    smap=set(scored_ids)
    scored_only_ranking=[d for d in ranking if d in smap]
    mix=ledger(pool,gold,scored_ids,ranking[:10],budget)
    so=ledger(pool,gold,scored_ids,scored_only_ranking[:10],budget)
    need(mix['exact_M'] and so['top10_length_ok'] and mix['top10_length_ok'],'cardinality')
    delta=so['b']-mix['b'];rhs=mix['ell']-so['ell']-mix['j'];need(delta==rhs,'paired BEF identity')
    return {'mixed':mix,'scored_only':so,'b_SO_minus_b_mix':delta,
            'ell_mix_minus_ell_SO_minus_j_mix':rhs,'identity_residual':delta-rhs,
            'additional_SO_score_requests':0,'SO_is_selection_strategy':False}

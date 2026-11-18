"""Frozen score calculations restricted to the released method paths.

Only the registered main methods and documented ablation score paths are exposed.
Unused historical branches and default tables were omitted; retained calculations
are unchanged and checked against the original pure functions.
"""
from __future__ import annotations
import math

def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_group(q, rows, hg, top_k):
    require(len(rows) == top_k, f'{q}: candidate depth {len(rows)} != {top_k}')
    docs = [r[0] for r in rows]
    require(len(set(docs)) == len(docs) and all(docs), f'{q}: duplicate/empty doc ids')
    require([r[1] for r in rows] == list(range(1, top_k + 1)), f'{q}: ranks not contiguous')
    require(all(math.isfinite(r[2]) for r in rows), f'{q}: non-finite BM25 scores')
    require(str(hg.get('query_id')) == q, f'{q}: HG qid mismatch')
    require([str(d) for d in hg.get('candidate_doc_ids', [])] == docs,
            f'{q}: candidate order differs from hypergraph')
    require(isinstance(hg.get('hyperedges'), list), f'{q}: missing hyperedges')
    allowed = set(docs)
    for edge in hg['hyperedges']:
        require('edge_type' in edge and isinstance(edge.get('members'), list),
                f'{q}: malformed hyperedge')
        ms = [str(d) for d in edge['members']]
        require(len(ms) == len(set(ms)), f'{q}: duplicate hyperedge members')
        require(set(ms) <= allowed, f'{q}: hyperedge has out-of-pool members')
    return docs


def touched_edges(edges, seeds):
    return [e for e in edges if e['edge_type'] == 'title_link'
            and len(e['members']) >= 2 and seeds.intersection(map(str, e['members']))]


def max_normalize(values):
    hi = max(values.values(), default=0.0)
    return {d: v / hi if hi > 0 else 0.0 for d, v in values.items()}


def edge_statistic(scope, edges, relevance, statistic):
    require(statistic == 'mean', 'Only mean edge statistics are included in this release')
    mu = sum(relevance[d] for d in scope) / len(scope)
    den = max(1e-12, 1 - mu)
    raw = dict.fromkeys(scope, 0.0)
    for edge in edges:
        ms = [str(d) for d in edge['members']]
        vals = sorted((relevance[d] for d in ms), reverse=True)
        v = sum(vals) / len(vals)
        u = max(0.0, (v - mu) / den)
        for d in ms:
            raw[d] += u
    return max_normalize(raw)


def anchor_strength(members, seeds, relevance, candidate):
    return max((relevance[s] for s in members if s in seeds and s != candidate), default=0.0)


def information_content(size, top_k):
    require(2 <= size <= top_k and top_k > 2, 'IC requires 2 <= edge size <= K, K > 2')
    return math.log(top_k / size) / math.log(top_k / 2)


def structural(method, scope, touched, seeds, relevance, top_k):
    raw = dict.fromkeys(scope, 0.0)
    if method == 'raw':
        return raw
    if method == 'edge_mean_touched':
        return edge_statistic(scope, touched, relevance, 'mean')
    if method in ('seed_affinity', 'anchor_ic_max', 'anchor_ic_or'):
        for e in touched:
            ms = [str(d) for d in e['members']]
            weight = 1.0 if method == 'seed_affinity' else information_content(len(ms), top_k)
            for d in ms:
                g = weight * anchor_strength(ms, seeds, relevance, d)
                if method == 'anchor_ic_or':
                    # Bounded saturating score, not a calibrated probability.
                    raw[d] = 1 - (1 - raw[d]) * (1 - g)
                else:
                    raw[d] = max(raw[d], g)
        return raw
    if method == 'diffusion_touched':
        # Source-normalized one-step seed -> edge -> OTHER vertex flow.
        # It is a projected-graph-equivalent linear baseline, not proof of
        # an irreducible high-order hypergraph advantage.
        degree = dict.fromkeys(seeds, 0)
        for e in touched:
            for s in map(str, e['members']):
                if s in seeds:
                    degree[s] += 1
        for e in touched:
            ms = [str(d) for d in e['members']]
            contributions = {s: relevance[s] / degree[s] / (len(ms) - 1)
                             for s in ms if s in seeds}
            total = sum(contributions.values())
            for d in ms:
                raw[d] += max(0.0, total - contributions.get(d, 0.0))
        return max_normalize(raw)
    raise ValueError(f'Unknown method: {method}')


def select(scope, relevance, scores, ranks, budget, lam):
    require(len(scope) >= budget, 'Scope smaller than budget')
    require(all(math.isfinite(scores[d]) and -1e-12 <= scores[d] <= 1+1e-12
                for d in scope), 'Invalid structural score')
    return sorted(scope, key=lambda d: (-(relevance[d] + lam * scores[d]), ranks[d], d))[:budget]


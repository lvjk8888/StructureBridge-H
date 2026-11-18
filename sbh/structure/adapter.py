"""Frozen candidate title/alias and shared-mention relation construction."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import re
import unicodedata
import pandas as pd

GENERIC_MENTION_STOPLIST = {
    "the", "a", "an", "this", "that", "these", "those",
    "he", "she", "it", "they", "his", "her", "its", "their",
    "united states", "united kingdom", "english", "american",
    "wikipedia", "article", "film", "television", "series",
}


CONNECTOR_TOKENS = {
    "of", "the", "and", "de", "da", "del", "van", "von",
    "la", "le", "di", "du", "des", "der", "den", "st",
}


TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’.-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*")


CAP_TOKEN_RE = re.compile(r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ0-9]*(?:['’.-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*$")


@dataclass
class AdapterConfig:
    min_title_chars: int = 3
    min_mention_chars: int = 4
    max_title_tokens: int = 12
    max_mention_tokens: int = 6
    max_edge_size: int = 100


def normalize_text(value: object) -> str:
    s = "" if value is None else str(value)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("_", " ")
    return " ".join(s.split())


def normalize_phrase(value: object) -> str:
    s = normalize_text(value).lower()
    tokens = TOKEN_RE.findall(s)
    return " ".join(t.lower() for t in tokens)


def title_aliases(title: object, cfg: AdapterConfig) -> List[Tuple[str, ...]]:
    raw = normalize_text(title)
    aliases = [raw]
    # Wikipedia disambiguation suffix: "Daniel Fernandes (footballer)" -> "Daniel Fernandes"
    base = re.sub(r"\s*\([^()]{1,80}\)\s*$", "", raw).strip()
    if base and base != raw:
        aliases.append(base)

    out: List[Tuple[str, ...]] = []
    seen = set()
    for alias in aliases:
        norm = normalize_phrase(alias)
        if len(norm) < cfg.min_title_chars or norm in GENERIC_MENTION_STOPLIST:
            continue
        toks = tuple(norm.split())
        if not toks or len(toks) > cfg.max_title_tokens:
            continue
        if toks in seen:
            continue
        seen.add(toks)
        out.append(toks)
    return out


def original_tokens(text: object) -> List[str]:
    return TOKEN_RE.findall(normalize_text(text))


def extract_capitalized_mentions(text: object, cfg: AdapterConfig) -> set[str]:
    toks = original_tokens(text)
    out: set[str] = set()
    n = len(toks)
    i = 0
    while i < n:
        if not CAP_TOKEN_RE.match(toks[i]):
            i += 1
            continue

        phrase = [toks[i]]
        j = i + 1
        proper_count = 1
        while j < n and len(phrase) < cfg.max_mention_tokens:
            tok = toks[j]
            low = tok.lower().rstrip(".")
            if CAP_TOKEN_RE.match(tok):
                phrase.append(tok)
                proper_count += 1
                j += 1
                continue
            if low in CONNECTOR_TOKENS and j + 1 < n and CAP_TOKEN_RE.match(toks[j + 1]):
                phrase.append(tok)
                j += 1
                continue
            break

        # Emit the longest phrase only. Singleton mentions are allowed, but only
        # if sufficiently long and shared by >=2 candidate docs later.
        norm = normalize_phrase(" ".join(phrase))
        if norm and len(norm) >= cfg.min_mention_chars:
            if norm not in GENERIC_MENTION_STOPLIST:
                if proper_count >= 2 or len(norm) >= 5:
                    out.add(norm)
        i += 1
    return out


class PhraseTrie:
    END = "__END__"

    def __init__(self) -> None:
        self.root: dict = {}

    def add(self, tokens: Sequence[str], phrase_key: str) -> None:
        node = self.root
        for tok in tokens:
            node = node.setdefault(tok, {})
        node.setdefault(self.END, set()).add(phrase_key)

    def find(self, text: object) -> set[str]:
        toks = [t.lower() for t in TOKEN_RE.findall(normalize_text(text))]
        found: set[str] = set()
        root = self.root
        for i in range(len(toks)):
            node = root.get(toks[i])
            if node is None:
                continue
            if self.END in node:
                found.update(node[self.END])
            j = i + 1
            while j < len(toks):
                node = node.get(toks[j])
                if node is None:
                    break
                if self.END in node:
                    found.update(node[self.END])
                j += 1
        return found


def build_query_hypergraph(group: pd.DataFrame, cfg: AdapterConfig) -> tuple[list[dict], dict]:
    docs = []
    for row in group.itertuples(index=False):
        docs.append(
            {
                "doc_id": str(row.doc_id),
                "title": str(row.doc_title),
                "text": str(row.doc_text),
                "rank": int(row.bm25_rank),
            }
        )

    # ---------- title-link edges ----------
    alias_to_owners: Dict[str, set[str]] = defaultdict(set)
    trie = PhraseTrie()

    for doc in docs:
        for toks in title_aliases(doc["title"], cfg):
            key = " ".join(toks)
            alias_to_owners[key].add(doc["doc_id"])
            trie.add(toks, key)

    title_members: Dict[str, set[str]] = {
        key: set(owners) for key, owners in alias_to_owners.items()
    }
    for doc in docs:
        for key in trie.find(doc["text"]):
            title_members[key].add(doc["doc_id"])

    # ---------- shared capitalized-mention edges ----------
    mention_members: Dict[str, set[str]] = defaultdict(set)
    for doc in docs:
        mentions = extract_capitalized_mentions(
            f'{doc["title"]}. {doc["text"]}', cfg
        )
        for mention in mentions:
            mention_members[mention].add(doc["doc_id"])

    edges: list[dict] = []
    dropped_too_large = Counter()
    dropped_singleton = Counter()

    def emit(edge_type: str, mapping: Dict[str, set[str]]) -> None:
        for key in sorted(mapping):
            members = sorted(mapping[key])
            if len(members) < 2:
                dropped_singleton[edge_type] += 1
                continue
            if len(members) > cfg.max_edge_size:
                dropped_too_large[edge_type] += 1
                continue
            edges.append(
                {
                    "edge_id": f"{edge_type}::{key}",
                    "edge_type": edge_type,
                    "edge_key": key,
                    "members": members,
                }
            )

    emit("title_link", title_members)
    emit("shared_mention", mention_members)

    stats = {
        "num_candidates": len(docs),
        "num_edges": len(edges),
        "num_memberships": int(sum(len(e["members"]) for e in edges)),
        "dropped_too_large": dict(dropped_too_large),
        "dropped_singleton": dict(dropped_singleton),
    }
    return edges, stats


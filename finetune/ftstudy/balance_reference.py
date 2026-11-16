"""Reference for the owner's part (a): the length-and-template balancing function.

Idea. A PAIR is two items of one premise and one hypothesis type: one valid (string q), one invalid (string t != q).
Any set of pairs is balanced within premise, and within hypothesis type x premise length exactly (both items share
the premise). What a set of pairs does not give by itself is balance within a hypothesis STRING, overall and at
every length: string q may be the valid member more often than the invalid one, and its valid items may sit on
shorter premises. So pairs are collected separately inside each (type, length bin), and inside each such cell every
string must be the valid member exactly as often as it is the invalid member. Seen as a directed graph on strings
(an edge q -> t per pair), that is "in-degree = out-degree at every node". A greedy pass adds edges that reduce the
imbalance; a trimming pass then removes edges until every node is balanced.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def length_bin_edges(lengths: Sequence[int], n_bins: int) -> list[float]:
    if n_bins <= 1 or len(lengths) == 0:
        return []
    return [float(x) for x in np.quantile(np.asarray(lengths, float), np.arange(1, n_bins) / n_bins)]


def balance_items(items: list[dict], premise_tokens: Sequence[int], *, n_length_bins: int,
                  max_pairs_per_premise: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    premise_tokens = np.asarray(premise_tokens)
    # strings that occur with both labels among the candidates: the others can never be balanced
    lab = defaultdict(set)
    for it in items:
        lab[it["query"]].add(it["label"])
    both = {q for q, s in lab.items() if s == {0, 1}}
    by_tmpl = defaultdict(list)
    for i, it in enumerate(items):
        if it["query"] in both:
            by_tmpl[it["template"]].append(i)
    chosen: list[tuple[int, int]] = []
    used_pairs = defaultdict(int)                       # premise -> pairs taken, over all types
    for tmpl in sorted(by_tmpl):
        idx = np.array(by_tmpl[tmpl])
        prem_len = {items[i]["group"]: int(premise_tokens[i]) for i in idx}
        edges = length_bin_edges(sorted(prem_len.values()), n_length_bins)     # quantiles over premises of this type
        by_cell = defaultdict(lambda: defaultdict(lambda: ([], [])))           # bin -> premise -> (valid idx, invalid idx)
        for i in idx:
            g = items[i]["group"]
            b = int(np.searchsorted(edges, prem_len[g], side="right"))
            by_cell[b][g][0 if items[i]["label"] == 1 else 1].append(int(i))
        for b in sorted(by_cell):
            d = defaultdict(int)                        # string -> (times valid member) - (times invalid member)
            pairs = []
            prems = sorted(by_cell[b]); rng.shuffle(prems)
            for g in prems:
                val, inv = (list(x) for x in by_cell[b][g])
                rng.shuffle(val); rng.shuffle(inv)
                while val and inv and used_pairs[g] < max_pairs_per_premise:
                    v = min(val, key=lambda i: d[items[i]["query"]])           # the string most in need of a valid member
                    cand = [i for i in inv if items[i]["query"] != items[v]["query"]]
                    if not cand:
                        break
                    w = max(cand, key=lambda i: d[items[i]["query"]])          # ... and of an invalid member
                    val.remove(v); inv.remove(w)
                    d[items[v]["query"]] += 1; d[items[w]["query"]] -= 1
                    pairs.append((v, w)); used_pairs[g] += 1
            # trim until every string is balanced inside this cell. Each step removes one pair, so it ends.
            while True:
                over = [q for q, x in d.items() if x > 0]
                if not over:
                    break
                q = max(over, key=lambda z: d[z])
                js = [j for j, (v, w) in enumerate(pairs) if items[v]["query"] == q]
                j = min(js, key=lambda j: d[items[pairs[j][1]]["query"]])     # prefer a partner string that is short of valid members
                v, w = pairs.pop(j)
                d[q] -= 1; d[items[w]["query"]] += 1
                used_pairs[items[v]["group"]] -= 1
            assert all(x == 0 for x in d.values())
            chosen.extend(pairs)
    return sorted(i for p in chosen for i in p)

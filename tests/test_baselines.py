"""Text-only baselines must not leak labels: on shuffled labels they must sit at chance.
(Guards against the target-encoding leak found and fixed on Fri 13 Nov; see notes/log.md.)"""
import numpy as np
import pytest

from causal_probe import data, probes, stats


@pytest.mark.parametrize("tag", ["c2c_probe", "cladder"])
def test_baselines_at_chance_on_shuffled_labels(tag):
    items = data.get_dataset(tag)[::2]      # the samples are sorted by stratum, so take every other item
    y = np.array([i["label"] for i in items])
    ys = np.random.default_rng(0).permutation(y)
    folds = probes.folds_for(items, 0)
    for fn in (probes.structural_oof, probes.hyp_prior_oof, probes.tfidf_query_oof, probes.query_only_oof):
        s = fn(items, ys, folds)
        acc = stats.accuracy(ys, (s > 0).astype(int))
        assert 0.42 < acc < 0.58, (fn.__name__, acc)


def test_structural_parser_reads_every_statement():
    import re
    for it in data.get_dataset("c2c_probe")[::50]:
        p = probes.parse_premise(it["context"])
        n = len(p["corr"]) + len(p["ind"]) + sum(len(v) for v in p["cind"].values())
        assert n == len(re.findall(r"\.", it["context"].split("as follows:")[1]))

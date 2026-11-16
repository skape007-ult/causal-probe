"""Which implementation of the owner's parts the pipeline uses.

The owner (Sahil) writes `balance.py` and `within_string.py`. Until a function there stops raising
NotImplementedError, the pipeline falls back to the reference in `*_reference.py`, and every output file records
which one produced it ("owner" or "reference"). finetune/tests run the same tests on both.
"""
from __future__ import annotations

import numpy as np

from . import balance, balance_reference, within_string, within_string_reference


def _owner_ready(fn, probe) -> bool:
    try:
        probe(fn)
    except NotImplementedError:
        return False
    except Exception:
        return True          # implemented (even if it fails on the probe input: the tests will say so, loudly)
    return True


def balance_items():
    """-> (function, "owner" | "reference")"""
    ok = _owner_ready(balance.balance_items, lambda f: f([], np.array([], int), n_length_bins=1, max_pairs_per_premise=1, seed=0))
    return (balance.balance_items, "owner") if ok else (balance_reference.balance_items, "reference")


def within_string_auroc_length_removed():
    ok = _owner_ready(within_string.within_string_auroc_length_removed,
                      lambda f: f(np.array([0, 1]), np.array([0.0, 1.0]), np.array(["a", "a"]), np.array([5, 6])))
    return ((within_string.within_string_auroc_length_removed, "owner") if ok
            else (within_string_reference.within_string_auroc_length_removed, "reference"))

"""OWNER'S PART (a). Sahil implements this; Claude does not. Spec and how to check: finetune/MY_PARTS.md, part (a).

Do not read balance_reference.py before your own version passes the tests.
"""
from __future__ import annotations

from typing import Sequence


def length_bin_edges(lengths: Sequence[int], n_bins: int) -> list[float]:
    """Inner edges of n_bins quantile bins of `lengths` (n_bins - 1 numbers, non-decreasing)."""
    raise NotImplementedError("owner's part (a): see finetune/MY_PARTS.md")


def balance_items(items: list[dict], premise_tokens: Sequence[int], *, n_length_bins: int,
                  max_pairs_per_premise: int, seed: int) -> list[int]:
    """Indices (sorted, no repeats) of a subset of `items` that is balanced as MY_PARTS.md, part (a), specifies."""
    raise NotImplementedError("owner's part (a): see finetune/MY_PARTS.md")

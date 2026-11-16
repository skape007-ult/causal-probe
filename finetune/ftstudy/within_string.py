"""OWNER'S PART (b). Sahil implements this from scratch; Claude does not. Spec: finetune/MY_PARTS.md, part (b).

Do not read within_string_reference.py or scripts/15_answer_bias.py before your own version passes the tests.
"""
from __future__ import annotations


def within_string_auroc_length_removed(labels, scores, strings, n_tokens) -> dict:
    """{"mean_auroc": float, "per_string": {string: auroc}, "n_strings": int}; see MY_PARTS.md, part (b)."""
    raise NotImplementedError("owner's part (b): see finetune/MY_PARTS.md")

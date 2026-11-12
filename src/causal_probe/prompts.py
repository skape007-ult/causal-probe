"""Prompt construction, chat templating and token-position lookup.

The user message is always  INSTRUCTION + context + query  with the query last, so that
"content_end" (last token of the user's text) is the end of the hypothesis / question, and
"final" (last token of the whole templated prompt) is the position whose next-token
prediction is the model's answer.
"""
from __future__ import annotations

import numpy as np

C2C_INSTR = ("You are given a premise that lists statistical relations among variables, and a "
             "hypothesis about a causal relation between them. Decide whether the hypothesis is "
             "necessarily true given the premise. Answer with only one word: Yes or No.")
CLADDER_INSTR = "Answer the following question with only one word: Yes or No."


def user_parts(item: dict) -> tuple[str, str, str]:
    """Return (prefix, context, query-part) whose concatenation is the user message."""
    if item["dataset"] == "c2c":
        return C2C_INSTR + "\n\nPremise: ", item["context"], "\nHypothesis: " + item["query"]
    return CLADDER_INSTR + "\n\n", item["context"], " " + item["query"]


def user_text(item: dict) -> str:
    return "".join(user_parts(item))


def answer_word(label: int) -> str:
    return "Yes" if label == 1 else "No"


def build_messages(item: dict, shots: list[dict] | None = None) -> list[dict]:
    """Zero-shot: one user turn. Few-shot: each shot is a user turn + assistant answer turn."""
    msgs = []
    for s in shots or []:
        msgs.append({"role": "user", "content": user_text(s)})
        msgs.append({"role": "assistant", "content": answer_word(s["label"])})
    msgs.append({"role": "user", "content": user_text(item)})
    return msgs


def render(tokenizer, item: dict, shots: list[dict] | None = None) -> str:
    """Apply the model's own chat template, ending with the assistant-turn header."""
    return tokenizer.apply_chat_template(build_messages(item, shots), tokenize=False,
                                         add_generation_prompt=True)


def _token_covering(offsets: np.ndarray, char_idx: int) -> int:
    """Index of the token whose character span covers `char_idx` (the last character of a
    text span). Tokenizers may merge that character with what follows (e.g. '.\\n' is one
    token), so 'the token covering the last character' is the well-defined choice."""
    ok = np.where((offsets[:, 0] <= char_idx) & (offsets[:, 1] > char_idx))[0]
    assert len(ok) == 1, (char_idx, ok)
    return int(ok[0])


def encode_with_positions(tokenizer, item: dict, shots: list[dict] | None = None) -> dict:
    """Tokenise one rendered prompt and locate the three analysis positions.

    Returns dict(text, input_ids, pos={final, content_end, premise_end}) with positions as
    indices into input_ids (unpadded).
    """
    text = render(tokenizer, item, shots)
    prefix, ctx, q = user_parts(item)
    user = prefix + ctx + q
    start = text.rfind(user)  # rfind: with few-shot the last user turn is the target item
    assert start >= 0, "user text not found verbatim in templated prompt"
    ctx_end = start + len(prefix) + len(ctx)
    content_end = start + len(user)
    # The template already contains BOS where the model needs one -> add_special_tokens=False.
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    off = np.array(enc["offset_mapping"])
    ids = enc["input_ids"]
    pos = {
        "final": len(ids) - 1,
        "content_end": _token_covering(off, content_end - 1),
        "premise_end": _token_covering(off, ctx_end - 1),
    }
    assert pos["premise_end"] < pos["content_end"] < pos["final"]
    return {"text": text, "input_ids": ids, "pos": pos}


def pick_shots(pool: list[dict], k: int, seed: int, max_chars: int | None = None) -> list[dict]:
    """k balanced demonstrations (alternating Yes/No), deterministic given the seed.
    `max_chars` keeps demonstrations short so the few-shot prompt stays affordable."""
    rng = np.random.default_rng(seed)
    out = []
    for j in range(k):
        lab = 1 - (j % 2)
        cand = [p for p in pool if p["label"] == lab and p not in out
                and (max_chars is None or len(p["context"]) <= max_chars)]
        out.append(cand[int(rng.integers(len(cand)))])
    return out

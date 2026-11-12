"""Prompt tests: chat template applied, positions correct, answer tokens sane.
Tokenizer-only (no model weights needed)."""
import pytest

from causal_probe import data, models, prompts
from causal_probe.config import MODELS


@pytest.fixture(scope="module", params=list(MODELS))
def tok(request):
    try:
        return models.load_tokenizer(request.param)
    except Exception as e:  # e.g. gated model without login
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def items():
    return data.load_c2c("test")[::200] + data.cladder_sample()[::500]


def test_chat_template_applied(tok, items):
    for it in items:
        text = prompts.render(tok, it)
        assert prompts.user_text(it) in text
        assert text != prompts.user_text(it)                # template added something
        gen_header = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                             add_generation_prompt=True).split("x")[-1]
        assert "assistant" in gen_header or "model" in gen_header
        assert text.endswith(gen_header)                    # ends with the assistant header
        assert "Hypothesis: " + it["query"] in text if it["dataset"] == "c2c" else it["query"] in text


def test_no_double_bos(tok, items):
    enc = prompts.encode_with_positions(tok, items[0])
    if tok.bos_token_id is not None:
        assert enc["input_ids"].count(tok.bos_token_id) <= 1


def test_positions(tok, items):
    for it in items:
        enc = prompts.encode_with_positions(tok, it)
        ids, pos = enc["input_ids"], enc["pos"]
        assert pos["final"] == len(ids) - 1
        # decoding up to content_end reproduces the text up to the end of the query
        upto = tok.decode(ids[: pos["content_end"] + 1])
        assert upto.rstrip().endswith(it["query"].rstrip()[-25:])
        upto = tok.decode(ids[: pos["premise_end"] + 1])
        assert upto.rstrip().endswith(it["context"].rstrip()[-25:])
        assert it["query"][-15:] not in upto[-40:]


def test_fewshot_format(tok):
    pool = data.load_c2c("dev")
    shots = prompts.pick_shots(pool, 4, seed=0, max_chars=400)
    assert [s["label"] for s in shots] == [1, 0, 1, 0]
    assert len({s["uid"] for s in shots}) == 4
    item = data.load_c2c("test")[5]
    msgs = prompts.build_messages(item, shots)
    assert len(msgs) == 9 and msgs[1]["content"] == "Yes" and msgs[3]["content"] == "No"
    enc = prompts.encode_with_positions(tok, item, shots)
    upto = tok.decode(enc["input_ids"][: enc["pos"]["content_end"] + 1])
    assert upto.rstrip().endswith(item["query"][-25:])      # positions refer to the *target* item


def test_answer_tokens(tok):
    ans = models.answer_token_ids(tok)
    assert "Yes" in ans["yes"] and "No" in ans["no"]
    assert not (set(ans["yes"].values()) & set(ans["no"].values()))
    for w, i in {**ans["yes"], **ans["no"]}.items():
        assert tok.decode([i]) == w
